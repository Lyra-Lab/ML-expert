# %% Imports
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from datasets import load_from_disk

# %% Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
OUTPUT_DIR = "deepseek-finetuned"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
BATCH_SIZE = 4
MICRO_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
MAX_SEQ_LENGTH = 512
LEARNING_RATE = 3e-4
NUM_EPOCHS = 3
WARMUP_STEPS = 100

# Data paths
DATA_DIR = 'data'  # Base directory where your processed data is stored
HF_DATASET_DIR = f"{DATA_DIR}/hf_dataset"  # Directory with HuggingFace dataset

# %% Custom Dataset class for HuggingFace datasets
class HFDataset(Dataset):
    def __init__(self, dataset, tokenizer, split="train", max_length=MAX_SEQ_LENGTH):
        self.dataset = dataset[split]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get text from the dataset
        text = self.dataset[idx]['text']

        # Tokenize the text
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            'input_ids': tokenized.input_ids[0],
            'attention_mask': tokenized.attention_mask[0],
            'labels': tokenized.input_ids[0].clone()
        }

# %% PyTorch Lightning module for fine-tuning
class DeepSeekFineTuner(pl.LightningModule):
    def __init__(self, model_name, lora_config, learning_rate, warmup_steps, total_steps):
        super().__init__()
        self.save_hyperparameters()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model with 4-bit quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,
            device_map="auto"
        )

        # Prepare model for LoRA training
        self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA configuration
        self.model = get_peft_model(self.model, lora_config)

        # Set model to training mode
        self.model.train()

        # Learning rate and warmup steps for scheduler
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        val_loss = outputs.loss
        self.log('val_loss', val_loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

# %% Main function
def main():
    # Set up LoRA configuration
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Make sure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the HuggingFace dataset
    try:
        print(f"Loading HuggingFace dataset from {HF_DATASET_DIR}")
        hf_dataset = load_from_disk(HF_DATASET_DIR)
        print(f"Dataset loaded with splits: {hf_dataset.keys()}")

        # Create datasets
        train_dataset = HFDataset(hf_dataset, tokenizer, split="train")
        print("Number of training samples:", len(train_dataset))

        val_dataset = HFDataset(hf_dataset, tokenizer, split="validation")
        print("Number of validation samples:", len(val_dataset))

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please make sure the HuggingFace dataset is properly created.")
        return

    # Create data loaders
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=data_collator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=data_collator
    )

    # Calculate total steps
    total_steps = len(train_loader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS

    # Create PyTorch Lightning model
    model = DeepSeekFineTuner(
        model_name=MODEL_NAME,
        lora_config=lora_config,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        total_steps=total_steps
    )

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(OUTPUT_DIR, 'checkpoints'),
        filename='deepseek-finetuned-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Set up logger
    logger = TensorBoardLogger(save_dir=OUTPUT_DIR, name='logs')

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator='gpu',
        devices=1,
        precision='bf16-mixed',
        gradient_clip_val=1.0,
        accumulate_grad_batches=GRADIENT_ACCUMULATION_STEPS,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10
    )

    # Start training
    trainer.fit(model, train_loader, val_loader)

    # Save the final model
    model.model.save_pretrained(os.path.join(OUTPUT_DIR, 'final_model'))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, 'final_model'))

    print(f"Model saved to {os.path.join(OUTPUT_DIR, 'final_model')}")

if __name__ == "__main__":
    main()
