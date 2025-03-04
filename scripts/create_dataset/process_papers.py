import argparse
import json
import logging
from pathlib import Path
import time
from tqdm import tqdm
import random
try:
    from datasets import Dataset, DatasetDict
except ImportError:
    logging.warning("HuggingFace datasets library not found. Install with 'pip install datasets'")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("paper_processing.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class PaperProcessor:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.pdf_dir = self.base_dir / "pdfs"  # Input PDFs should be here

        # Output directories
        self.processed_dir = self.base_dir / "processed"
        self.dataset_dir = self.base_dir / "dataset"
        self.hf_dataset_dir = self.base_dir / "hf_dataset"

        # Train/val/test split directories
        self.train_dir = self.dataset_dir / "train"
        self.val_dir = self.dataset_dir / "val"
        self.test_dir = self.dataset_dir / "test"

        # Create all required directories
        for directory in [self.processed_dir, self.dataset_dir, self.hf_dataset_dir,
                         self.train_dir, self.val_dir, self.test_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Check if PDF input directory exists
        if not self.pdf_dir.exists():
            logger.error(f"PDF directory not found: {self.pdf_dir}")
            raise FileNotFoundError(f"PDF directory not found: {self.pdf_dir}")

        # Load metadata if exists
        self.papers_metadata = []
        metadata_path = self.base_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.papers_metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.papers_metadata)} papers")
        else:
            logger.warning(f"No metadata file found at {metadata_path}. Processing will continue without metadata.")

    def process_pdfs_simple(self):
        """Process PDFs using PyPDF2 instead of olmocr."""
        logger.info("Processing PDFs with PyPDF2 (simple approach)...")

        try:
            from PyPDF2 import PdfReader
        except ImportError:
            logger.error("PyPDF2 is not installed. Install with 'pip install PyPDF2'")
            return []

        # Get list of PDFs to process
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDFs to process")

        if len(pdf_files) == 0:
            logger.warning(f"No PDF files found in {self.pdf_dir}")
            return []

        processed_docs = []

        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                # Get the base name for output files
                pdf_id = pdf_path.stem
                output_path = self.processed_dir / f"{pdf_id}.json"

                # Skip if already processed
                if output_path.exists():
                    logger.debug(f"Skipping already processed PDF: {pdf_path}")
                    continue

                # Extract text with PyPDF2
                reader = PdfReader(str(pdf_path))
                text = ""
                total_pages = len(reader.pages)

                if total_pages == 0:
                    logger.warning(f"Skipping empty PDF: {pdf_path}")
                    continue

                # Extract text from each page
                for page_num in range(total_pages):
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"

                if not text.strip():
                    logger.warning(f"No text content extracted from {pdf_path}")
                    continue

                # Find matching metadata if available
                metadata = {}
                if self.papers_metadata:
                    pdf_id_str = str(pdf_id)
                    metadata = next(
                        (item for item in self.papers_metadata 
                         if pdf_id_str.endswith(str(item.get('id', '')))),
                        {}
                    )

                # Create document structure
                document = {
                    'id': pdf_id,
                    'text': text,
                    'metadata': {
                        'source_file': str(pdf_path),
                        'total_pages': total_pages,
                        'title': metadata.get('title', ''),
                        'authors': metadata.get('authors', []),
                        'abstract': metadata.get('abstract', ''),
                        'published_date': metadata.get('published', ''),
                        'source': metadata.get('source', '')
                    }
                }

                # Save processed document
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(document, f, ensure_ascii=False, indent=2)

                processed_docs.append(document)
                logger.info(f"Successfully processed PDF: {pdf_path}")

            except Exception as e:
                logger.error(f"Error processing PDF {pdf_path}: {str(e)}")

        logger.info(f"Successfully processed {len(processed_docs)} PDFs")
        return processed_docs

    def create_dataset(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, format_type="causal_lm"):
        """Create dataset splits from processed documents."""
        logger.info("Creating train/val/test datasets from processed documents...")

        # Verify ratios add up to 1.0
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            logger.warning("Train, validation, and test ratios do not sum to 1.0. Normalizing...")
            total = train_ratio + val_ratio + test_ratio
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total

        # Collect processed documents
        corpus_data = []
        processed_files = list(self.processed_dir.glob("*.json"))

        if len(processed_files) == 0:
            logger.warning(f"No processed files found in {self.processed_dir}")
            return

        logger.info(f"Found {len(processed_files)} processed documents")

        for proc_file in tqdm(processed_files, desc="Loading processed files"):
            try:
                with open(proc_file, 'r', encoding='utf-8') as f:
                    doc = json.load(f)

                if doc.get('text'):  # Make sure we have text content
                    corpus_data.append(doc)
                else:
                    logger.warning(f"Skipping document without text: {proc_file}")

            except Exception as e:
                logger.error(f"Error loading {proc_file}: {e}")

        if not corpus_data:
            logger.error("No valid documents found. Cannot create dataset.")
            return

        # Shuffle data for random split
        random.shuffle(corpus_data)

        # Calculate split sizes
        total = len(corpus_data)
        train_size = int(train_ratio * total)
        val_size = int(val_ratio * total)

        # Split into train/val/test
        train_data = corpus_data[:train_size]
        val_data = corpus_data[train_size:train_size+val_size]
        test_data = corpus_data[train_size+val_size:]

        # Save splits in traditional format
        with open(self.train_dir / "train_corpus.json", 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

        with open(self.val_dir / "val_corpus.json", 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)

        with open(self.test_dir / "test_corpus.json", 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        # Create HuggingFace dataset format
        try:
            # Format the data based on the desired output format
            if format_type == "causal_lm":
                # For causal language modeling, we just need the text
                train_formatted = [{"text": item["text"]} for item in train_data]
                val_formatted = [{"text": item["text"]} for item in val_data]
                test_formatted = [{"text": item["text"]} for item in test_data]
            elif format_type == "summarization":
                # For summarization, use abstract as target and full text as input
                train_formatted = [
                    {"document": item["text"], "summary": item["metadata"]["abstract"]} 
                    for item in train_data if item["metadata"].get("abstract")
                ]
                val_formatted = [
                    {"document": item["text"], "summary": item["metadata"]["abstract"]} 
                    for item in val_data if item["metadata"].get("abstract")
                ]
                test_formatted = [
                    {"document": item["text"], "summary": item["metadata"]["abstract"]} 
                    for item in test_data if item["metadata"].get("abstract")
                ]
            else:
                # Default to basic text format
                train_formatted = train_data
                val_formatted = val_data
                test_formatted = test_data

            # Save as JSONL (one JSON object per line) for HuggingFace compatibility
            train_jsonl_path = self.hf_dataset_dir / "train.jsonl"
            val_jsonl_path = self.hf_dataset_dir / "validation.jsonl"
            test_jsonl_path = self.hf_dataset_dir / "test.jsonl"

            # Write JSONL files
            with open(train_jsonl_path, 'w', encoding='utf-8') as f:
                for item in train_formatted:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            with open(val_jsonl_path, 'w', encoding='utf-8') as f:
                for item in val_formatted:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            with open(test_jsonl_path, 'w', encoding='utf-8') as f:
                for item in test_formatted:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            # Try to create HuggingFace datasets object
            try:
                from datasets import Dataset, DatasetDict

                # Create dataset dictionary from JSONL files
                dataset_dict = DatasetDict({
                    "train": Dataset.from_json(str(train_jsonl_path)),
                    "validation": Dataset.from_json(str(val_jsonl_path)),
                    "test": Dataset.from_json(str(test_jsonl_path))
                })

                # Save the dataset in HuggingFace's .arrow format
                dataset_dict.save_to_disk(self.hf_dataset_dir)
                logger.info(f"HuggingFace dataset saved to {self.hf_dataset_dir}")

                # Create dataset card (README.md)
                creation_time = time.strftime('%Y-%m-%d %H:%M:%S')
                dataset_card = f"""
# ML Research Papers Dataset

## Dataset Description

- **Source**: Scientific papers extracted from research repositories
- **Format**: {format_type}
- **Size**: {total} documents ({len(train_formatted)} train, {len(val_formatted)} validation, {len(test_formatted)} test)
- **Created**: {creation_time}

## Usage

```python
from datasets import load_from_disk

# Load the dataset
dataset = load_from_disk("{self.hf_dataset_dir}")

# Access splits
train_data = dataset["train"]
validation_data = dataset["validation"]
test_data = dataset["test"]
```

## Dataset Structure

```
{format_type} format with fields:
{list(train_formatted[0].keys()) if train_formatted else "No fields available"}
```

## Data Fields

- `text`: The full text content of the paper
- Additional metadata fields as available
                """

                with open(self.hf_dataset_dir / "README.md", 'w', encoding='utf-8') as f:
                    f.write(dataset_card)

            except ImportError:
                logger.warning("HuggingFace datasets package not available. Only JSONL files created.")
                logger.info("To install: pip install datasets")
            except Exception as e:
                logger.error(f"Error creating HuggingFace dataset: {e}")

        except Exception as e:
            logger.error(f"Error creating HuggingFace format: {e}")

        # Create the combined dataset info file
        combined_path = self.dataset_dir / "dataset_info.json"
        train_jsonl_str = str(train_jsonl_path) if 'train_jsonl_path' in locals() else None
        val_jsonl_str = str(val_jsonl_path) if 'val_jsonl_path' in locals() else None
        test_jsonl_str = str(test_jsonl_path) if 'test_jsonl_path' in locals() else None

        dataset_info = {
            "dataset_path": str(self.dataset_dir),
            "hf_dataset_path": str(self.hf_dataset_dir),
            "train_file": str(self.train_dir / "train_corpus.json"),
            "val_file": str(self.val_dir / "val_corpus.json"),
            "test_file": str(self.test_dir / "test_corpus.json"),
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
            "total_samples": total,
            "format_type": format_type,
            "hf_train_jsonl": train_jsonl_str,
            "hf_val_jsonl": val_jsonl_str,
            "hf_test_jsonl": test_jsonl_str
        }

        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)

        logger.info("Created dataset with:")
        logger.info(f"  - Train split: {len(train_data)} documents")
        logger.info(f"  - Validation split: {len(val_data)} documents")
        logger.info(f"  - Test split: {len(test_data)} documents")
        logger.info(f"Dataset info saved to: {combined_path}")
        logger.info(f"HuggingFace-ready dataset saved to: {self.hf_dataset_dir}")

    def run_pipeline(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, format_type="causal_lm"):
        """
        Run the complete processing pipeline
        """
        logger.info(f"Starting processing pipeline in {self.base_dir}...")

        # Process PDFs with PyPDF2 (simple approach)
        self.process_pdfs_simple()

        # Create dataset
        self.create_dataset(train_ratio, val_ratio, test_ratio, format_type)

        logger.info("Processing pipeline completed successfully")
        logger.info(f"Dataset created in {self.dataset_dir}")
        logger.info(f"HuggingFace-ready dataset created in {self.hf_dataset_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Process PDFs and create a dataset for ML training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory to process. Should contain a 'pdfs/' folder with input files."
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of data to use for training (0.0-1.0)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio of data to use for validation (0.0-1.0)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Ratio of data to use for testing (0.0-1.0)"
    )
    parser.add_argument(
        "--skip_processing",
        action="store_true",
        help="Skip PDF processing and only create dataset from existing processed files"
    )
    parser.add_argument(
        "--format_type",
        type=str,
        default="causal_lm",
        choices=["causal_lm", "summarization", "basic"],
        help="Format type for HuggingFace dataset"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push dataset to HuggingFace Hub (requires login)"
    )
    parser.add_argument(
        "--hub_name",
        type=str,
        help="Dataset name on HuggingFace Hub (username/dataset-name)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible dataset splits"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output"
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Check for PyPDF2
        try:
            from PyPDF2 import PdfReader
            logger.info("PyPDF2 is available")
        except ImportError:
            logger.error("PyPDF2 is required but not installed. Install with: pip install PyPDF2")
            return

        processor = PaperProcessor(args.base_dir)
        if not args.skip_processing:
            processor.process_pdfs_simple()
        processor.create_dataset(args.train_ratio, args.val_ratio, args.test_ratio, args.format_type)

        # Push to HuggingFace Hub if requested
        if args.push_to_hub and args.hub_name:
            try:
                from huggingface_hub import HfApi
                from datasets import load_from_disk

                # Load the dataset from disk
                dataset_path = processor.hf_dataset_dir
                dataset = load_from_disk(dataset_path)

                # Push to Hub
                dataset.push_to_hub(args.hub_name)

                # Upload README separately to ensure it's properly displayed
                api = HfApi()
                api.upload_file(
                    path_or_fileobj=str(dataset_path / "README.md"),
                    path_in_repo="README.md",
                    repo_id=args.hub_name,
                    repo_type="dataset"
                )

                logger.info(f"Dataset successfully pushed to HuggingFace Hub: {args.hub_name}")
            except ImportError:
                logger.error("Missing dependencies for HuggingFace Hub upload. Install with: pip install huggingface_hub datasets")
            except Exception as e:
                logger.error(f"Failed to push to HuggingFace Hub: {e}")

        logger.info(f"Dataset successfully created in {processor.dataset_dir}")
        logger.info("You can now use this dataset for training your model.")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
