
# ML Research Papers Dataset

## Dataset Description

- **Source**: Scientific papers extracted from research repositories
- **Format**: causal_lm
- **Size**: 301 documents (240 train, 30 validation, 31 test)
- **Created**: 2025-03-04 15:29:18

## Usage

```python
from datasets import load_from_disk

# Load the dataset
dataset = load_from_disk("dataset/hf_dataset")

# Access splits
train_data = dataset["train"]
validation_data = dataset["validation"]
test_data = dataset["test"]
```

## Dataset Structure

```
causal_lm format with fields:
['text']
```

## Data Fields

- `text`: The full text content of the paper
- Additional metadata fields as available
                