import argparse
import json
import logging
from pathlib import Path
import asyncio
import subprocess
import time
import os
from tqdm import tqdm
import random
from olmocr.pipeline import process_pdf
from olmocr.filter import PdfFilter
from enum import Enum
try:
    from datasets import Dataset, DatasetDict
except ImportError:
    logging.warning("HuggingFace datasets library not found. Install with 'pip install datasets'")

class Language(Enum):
    ENGLISH = "english"

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

        # Initialize PDF filter
        self.pdf_filter = PdfFilter(
            languages_to_keep={Language.ENGLISH},  # Keep English documents
            apply_form_check=True,  # Filter out forms
            apply_download_spam_check=True  # Filter out spam PDFs
        )

        # Initialize variables for sglang server
        self.sglang_process = None
        self.sglang_port = 30000  # Default port

    async def start_sglang_server(self):
        """Start the sglang server if not already running"""
        logger.info("Starting sglang server...")

        # Check if server is already running
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            result = s.connect_ex(('localhost', self.sglang_port))
            s.close()

            if result == 0:
                logger.info("sglang server is already running")
                return True
        except Exception as e:
            logger.warning(f"Error checking server status: {e}")

        # Try to start the server
        try:
            # Set environment variables
            os.environ["SGLANG_DISABLE_MODEL_UPLOAD"] = "1"

            # Start the server process
            self.sglang_process = subprocess.Popen(
                ["python", "-m", "sglang.launch_server", "--port", str(self.sglang_port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            logger.info(f"sglang server process started with PID {self.sglang_process.pid}")

            # Give it time to start
            await asyncio.sleep(10)

            # Check if it's running
            for i in range(5):
                try:
                    from olmocr.pipeline import sglang_server_ready
                    await sglang_server_ready()
                    logger.info("sglang server is ready")
                    return True
                except Exception as e:
                    logger.warning(f"Server not ready yet (attempt {i+1}/5): {e}")
                    await asyncio.sleep(5)

            # If we get here, server couldn't be started or verified
            logger.warning("Could not verify sglang server is running, but will try to continue")
            return True  # Return True anyway to attempt processing

        except Exception as e:
            logger.error(f"Failed to start sglang server: {e}")
            return False

    async def initialize_sglang_server(self):
        """Initialize the sglang server for OCR processing"""
        await self.start_sglang_server()  # Start our own server

        from olmocr.pipeline import sglang_server_ready

        # Try to connect to the server
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                await sglang_server_ready()
                logger.info("sglang server connection established")
                return
            except Exception as e:
                logger.warning(f"Failed to connect to sglang server (attempt {attempt+1}/{max_attempts}): {e}")
                if attempt < max_attempts - 1:
                    logger.info("Restarting sglang server...")
                    await self.start_sglang_server()
                    await asyncio.sleep(5)

        logger.error("All attempts to connect to sglang server failed")
        raise RuntimeError("Could not connect to sglang server after multiple attempts")

    async def process_single_pdf(self, pdf_path):
        """Helper function to process a single PDF"""
        try:
            # Get the base name for output files
            pdf_id = pdf_path.stem
            output_path = self.processed_dir / f"{pdf_id}.json"

            # Skip if already processed
            if output_path.exists():
                logger.debug(f"Skipping already processed PDF: {pdf_path}")
                return

            # Validate PDF before processing
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(str(pdf_path))
                num_pages = len(reader.pages)
                if num_pages == 0:
                    logger.warning(f"Skipping empty PDF: {pdf_path}")
                    return
            except Exception as e:
                logger.warning(f"Invalid or corrupt PDF {pdf_path}: {e}")
                return

            # Process PDF using olmocr pipeline
            result = await process_pdf(
                str(pdf_path),
                worker_id=0,
                pdf_orig_path=str(pdf_path),
            )

            if result is None:
                logger.warning(f"Failed to process PDF: {pdf_path}")
                return

            # Extract text and validate content
            text = result.get('text', '').strip()
            if not text:
                logger.warning(f"No text content extracted from {pdf_path}")
                return

            # Find matching metadata if available
            metadata = {}
            if self.papers_metadata:
                metadata = next(
                    (item for item in self.papers_metadata
                     if any(pdf_id.endswith(str(item.get('id', '')))
                     for source in ['arxiv_', 'openreview_', 'pwc_'])),
                    {}
                )

            # Create document structure
            document = {
                'id': pdf_id,
                'text': text,
                'metadata': {
                    'source_file': str(pdf_path),
                    'total_pages': result.get('pdf-total-pages', 0),
                    'processed_pages': len(result.get('page_responses', [])),
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

            logger.info(f"Successfully processed PDF: {pdf_path}")
            return document

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return None

    def process_pdfs(self):
        """Process collected PDFs using olmocr pipeline."""
        logger.info("Processing PDFs with olmocr...")

        # Get list of PDFs to process
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDFs to process")

        if len(pdf_files) == 0:
            logger.warning(f"No PDF files found in {self.pdf_dir}")
            return []

        # Process PDFs using asyncio
        async def process_all_pdfs():
            # Initialize sglang server first
            await self.initialize_sglang_server()

            # Process in smaller batches to manage memory better
            batch_size = 5  # Reduced batch size for better stability
            results = []

            for i in range(0, len(pdf_files), batch_size):
                batch = pdf_files[i:i + batch_size]
                tasks = [asyncio.create_task(self.process_single_pdf(pdf_file))
                        for pdf_file in batch]

                # Process batch with progress bar
                for task in tqdm(
                    asyncio.as_completed(tasks),
                    total=len(tasks),
                    desc=f"Processing PDFs (batch {i//batch_size + 1}/{len(pdf_files)//batch_size + 1})"
                ):
                    try:
                        result = await task
                        if result:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Batch processing error: {e}")
                        continue

                # Small delay between batches to prevent resource exhaustion
                await asyncio.sleep(2)

            return results

        try:
            # Run processing
            results = asyncio.run(process_all_pdfs())
            logger.info(f"Successfully processed {len(results)} PDFs")
            return results
        finally:
            # Clean up sglang server process if we started it
            if self.sglang_process is not None:
                try:
                    logger.info("Stopping sglang server...")
                    self.sglang_process.terminate()
                    self.sglang_process.wait(timeout=10)
                    logger.info("sglang server stopped")
                except Exception as e:
                    logger.warning(f"Error stopping sglang server: {e}")
                    try:
                        self.sglang_process.kill()
                    except:
                        pass

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
                import datasets
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
            "hf_train_jsonl": str(train_jsonl_path) if 'train_jsonl_path' in locals() else None,
            "hf_val_jsonl": str(val_jsonl_path) if 'val_jsonl_path' in locals() else None,
            "hf_test_jsonl": str(test_jsonl_path) if 'test_jsonl_path' in locals() else None
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

        # Process PDFs with olmocr
        self.process_pdfs()

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
        # Check required dependencies
        from olmocr.check import (
            check_poppler_version,
            check_sglang_version,
            check_torch_gpu_available
        )

        # Verify requirements
        check_poppler_version()
        check_sglang_version()
        check_torch_gpu_available()

        processor = PaperProcessor(args.base_dir)
        if not args.skip_processing:
            processor.process_pdfs()
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
