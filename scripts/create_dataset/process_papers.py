import argparse
import json
import logging
from pathlib import Path
import asyncio
from tqdm import tqdm
from olmocr.pipeline import process_pdf
from olmocr.filter import PdfFilter
from enum import Enum

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
    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)
        self.pdf_dir = self.input_dir / "pdfs"
        self.processed_dir = self.input_dir / "processed"
        self.corpus_dir = self.input_dir / "corpus"
        self.train_dir = self.corpus_dir / "train"
        self.val_dir = self.corpus_dir / "val"
        self.test_dir = self.corpus_dir / "test"

        # Create directories
        for directory in [self.processed_dir, self.train_dir, self.val_dir, self.test_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Load metadata
        self.papers_metadata = []
        metadata_path = self.input_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.papers_metadata = json.load(f)

        # Initialize PDF filter
        self.pdf_filter = PdfFilter(
            languages_to_keep={Language.ENGLISH},  # Keep English documents
            apply_form_check=True,  # Filter out forms
            apply_download_spam_check=True  # Filter out spam PDFs
        )

    async def initialize_sglang_server(self):
            """Initialize the sglang server for OCR processing"""
            from olmocr.pipeline import sglang_server_ready

            # Wait for server to be ready
            try:
                await sglang_server_ready()
                logger.info("sglang server initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize sglang server: {e}")
                raise

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

                    # Extract metadata
                    metadata = next(
                        (item for item in self.papers_metadata
                         if any(pdf_id.endswith(str(item['id']))
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

        # Run processing
        results = asyncio.run(process_all_pdfs())
        logger.info(f"Successfully processed {len(results)} PDFs")

    def create_corpus(self):
        """Create training corpus from processed documents."""
        logger.info("Creating training, validation and test corpora from processed documents...")

        corpus_data = []
        processed_files = list(self.processed_dir.glob("*.json"))

        for proc_file in tqdm(processed_files, desc="Creating corpus"):
            try:
                with open(proc_file, 'r', encoding='utf-8') as f:
                    doc = json.load(f)

                doc_id = proc_file.stem
                metadata = next(
                    (item for item in self.papers_metadata
                    if any(doc_id.endswith(str(item['id']))
                    for source in ['arxiv_', 'openreview_', 'pwc_'])),
                    None
                )

                if metadata and doc.get('text'):
                    corpus_entry = {
                        'id': doc['id'],
                        'text': doc['text'],
                        'metadata': {
                            **doc['metadata'],
                            'title': metadata.get('title', ''),
                            'authors': metadata.get('authors', []),
                            'abstract': metadata.get('abstract', ''),
                            'published_date': metadata.get('published', ''),
                            'source': metadata.get('source', '')
                        }
                    }
                    corpus_data.append(corpus_entry)

            except Exception as e:
                logger.error(f"Error processing {proc_file}: {e}")

        # Split into train/val/test
        total = len(corpus_data)
        train_size = int(0.8 * total)
        val_size = int(0.1 * total)

        train_data = corpus_data[:train_size]
        val_data = corpus_data[train_size:train_size+val_size]
        test_data = corpus_data[train_size+val_size:]

        # Save splits
        with open(self.train_dir / "train.json", 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

        with open(self.val_dir / "val.json", 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)

        with open(self.test_dir / "test.json", 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Created train corpus with {len(train_data)} documents")
        logger.info(f"Created validation corpus with {len(val_data)} documents")
        logger.info(f"Created test corpus with {len(test_data)} documents")

    def run_pipeline(self):
        """
        Run the processing pipeline
        """
        logger.info("Starting processing pipeline...")

        # Process PDFs with olmocr
        self.process_pdfs()

        # Create training corpus
        self.create_corpus()

        logger.info("Processing pipeline completed successfully")

def main():
    parser = argparse.ArgumentParser(
        description="Process downloaded papers and create corpus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing downloaded papers"
    )
    parser.add_argument(
        "--skip_processing",
        action="store_true",
        help="Skip PDF processing and only create corpus from existing processed files"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output"
    )

    args = parser.parse_args()

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

        processor = PaperProcessor(args.input_dir)
        if not args.skip_processing:
            processor.process_pdfs()
        processor.create_corpus()
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
