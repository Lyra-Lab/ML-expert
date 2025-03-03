import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Set
from tqdm import tqdm
import arxiv
import requests
import random
import time
import asyncio
from olmocr.pipeline import build_dolma_document, process_pdf
from olmocr.filter import PdfFilter

from enum import Enum
class Language(Enum):
    ENGLISH = "english"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("paper_collection.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class MLResourceCollector:
    """
    Collects machine learning research papers and processes them using olmocr.
    """
    def __init__(self, output_dir: str, max_papers: int = 100):
        self.output_dir = Path(output_dir)
        self.pdf_dir = self.output_dir / "pdfs"
        self.processed_dir = self.output_dir / "processed"
        self.corpus_dir = self.output_dir / "corpus"

        # Create directories
        for directory in [self.output_dir, self.pdf_dir, self.processed_dir, self.corpus_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        self.max_papers = max_papers
        self.papers_metadata = []

        # Initialize PDF filter
        self.pdf_filter = PdfFilter(
            languages_to_keep={Language.ENGLISH},
            apply_form_check=True
        )

    def collect_from_arxiv(self, query: str, max_results: int = 50) -> List[Dict]:
        logger.info(f"Collecting papers from arXiv with query: {query}")
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        papers = []
        for result in tqdm(client.results(search), desc="Collecting arXiv papers"):
            paper_info = {
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "abstract": result.summary,
                "pdf_url": result.pdf_url,
                "published": result.published.strftime("%Y-%m-%d"),
                "source": "arxiv",
                "id": result.entry_id.split("/")[-1],
                "categories": result.categories
            }
            papers.append(paper_info)

            # Download PDF
            pdf_path = self.pdf_dir / f"arxiv_{paper_info['id']}.pdf"
            if not pdf_path.exists():
                try:
                    response = requests.get(result.pdf_url)
                    with open(pdf_path, "wb") as f:
                        f.write(response.content)
                    logger.info(f"Downloaded: {pdf_path}")
                    time.sleep(random.uniform(1, 3))
                except Exception as e:
                    logger.error(f"Failed to download {result.pdf_url}: {e}")

        self.papers_metadata.extend(papers)
        return papers

    def collect_from_openreview(self, venue: str, limit: int = 50) -> List[Dict]:
        logger.info(f"Collecting papers from OpenReview with venue: {venue}")
        base_url = "https://api.openreview.net/notes"
        params = {
            "content.venue": venue,
            "details": "replyCount,directReplyCount",
            "sort": "cdate",
            "limit": limit,
            "offset": 0
        }
        papers = []
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            for note in tqdm(data.get("notes", []), desc="Collecting OpenReview papers"):
                content = note.get("content", {})
                paper_info = {
                    "title": content.get("title", ""),
                    "authors": content.get("authors", []),
                    "abstract": content.get("abstract", ""),
                    "pdf_url": f"https://openreview.net/pdf?id={note['id']}",
                    "published": note.get("cdate", ""),
                    "source": "openreview",
                    "id": note["id"],
                    "venue": venue
                }
                papers.append(paper_info)

                # Download PDF
                pdf_path = self.pdf_dir / f"openreview_{paper_info['id']}.pdf"
                if not pdf_path.exists():
                    try:
                        response = requests.get(paper_info["pdf_url"])
                        with open(pdf_path, "wb") as f:
                            f.write(response.content)
                        logger.info(f"Downloaded: {pdf_path}")
                        time.sleep(random.uniform(1, 3))
                    except Exception as e:
                        logger.error(f"Failed to download {paper_info['pdf_url']}: {e}")
        except Exception as e:
            logger.error(f"Failed to collect from OpenReview: {e}")

        self.papers_metadata.extend(papers)
        return papers

    def collect_from_paperswithcode(self, topic: str, limit: int = 50) -> List[Dict]:
        logger.info(f"Collecting papers from PapersWithCode with topic: {topic}")
        url = f"https://paperswithcode.com/api/v1/papers/?topics={topic}&limit={limit}"
        papers = []
        try:
            response = requests.get(url)
            data = response.json()
            for paper in tqdm(data.get("results", []), desc="Collecting PapersWithCode papers"):
                paper_info = {
                    "title": paper.get("title", ""),
                    "authors": [author.get("name", "") for author in paper.get("authors", [])],
                    "abstract": paper.get("abstract", ""),
                    "pdf_url": paper.get("arxiv_url", "").replace("abs", "pdf") + ".pdf" if paper.get("arxiv_url") else "",
                    "published": paper.get("published", ""),
                    "source": "paperswithcode",
                    "id": paper.get("id", ""),
                    "repository_url": paper.get("repository_url", "")
                }
                if paper_info["pdf_url"]:
                    papers.append(paper_info)

                    # Download PDF
                    pdf_path = self.pdf_dir / f"pwc_{paper_info['id']}.pdf"
                    if not pdf_path.exists() and paper_info["pdf_url"]:
                        try:
                            response = requests.get(paper_info["pdf_url"])
                            with open(pdf_path, "wb") as f:
                                f.write(response.content)
                            logger.info(f"Downloaded: {pdf_path}")
                            time.sleep(random.uniform(1, 3))
                        except Exception as e:
                            logger.error(f"Failed to download {paper_info['pdf_url']}: {e}")
        except Exception as e:
            logger.error(f"Failed to collect from PapersWithCode: {e}")

        self.papers_metadata.extend(papers)
        return papers

    async def process_single_pdf(self, pdf_path):
        """Helper function to process a single PDF"""
        try:
            dolma_doc = await process_pdf(
                args=type('Args', (), {
                    'target_longest_image_dim': 1024,
                    'target_anchor_text_len': 6000,
                    'max_page_retries': 3,
                    'max_page_error_rate': 0.004,
                    'model_max_context': 8192,
                    'apply_filter': True
                }),
                worker_id=0,
                pdf_orig_path=str(pdf_path)
            )

            if dolma_doc:
                output_path = self.processed_dir / f"{pdf_path.stem}.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(dolma_doc, f, ensure_ascii=False, indent=2)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")
            return False

    def process_pdfs(self):
        """Process collected PDFs using olmocr pipeline."""
        logger.info("Processing PDFs with olmocr...")

        async def process_all_pdfs():
            pdf_files = list(self.pdf_dir.glob("*.pdf"))
            logger.info(f"Found {len(pdf_files)} PDFs to process")

            tasks = []
            async with asyncio.TaskGroup() as tg:
                for pdf_file in pdf_files:
                    task = tg.create_task(self.process_single_pdf(pdf_file))
                    tasks.append(task)

            successful = sum(1 for task in tasks if task.result())
            logger.info(f"Successfully processed {successful} out of {len(pdf_files)} PDFs")

        asyncio.run(process_all_pdfs())

    def create_corpus(self):
        """Create training corpus from processed documents."""
        logger.info("Creating training corpus from processed documents...")

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

        corpus_file = self.corpus_dir / "ml_papers_corpus.json"
        with open(corpus_file, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Created corpus with {len(corpus_data)} documents at {corpus_file}")

    def run_pipeline(self, arxiv_queries: List[str], openreview_venues: List[str], pwc_topics: List[str]):
        """
        Run the complete pipeline
        """
        logger.info("Starting pipeline run...")

        # Collect papers from all sources with balanced distribution
        papers_per_source = self.max_papers // 3
        collected_count = 0

        # Collect from arXiv
        for query in arxiv_queries:
            if collected_count >= papers_per_source:
                break
            papers = self.collect_from_arxiv(query, max_results=min(50, papers_per_source - collected_count))
            collected_count += len(papers)
            logger.info(f"Collected {len(papers)} papers from arXiv query: {query}")

        # Reset counter for OpenReview
        collected_count = 0
        for venue in openreview_venues:
            if collected_count >= papers_per_source:
                break
            papers = self.collect_from_openreview(venue, limit=min(50, papers_per_source - collected_count))
            collected_count += len(papers)
            logger.info(f"Collected {len(papers)} papers from OpenReview venue: {venue}")

        # Reset counter for PapersWithCode
        collected_count = 0
        for topic in pwc_topics:
            if collected_count >= papers_per_source:
                break
            papers = self.collect_from_paperswithcode(topic, limit=min(50, papers_per_source - collected_count))
            collected_count += len(papers)
            logger.info(f"Collected {len(papers)} papers from PapersWithCode topic: {topic}")

        total_papers = len(self.papers_metadata)
        logger.info(f"Total papers collected: {total_papers}")

        # Save metadata
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.papers_metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")

        # Process PDFs with olmocr
        logger.info("Starting PDF processing...")
        self.process_pdfs()

        # Create training corpus
        logger.info("Creating training corpus...")
        self.create_corpus()

        logger.info("Pipeline completed successfully")

def main():
    parser = argparse.ArgumentParser(description="Collect ML research papers and create a training corpus")
    parser.add_argument("--output_dir", type=str, default="ml_corpus",
                        help="Output directory for collected papers")
    parser.add_argument("--max_papers", type=int, default=100,
                        help="Maximum number of papers to collect")
    parser.add_argument("--arxiv_queries", type=str, nargs="+",
                        default=["Large Language Models", "Transformers", "Deep Learning"],
                        help="List of arXiv queries")
    parser.add_argument("--openreview_venues", type=str, nargs="+",
                        default=["ICLR 2023", "NeurIPS 2023"],
                        help="List of OpenReview venues")
    parser.add_argument("--pwc_topics", type=str, nargs="+",
                        default=["Language Modeling", "Transformers", "Fine-tuning"],
                        help="List of PapersWithCode topics")

    args = parser.parse_args()

    collector = MLResourceCollector(args.output_dir, args.max_papers)
    collector.run_pipeline(args.arxiv_queries, args.openreview_venues, args.pwc_topics)

if __name__ == "__main__":
    main()
