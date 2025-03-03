import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import arxiv
import requests
import random
import time

from olmocr.pipeline import process_pdf_batch
from olmocr.filter import PdfFilter
from olmocr.datatypes import Language

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

    def process_pdfs(self):
        """
        Process collected PDFs using olmocr pipeline
        """
        logger.info("Processing PDFs with olmocr pipeline")
        pdf_files = list(self.pdf_dir.glob("*.pdf"))

        if not pdf_files:
            logger.warning("No PDFs found for processing")
            return

        try:
            # Process PDFs in batches
            results = process_pdf_batch(
                pdf_paths=pdf_files,
                output_dir=str(self.processed_dir),
                content_filter=self.pdf_filter,
                model_name="allenai/olmOCR-7B-0225-preview"
            )
            logger.info(f"Successfully processed {len(results)} PDFs")
            return results
        except Exception as e:
            logger.error(f"Failed to process PDFs: {e}")
            return []

    def create_corpus(self):
        """
        Create training corpus from processed documents
        """
        logger.info("Creating training corpus")
        corpus_data = []

        # Read processed JSONL files
        for jsonl_file in self.processed_dir.glob("*.jsonl"):
            try:
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        doc_data = json.loads(line)

                        # Extract text from processed pages
                        text_content = ""
                        for page in doc_data.get("pages", []):
                            for block in page.get("blocks", []):
                                if block.get("type") == "text":
                                    text_content += block.get("text", "") + "\n"

                        # Match with metadata
                        doc_id = jsonl_file.stem
                        metadata = next((p for p in self.papers_metadata if str(p["id"]) in doc_id), {})

                        entry = {
                            "title": metadata.get("title", ""),
                            "authors": metadata.get("authors", []),
                            "abstract": metadata.get("abstract", ""),
                            "text": text_content,
                            "source": metadata.get("source", ""),
                            "id": metadata.get("id", ""),
                            "published": metadata.get("published", "")
                        }
                        corpus_data.append(entry)
            except Exception as e:
                logger.error(f"Failed to process {jsonl_file}: {e}")

        # Create train/val split
        if corpus_data:
            random.shuffle(corpus_data)
            split_idx = int(len(corpus_data) * 0.9)
            train_data = corpus_data[:split_idx]
            val_data = corpus_data[split_idx:]

            # Save splits
            with open(self.corpus_dir / "train_corpus.json", "w", encoding="utf-8") as f:
                json.dump(train_data, f, ensure_ascii=False, indent=2)
            with open(self.corpus_dir / "val_corpus.json", "w", encoding="utf-8") as f:
                json.dump(val_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Created corpus with {len(train_data)} training and {len(val_data)} validation examples")

    def run_pipeline(self, arxiv_queries: List[str], openreview_venues: List[str], pwc_topics: List[str]):
        """
        Run the complete pipeline
        """
        # Collect papers from all sources
        collected_count = 0

        # Collect from arXiv
        for query in arxiv_queries:
            if collected_count >= self.max_papers:
                break
            papers = self.collect_from_arxiv(query, max_results=min(50, self.max_papers - collected_count))
            collected_count += len(papers)

        # Collect from OpenReview
        for venue in openreview_venues:
            if collected_count >= self.max_papers:
                break
            papers = self.collect_from_openreview(venue, limit=min(50, self.max_papers - collected_count))
            collected_count += len(papers)

        # Collect from PapersWithCode
        for topic in pwc_topics:
            if collected_count >= self.max_papers:
                break
            papers = self.collect_from_paperswithcode(topic, limit=min(50, self.max_papers - collected_count))
            collected_count += len(papers)

        # Save metadata
        with open(self.output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.papers_metadata, f, ensure_ascii=False, indent=2)

        # Process PDFs with olmocr
        self.process_pdfs()

        # Create training corpus
        self.create_corpus()

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