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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("paper_collection.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class ArxivPaperDownloader:
    def __init__(self, output_dir: str, max_papers: int = 100):
        self.output_dir = Path(output_dir)
        self.pdf_dir = self.output_dir / "pdfs"

        # Create directories
        for directory in [self.output_dir, self.pdf_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        self.max_papers = max_papers
        self.papers_metadata = []

    def collect_from_arxiv(self, query: str, max_results: int = 50) -> List[Dict]:
        """Collect papers from arXiv based on the provided query."""
        logger.info(f"Collecting papers from arXiv with query: {query}")
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        papers = []
        for result in tqdm(client.results(search), desc=f"Collecting arXiv papers for '{query}'"):
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
                    time.sleep(random.uniform(1, 3))  # Be nice to arXiv servers
                except Exception as e:
                    logger.error(f"Failed to download {result.pdf_url}: {e}")

        self.papers_metadata.extend(papers)
        return papers

    def run_pipeline(self, arxiv_queries: List[str]):
        """Run the download pipeline with only arXiv sources."""
        logger.info("Starting download pipeline...")

        papers_per_query = self.max_papers // len(arxiv_queries) if arxiv_queries else 0
        remaining_papers = self.max_papers

        # Collect from arXiv
        for query in arxiv_queries:
            papers_to_collect = min(papers_per_query, remaining_papers)
            if papers_to_collect <= 0:
                break

            papers = self.collect_from_arxiv(query, max_results=papers_to_collect)
            remaining_papers -= len(papers)
            logger.info(f"Collected {len(papers)} papers from arXiv query: {query}")

        # Save metadata
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.papers_metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved metadata for {len(self.papers_metadata)} papers to {metadata_path}")
        logger.info("Download pipeline completed successfully")

def main():
    parser = argparse.ArgumentParser(
        description="Download ML research papers from arXiv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="arxiv_papers",
        help="Output directory for downloaded papers"
    )
    parser.add_argument(
        "--max_papers",
        type=int,
        default=100,
        help="Maximum total number of papers to download"
    )
    parser.add_argument(
        "--arxiv_queries",
        type=str,
        nargs="+",
        default=["Large Language Models", "Transformers", "Deep Learning", "Machine Learning"],
        help="List of arXiv search queries"
    )

    args = parser.parse_args()

    try:
        downloader = ArxivPaperDownloader(args.output_dir, args.max_papers)
        downloader.run_pipeline(args.arxiv_queries)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info("Pipeline execution completed")

if __name__ == "__main__":
    main()
