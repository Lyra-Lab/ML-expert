import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass
from functools import partial
import argparse
from typing import Optional
import json

# Import necessary components from olmocr
from olmocr.pipeline import (
    sglang_server_host, 
    sglang_server_ready, 
    build_page_query,
    apost,
    SGLANG_SERVER_PORT,
    MetricsKeeper,
    WorkerTracker
)
from olmocr.prompts import PageResponse


# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("olmocr_runner")

# Basic configuration
@dataclass
class Args:
    model: str = "allenai/olmOCR-7B-0225-preview"
    model_chat_template: str = "qwen2-vl"
    model_max_context: int = 8192
    target_longest_image_dim: int = 1024
    target_anchor_text_len: int = 6000

async def run_olmocr(pdf_path: str, page_num: int = 1, temperature: float = 0.8) -> str:
    """
    Process a single page of a PDF using the olmocr pipeline.
    
    Args:
        pdf_path: Path to the PDF file
        page_num: Page number to process (1-indexed)
        temperature: Temperature parameter for the model
        
    Returns:
        The extracted text from the page
    """
    # Ensure global variables are initialized
    global metrics, tracker
    if 'metrics' not in globals() or metrics is None:
        metrics = MetricsKeeper(window=60*5)
    if 'tracker' not in globals() or tracker is None:
        tracker = WorkerTracker()
    
    args = Args()
    semaphore = asyncio.Semaphore(1)
    
    # Ensure server is running
    server_task = None
    try:
        await asyncio.wait_for(sglang_server_ready(), timeout=5)
        print("Using existing sglang server")
    except Exception:
        print("Starting new sglang server")
        server_task = asyncio.create_task(sglang_server_host(args, semaphore))
        await sglang_server_ready()
    
    try:
        # Process the page
        query = await build_page_query(
            pdf_path, 
            page_num, 
            args.target_longest_image_dim, 
            args.target_anchor_text_len
        )
        query["temperature"] = temperature
        
        # Make request and get response
        url = f"http://localhost:{SGLANG_SERVER_PORT}/v1/chat/completions"
        status_code, response_body = await apost(url, json_data=query)
        
        if status_code != 200:
            return f"Error: HTTP status {status_code}"
        
        # Parse response
        response_data = json.loads(response_body)
        content = response_data["choices"][0]["message"]["content"]
        model_json = json.loads(content)
        page_response = PageResponse(**model_json)
        
        # Update metrics
        metrics.add_metrics(
            sglang_input_tokens=response_data["usage"].get("prompt_tokens", 0),
            sglang_output_tokens=response_data["usage"].get("completion_tokens", 0)
        )
        
        return page_response.natural_text
        
    except Exception as e:
        return f"Error: {type(e).__name__} - {str(e)}"
        
    finally:
        # We leave the server running for potential reuse
        # This is more efficient if multiple pages will be processed
        pass