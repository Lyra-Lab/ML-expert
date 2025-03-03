import argparse
import glob
import importlib
import os
import asyncio
import inspect

from tqdm import tqdm


def parse_method_arg(method_arg):
    """
    Parse a method configuration string of the form:
       method_name[:key=value[:key2=value2...]]
    Returns:
       (method_name, kwargs_dict, folder_name)
    """
    parts = method_arg.split(":")
    name = parts[0]
    kwargs = {}
    folder_name = name  # Default folder name is the method name
    
    for extra in parts[1:]:
        if "=" in extra:
            key, value = extra.split("=", 1)
            if key == "name":
                folder_name = value
                continue
                
            try:
                converted = int(value)
            except ValueError:
                try:
                    converted = float(value)
                except ValueError:
                    converted = value
            kwargs[key] = converted
        else:
            raise ValueError(f"Extra argument '{extra}' is not in key=value format")
    
    return name, kwargs, folder_name


async def process_pdfs(config, pdf_directory, data_directory, repeats):
    """Process PDFs with both sync and async functions"""
    for candidate in config.keys():
        print(f"Starting conversion using {candidate} with kwargs: {config[candidate]['kwargs']}")
        folder_name = config[candidate]["folder_name"]
        candidate_output_dir = os.path.join(data_directory, folder_name)
        os.makedirs(candidate_output_dir, exist_ok=True)
        
        method = config[candidate]["method"]
        kwargs = config[candidate]["kwargs"]
        is_async = asyncio.iscoroutinefunction(method)
        
        for pdf_path in tqdm(glob.glob(os.path.join(pdf_directory, "*.pdf")), desc=candidate):
            base_name = os.path.basename(pdf_path).replace(".pdf", "")
            
            for i in range(1, repeats + 1):
                if is_async:
                    # Run async function
                    markdown = await method(pdf_path, page_num=1, **kwargs)
                else:
                    # Run synchronous function
                    markdown = method(pdf_path, page_num=1, **kwargs)
                
                output_filename = f"{base_name}_{i}.md"
                output_path = os.path.join(candidate_output_dir, output_filename)
                with open(output_path, "w") as out_f:
                    out_f.write(markdown)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PDF conversion using specified OCR methods and extra parameters.")
    parser.add_argument("methods", nargs="+", help="Methods to run in the format method[:key=value ...]. "
                       "Example: gotocr mineru:temperature=2 marker:runs=3. "
                       "Use 'name=folder_name' to specify a custom output folder name.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of times to repeat the conversion for each PDF.")
    args = parser.parse_args()

    # Mapping of method names to a tuple: (module path, function name)
    available_methods = {
        "olmocr": ("olmocr.bench.runners.run_olmocr", "run_olmocr"),
        "gotocr": ("olmocr.bench.runners.run_gotocr", "run_gotocr"),
        "marker": ("olmocr.bench.runners.run_marker", "run_marker"),
        "mineru": ("olmocr.bench.runners.run_mineru", "run_mineru"),
        "chatgpt": ("olmocr.bench.runners.run_chatgpt", "run_chatgpt"),
    }

    # Build config by importing only requested methods.
    config = {}
    for method_arg in args.methods:
        method_name, extra_kwargs, folder_name = parse_method_arg(method_arg)
        if method_name not in available_methods:
            parser.error(f"Unknown method: {method_name}. " f"Available methods: {', '.join(available_methods.keys())}")
        module_path, function_name = available_methods[method_name]
        # Dynamically import the module and get the function.
        module = importlib.import_module(module_path)
        function = getattr(module, function_name)
        config[method_name] = {"method": function, "kwargs": extra_kwargs, "folder_name": folder_name}

    data_directory = os.path.join(os.path.dirname(__file__), "sample_data")
    pdf_directory = os.path.join(data_directory, "pdfs")

    # Run the async process function
    asyncio.run(process_pdfs(config, pdf_directory, data_directory, args.repeats))