#!/usr/bin/env python3
"""
This script runs olmocr bench.
It will take as an argument a folder, and scan it for .jsonl files which contain the various rules and properties that we will check.
It will then validate the JSON files to make sure they are all valid.
Then, each other folder in there (besides /pdfs) represents a pipeline tool that we will evaluate.
We will validate that each one of those contains at least one .md file (or repeated generations, e.g. _1.md, _2.md, etc.)
corresponding to its parse for every .pdf in the /pdfs folder.
Then, we will read each one, and check if they pass against all the rules.
If a rule fails on some of the repeats, a short explanation is printed.
The final score is averaged over the repeated generations.
"""

import argparse
import glob
import os
import sys
from typing import Tuple, List, Dict

from .tests import BasePDFTest, load_tests


def evaluate_candidate(candidate_folder: str, all_tests: List[BasePDFTest], pdf_basenames: List[str]) -> Tuple[float, int, List[str], List[str], Dict[str, List[float]]]:
    """
    For the candidate folder (pipeline tool output), validate that it contains at least one .md file
    (i.e. repeated generations like _1.md, _2.md, etc.) for every PDF in the pdf folder.
    Then, run each rule against all corresponding .md files and average the results.

    Returns a tuple:
      (overall_score, total_tests, candidate_errors, test_failures, test_type_breakdown)

      - overall_score: Average fraction of tests passed (averaged over repeats and tests).
      - total_tests: Total number of tests evaluated.
      - candidate_errors: List of candidate errors (e.g. missing files).
      - test_failures: List of failure messages for tests not passing on all repeats.
      - test_type_breakdown: Dictionary mapping test type to list of average pass ratios for tests of that type.
    """
    candidate_errors = []
    test_failures = []
    test_type_breakdown = {}  # key: test type, value: list of average pass ratios
    candidate_name = os.path.basename(candidate_folder)

    # Map each PDF to its corresponding MD repeats (e.g., doc1_1.md, doc1_2.md, etc.)
    pdf_to_md_files = {}
    for pdf_name in pdf_basenames:
        md_base = os.path.splitext(pdf_name)[0]
        md_pattern = os.path.join(candidate_folder, f"{md_base}_*.md")
        md_files = glob.glob(md_pattern)
        if not md_files:
            candidate_errors.append(f"Candidate '{candidate_name}' is missing MD repeats for {pdf_name} (expected files matching {md_base}_*.md).")
        else:
            pdf_to_md_files[pdf_name] = md_files

    if candidate_errors:
        return (0.0, len(all_tests), candidate_errors, test_failures, test_type_breakdown)

    total_test_score = 0.0

    # Evaluate each test. Each test references a PDF (e.g., "doc1.pdf") so we get all its MD repeats.
    for test in all_tests:
        test_type = test.type
        if test_type not in test_type_breakdown:
            test_type_breakdown[test_type] = []
        pdf_name = test.pdf
        md_base = os.path.splitext(pdf_name)[0]
        md_files = pdf_to_md_files.get(pdf_name, [])
        if not md_files:
            continue  # Should not occur due to earlier check.
        repeat_passes = 0
        num_repeats = 0
        explanations = []
        for md_path in md_files:
            num_repeats += 1
            try:
                with open(md_path, "r", encoding="utf-8") as f:
                    md_content = f.read()
            except Exception as e:
                candidate_errors.append(f"Error reading {md_path}: {e}")
                continue

            try:
                # Use the test's run method to evaluate the content
                passed, explanation = test.run(md_content)
                if passed:
                    repeat_passes += 1
                else:
                    explanations.append(explanation)
            except Exception as e:
                candidate_errors.append(f"Error running test {test.id} on {md_path}: {e}")
                explanations.append(str(e))
        
        test_avg = repeat_passes / num_repeats if num_repeats > 0 else 0.0
        total_test_score += test_avg
        if test_avg < 1.0:
            test_failures.append(
                f"Test {test.id} on {md_base} average pass ratio: {test_avg:.3f} ({repeat_passes}/{num_repeats} repeats passed). "
                f"Example explanation: {explanations[0] if explanations else 'No explanation'}"
            )
        test_type_breakdown[test_type].append(test_avg)

    overall_score = total_test_score / len(all_tests) if all_tests else 0.0
    return (overall_score, len(all_tests), candidate_errors, test_failures, test_type_breakdown)


def main():
    parser = argparse.ArgumentParser(description="Run OLMOCR Bench.")
    parser.add_argument(
        "--input_folder",
        default=os.path.join(os.path.dirname(__file__), "sample_data"),
        help="Path to the folder containing .jsonl files, /pdfs folder, and pipeline tool subfolders.",
    )
    args = parser.parse_args()

    input_folder = args.input_folder
    pdf_folder = os.path.join(input_folder, "pdfs")

    # Check that the pdfs folder exists
    if not os.path.exists(pdf_folder):
        print("Error: /pdfs folder must exist in your data directory.", file=sys.stderr)
        sys.exit(1)

    # Find all pdf files in the pdf folder
    all_pdf_files = list(glob.glob(os.path.join(pdf_folder, "*.pdf")))
    if not all_pdf_files:
        print(f"Error: No PDF files found in {pdf_folder}", file=sys.stderr)
        sys.exit(1)

    # Get PDF basenames (e.g. "doc1.pdf")
    pdf_basenames = [os.path.basename(p) for p in all_pdf_files]

    # Find and validate .jsonl files in the inp∂ut folder
    jsonl_files = glob.glob(os.path.join(input_folder, "*.jsonl"))
    if not jsonl_files:
        print(f"Error: No .jsonl files found in {input_folder}.", file=sys.stderr)
        sys.exit(1)

    # Load and concatenate all test rules from JSONL files
    all_tests = []
    for jsonl_path in jsonl_files:
        tests = load_tests(jsonl_path)
        all_tests.extend(tests)

    if not all_tests:
        print("No valid tests found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Identify candidate pipeline folders (subdirectories of input_folder excluding /pdfs)
    candidate_folders = []
    for entry in os.listdir(input_folder):
        full_path = os.path.join(input_folder, entry)
        if os.path.isdir(full_path) and entry != "pdfs":
            candidate_folders.append(full_path)

    if not candidate_folders:
        print("Error: No candidate pipeline folders found (subdirectories besides 'pdfs').", file=sys.stderr)
        sys.exit(1)

    # Evaluate each candidate
    summary = []
    print("\nRunning tests for each candidate:")
    for candidate in candidate_folders:
        candidate_name = os.path.basename(candidate)
        overall_score, total_tests, candidate_errors, test_failures, test_type_breakdown = evaluate_candidate(
            candidate, all_tests, pdf_basenames
        )
        summary.append((candidate_name, overall_score, total_tests, candidate_errors, test_failures, test_type_breakdown))
        print(f"\nCandidate: {candidate_name}")
        if candidate_errors:
            for err in candidate_errors:
                print(f"  [ERROR] {err}")
        else:
            if test_failures:
                for fail in test_failures:
                    print(f"  [FAIL] {fail}")
            print(f"  Average Score: {overall_score * 100:.1f}% over {total_tests} tests.")

    # Print final summary with breakdown by test type
    print("\n" + "=" * 50)
    print("Final Summary:")
    for candidate_name, overall_score, total_tests, candidate_errors, _, test_type_breakdown in summary:
        if candidate_errors:
            status = "FAILED (errors)"
        else:
            status = f"{overall_score * 100:0.1f}%"
        print(f"{candidate_name:20s} : Average Score: {overall_score * 100:0.1f}% over {total_tests:3d} tests - {status}")
        print("  Breakdown by test type:")
        for ttype, scores in test_type_breakdown.items():
            if scores:
                avg = sum(scores) / len(scores) * 100
            else:
                avg = 0.0
            print(f"    {ttype:8s}: {avg:0.1f}% average pass rate over {len(scores)} tests")
        print("")
    print("=" * 50)


if __name__ == "__main__":
    main()