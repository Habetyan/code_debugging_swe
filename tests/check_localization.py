"""
Localization Checker: Benchmarks the accuracy of file localization across pipelines.
Compares predicted faulty files against ground truth usage in patches.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_swe_bench_lite
from src.llm import LLMProvider
from src.retrieval.source_code import RepoManager
from src.retrieval.example_retriever import ExampleRetriever
from src.pipelines.baseline import BaselinePipeline
from src.pipelines.rag import RAGPipeline
from src.pipelines.cot import CoTPipeline
from src.pipelines.agentic import AgenticPipeline


def extract_files_from_patch(patch: str) -> list[str]:
    """Extract file paths from a unified diff patch."""
    if not patch:
        return []
    # Pattern to match:
    # --- a/path/to/file.py
    # +++ b/path/to/file.py
    # diff --git a/path/to/file.py b/path/to/file.py
    pattern = r'^(?:---|\+\+\+|diff --git) [ab]/(.+?)(?: |$|\n)'
    files = set()
    for line in patch.split('\n'):
        match = re.match(pattern, line)
        if match:
            f = match.group(1).strip()
            if f.endswith('.py'):
                files.add(f)
    return list(files)


def check_from_results_file(results_path: Path):
    """Check localization accuracy from a results JSON file."""
    print(f"Loading results from {results_path}...")
    with open(results_path) as f:
        data = json.load(f)
    
    print("Loading SWE-bench Lite dataset (for ground truth)...")
    dataset = load_swe_bench_lite()
    
    # Create ground truth map
    gt_map = {}
    for inst in dataset:
        gt_files = extract_files_from_patch(inst.patch)
        if gt_files:
            gt_map[inst.instance_id] = gt_files
    
    # Check each result
    correct = 0
    wrong = 0
    empty = 0
    total = 0
    
    print("\n" + "=" * 80)
    print(f"{'Instance ID':<40} | {'Status':<10} | {'Predicted':<30}")
    print("=" * 80)
    
    for inst in data.get("instances", []):
        instance_id = inst["instance_id"]
        gt_files = gt_map.get(instance_id, [])
        
        if not gt_files:
            continue
        
        # Get prediction
        attempts = inst.get("attempts", [])
        if not attempts or not attempts[0].get("generated_patch"):
            pred_files = []
        else:
            patch = attempts[0]["generated_patch"]
            pred_files = extract_files_from_patch(patch)
        
        # Check correctness
        total += 1
        if not pred_files:
            status = " EMPTY"
            empty += 1
        elif any(pf in gt_files for pf in pred_files):
            status = " CORRECT"
            correct += 1
        else:
            status = " WRONG"
            wrong += 1
        
        pred_str = ", ".join(pred_files) if pred_files else "None"
        print(f"{instance_id:<40} | {status:<10} | {pred_str:<30}")
        
        if status == " WRONG":
            print(f"   Expected: {', '.join(gt_files)}")
    
    # Summary
    print("=" * 80)
    print(f"Total:       {total}")
    print(f"Correct:     {correct} ({100*correct/total:.1f}%)" if total else "N/A")
    print(f"Wrong:       {wrong} ({100*wrong/total:.1f}%)" if total else "N/A")
    print(f"Empty:       {empty} ({100*empty/total:.1f}%)" if total else "N/A")
    print(f"\nLocalization Accuracy: {correct}/{total} ({100*correct/total:.1f}%)" if total else "N/A")


def run_live_localization_test(approach: str, n: int, model: str):
    """Run live localization test by calling pipeline's localization phase."""
    print(f"\nTesting {approach.upper()} localization on {n} instances...\n")
    
    # Load dataset
    dataset = load_swe_bench_lite()
    test_instances = dataset[:n]
    
    # Initialize components
    llm = LLMProvider(model=model)
    repo_manager = RepoManager()
    
    # Initialize pipeline
    if approach == "baseline":
        pipeline = BaselinePipeline(llm_provider=llm)
        localize_method = None  # Baseline doesn't have localization
    elif approach == "rag":
        pipeline = RAGPipeline(llm_provider=llm, repo_manager=repo_manager)
        localize_method = pipeline._find_primary_file
    elif approach == "cot":
        pipeline = CoTPipeline(llm_provider=llm, repo_manager=repo_manager)
        localize_method = pipeline._localize_file
    elif approach == "agentic":
        retriever = ExampleRetriever()
        pipeline = AgenticPipeline(llm_provider=llm, repo_manager=repo_manager, example_retriever=retriever)
        localize_method = pipeline._phase_localize_candidates
    else:
        print(f"Unknown approach: {approach}")
        return
    
    if not localize_method and approach != "baseline":
        print(f"Error: {approach} pipeline doesn't have a localization method!")
        return
    
    # Run localization tests
    correct = 0
    wrong = 0
    not_found = 0
    
    for i, instance in enumerate(test_instances):
        print(f"[{i+1}/{n}] {instance.instance_id}")
        
        # Get ground truth
        gt_files = extract_files_from_patch(instance.patch)
        if not gt_files:
            print(f"Could not parse ground truth")
            continue
        
        # Get repo
        try:
            repo_path = repo_manager.get_repo_path(instance.repo, instance.base_commit)
        except Exception as e:
            print(f"Failed to clone: {e}")
            continue
        
        # Run localization
        if approach == "baseline":
            # For Baseline, we must run the full pipeline to see what file it patches
            try:
                result = pipeline.run(instance)
                predicted_files = extract_files_from_patch(result.generated_patch)
            except Exception as e:
                print(f"Error: {e}")
                predicted_files = []
        else:
            try:
                if approach == "agentic":
                    # Agentic returns list of candidates
                    predicted = localize_method(instance, repo_path)
                    # Normalize paths
                    if isinstance(predicted, list):
                        predicted_files = [p.lstrip('./') for p in predicted]
                    else:
                        predicted_files = [predicted.lstrip('./')] if predicted else []
                else:
                    # RAG and CoT return a single string or None
                    predicted_file = localize_method(instance.problem_statement, repo_path)
                    predicted_files = [predicted_file] if predicted_file else []
            except Exception as e:
                print(f" Error: {e}")
                predicted_files = []
        
        # Check correctness
        if not predicted_files or predicted_files == [None]:
            not_found += 1
            print(f" NOT FOUND")
            print(f"     Expected: {gt_files}")
        elif any(pf in gt_files for pf in predicted_files):
            correct += 1
            print(f"CORRECT: {predicted_files[0] if predicted_files else 'None'}")
        else:
            wrong += 1
            print(f" WRONG: {predicted_files[0] if predicted_files else 'None'}")
            print(f"     Expected: {gt_files}")
        print()
    
    # Summary
    total = correct + wrong + not_found
    print("=" * 60)
    print(f"{approach.upper()} LOCALIZATION SUMMARY")
    print("=" * 60)
    print(f"Total:       {total}")
    if total > 0:
        print(f"Correct:     {correct} ({100*correct/total:.1f}%)")
        print(f"Wrong:       {wrong} ({100*wrong/total:.1f}%)")
        print(f"Not Found:   {not_found} ({100*not_found/total:.1f}%)")
        print(f"\nAccuracy: {100*correct/total:.1f}%")
    else:
        print("No instances evaluated.")


def main():
    parser = argparse.ArgumentParser(
        description="Check localization accuracy from results or run live test"
    )
    
    # Mode selection
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--results", "-r",
        type=str,
        help="Path to results JSON file to analyze"
    )
    mode.add_argument(
        "--approach", "-a",
        type=str,
        choices=["baseline", "rag", "cot", "agentic"],
        help="Pipeline approach to test live"
    )
    
    # Live test options
    parser.add_argument(
        "--n", "-n",
        type=int,
        default=100,
        help="Number of instances for live test (default: 100)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="deepseek/deepseek-chat",
        help="LLM model for live test"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LOCALIZATION ACCURACY CHECKER")
    print("=" * 60)
    
    if args.results:
        # Check from results file
        results_path = Path(args.results)
        if not results_path.exists():
            print(f"Error: File not found: {results_path}")
            return
        check_from_results_file(results_path)
    
    elif args.approach:
        # Run live test
        run_live_localization_test(args.approach, args.n, args.model)


if __name__ == "__main__":
    main()
