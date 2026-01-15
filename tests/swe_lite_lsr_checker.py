"""
SWE-bench Lite Localization Success Rate (LSR) Checker
Runs localization tests on all 23 dev split instances for different approaches.
Saves results to JSON file.
"""
import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_swe_bench_lite
from src.llm import LLMProvider
from src.retrieval.source_code import RepoManager
from src.retrieval.example_retriever import ExampleRetriever
from src.pipelines.baseline import BaselinePipeline
from src.pipelines.rag import RAGPipeline
from src.pipelines.cot import CoTPipeline
from src.pipelines.agentic import AgenticPipeline


def extract_files_from_patch(patch: str) -> set[str]:
    """Extract Python file paths from a unified diff patch."""
    if not patch:
        return set()
    pattern = r'^(?:---|\+\+\+|diff --git) [ab]/(.+?\.py)'
    files = set()
    for line in patch.split('\n'):
        match = re.match(pattern, line)
        if match:
            files.add(match.group(1).strip())
    return files


def run_localization_test(approach: str, model: str):
    """
    Run localization test on all SWE-bench Lite dev split instances.

    Args:
        approach: Pipeline approach (baseline, rag, cot, agentic)
        model: LLM model name

    Returns:
        dict with results
    """
    print("=" * 80)
    print(f"SWE-bench Lite Dev Split Localization Test - {approach.upper()}")
    print("=" * 80)

    # Load dev split (23 instances)
    print(f"\nLoading SWE-bench Lite dev split...")
    instances = load_swe_bench_lite(split='dev')
    print(f"Loaded {len(instances)} instances\n")

    # Initialize components
    print(f"Initializing LLM provider ({model})...")
    llm = LLMProvider(model=model)
    repo_manager = RepoManager()

    # Initialize pipeline
    print(f"Initializing {approach.upper()} pipeline...")
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
        raise ValueError(f"Unknown approach: {approach}")

    # Results storage
    results = {
        "experiment_name": f"lsr_{approach}_dev",
        "approach": approach,
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "dataset": "princeton-nlp/SWE-bench_Lite",
        "split": "dev",
        "total_instances": len(instances),
        "instances": []
    }

    # Run localization on each instance
    print(f"\nRunning localization tests...\n")
    print(f"{'Instance ID':<45} | {'Status':<10} | {'Predicted File'}")
    print("=" * 80)

    correct = 0
    wrong = 0
    not_found = 0
    errors = 0

    for i, instance in enumerate(instances):
        instance_result = {
            "instance_id": instance.instance_id,
            "repo": instance.repo,
            "predicted_file": None,
            "ground_truth_files": list(extract_files_from_patch(instance.patch)),
            "status": None,
            "error": None
        }

        # Get ground truth files
        gt_files = extract_files_from_patch(instance.patch)
        if not gt_files:
            instance_result["status"] = "no_gt"
            instance_result["error"] = "Could not parse ground truth patch"
            results["instances"].append(instance_result)
            errors += 1
            print(f"{instance.instance_id:<45} | {'ERROR':<10} | No ground truth")
            continue

        # Get repo
        try:
            repo_path = repo_manager.get_repo_path(instance.repo, instance.base_commit)
        except Exception as e:
            instance_result["status"] = "error"
            instance_result["error"] = f"Failed to clone repo: {str(e)}"
            results["instances"].append(instance_result)
            errors += 1
            print(f"{instance.instance_id:<45} | {'ERROR':<10} | Clone failed")
            continue

        # Run localization
        predicted_files = []
        try:
            if approach == "baseline":
                # For Baseline, run full pipeline to see what file it patches
                result = pipeline.run(instance)
                predicted_files = list(extract_files_from_patch(result.generated_patch))
            elif approach == "agentic":
                # Agentic returns list of candidates
                predicted = localize_method(instance, repo_path)
                if isinstance(predicted, list):
                    predicted_files = [p.lstrip('./') for p in predicted if p]
                elif predicted:
                    predicted_files = [predicted.lstrip('./')]
            else:
                # RAG and CoT return a single string or None
                predicted_file = localize_method(instance.problem_statement, repo_path)
                if predicted_file:
                    predicted_files = [predicted_file.lstrip('./')]
        except Exception as e:
            instance_result["status"] = "error"
            instance_result["error"] = f"Localization failed: {str(e)}"
            results["instances"].append(instance_result)
            errors += 1
            print(f"{instance.instance_id:<45} | {'ERROR':<10} | {str(e)[:30]}")
            continue

        # Store predicted file
        instance_result["predicted_file"] = predicted_files[0] if predicted_files else None

        # Check correctness
        if not predicted_files or predicted_files == [None]:
            instance_result["status"] = "not_found"
            not_found += 1
            status = "NOT FOUND"
            print(f"{instance.instance_id:<45} | {status:<10} | None")
        elif any(pf in gt_files or any(pf in gf or gf in pf for gf in gt_files) for pf in predicted_files):
            instance_result["status"] = "correct"
            correct += 1
            status = "✓ CORRECT"
            pred_str = predicted_files[0] if predicted_files else "None"
            print(f"{instance.instance_id:<45} | {status:<10} | {pred_str}")
        else:
            instance_result["status"] = "wrong"
            wrong += 1
            status = "✗ WRONG"
            pred_str = predicted_files[0] if predicted_files else "None"
            print(f"{instance.instance_id:<45} | {status:<10} | {pred_str}")
            print(f"{'':>45} | {'':>10} | Expected: {', '.join(gt_files)}")

        results["instances"].append(instance_result)

    # Calculate summary
    total = correct + wrong + not_found
    results["summary"] = {
        "total_evaluated": total,
        "correct": correct,
        "wrong": wrong,
        "not_found": not_found,
        "errors": errors,
        "accuracy": round(100 * correct / total, 2) if total > 0 else 0.0
    }

    # Print summary
    print("=" * 80)
    print(f"\n{approach.upper()} LOCALIZATION SUMMARY")
    print("=" * 80)
    print(f"Total Evaluated:  {total}")
    print(f"Correct:          {correct} ({100*correct/total:.1f}%)" if total > 0 else "N/A")
    print(f"Wrong:            {wrong} ({100*wrong/total:.1f}%)" if total > 0 else "N/A")
    print(f"Not Found:        {not_found} ({100*not_found/total:.1f}%)" if total > 0 else "N/A")
    print(f"Errors:           {errors}")
    print(f"\nLocalization Success Rate (LSR): {100*correct/total:.2f}%" if total > 0 else "N/A")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="SWE-bench Lite Dev Split Localization Success Rate Checker"
    )
    parser.add_argument(
        "--approach", "-a",
        type=str,
        required=True,
        choices=["baseline", "rag", "cot", "agentic"],
        help="Pipeline approach to test"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="deepseek/deepseek-chat",
        help="LLM model to use (default: deepseek/deepseek-chat)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path (default: results/lsr_{approach}_dev.json)"
    )

    args = parser.parse_args()

    # Run test
    results = run_localization_test(args.approach, args.model)

    # Save results
    output_path = args.output or f"results/lsr_{args.approach}_dev.json"
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[SUCCESS] Results saved to {output_file}")


if __name__ == "__main__":
    main()
