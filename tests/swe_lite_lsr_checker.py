"""
SWE-bench Lite Localization Success Rate (LSR) Checker
Runs localization tests on all 23 dev split instances for different approaches.
Saves results to JSON file.

Supports multiple LLM backends:
- deepseek/deepseek-chat (default, via OpenRouter)
- local-qwen (Ollama qwen2.5-coder:7b-instruct)
- Any OpenRouter model name
"""
import argparse
import json
import re
import sys
import signal
import time
import traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_swe_bench_lite
from src.llm import LLMProvider, OllamaProvider
from src.retrieval.source_code import RepoManager
from src.retrieval.example_retriever import ExampleRetriever
from src.pipelines.baseline import BaselinePipeline
from src.pipelines.rag import RAGPipeline
from src.pipelines.cot import CoTPipeline
from src.pipelines.agentic import AgenticPipeline


MODEL_PRESETS = {
    "deepseek": "deepseek/deepseek-chat",
    "deepseek-r1": "deepseek/deepseek-reasoner",
    "gpt4": "openai/gpt-4-turbo",
    "gpt4o": "openai/gpt-4o",
    "claude": "anthropic/claude-3.5-sonnet",
    "qwen-72b": "qwen/qwen-2.5-72b-instruct",
    "qwen-32b": "qwen/qwen-2.5-32b-instruct",
    "local-qwen": "ollama:qwen2.5-coder:7b-instruct", 
    "local-qwen-32b": "ollama:qwen2.5-coder:32b-instruct",
}


def timeout_handler(signum, frame):
    """Handler for timeout signal."""
    raise TimeoutError("Localization timeout exceeded")


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


def get_llm_provider(model: str):
    """
    Get the appropriate LLM provider based on model name.

    Args:
        model: Model name or preset. Use "ollama:" prefix for local Ollama models.

    Returns:
        LLMProvider or OllamaProvider instance
    """
    # Check if it's a preset
    if model in MODEL_PRESETS:
        model = MODEL_PRESETS[model]

    # Check if it's an Ollama model
    if model.startswith("ollama:"):
        ollama_model = model[7:]  # Remove "ollama:" prefix
        print(f"Using local Ollama model: {ollama_model}")
        return OllamaProvider(model=ollama_model)

    # Default to OpenRouter
    return LLMProvider(model=model)


def run_localization_test(approach: str, model: str, debug_mode: bool = False,
                          cot_adversarial: bool = False, cot_rubric: bool = False):
    """
    Run localization test on all SWE-bench Lite dev split instances.

    Args:
        approach: Pipeline approach (baseline, rag, cot, agentic)
        model: LLM model name or preset
        debug_mode: If True, save detailed error analysis
        cot_adversarial: [CoT only] Enable adversarial critique loop
        cot_rubric: [CoT only] Enable rubric checklist validation

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
    resolved_model = MODEL_PRESETS.get(model, model)
    print(f"Initializing LLM provider ({resolved_model})...")
    llm = get_llm_provider(model)
    repo_manager = RepoManager()

    # Initialize pipeline
    print(f"Initializing {approach.upper()} pipeline...")
    if approach == "baseline":
        pipeline = BaselinePipeline(llm_provider=llm)
        localize_method = None  # Baseline doesn't have localization
    elif approach == "rag":
        pipeline = RAGPipeline(llm_provider=llm, repo_manager=repo_manager, use_repo_embedding=True)
        localize_method = pipeline._find_primary_file
    elif approach == "cot":
        pipeline = CoTPipeline(
            llm_provider=llm, repo_manager=repo_manager,
            use_adversarial=cot_adversarial, use_rubric=cot_rubric
        )
        localize_method = pipeline._localize_file
        if cot_adversarial:
            print("  [CoT Variant] Adversarial critique enabled")
        if cot_rubric:
            print("  [CoT Variant] Rubric checklist enabled")
    elif approach == "agentic":
        retriever = ExampleRetriever()
        pipeline = AgenticPipeline(llm_provider=llm, repo_manager=repo_manager, example_retriever=retriever)
        localize_method = pipeline._phase_localize_candidates
    else:
        raise ValueError(f"Unknown approach: {approach}")

    # Build experiment name with variant suffix
    exp_suffix = ""
    if approach == "cot":
        if cot_adversarial:
            exp_suffix += "_adversarial"
        if cot_rubric:
            exp_suffix += "_rubric"

    # Results storage
    results = {
        "experiment_name": f"lsr_{approach}{exp_suffix}_dev",
        "approach": approach,
        "model": resolved_model,
        "timestamp": datetime.now().isoformat(),
        "dataset": "princeton-nlp/SWE-bench_Lite",
        "split": "dev",
        "total_instances": len(instances),
        "cot_adversarial": cot_adversarial if approach == "cot" else None,
        "cot_rubric": cot_rubric if approach == "cot" else None,
        "instances": []
    }

    # Error analysis storage (for debug mode)
    error_analysis = {
        "experiment_name": f"error_analysis_{approach}_dev",
        "approach": approach,
        "model": resolved_model,
        "timestamp": datetime.now().isoformat(),
        "errors": []
    }

    # Run localization on each instance
    print(f"\nRunning localization tests...\n")
    print(f"{'Instance ID':<45} | {'Status':<10} | {'Time':<8} | {'LLM':<5} | {'Predicted File'}")
    print("=" * 110)

    correct = 0
    wrong = 0
    not_found = 0
    errors = 0
    total_time = 0.0
    instance_times = []
    total_llm_calls = 0
    total_api_calls = 0
    total_cache_hits = 0

    # Reset LLM stats before starting
    llm.reset_call_stats()

    for instance in instances:
        # Track LLM calls for this instance
        llm_calls_before = llm.call_count

        instance_result = {
            "instance_id": instance.instance_id,
            "repo": instance.repo,
            "predicted_file": None,
            "ground_truth_files": list(extract_files_from_patch(instance.patch)),
            "status": None,
            "error": None,
            "localization_time_sec": None,
            "llm_calls": 0
        }

        # Get ground truth files
        gt_files = extract_files_from_patch(instance.patch)
        if not gt_files:
            instance_result["status"] = "no_gt"
            instance_result["error"] = "Could not parse ground truth patch"
            results["instances"].append(instance_result)
            errors += 1
            print(f"{instance.instance_id:<45} | {'ERROR':<10} | No ground truth")

            if debug_mode:
                error_analysis["errors"].append({
                    "instance_id": instance.instance_id,
                    "fail_type": "no_ground_truth",
                    "error_message": "Could not parse ground truth patch",
                    "problem_statement": instance.problem_statement[:1000],
                    "ground_truth_patch": instance.patch,
                })
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

            if debug_mode:
                error_analysis["errors"].append({
                    "instance_id": instance.instance_id,
                    "fail_type": "repo_clone_failed",
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                    "repo": instance.repo,
                    "base_commit": instance.base_commit,
                })
            continue

        # Run localization with timeout (5 minutes max per instance)
        predicted_files = []
        raw_llm_response = None
        start_time = time.time()
        try:
            # Set timeout alarm
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(300)  # 5 minute timeout

            try:
                if approach == "baseline":
                    # For Baseline, run full pipeline to see what file it patches
                    result = pipeline.run(instance)
                    predicted_files = list(extract_files_from_patch(result.generated_patch))
                    raw_llm_response = result.raw_response
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
            finally:
                # Cancel alarm
                signal.alarm(0)
                elapsed_time = time.time() - start_time
                instance_llm_calls = llm.call_count - llm_calls_before
                instance_result["localization_time_sec"] = round(elapsed_time, 2)
                instance_result["llm_calls"] = instance_llm_calls
                instance_times.append(elapsed_time)

        except TimeoutError:
            elapsed_time = time.time() - start_time
            instance_llm_calls = llm.call_count - llm_calls_before
            instance_result["status"] = "error"
            instance_result["error"] = "Localization timeout (5 min)"
            instance_result["localization_time_sec"] = round(elapsed_time, 2)
            instance_result["llm_calls"] = instance_llm_calls
            results["instances"].append(instance_result)
            errors += 1
            print(f"{instance.instance_id:<45} | {'TIMEOUT':<10} | {elapsed_time:>6.1f}s | {instance_llm_calls:>4} | Exceeded 5 min limit")

            if debug_mode:
                error_analysis["errors"].append({
                    "instance_id": instance.instance_id,
                    "fail_type": "timeout",
                    "error_message": "Localization timeout (5 min)",
                    "problem_statement": instance.problem_statement[:1000],
                    "ground_truth_files": list(gt_files),
                })
            continue
        except Exception as e:
            elapsed_time = time.time() - start_time
            instance_llm_calls = llm.call_count - llm_calls_before
            instance_result["status"] = "error"
            instance_result["error"] = f"Localization failed: {str(e)}"
            instance_result["localization_time_sec"] = round(elapsed_time, 2)
            instance_result["llm_calls"] = instance_llm_calls
            results["instances"].append(instance_result)
            errors += 1
            print(f"{instance.instance_id:<45} | {'ERROR':<10} | {elapsed_time:>6.1f}s | {instance_llm_calls:>4} | {str(e)[:30]}")

            if debug_mode:
                error_analysis["errors"].append({
                    "instance_id": instance.instance_id,
                    "fail_type": "localization_exception",
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                    "problem_statement": instance.problem_statement[:1000],
                    "ground_truth_files": list(gt_files),
                })
            continue

        # Store predicted file
        instance_result["predicted_file"] = predicted_files[0] if predicted_files else None

        # Check correctness
        time_str = f"{elapsed_time:>6.1f}s"
        llm_str = f"{instance_llm_calls:>4}"
        if not predicted_files or predicted_files == [None]:
            instance_result["status"] = "not_found"
            not_found += 1
            status = "NOT FOUND"
            print(f"{instance.instance_id:<45} | {status:<10} | {time_str} | {llm_str} | None")

            if debug_mode:
                error_analysis["errors"].append({
                    "instance_id": instance.instance_id,
                    "fail_type": "localization_not_found",
                    "error_message": "No file predicted",
                    "predicted_file": None,
                    "ground_truth_files": list(gt_files),
                    "problem_statement": instance.problem_statement[:1000],
                    "raw_llm_response": raw_llm_response[:2000] if raw_llm_response else None,
                })
        elif any(pf in gt_files or any(pf in gf or gf in pf for gf in gt_files) for pf in predicted_files):
            instance_result["status"] = "correct"
            correct += 1
            status = "✓ CORRECT"
            pred_str = predicted_files[0] if predicted_files else "None"
            print(f"{instance.instance_id:<45} | {status:<10} | {time_str} | {llm_str} | {pred_str}")
        else:
            instance_result["status"] = "wrong"
            wrong += 1
            status = "✗ WRONG"
            pred_str = predicted_files[0] if predicted_files else "None"
            print(f"{instance.instance_id:<45} | {status:<10} | {time_str} | {llm_str} | {pred_str}")
            print(f"{'':>45} | {'':>10} | {'':>8} | {'':>5} | Expected: {', '.join(gt_files)}")

            if debug_mode:
                error_analysis["errors"].append({
                    "instance_id": instance.instance_id,
                    "fail_type": "localization_wrong_file",
                    "error_message": f"Predicted wrong file",
                    "predicted_file": pred_str,
                    "predicted_files_all": predicted_files,
                    "ground_truth_files": list(gt_files),
                    "problem_statement": instance.problem_statement[:1000],
                    "raw_llm_response": raw_llm_response[:2000] if raw_llm_response else None,
                    "ground_truth_patch": instance.patch,
                })

        results["instances"].append(instance_result)

    # Calculate summary
    total = correct + wrong + not_found
    total_time = sum(instance_times) if instance_times else 0.0
    avg_time = total_time / len(instance_times) if instance_times else 0.0

    # Get final LLM stats
    llm_stats = llm.get_call_stats()

    results["summary"] = {
        "total_evaluated": total,
        "correct": correct,
        "wrong": wrong,
        "not_found": not_found,
        "errors": errors,
        "accuracy": round(100 * correct / total, 2) if total > 0 else 0.0,
        "total_time_sec": round(total_time, 2),
        "avg_time_sec": round(avg_time, 2),
        "min_time_sec": round(min(instance_times), 2) if instance_times else 0.0,
        "max_time_sec": round(max(instance_times), 2) if instance_times else 0.0,
        "llm_total_calls": llm_stats["total_calls"],
        "llm_api_calls": llm_stats["api_calls"],
        "llm_cache_hits": llm_stats["cache_hits"],
    }

    # Print summary
    print("=" * 110)
    print(f"\n{approach.upper()} LOCALIZATION SUMMARY")
    print("=" * 110)
    print(f"Total Evaluated:  {total}")
    print(f"Correct:          {correct} ({100*correct/total:.1f}%)" if total > 0 else "N/A")
    print(f"Wrong:            {wrong} ({100*wrong/total:.1f}%)" if total > 0 else "N/A")
    print(f"Not Found:        {not_found} ({100*not_found/total:.1f}%)" if total > 0 else "N/A")
    print(f"Errors:           {errors}")
    print(f"\nLocalization Success Rate (LSR): {100*correct/total:.2f}%" if total > 0 else "N/A")
    print(f"\n--- Timing Statistics ---")
    print(f"Total Time:       {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Average Time:     {avg_time:.2f}s per instance")
    if instance_times:
        print(f"Min Time:         {min(instance_times):.2f}s")
        print(f"Max Time:         {max(instance_times):.2f}s")
    print(f"\n--- LLM Call Statistics ---")
    print(f"Total Calls:      {llm_stats['total_calls']}")
    print(f"API Calls:        {llm_stats['api_calls']} (actual requests)")
    print(f"Cache Hits:       {llm_stats['cache_hits']}")
    if total > 0:
        print(f"Avg Calls/Inst:   {llm_stats['total_calls']/total:.1f}")

    # Add error analysis summary
    if debug_mode:
        error_analysis["summary"] = {
            "total_errors": len(error_analysis["errors"]),
            "by_type": {}
        }
        for err in error_analysis["errors"]:
            fail_type = err["fail_type"]
            error_analysis["summary"]["by_type"][fail_type] = \
                error_analysis["summary"]["by_type"].get(fail_type, 0) + 1

    return results, error_analysis if debug_mode else None


def main():
    parser = argparse.ArgumentParser(
        description="SWE-bench Lite Dev Split Localization Success Rate Checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model presets:
  deepseek      deepseek/deepseek-chat (default)
  deepseek-r1   deepseek/deepseek-reasoner
  gpt4          openai/gpt-4-turbo
  gpt4o         openai/gpt-4o
  claude        anthropic/claude-3.5-sonnet
  qwen-72b      qwen/qwen-2.5-72b-instruct
  qwen-32b      qwen/qwen-2.5-32b-instruct
  local-qwen    ollama:qwen2.5-coder:7b-instruct (local Ollama)
  local-qwen-32b ollama:qwen2.5-coder:32b-instruct (local Ollama)

Or specify any OpenRouter model directly (e.g., meta-llama/llama-3.1-70b-instruct)
For local Ollama models, use prefix: ollama:model-name

Examples:
  python swe_lite_lsr_checker.py -a rag -m deepseek
  python swe_lite_lsr_checker.py -a cot -m local-qwen --debug
  python swe_lite_lsr_checker.py -a agentic -m ollama:codellama:34b
"""
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
        default="deepseek",
        help="LLM model preset or full name (default: deepseek)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path (default: results/lsr_{approach}_dev.json)"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug mode: save detailed error analysis to error_analysis/"
    )
    parser.add_argument(
        "--cot-adversarial",
        action="store_true",
        help="[CoT only] Enable adversarial critique loop"
    )
    parser.add_argument(
        "--cot-rubric",
        action="store_true",
        help="[CoT only] Enable rubric checklist validation"
    )

    args = parser.parse_args()

    # Run test
    cot_adv = getattr(args, 'cot_adversarial', False)
    cot_rub = getattr(args, 'cot_rubric', False)
    if args.debug:
        results, error_analysis = run_localization_test(
            args.approach, args.model, debug_mode=True,
            cot_adversarial=cot_adv, cot_rubric=cot_rub
        )
    else:
        results, _ = run_localization_test(
            args.approach, args.model, debug_mode=False,
            cot_adversarial=cot_adv, cot_rubric=cot_rub
        )
        error_analysis = None

    # Save results - add suffix for CoT variants
    if args.output:
        output_path = args.output
    else:
        suffix = ""
        if args.approach == "cot":
            if cot_adv:
                suffix += "_adversarial"
            if cot_rub:
                suffix += "_rubric"
        output_path = f"results/lsr_{args.approach}{suffix}_dev.json"
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[SUCCESS] Results saved to {output_file}")

    # Save error analysis if debug mode
    if args.debug and error_analysis:
        # Create error_analysis directory
        error_dir = Path("error_analysis")
        error_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp and model
        model_short = args.model.replace("/", "_").replace(":", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_file = error_dir / f"{args.approach}_{model_short}_{timestamp}.json"

        with open(error_file, 'w') as f:
            json.dump(error_analysis, f, indent=2)

        print(f"[DEBUG] Error analysis saved to {error_file}")
        print(f"[DEBUG] Total errors captured: {len(error_analysis['errors'])}")
        if error_analysis["summary"]["by_type"]:
            print(f"[DEBUG] Error breakdown:")
            for fail_type, count in error_analysis["summary"]["by_type"].items():
                print(f"         - {fail_type}: {count}")


if __name__ == "__main__":
    main()
