"""
Agentic Experiment: Runs the Agentic pipeline (ReAct loop) on SWE-bench.
Uses fuzzy patching for validation.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.swe_bench import load_swe_bench_dataset, load_swe_bench_lite, create_stratified_subset
from src.llm import LLMProvider
from src.pipelines.agentic import AgenticPipeline
from src.evaluation.runner import ExperimentRunner
from src.utils.fuzzy_patch import apply_patch_fuzzy, strictify_patch





def main():
    parser = argparse.ArgumentParser(description="Run Agentic Pipeline (Best Pass@1)")
    parser.add_argument("--n", "-n", type=int, default=5,
                        help="Number of instances to run")
    parser.add_argument("--model", "-m", type=str, default="meta-llama/llama-3.1-8b-instruct",
                        help="Model for localization (small/fast)")
    parser.add_argument("--patch-model", "-pm", type=str, default="deepseek/deepseek-chat",
                        help="Model for patch generation (stronger)")
    parser.add_argument("--experiment-name", "-e", type=str, default="agentic_run",
                        help="Name for the experiment")
    parser.add_argument("--instance-id", type=str, default=None,
                        help="Specific instance ID to run (comma-separated for multiple)")
    parser.add_argument("--temperature", "-t", type=float, default=0.1,
                        help="LLM temperature (lower is more deterministic)")
    parser.add_argument("--dataset", "-d", type=str, default="lite",
                        choices=["lite", "dev"],
                        help="Dataset to use: lite (has working docker images) or dev")
    parser.add_argument("--split", "-s", type=str, default=None,
                        help="Dataset split (lite: test/dev, dev: dev). Default: test for lite, dev for dev")
    parser.add_argument("--self-critique", action="store_true",
                        help="Enable LLM self-critique for patch validation (adds extra LLM call)")
    parser.add_argument("--repo-embedding", action="store_true",
                        help="Use repo file embedding for localization")
    parser.add_argument("--no-harness", action="store_true",
                        help="Disable SWE-bench harness verification (faster, just generate patches)")
    args = parser.parse_args()

    print("=" * 60)
    print("Agentic Pipeline (Tool-Use Loop for High Pass@1)")
    print("=" * 60)

    # Load dataset
    if args.dataset == "lite":
        split = args.split or "test"  # default to test for lite
        print(f"\nLoading SWE-bench Lite dataset [{split}]...")
        all_instances = load_swe_bench_lite(split=split)
        dataset_name = "princeton-nlp/SWE-bench_Lite"
    else:
        split = args.split or "dev"  # default to dev for full
        print(f"\nLoading SWE-bench Full dataset [{split}]...")
        all_instances = load_swe_bench_dataset("princeton-nlp/SWE-bench", split=split)
        dataset_name = "princeton-nlp/SWE-bench"

    # Extract IDs for data leakage prevention
    exclude_ids = {inst.instance_id for inst in all_instances}
    print(f"Excluding {len(exclude_ids)} instances from RAG training corpus...")

    # Filter instances
    if args.instance_id:
        instance_ids = [x.strip() for x in args.instance_id.split(",")]
        instances = [i for i in all_instances if i.instance_id in instance_ids]
        if not instances:
            print(f"No instances found matching: {args.instance_id}")
            return
    else:
        instances = create_stratified_subset(all_instances, n=args.n)

    print(f"Running {len(instances)} instances")

    # Initialize LLMs (dual-model: small for localization, strong for patches)
    print(f"\nInitializing LLM providers...")
    print(f"  Localization model: {args.model}")
    print(f"  Patch model: {args.patch_model}")
    llm = LLMProvider(model=args.model)
    patch_llm = LLMProvider(model=args.patch_model)

    print("Initializing Agentic pipeline...")
    print(f"  Self-critique: {'ENABLED' if args.self_critique else 'DISABLED'}")
    print(f"  Repo embedding: {'ENABLED' if args.repo_embedding else 'DISABLED'}")
    print(f"  Harness verification: {'DISABLED' if args.no_harness else 'ENABLED'}")
    pipeline = AgenticPipeline(
        llm_provider=llm,
        patch_llm_provider=patch_llm,
        temperature=args.temperature,
        exclude_example_ids=exclude_ids,
        harness_dataset=dataset_name,
        harness_split=split,
        use_self_critique=args.self_critique,
        use_repo_embedding=args.repo_embedding,
        use_harness=not args.no_harness,
    )
    
    # Run experiment
    print("\nRunning experiment...")
    runner = ExperimentRunner(experiment_name=args.experiment_name)
    results = runner.run_baseline_experiment(
        instances=instances,
        pipeline=pipeline,
        attempts_per_instance=1,
    )
    
    # Strictify patches
    print("\nStrictifying patches for evaluation...")
    count = 0
    for inst in results['instances']:
        if inst['attempts'] and inst['attempts'][0].get('generated_patch'):
            original_patch = inst['attempts'][0]['generated_patch']
            repo = inst['repo']
            base_commit = inst.get('base_commit')

            if not base_commit:
                print(f"[ERROR] Missing base_commit for {inst['instance_id']}, skipping strictification")
                continue

            strict_patch = strictify_patch(original_patch, repo, base_commit)
            if strict_patch:
                inst['attempts'][0]['generated_patch'] = strict_patch
                count += 1
                print(f"[OK] Strictified {inst['instance_id']}")
            else:
                print(f"[WARN] Could not strictify {inst['instance_id']}")
    
    # Save updated results
    with open(runner.results_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n[SUCCESS] Saved results with strict patches to {runner.results_file}")


if __name__ == "__main__":
    main()
