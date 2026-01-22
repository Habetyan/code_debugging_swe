"""
CoT Experiment: Runs the Chain-of-Thought pipeline on SWE-bench.
Uses fuzzy patching to validate/strictify patches before submission.
"""
import argparse
import subprocess
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.swe_bench import load_swe_bench_dev, create_stratified_subset
from src.llm import LLMProvider
from src.pipelines.cot import CoTPipeline
from src.evaluation.runner import ExperimentRunner
from src.utils.fuzzy_patch import apply_patch_fuzzy, strictify_patch





def main():
    parser = argparse.ArgumentParser(description="Run CoT Pipeline Experiment (SWE-bench Full Dev)")
    parser.add_argument("--n", "-n", type=int, default=10,
                        help="Number of instances to run")
    parser.add_argument("--model", "-m", type=str, default="deepseek/deepseek-chat",
                        help="Model to use")
    parser.add_argument("--experiment-name", "-e", type=str, default="cot_dev",
                        help="Name for the experiment")
    parser.add_argument("--instance-id", type=str, default=None,
                        help="Specific instance ID to run (comma-separated for multiple)")
    parser.add_argument("--temperature", "-t", type=float, default=0.2,
                        help="LLM temperature")
    args = parser.parse_args()

    print("=" * 60)
    print("CoT Pipeline Experiment (SWE-bench Full Dev)")
    print("=" * 60)

    # Load dataset
    print("\nLoading SWE-bench Full dev dataset...")
    all_dev_instances = load_swe_bench_dev()

    # Extract dev IDs for data leakage prevention
    dev_ids = {inst.instance_id for inst in all_dev_instances}
    print(f"Excluding {len(dev_ids)} dev instances from RAG training corpus...")

    # Filter instances
    if args.instance_id:
        instance_ids = [x.strip() for x in args.instance_id.split(",")]
        instances = [i for i in all_dev_instances if i.instance_id in instance_ids]
        if not instances:
            print(f"No instances found matching: {args.instance_id}")
            return
    else:
        instances = create_stratified_subset(all_dev_instances, n=args.n)

    print(f"Running {len(instances)} instances")

    # Initialize LLM
    print(f"\nInitializing LLM provider ({args.model})...")
    llm = LLMProvider(model=args.model)

    # Initialize CoT Pipeline
    print("Initializing CoT pipeline...")
    pipeline = CoTPipeline(
        llm_provider=llm,
        temperature=args.temperature,
        exclude_example_ids=dev_ids,
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
