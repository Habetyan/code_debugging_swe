"""
Baseline Experiment: Runs the Baseline pipeline on SWE-bench.
Establishes a lower bound for performance comparison.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_swe_bench_dev, create_stratified_subset
from src.llm import LLMProvider
from src.pipelines import BaselinePipeline
from src.evaluation import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline experiment on SWE-bench Full dev split"
    )
    parser.add_argument(
        "--num-instances", "-n",
        type=int,
        default=5,
        help="Number of instances to process (default: 5)"
    )
    parser.add_argument(
        "--attempts", "-a",
        type=int,
        default=3,
        help="Attempts per instance for Pass@k (default: 3)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="deepseek/deepseek-chat",
        help="Model to use via OpenRouter"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run on single instance for testing"
    )
    parser.add_argument(
        "--experiment-name", "-e",
        type=str,
        default=None,
        help="Name for this experiment"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Baseline Experiment (SWE-bench Full Dev)")
    print("=" * 60)

    # Load dataset
    print("\nLoading SWE-bench Full dev dataset...")
    try:
        all_instances = load_swe_bench_dev()
        print(f"Loaded {len(all_instances)} total instances")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Make sure you have internet connection for first download.")
        sys.exit(1)
    
    # Create subset
    instances = create_stratified_subset(all_instances, n=args.num_instances)
    print(f"   Selected {len(instances)} instances for experiment")
    
    # Show repo distribution
    repos = {}
    for inst in instances:
        repos[inst.repo] = repos.get(inst.repo, 0) + 1
    print(f"   Repos: {dict(repos)}")
    
    # Initialize provider
    print(f"\nInitializing LLM provider...")
    print(f"   Model: {args.model}")
    
    try:
        llm = LLMProvider(model=args.model)
        print("Provider initialized successfully")
    except ValueError as e:
        print(f"Failed to initialize provider: {e}")
        print("Please add your API key to .env file:")
        print("OPENROUTER_API_KEY=your-key-here")
        sys.exit(1)
    
    # Initialize pipeline
    pipeline = BaselinePipeline(
        llm_provider=llm,
        temperature=args.temperature,
    )
    
    # Run experiment
    print(f"\nRunning experiment...")
    print(f"   Attempts per instance: {args.attempts}")
    print(f"   Dry run: {args.dry_run}")
    
    runner = ExperimentRunner(experiment_name=args.experiment_name)
    results = runner.run_baseline_experiment(
        instances=instances,
        pipeline=pipeline,
        attempts_per_instance=args.attempts,
        dry_run=args.dry_run,
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("Experiment Summary")
    print("=" * 60)
    print(f"Instances processed: {len(results['instances'])}")
    print(f"Results saved to: {runner.results_file}")
    
    # Show sample output
    if results['instances']:
        first = results['instances'][0]
        print(f"\n   Sample (first instance: {first['instance_id']}):")
        if first['attempts']:
            patch_preview = first['attempts'][0]['generated_patch'][:200]
            if patch_preview:
                print(f"Generated patch preview:\n   {patch_preview}...")
            else:
                print("No patch generated (empty)")
    
    print("\nExperiment complete!")
    print(f"Cache stats: {llm.get_cache_stats()}")


if __name__ == "__main__":
    main()
