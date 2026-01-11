import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_swe_bench_lite, create_stratified_subset
from src.llm import LLMProvider
from src.pipelines import SearchReplacePipeline
from src.evaluation import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="Run search-replace experiment")
    parser.add_argument("--num-instances", "-n", type=int, default=5)
    parser.add_argument("--attempts", "-a", type=int, default=1)
    parser.add_argument("--model", "-m", type=str, default="deepseek/deepseek-chat")
    parser.add_argument("--temperature", "-t", type=float, default=0.2)
    parser.add_argument("--experiment-name", "-e", type=str, default=None)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Search-and-Replace Experiment")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading SWE-bench-Lite dataset...")
    all_instances = load_swe_bench_lite()
    instances = create_stratified_subset(all_instances, n=args.num_instances)
    print(f"Selected {len(instances)} instances")
    
    # Initialize
    print(f"\nInitializing LLM provider (model: {args.model})...")
    llm = LLMProvider(model=args.model)
    
    print("Initializing Search-Replace pipeline...")
    pipeline = SearchReplacePipeline(
        llm_provider=llm,
        temperature=args.temperature,
    )
    
    # Run
    print(f"\nRunning experiment...")
    exp_name = args.experiment_name or "search_replace"
    runner = ExperimentRunner(experiment_name=exp_name)
    results = runner.run_baseline_experiment(
        instances=instances,
        pipeline=pipeline,
        attempts_per_instance=args.attempts,
    )
    
    print("\n" + "=" * 60)
    print("Experiment Summary")
    print("=" * 60)
    print(f"Instances: {len(results['instances'])}")
    print(f"Results: {runner.results_file}")
    print("\nDone!")


if __name__ == "__main__":
    main()
