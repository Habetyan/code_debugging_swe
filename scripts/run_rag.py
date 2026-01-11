import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_swe_bench_lite, create_stratified_subset
from src.llm import LLMProvider
from src.pipelines import RAGPipeline
from src.evaluation import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG experiment on SWE-bench-Lite"
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
        "--retrieval-k", "-k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )
    parser.add_argument(
        "--retrieval-method",
        type=str,
        choices=["embedding", "bm25", "hybrid"],
        default="hybrid",
        help="Retrieval method (default: hybrid)"
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
    print("CoT-RACG RAG Experiment")
    print("=" * 60)
    
    # Load dataset
    print("\n Loading SWE-bench-Lite dataset...")
    try:
        all_instances = load_swe_bench_lite()
        print(f"Loaded {len(all_instances)} total instances")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)
    
    # Create subset
    instances = create_stratified_subset(all_instances, n=args.num_instances)
    print(f"Selected {len(instances)} instances for experiment")
    
    # Initialize provider
    print(f"Initializing LLM provider...")
    print(f"Model: {args.model}")
    
    try:
        llm = LLMProvider(model=args.model)
        print("Provider initialized successfully")
    except ValueError as e:
        print(f" {e}")
        sys.exit(1)
    
    # Initialize RAG pipeline
    print(f"Initializing RAG pipeline...")
    print(f"Retrieval k: {args.retrieval_k}")
    print(f"Retrieval method: {args.retrieval_method}")
    
    pipeline = RAGPipeline(
        llm_provider=llm,
        retrieval_k=args.retrieval_k,
        retrieval_method=args.retrieval_method,
        temperature=args.temperature,
    )
    
    # Run experiment
    print("Running experiment...")
    print(f"Attempts per instance: {args.attempts}")
    print(f"Dry run: {args.dry_run}")
    
    exp_name = args.experiment_name or f"rag_k{args.retrieval_k}_{args.retrieval_method}"
    runner = ExperimentRunner(experiment_name=exp_name)
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
    
    print("Experiment complete!")
    print(f"Cache stats: {llm.get_cache_stats()}")


if __name__ == "__main__":
    main()
