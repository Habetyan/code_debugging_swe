import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_swe_bench_lite, create_stratified_subset
from src.llm import LLMProvider
from src.pipelines import RAGWithCodePipeline
from src.evaluation import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="Run RAG with source code")
    parser.add_argument("--num-instances", "-n", type=int, default=5)
    parser.add_argument("--attempts", "-a", type=int, default=1)
    parser.add_argument("--model", "-m", type=str, default="deepseek/deepseek-chat")
    parser.add_argument("--retrieval-k", "-k", type=int, default=3)
    parser.add_argument("--temperature", "-t", type=float, default=0.3)
    parser.add_argument("--no-docs", action="store_true", help="Skip documentation retrieval")
    parser.add_argument("--experiment-name", "-e", type=str, default=None)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CoT-RACG: RAG with Source Code")
    print("=" * 60)
    
    # Load dataset
    print("\n📥 Loading SWE-bench-Lite dataset...")
    all_instances = load_swe_bench_lite()
    instances = create_stratified_subset(all_instances, n=args.num_instances)
    print(f"   Selected {len(instances)} instances")
    
    # Initialize
    print(f"\n🤖 Initializing LLM provider...")
    llm = LLMProvider(model=args.model)
    
    print(f"\n📁 Initializing RAG with source code...")
    print(f"   Include docs: {not args.no_docs}")
    
    pipeline = RAGWithCodePipeline(
        llm_provider=llm,
        retrieval_k=args.retrieval_k,
        temperature=args.temperature,
        include_docs=not args.no_docs,
    )
    
    # Run experiment
    print(f"\n🚀 Running experiment (this may take time to clone repos)...")
    
    exp_name = args.experiment_name or "rag_with_code"
    runner = ExperimentRunner(experiment_name=exp_name)
    results = runner.run_baseline_experiment(
        instances=instances,
        pipeline=pipeline,
        attempts_per_instance=args.attempts,
    )
    
    print("\n" + "=" * 60)
    print("📊 Experiment Summary")
    print("=" * 60)
    print(f"   Instances: {len(results['instances'])}")
    print(f"   Results: {runner.results_file}")
    
    # Show sample
    if results['instances']:
        first = results['instances'][0]
        print(f"\n   First instance: {first['instance_id']}")
        if first['attempts']:
            patch = first['attempts'][0].get('generated_patch', '')[:300]
            print(f"   Patch preview:\n{patch}...")
    
    print("\n✅ Complete!")


if __name__ == "__main__":
    main()
