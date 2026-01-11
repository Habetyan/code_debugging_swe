import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#!/usr/bin/env python3
"""
Run CoT-RAG Experiment

Entry point for running the Chain-of-Thought RAG pipeline.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_swe_bench_lite, create_stratified_subset
from src.llm import LLMProvider
from src.pipelines import CoTRAGPipeline
from src.evaluation import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="Run CoT-RAG experiment")
    parser.add_argument("--num-instances", "-n", type=int, default=5)
    parser.add_argument("--attempts", "-a", type=int, default=1)
    parser.add_argument("--model", "-m", type=str, default="deepseek/deepseek-chat")
    parser.add_argument("--retrieval-k", "-k", type=int, default=3)
    parser.add_argument("--temperature", "-t", type=float, default=0.7)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--experiment-name", "-e", type=str, default=None)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CoT-RACG Chain-of-Thought Experiment")
    print("=" * 60)
    
    # Load dataset
    print("\n📥 Loading SWE-bench-Lite dataset...")
    all_instances = load_swe_bench_lite()
    instances = create_stratified_subset(all_instances, n=args.num_instances)
    print(f"   Selected {len(instances)} instances")
    
    # Initialize
    print(f"\n🤖 Initializing LLM provider...")
    llm = LLMProvider(model=args.model)
    
    print(f"\n🧠 Initializing CoT-RAG pipeline...")
    print(f"   Retrieval k: {args.retrieval_k}")
    
    pipeline = CoTRAGPipeline(
        llm_provider=llm,
        retrieval_k=args.retrieval_k,
        temperature=args.temperature,
    )
    
    # Run with verbose output if requested
    if args.verbose and not args.dry_run:
        print("\n🔍 Running with verbose output...")
        for inst in instances[:1]:  # Just first one for verbose
            result = pipeline.run_with_details(inst)
            print(f"\n📋 Instance: {result.instance_id}")
            print(f"\n🔎 Diagnosis:\n{result.diagnosis[:500]}...")
            print(f"\n🔍 Queries: {result.queries}")
            print(f"\n📚 Retrieved: {result.retrieved_docs}")
            print(f"\n🔧 Patch preview:\n{result.generated_patch[:300]}...")
    
    # Run experiment
    print(f"\n🚀 Running experiment...")
    exp_name = args.experiment_name or "cot_rag"
    runner = ExperimentRunner(experiment_name=exp_name)
    results = runner.run_baseline_experiment(
        instances=instances,
        pipeline=pipeline,
        attempts_per_instance=args.attempts,
        dry_run=args.dry_run,
    )
    
    print("\n" + "=" * 60)
    print("📊 Experiment Summary")
    print("=" * 60)
    print(f"   Instances: {len(results['instances'])}")
    print(f"   Results: {runner.results_file}")
    print("\n✅ Complete!")


if __name__ == "__main__":
    main()
