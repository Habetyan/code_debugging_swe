import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import random
from src.data import load_swe_bench_lite
from src.llm.provider import LLMProvider
from src.retrieval.indexer import HybridRetriever
from src.retrieval.corpus import create_sample_corpus
from src.retrieval.source_code import RepoManager
from src.pipelines.rag_with_code import RAGWithCodePipeline
from src.evaluation.runner import ExperimentRunner

def main():
    parser = argparse.ArgumentParser(description="Run RAG with Code Pipeline (Pass@k)")
    parser.add_argument("--num_instances", "-n", type=int, default=10, help="Number of instances to run")
    parser.add_argument("--attempts", "-k", type=int, default=3, help="Attempts per instance (Pass@k)")
    parser.add_argument("--model", "-m", type=str, default="deepseek/deepseek-chat")
    parser.add_argument("--temperature", "-t", type=float, default=0.7)
    parser.add_argument("--output", "-o", type=str, default="results/rag_pass_k.json")
    args = parser.parse_args()

    print(f"Loading SWE-bench-Lite dataset...")
    dataset = load_swe_bench_lite()
    
    # Use the same 10 instances as before for consistency
    # We can filter by the IDs we used in full_eval_strict
    target_ids = [
        "mwaskom__seaborn-3010",
        "sphinx-doc__sphinx-8713",
        "scikit-learn__scikit-learn-10508",
        "django__django-11019",
        "astropy__astropy-7746",
        "matplotlib__matplotlib-23913",
        "pallets__flask-4045",
        "psf__requests-2148",
        "pytest-dev__pytest-5103",
        "scikit-learn__scikit-learn-15535",
        "sympy__sympy-12171"
    ]
    
    instances = [i for i in dataset if i.instance_id in target_ids]
    # If we have more than requested, slice it
    instances = instances[:args.num_instances]
    
    print(f"Running on {len(instances)} instances with k={args.attempts}...")

    print(f"\nInitializing LLM provider (model: {args.model})...")
    llm = LLMProvider(model=args.model)

    print("\nInitializing Retriever (Docs)...")
    corpus = create_sample_corpus()
    retriever = HybridRetriever(corpus)
    
    print("\nInitializing Repo Manager...")
    repo_manager = RepoManager()

    print("\nInitializing RAG Pipeline...")
    pipeline = RAGWithCodePipeline(
        llm_provider=llm,
        retriever=retriever,
        repo_manager=repo_manager,
        temperature=args.temperature
    )

    runner = ExperimentRunner(experiment_name="rag_pass_k")
    runner.results_file = args.output # Override output path

    print("\nRunning experiment...")
    results = runner.run_baseline_experiment(
        instances=instances,
        pipeline=pipeline,
        attempts_per_instance=args.attempts
    )

    print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()
