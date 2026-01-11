import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#!/usr/bin/env python3
"""
Run Graph-Enhanced RAG Experiment
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_swe_bench_lite
from src.llm import LLMProvider
from src.pipelines import GraphRAGPipeline
from src.evaluation import ExperimentRunner
from src.retrieval import HybridRetriever
from src.retrieval.corpus import create_sample_corpus

def main():
    parser = argparse.ArgumentParser(description="Run Graph RAG experiment")
    parser.add_argument("--instance-id", type=str, default="scikit-learn__scikit-learn-10508")
    parser.add_argument("--model", "-m", type=str, default="deepseek/deepseek-chat")
    parser.add_argument("--temperature", "-t", type=float, default=0.2)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Graph-Enhanced RAG Experiment")
    print("=" * 60)
    
    # Load dataset
    print(f"\nLoading instance {args.instance_id}...")
    all_instances = load_swe_bench_lite()
    instances = [i for i in all_instances if i.instance_id == args.instance_id]
    
    if not instances:
        print(f"Instance {args.instance_id} not found!")
        return

    print(f"Found instance: {instances[0].instance_id}")
    
    # Initialize
    print(f"\nInitializing LLM provider (model: {args.model})...")
    llm = LLMProvider(model=args.model)
    
    print("Initializing Retriever (Docs)...")
    corpus = create_sample_corpus()
    retriever = HybridRetriever(corpus)
    
    print("Initializing Graph RAG pipeline...")
    pipeline = GraphRAGPipeline(
        llm_provider=llm,
        retriever=retriever,
        temperature=args.temperature,
    )
    
    # Run
    print(f"\nRunning experiment...")
    runner = ExperimentRunner(experiment_name="graph_rag")
    results = runner.run_baseline_experiment(
        instances=instances,
        pipeline=pipeline,
        attempts_per_instance=1,
    )
    
    print("\n" + "=" * 60)
    print("Experiment Summary")
    print("=" * 60)
    print(f"Results: {runner.results_file}")
    
    # Check fuzzy apply
    from src.utils.fuzzy_patch import test_patch_fuzzy
    import os
    
    patch = results['instances'][0]['attempts'][0].get('generated_patch', '')
    repo = instances[0].repo.replace('/', '__')
    cache_dirs = [d for d in os.listdir('repo_cache') if repo in d]
    
    if cache_dirs:
        repo_path = f'repo_cache/{cache_dirs[0]}'
            
        success, msg = test_patch_fuzzy(patch, repo_path, threshold=0.6)
        print(f"\nFuzzy Patch Application: {'SUCCESS' if success else 'FAILED'}")
        print(f"Message: {msg}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
