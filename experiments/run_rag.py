import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import sys
import subprocess
from pathlib import Path
from src.data import load_swe_bench_lite, create_stratified_subset
from src.llm import LLMProvider
from src.pipelines.rag import RAGPipeline
from src.evaluation import ExperimentRunner
from src.utils.fuzzy_patch import apply_patch_fuzzy

def strictify_patch(patch: str, repo: str) -> str:
    """
    Converts a fuzzy patch to a strict git diff by applying it fuzzily 
    to a cached repo and then running git diff.
    """
    if not patch.strip():
        return ""
        
    # Find cached repo
    repo_name = repo.replace('/', '__')
    cache_dirs = [d for d in os.listdir('repo_cache') if repo_name in d]
    
    if not cache_dirs:
        return "" # Repo not cached, can't strictify
    
    repo_path = Path('repo_cache') / cache_dirs[0]
    
    # Reset repo first
    subprocess.run(['git', 'reset', '--hard', 'HEAD'], cwd=repo_path, capture_output=True)
    subprocess.run(['git', 'clean', '-fd'], cwd=repo_path, capture_output=True)
    
    # Apply fuzzy
    success, msg, files = apply_patch_fuzzy(patch, str(repo_path), threshold=0.5)
    
    strict_patch = ""
    if success:
        # Generate strict diff
        result = subprocess.run(
            ['git', 'diff'], 
            cwd=repo_path, 
            capture_output=True, 
            text=True
        )
        strict_patch = result.stdout
    
    # Reset repo again
    subprocess.run(['git', 'reset', '--hard', 'HEAD'], cwd=repo_path, capture_output=True)
    subprocess.run(['git', 'clean', '-fd'], cwd=repo_path, capture_output=True)
    
    return strict_patch

def main():
    parser = argparse.ArgumentParser(description="Run RAG Experiment (Best Configuration)")
    parser.add_argument("--num-instances", "-n", type=int, default=1, help="Number of instances to run")
    parser.add_argument("--instance-id", type=str, help="Specific instance ID to run")
    parser.add_argument("--model", "-m", type=str, default="deepseek/deepseek-chat")
    parser.add_argument("--temperature", "-t", type=float, default=0.2)
    parser.add_argument("--experiment-name", "-e", type=str, default="rag_best")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RAG Experiment (Graph + Code + Fuzzy Patching)")
    print("=" * 60)
    
    # Load dataset
    print(f"\\nLoading dataset...")
    all_instances = load_swe_bench_lite()
    
    if args.instance_id:
        instances = [i for i in all_instances if i.instance_id == args.instance_id]
        if not instances:
            print(f"[ERROR] Instance {args.instance_id} not found!")
            return
        print(f"Running single instance: {instances[0].instance_id}")
    else:
        instances = create_stratified_subset(all_instances, n=args.num_instances)
        print(f"Running {len(instances)} instances")
    
    # Initialize
    print(f"\\nInitializing LLM provider ({args.model})...")
    try:
        llm = LLMProvider(model=args.model)
    except Exception as e:
        print(f"[ERROR] Failed to init LLM: {e}")
        return
    
    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline(
        llm_provider=llm,
        temperature=args.temperature,
    )
    
    # Run
    print(f"\\nRunning experiment...")
    runner = ExperimentRunner(experiment_name=args.experiment_name)
    results = runner.run_baseline_experiment(
        instances=instances,
        pipeline=pipeline,
        attempts_per_instance=1,
    )
    
    # Strictify Patches
    print(f"\\nStrictifying patches for evaluation...")
    count = 0
    for inst in results['instances']:
        if inst['attempts'] and inst['attempts'][0].get('generated_patch'):
            original_patch = inst['attempts'][0]['generated_patch']
            repo = inst['repo']
            
            strict_patch = strictify_patch(original_patch, repo)
            
            if strict_patch:
                inst['attempts'][0]['generated_patch'] = strict_patch
                count += 1
                print(f"[OK] Strictified {inst['instance_id']}")
            else:
                print(f"[WARN] Could not strictify {inst['instance_id']} (Fuzzy apply failed or empty diff)")
    
    # Save updated results
    import json
    with open(runner.results_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\\n[SUCCESS] Saved results with strict patches to {runner.results_file}")

if __name__ == "__main__":
    main()
