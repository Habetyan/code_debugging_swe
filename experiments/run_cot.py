import argparse
import os
import subprocess
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.swe_bench import load_swe_bench_lite, create_stratified_subset
from src.llm import LLMProvider
from src.pipelines.cot import CoTPipeline
from src.evaluation.runner import ExperimentRunner
from src.utils.fuzzy_patch import apply_patch_fuzzy


def strictify_patch(patch: str, repo: str, base_commit: str) -> str:
    """
    Converts a fuzzy patch to a strict git diff by applying it fuzzily 
    to a cached repo and then running git diff.
    """
    if not patch.strip():
        return ""
        
    # Find cached repo
    repo_name = repo.replace('/', '__')
    dir_name = f"{repo_name}__{base_commit[:8]}"
    repo_path = Path('repo_cache') / dir_name
    
    if not repo_path.exists():
        # Fallback: try finding any dir with repo name if specific commit not found 
        # (though this should ideally not happen if pipeline just ran)
        if not os.path.exists('repo_cache'):
            return ""
        cache_dirs = [d for d in os.listdir('repo_cache') if repo_name in d]
        if not cache_dirs:
            return ""
        repo_path = Path('repo_cache') / cache_dirs[0]
    
    # Reset repo first
    subprocess.run(['git', 'reset', '--hard', 'HEAD'], cwd=repo_path, capture_output=True)
    subprocess.run(['git', 'clean', '-fd'], cwd=repo_path, capture_output=True)
    
    # Apply fuzzy
    success, msg, files = apply_patch_fuzzy(patch, str(repo_path), threshold=0.6)
    
    strict_patch = ""
    if success and files:
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
    parser = argparse.ArgumentParser(description="Run CoT Pipeline Experiment")
    parser.add_argument("--n", "-n", type=int, default=10,
                        help="Number of instances to run")
    parser.add_argument("--model", "-m", type=str, default="deepseek/deepseek-chat",
                        help="Model to use")
    parser.add_argument("--experiment-name", "-e", type=str, default="cot_run",
                        help="Name for the experiment")
    parser.add_argument("--instance-id", type=str, default=None,
                        help="Specific instance ID to run (comma-separated for multiple)")
    parser.add_argument("--temperature", "-t", type=float, default=0.2,
                        help="LLM temperature")
    args = parser.parse_args()

    print("=" * 60)
    print("CoT Pipeline Experiment (Two-Stage: Localize + Fix)")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading dataset...")
    instances = load_swe_bench_lite()
    
    # Filter instances
    if args.instance_id:
        instance_ids = [x.strip() for x in args.instance_id.split(",")]
        instances = [i for i in instances if i.instance_id in instance_ids]
        if not instances:
            print(f"No instances found matching: {args.instance_id}")
            return
    else:
        instances = create_stratified_subset(instances, n=args.n)
    
    print(f"Running {len(instances)} instances")
    
    # Initialize LLM
    print(f"\nInitializing LLM provider ({args.model})...")
    llm = LLMProvider(model=args.model)
    
    # Initialize CoT Pipeline
    print("Initializing CoT pipeline...")
    pipeline = CoTPipeline(
        llm_provider=llm,
        temperature=args.temperature,
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
            
            base_commit = inst['base_commit']
            
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
