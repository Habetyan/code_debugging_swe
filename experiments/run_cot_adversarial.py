"""
Adversarial Pipeline Experiment Runner.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.swe_bench import load_swe_bench_dev, create_stratified_subset
from src.llm import LLMProvider
from src.pipelines.adversarial import AdversarialPipeline
from src.evaluation.runner import ExperimentRunner
from src.utils.fuzzy_patch import apply_patch_fuzzy
import subprocess
import json

def strictify_patch(patch: str, repo: str, base_commit: str) -> str:
    if not patch.strip(): return ""
    
    repo_name = repo.replace('/', '__')
    dir_name = f"{repo_name}__{base_commit[:8]}"
    repo_path = Path('repo_cache') / dir_name
    
    if not repo_path.exists(): return ""
    
    subprocess.run(['git', 'reset', '--hard', 'HEAD'], cwd=repo_path, capture_output=True)
    subprocess.run(['git', 'clean', '-fd'], cwd=repo_path, capture_output=True)
    
    success, msg, files = apply_patch_fuzzy(patch, str(repo_path), threshold=0.6)
    
    strict_patch = ""
    if success and files:
        result = subprocess.run(['git', 'diff'], cwd=repo_path, capture_output=True, text=True)
        strict_patch = result.stdout
    
    subprocess.run(['git', 'reset', '--hard', 'HEAD'], cwd=repo_path, capture_output=True)
    subprocess.run(['git', 'clean', '-fd'], cwd=repo_path, capture_output=True)
    
    return strict_patch

def main():
    parser = argparse.ArgumentParser(description="Run Adversarial Pipeline Experiment")
    parser.add_argument("--n", "-n", type=int, default=10)
    parser.add_argument("--model", "-m", type=str, default="deepseek/deepseek-chat")
    parser.add_argument("--experiment-name", "-e", type=str, default="adversarial_dev")
    # Added instance-id following run_baseline.py logic
    parser.add_argument("--instance-id", type=str, default=None)
    parser.add_argument("--temperature", "-t", type=float, default=0.2)
    args = parser.parse_args()

    print("=" * 60)
    print("Adversarial Pipeline Experiment (SWE-bench Full Dev)")
    print("=" * 60)

    try:
        all_dev_instances = load_swe_bench_dev()
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)
    
    # Filter or Create subset
    if args.instance_id:
        instance_ids = [x.strip() for x in args.instance_id.split(",")]
        # Using logic from run_baseline.py that is confirmed to work
        instances = [i for i in all_dev_instances if i.instance_id in instance_ids]
        if not instances:
            print(f"No instances found matching: {args.instance_id}")
            return
    else:
        instances = create_stratified_subset(all_dev_instances, n=args.n)

    print(f"Running {len(instances)} instances")

    llm = LLMProvider(model=args.model)
    pipeline = AdversarialPipeline(llm_provider=llm, temperature=args.temperature)
    runner = ExperimentRunner(experiment_name=args.experiment_name)
    
    results = runner.run_baseline_experiment(
        instances=instances,
        pipeline=pipeline,
        attempts_per_instance=1,
    )
    
    for inst in results['instances']:
        if inst['attempts'] and inst['attempts'][0].get('generated_patch'):
            original_patch = inst['attempts'][0]['generated_patch']
            strict = strictify_patch(original_patch, inst['repo'], inst.get('base_commit'))
            if strict: inst['attempts'][0]['generated_patch'] = strict

    with open(runner.results_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
