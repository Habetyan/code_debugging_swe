import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import os
from pathlib import Path
from src.utils.fuzzy_patch import test_patch_fuzzy
from src.retrieval.source_code import RepoManager

def main():
    baseline_file = 'results/baseline_20260110_231343.json'
    print(f"Checking baseline patches from {baseline_file}...")
    
    with open(baseline_file) as f:
        data = json.load(f)
    
    repo_manager = RepoManager()
    
    total = 0
    applied = 0
    
    for inst in data['instances']:
        instance_id = inst['instance_id']
        repo = inst['repo']
        # We need the base commit, but it's not in the results file (oops).
        # We'll load it from the dataset.
        pass

    # Load dataset to get commits
    from src.data import load_swe_bench_lite
    dataset = load_swe_bench_lite()
    dataset_map = {i.instance_id: i for i in dataset}
    
    print(f"{'Instance':<40} | {'Strict Apply':<15} | {'Fuzzy Apply':<15}")
    print("-" * 75)
    
    for inst in data['instances']:
        instance_id = inst['instance_id']
        if instance_id not in dataset_map:
            continue
            
        swe_inst = dataset_map[instance_id]
        repo_path = repo_manager.get_repo_path(swe_inst.repo, swe_inst.base_commit)
        
        if not repo_path:
            print(f"{instance_id:<40} | {'Repo Error':<15} | {'-'}")
            continue
            
        patch = inst['attempts'][0].get('generated_patch', '')
        if not patch:
            print(f"{instance_id:<40} | {'No Patch':<15} | {'-'}")
            continue
            
        # Check Strict (Threshold 1.0)
        strict_ok, _ = test_patch_fuzzy(patch, str(repo_path), threshold=1.0)
        
        # Check Fuzzy (Threshold 0.6)
        fuzzy_ok, _ = test_patch_fuzzy(patch, str(repo_path), threshold=0.6)
        
        print(f"{instance_id:<40} | {'YES' if strict_ok else 'NO':<15} | {'YES' if fuzzy_ok else 'NO':<15}")
        
        total += 1
        if strict_ok:
            applied += 1
            
    print("-" * 75)
    print(f"Total: {total}")
    print(f"Strict Application Rate: {applied}/{total} ({applied/total*100:.1f}%)")

if __name__ == "__main__":
    main()
