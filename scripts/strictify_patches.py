import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
Strictify Patches

Converts fuzzy-matched patches into strict git diffs that can be applied by `git apply`.
This is necessary for the Docker-based evaluation harness.
"""

import json
import subprocess
import os
from pathlib import Path
from src.utils.fuzzy_patch import apply_patch_fuzzy


def strictify_patches(input_file: str, output_file: str):
    """
    Load results, apply patches fuzzily to cached repos, generate strict diffs, and save.
    """
    with open(input_file) as f:
        data = json.load(f)
    
    print(f"Loaded {len(data['instances'])} instances from {input_file}", flush=True)
    
    converted_count = 0
    
    for inst in data['instances']:
        instance_id = inst['instance_id']
        print(f"Processing {instance_id}...", flush=True)
        patch = inst['attempts'][0].get('generated_patch', '')
        
        if not patch.strip():
            print(f"Skipping {instance_id}: Empty patch")
            continue
        
        # Find cached repo
        repo = inst['repo'].replace('/', '__')
        cache_dirs = [d for d in os.listdir('repo_cache') if repo in d]
        
        if not cache_dirs:
            print(f"Skipping {instance_id}: Repo not cached")
            continue
        
        repo_path = Path('repo_cache') / cache_dirs[0]
        
        # Reset repo first
        subprocess.run(['git', 'reset', '--hard', 'HEAD'], cwd=repo_path, capture_output=True)
        subprocess.run(['git', 'clean', '-fd'], cwd=repo_path, capture_output=True)
        
        # Apply fuzzy
        success, msg, files = apply_patch_fuzzy(patch, str(repo_path), threshold=0.5)
        
        if success:
            # Generate strict diff
            result = subprocess.run(
                ['git', 'diff'], 
                cwd=repo_path, 
                capture_output=True, 
                text=True
            )
            strict_patch = result.stdout
            
            if strict_patch.strip():
                inst['attempts'][0]['generated_patch'] = strict_patch
                converted_count += 1
                print(f"✓ Converted {instance_id}")
            else:
                print(f"⚠ Applied but empty diff for {instance_id}")
        else:
            print(f"✗ Failed to apply fuzzy patch for {instance_id}: {msg}")
            
        # Reset repo again
        subprocess.run(['git', 'reset', '--hard', 'HEAD'], cwd=repo_path, capture_output=True)
        subprocess.run(['git', 'clean', '-fd'], cwd=repo_path, capture_output=True)
    
    # Save new results
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSaved {converted_count} strictified patches to {output_file}")


if __name__ == "__main__":
    strictify_patches("results/full_eval.json", "results/full_eval_strict.json")
