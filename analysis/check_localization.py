import json
import re
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.swe_bench import load_swe_bench_lite

def get_modified_file(patch: str) -> str:
    """Extract the first modified file from a git diff."""
    if not patch:
        return None
    # Look for: diff --git a/path/to/file.py b/path/to/file.py
    match = re.search(r'diff --git a/(.*?) b/', patch)
    if match:
        return match.group(1)
    return None

def main():
    parser = argparse.ArgumentParser(description="Check Localization Accuracy")
    parser.add_argument("results_file", help="Path to the results JSON file")
    args = parser.parse_args()
    
    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"File not found: {results_path}")
        return

    print(f"Loading results from {results_path}...")
    with open(results_path) as f:
        data = json.load(f)
        
    print("Loading SWE-bench Lite dataset (for Ground Truth)...")
    dataset = load_swe_bench_lite()
    # Create lookup map
    # SWE-bench Lite instances have 'problem_statement' + 'patch' (gold solution)
    # We can infer the gold file from the gold patch
    gold_map = {}
    for inst in dataset:
        gold_file = get_modified_file(inst.patch)
        if gold_file:
            gold_map[inst.instance_id] = gold_file

    correct_count = 0
    total_count = 0
    
    print("\n" + "="*80)
    print(f"{'Instance ID':<40} | {'Status':<10} | {'Predicted File':<25}")
    print("="*80)
    
    for inst in data.get("instances", []):
        instance_id = inst["instance_id"]
        
        # Get ground truth
        gold_file = gold_map.get(instance_id)
        if not gold_file:
            continue # Should not happen usually
            
        # Get prediction
        attempts = inst.get("attempts", [])
        if not attempts:
            predicted_file = None
        else:
            patch = attempts[0].get("generated_patch", "")
            predicted_file = get_modified_file(patch)
            
        is_correct = (predicted_file == gold_file)
        
        status = "✅ PASS" if is_correct else "❌ FAIL"
        if predicted_file is None:
            status = "⚠ EMPTY"
            
        print(f"{instance_id:<40} | {status:<10} | {str(predicted_file):<25}")
        
        if not is_correct:
             print(f"   Expected: {gold_file}")
             
        if predicted_file is not None or is_correct: 
             # Count denominator as instances where we tried to generate something
             # Or should we count all? The list includes all run instances.
             pass
        
        total_count += 1
        if is_correct:
            correct_count += 1
            
    print("="*80)
    if total_count > 0:
        print(f"Localization Accuracy: {correct_count}/{total_count} ({correct_count/total_count:.1%})")
    else:
        print("No instances processed.")

if __name__ == "__main__":
    main()
