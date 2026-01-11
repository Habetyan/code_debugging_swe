import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import argparse
import sys
from pathlib import Path
from difflib import SequenceMatcher

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_swe_bench_lite, get_instance_by_id

def main():
    parser = argparse.ArgumentParser(description="Inspect experiment results")
    parser.add_argument("results_file", help="Path to results JSON file")
    args = parser.parse_args()
    
    # Load results
    try:
        with open(args.results_file, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Failed to load results: {e}")
        sys.exit(1)
        
    # Load dataset (to get ground truth)
    print("Loading dataset for comparison...")
    dataset = load_swe_bench_lite()
    
    print(f"\n Inspection: {results['experiment_name']}")
    print("=" * 80)
    
    for inst_res in results['instances']:
        instance_id = inst_res['instance_id']
        instance = get_instance_by_id(dataset, instance_id)
        
        if not instance:
            print(f"Instance {instance_id} not found in dataset")
            continue
            
        print(f"\nInstance: {instance_id}")
        print(f"   Repo: {instance.repo}")
        print(f"   Problem: {instance.problem_statement.splitlines()[0]}...")
        
        # Get best attempt (first successful one)
        best_attempt = None
        for attempt in inst_res['attempts']:
            if attempt['success'] and attempt['generated_patch']:
                best_attempt = attempt
                break
        
        if not best_attempt:
            print("No successful patch generated")
            continue
            
        gen_patch = best_attempt['generated_patch']
        gt_patch = instance.patch
        
        # Calculate similarity
        similarity = SequenceMatcher(None, gen_patch, gt_patch).ratio()
        print(f"Similarity to Ground Truth: {similarity:.2%}")
        
        print("\nGround Truth Patch:")
        print("-" * 40)
        for line in gt_patch.splitlines()[:10]:
            print(f"{line}")
        if len(gt_patch.splitlines()) > 10:
            print("... (truncated)")
            
        print("\n🤖 Generated Patch:")
        print("-" * 40)
        for line in gen_patch.splitlines()[:10]:
            print(f"{line}")
        if len(gen_patch.splitlines()) > 10:
            print("... (truncated)")
            
        print("=" * 80)

if __name__ == "__main__":
    main()
