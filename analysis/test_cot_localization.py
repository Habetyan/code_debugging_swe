import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.swe_bench import load_swe_bench_lite
from src.retrieval.source_code import RepoManager
from src.llm import LLMProvider
from src.pipelines.cot import CoTPipeline


def extract_files_from_patch(patch: str) -> list[str]:
    """Extract file paths from a unified diff patch."""
    pattern = r'^(?:---|\+\+\+) [ab]/(.+\.py)$'
    files = set()
    for line in patch.split('\n'):
        match = re.match(pattern, line)
        if match:
            files.add(match.group(1))
    return list(files)


def main():
    print("Loading SWE-bench Lite...")
    instances = load_swe_bench_lite()
    
    # Test on first N instances
    n = 20
    test_instances = instances[:n]
    
    print(f"Testing CoT localization on {n} instances...\n")
    
    # Initialize components
    llm = LLMProvider()
    repo_manager = RepoManager()
    
    # Create CoT pipeline
    pipeline = CoTPipeline(
        llm_provider=llm,
        repo_manager=repo_manager,
    )
    
    correct = 0
    wrong_file = 0
    not_found = 0
    
    results = []
    
    for i, instance in enumerate(test_instances):
        print(f"[{i+1}/{n}] {instance.instance_id}")
        
        # Get the ground truth files from the patch
        gt_files = extract_files_from_patch(instance.patch)
        if not gt_files:
            print(f"  ⚠ Could not parse ground truth files from patch")
            continue
        
        # Clone the repo
        try:
            repo_path = repo_manager.get_repo_path(instance.repo, instance.base_commit)
        except Exception as e:
            print(f"  ✗ Failed to clone: {e}")
            continue
        
        # Run CoT localization (Stage 1 only)
        try:
            predicted_file = pipeline._localize_file(instance.problem_statement, repo_path)
        except Exception as e:
            print(f"  ✗ LLM Error: {e}")
            predicted_file = None
        
        # Check if prediction matches ground truth
        if predicted_file is None:
            status = "NOT_FOUND"
            not_found += 1
            print(f"  ✗ NOT FOUND")
            print(f"    Ground Truth: {gt_files}")
        elif predicted_file in gt_files:
            status = "CORRECT"
            correct += 1
            print(f"  ✓ CORRECT: {predicted_file}")
        else:
            status = "WRONG"
            wrong_file += 1
            print(f"  ✗ WRONG: {predicted_file}")
            print(f"    Ground Truth: {gt_files}")
        
        results.append({
            "instance_id": instance.instance_id,
            "status": status,
            "predicted": predicted_file,
            "ground_truth": gt_files,
        })
        print()
    
    # Summary
    total = correct + wrong_file + not_found
    print("=" * 60)
    print("COT LOCALIZATION SUMMARY")
    print("=" * 60)
    print(f"Total:     {total}")
    print(f"Correct:   {correct} ({100*correct/total:.1f}%)")
    print(f"Wrong:     {wrong_file} ({100*wrong_file/total:.1f}%)")
    print(f"Not Found: {not_found} ({100*not_found/total:.1f}%)")
    
    # Compare with previous baseline
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print("Grep-based localization: ~30% accuracy")
    print(f"CoT-based localization:  {100*correct/total:.1f}% accuracy")
    
    # Show failure cases for analysis
    if wrong_file + not_found > 0:
        print("\n" + "=" * 60)
        print("FAILURE CASES (for debugging)")
        print("=" * 60)
        for r in results:
            if r["status"] != "CORRECT":
                print(f"\n{r['instance_id']}:")
                print(f"  Predicted: {r['predicted']}")
                print(f"  Expected:  {r['ground_truth']}")


if __name__ == "__main__":
    main()
