import argparse
import json
import sys
import os
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_swe_bench_lite, get_instance_by_id

def read_file_content(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(errors='replace')
    except Exception:
        return ""

def main():
    parser = argparse.ArgumentParser(description="Analyze Pass@1 Results in Detail")
    parser.add_argument("--run-id", required=True, help="Run ID (e.g., full_eval_strict)")
    parser.add_argument("--instance-id", help="Specific instance to analyze (optional)")
    parser.add_argument("--logs-dir", default="logs", help="Logs directory")
    args = parser.parse_args()

    # Base path for logs
    # Structure: logs/run_evaluation/<run_id>/<model_name>/<instance_id>
    # We need to find the model name directory
    eval_base = Path(args.logs_dir) / "run_evaluation" / args.run_id
    
    if not eval_base.exists():
        print(f"Error: Run directory not found: {eval_base}")
        sys.exit(1)
        
    # Find model directory (usually just one)
    model_dirs = [d for d in eval_base.iterdir() if d.is_dir()]
    if not model_dirs:
        print("Error: No model directory found in run folder")
        sys.exit(1)
    
    model_dir = model_dirs[0] # Assume first one
    
    # Load dataset for context
    print("Loading dataset...")
    dataset = load_swe_bench_lite()
    
    # Get instances to analyze
    if args.instance_id:
        instance_dirs = [model_dir / args.instance_id]
        if not instance_dirs[0].exists():
            print(f"Instance {args.instance_id} not found in logs")
            sys.exit(1)
    else:
        instance_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()])

    print(f"\nAnalyzing {len(instance_dirs)} instances from {args.run_id}...\n")

    for inst_dir in instance_dirs:
        instance_id = inst_dir.name
        
        # Load report.json
        report_path = inst_dir / "report.json"
        report = {}
        if report_path.exists():
            try:
                with open(report_path) as f:
                    report = json.load(f)
                    # Report format is {instance_id: {...}}
                    report = report.get(instance_id, {})
            except:
                pass
        
        resolved = report.get("resolved", False)
        applied = report.get("patch_successfully_applied", False)
        
        status_icon = "[PASS]" if resolved else ("[APPLIED]" if applied else "[FAIL]")
        print(f"{'='*80}")
        print(f"{status_icon} Instance: {instance_id}")
        print(f"{'='*80}")
        
        # 1. Problem Description
        instance = get_instance_by_id(dataset, instance_id)
        if instance:
            print("\n Problem Statement (Snippet):")
            print("-" * 40)
            print('\n'.join(instance.problem_statement.splitlines()[:10]))
            print("... (truncated)")
        
        # 2. Generated Patch
        patch_path = inst_dir / "patch.diff"
        patch_content = read_file_content(patch_path)
        if patch_content:
            print("\n Generated Patch:")
            print("-" * 40)
            print(patch_content.strip())
        else:
            print("\n [NO PATCH] No patch found.")

        # 3. Test Output (Error Logs)
        test_output_path = inst_dir / "test_output.txt"
        test_log = read_file_content(test_output_path)
        
        if test_log:
            print("\n Test Output (Snippet):")
            print("-" * 40)
            
            # Try to find the error summary or failure section
            lines = test_log.splitlines()
            
            # Simple heuristic: Look for "FAIL" or "ERROR"
            error_lines = []
            capture = False
            for line in lines:
                if "FAIL" in line or "ERROR" in line or "Traceback" in line:
                    capture = True
                if capture:
                    error_lines.append(line)
                    if len(error_lines) > 20: # Limit output
                        error_lines.append("... (truncated)")
                        break
            
            if error_lines:
                print('\n'.join(error_lines))
            else:
                # If no obvious error, show last 20 lines
                print('\n'.join(lines[-20:]))
        else:
            print("\n [WARN] No test output found.")
            
        print("\n")

if __name__ == "__main__":
    main()
