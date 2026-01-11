import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#!/usr/bin/env python3
"""
Compute Final Evaluation Metrics

Summarizes execution-based evaluation results from SWE-bench harness.
Calculates Pass@1, Pass@3, and generates a report.
"""

import json
import argparse
from pathlib import Path
from typing import Optional


def find_reports(logs_dir: str, run_id: str) -> list[Path]:
    """Find all report.json files for a given run."""
    base = Path(logs_dir) / "run_evaluation" / run_id
    if not base.exists():
        return []
    
    reports = list(base.rglob("report.json"))
    return reports


def compute_metrics(reports: list[Path]) -> dict:
    """Compute metrics from report files."""
    total = len(reports)
    resolved = 0
    patch_applied = 0
    
    results = []
    
    for report_path in reports:
        with open(report_path) as f:
            data = json.load(f)
        
        for instance_id, result in data.items():
            entry = {
                "instance_id": instance_id,
                "patch_applied": result.get("patch_successfully_applied", False),
                "resolved": result.get("resolved", False),
            }
            results.append(entry)
            
            if entry["patch_applied"]:
                patch_applied += 1
            if entry["resolved"]:
                resolved += 1
    
    return {
        "total_instances": total,
        "patches_applied": patch_applied,
        "resolved": resolved,
        "pass_at_1": resolved / total if total > 0 else 0,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute evaluation metrics")
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs",
        help="Logs directory"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID to analyze"
    )
    
    args = parser.parse_args()
    
    print(f"Looking for reports in: {args.logs_dir}/run_evaluation/{args.run_id}")
    
    reports = find_reports(args.logs_dir, args.run_id)
    
    if not reports:
        print("❌ No reports found. Make sure:")
        print("   1. Docker is running")
        print("   2. The evaluation completed")
        return
    
    print(f"Found {len(reports)} reports")
    
    metrics = compute_metrics(reports)
    
    print("\n" + "=" * 60)
    print("📊 EVALUATION METRICS")
    print("=" * 60)
    print(f"   Total Instances:  {metrics['total_instances']}")
    print(f"   Patches Applied:  {metrics['patches_applied']}")
    print(f"   Resolved (Pass):  {metrics['resolved']}")
    print(f"   Pass@1:           {metrics['pass_at_1']:.2%}")
    print("=" * 60)
    
    print("\nPer-Instance Results:")
    for r in metrics["results"]:
        status = "✅" if r["resolved"] else ("⚠️" if r["patch_applied"] else "❌")
        print(f"   {status} {r['instance_id']}")


if __name__ == "__main__":
    main()
