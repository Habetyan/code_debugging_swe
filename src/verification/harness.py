"""
Verification Harness: Wrapper around official SWE-bench harness for verifying patches.
"""
import json
import sys
import subprocess
import time
from pathlib import Path

class VerificationHarness:
    """
    Wrapper around official SWE-bench harness for verifying patches.
    """

    def __init__(self, output_dir: str = "verification_results",
                 dataset_name: str = "princeton-nlp/SWE-bench",
                 split: str = "dev"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name
        self.split = split

    def verify_patch(self, instance, patch: str) -> bool:
        success, _ = self.verify_patch_with_logs(instance, patch)
        return success

    def verify_patch_with_logs(self, instance, patch: str) -> tuple[bool, str]:
        """
        Run verification and return (success, log_content).
        """
        # 1. Create prediction file
        pred_data = {
            instance.instance_id: {
                "model_name_or_path": "agent",
                "model_patch": patch,
                "instance_id": instance.instance_id
            }
        }

        run_id = f"verify_{instance.instance_id}_{int(time.time())}"
        pred_file = self.output_dir / f"{run_id}_pred.json"

        with open(pred_file, "w") as f:
            json.dump(pred_data, f)

        report_dir = self.output_dir / "reports"

        cmd = [
            sys.executable, "-m", "swebench.harness.run_evaluation",
            "--dataset_name", self.dataset_name,
            "--split", self.split,
            "--instance_ids", instance.instance_id,
            "--predictions_path", str(pred_file),
            "--max_workers", "1",
            "--force_rebuild", "False",
            "--cache_level", "env",
            "--clean", "False",
            "--open_file_limit", "4096",
            "--run_id", run_id,
            "--timeout", "1800",
            "--namespace", "swebench",
            "--instance_image_tag", "latest",
            "--env_image_tag", "latest",
            "--report_dir", str(report_dir)
        ]
        
        print(f"  [Harness] Launching subprocess: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, capture_output=True, text=True)
        except Exception as e:
            return False, f"Subprocess Error: {e}"

        # 3. Determine Result
        report_path = report_dir / run_id / "agent" / instance.instance_id / "report.json"
        
        log_path = Path("logs/run_evaluation") / run_id / "agent" / instance.instance_id / "run_instance.log"
        
        log_content = ""
        if log_path.exists():
            try: log_content = log_path.read_text()
            except: log_content = "Could not read log file."
        else:
            log_content = f"Log file not found at {log_path}"
            
        success = False
        if report_path.exists():
            try:
                with open(report_path) as f:
                    report = json.load(f)
                    success = report.get(instance.instance_id, {}).get("resolved", False)
            except: pass
        else:
             print(f"  [Harness] Report file not found at {report_path}")
             # Fallback log parsing
             if f"Result for {instance.instance_id}: resolved: True" in log_content:
                 print(f"  [Harness] Found success confirmation in logs!")
                 success = True
             elif log_path.exists():
                 print(f"  [Harness] Log indicates failure or no result found.")
        
        return success, log_content
