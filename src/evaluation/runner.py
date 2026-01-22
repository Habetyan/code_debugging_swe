import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Optional

from src.data import SWEBenchInstance
from src.pipelines.baseline import BaselinePipeline


class ExperimentRunner:
    """
    Runs experiments on SWE-bench instances and saves results.
    """
    
    def __init__(
        self,
        output_dir: str = "results",
        experiment_name: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"baseline_{timestamp}"
        
        self.experiment_name = experiment_name
        self.results_file = self.output_dir / f"{experiment_name}.json"
    
    def run_baseline_experiment(
        self,
        instances: list[SWEBenchInstance],
        pipeline: Optional[BaselinePipeline] = None,
        attempts_per_instance: int = 3,
        dry_run: bool = False,
        harness: any = None,
    ) -> dict:
        """Run the experiment on a list of instances."""
        results = {
            "experiment_name": self.experiment_name,
            "pipeline_type": pipeline.__class__.__name__.lower(),
            "config": {
                "attempts_per_instance": attempts_per_instance,
                "model": getattr(pipeline.llm, "model", "unknown") if hasattr(pipeline, "llm") else "unknown",
                "temperature": getattr(pipeline, "temperature", 0.0),
            },
            "timestamp": datetime.now().isoformat(),
            "instances": [],
        }
        
        instances_to_run = instances[:1] if dry_run else instances
        
        for instance in tqdm(instances_to_run, desc="Processing instances"):
            instance_result = {
                "instance_id": instance.instance_id,
                "repo": instance.repo,
                "base_commit": instance.base_commit,
                "attempts": [],
                "verified": False
            }
            
            num_attempts = 1 if dry_run else attempts_per_instance
            
            for attempt_idx in range(num_attempts):
                result = pipeline.run(instance)
                
                # Verify if harness is provided
                verified = False
                if harness and result.success and result.generated_patch:
                    print(f"\n[HARNESS] Verifying patch for {instance.instance_id}...")
                    try:
                        verified = harness.verify_patch(instance, result.generated_patch)
                        if verified:
                            print(f"[HARNESS] RESOLVED: {instance.instance_id}")
                        else:
                            print(f"[HARNESS] FAILED: {instance.instance_id}")
                    except Exception as e:
                        print(f"[HARNESS] ERROR: Verification failed: {e}")
                
                instance_result["attempts"].append({
                    "attempt": attempt_idx + 1,
                    "success": result.success,
                    "generated_patch": result.generated_patch,
                    "error": result.error,
                    "verified": verified
                })
                
                if verified:
                    instance_result["verified"] = True
            
            results["instances"].append(instance_result)
            
            # Save checkpoints after each instance
            self._save_results(results)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: dict):
        """Save results to JSON file."""
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {self.results_file}")
    
    def load_results(self) -> dict:
        """Load results from file."""
        with open(self.results_file, 'r') as f:
            return json.load(f)
