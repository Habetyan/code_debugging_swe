"""
Multi-Viewpoint Pipeline (The 3D Glasses).
Analyzes the bug from Data, Logic, and Environment perspectives.
"""
from src.data import SWEBenchInstance
from .cot import CoTPipeline, PipelineResult
from pathlib import Path

MULTIVIEW_ANALYSIS_PROMPT = """# Bug Report
{problem_statement}

# File Content ({file_path})
```python
{file_content}
```

# Task
Fix the bug in '{file_path}' by analyzing it from three distinct perspectives.

## Perspective 1: Data Flow
Trace the data. What values are being passed? Are types correct? Is there data corruption?

## Perspective 2: Logic & Control Flow
Trace the execution path. Are conditionals correct? Is the loop logic sound? Are there off-by-one errors?

## Perspective 3: System, Environment & Localization
Consider external factors. Is this an OS issue? Resource limit?
CRITICAL: Confirm this is the correct file. Does the import structure and stack trace align with this file being the root cause?

## Synthesis & Fix
Combine insights from all three perspectives to determine the root cause and generate a fix.

Output Format:
<data_view>...</data_view>
<logic_view>...</logic_view>
<system_view>...</system_view>
<synthesis>...</synthesis>

```diff
--- a/{file_path}
+++ b/{file_path}
@@ ... @@
```
"""

class MultiViewPipeline(CoTPipeline):
    """
    Pipeline that forces multi-perspective analysis.
    """
    
    def run(self, instance: SWEBenchInstance) -> PipelineResult:
        try:
            repo_path = self.repo_manager.get_repo_path(instance.repo, instance.base_commit)
            
            # 1. Localization
            target_file = self._localize_file(instance.problem_statement, repo_path)
            if not target_file:
                return PipelineResult(instance.instance_id, "", instance.patch, "Localization Failed", False, "Localization Error")
            
            # 2. Read
            file_path = Path(repo_path) / target_file
            content = file_path.read_text(errors='ignore')
            
            # 3. Generate
            prompt = MULTIVIEW_ANALYSIS_PROMPT.format(
                problem_statement=instance.problem_statement,
                file_path=target_file,
                file_content=content
            )
            
            response = self.llm.generate(
                prompt=prompt,
                system_prompt="You are a holistic systems architect.",
                temperature=self.temperature,
                max_tokens=self.max_tokens + 1024
            )
            
            # 4. Patch
            patch = self._extract_patch(response)
            
            return PipelineResult(
                instance_id=instance.instance_id,
                generated_patch=patch,
                ground_truth_patch=instance.patch,
                raw_response=response,
                success=True
            )
            
        except Exception as e:
            return PipelineResult(instance.instance_id, "", instance.patch, "", False, str(e))
