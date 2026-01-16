"""
Rubric Filtered Pipeline (The Quality Checklist).
Generates a solution and safeguards it with a quality checklist.
"""
from typing import Optional
from pathlib import Path
from src.data import SWEBenchInstance
from src.llm import LLMProvider
from .cot import CoTPipeline, PipelineResult, COT_SYSTEM_PROMPT

RUBRIC_CHECKLIST = """
RUBRIC CHECKLIST:
1. Localization: Is the fix applied to the correct file and function based on the bug report?
2. Correctness: Does the fix directly address the reported issue?
3. Safety: Does it avoid introducing new bugs?
4. Minimal: Is the change minimal and focused?
5. Readable: Is the code clean and easy to understand?
6. Testable: Can this change be verified?
"""

RUBRIC_EVAL_PROMPT = """# Bug Report
{problem_statement}

# Original File ({file_path})
```python
{file_content}
```

# Proposed Fix
{generated_patch}

# Task
Evaluate this fix against the following Rubric:
{rubric}

If the fix fails ANY criteria, explain why and generate a BETTER fix.
If the fix passes ALL criteria, output "APPROVED" and repeat the patch.

Output Format:
<evaluation>
1. Localization: [Pass/Fail] - Reason
2. Correctness: [Pass/Fail] - Reason
...
</evaluation>

<final_decision>
[APPROVED / REJECTED]
</final_decision>

```diff
--- a/{file_path}
+++ b/{file_path}
@@ ... @@
```
"""

class RubricPipeline(CoTPipeline):
    """
    Pipeline that checks generated patches against a quality rubric.
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
            
            # 3. Initial Generation (using base CoT)
            cot_prompt = self._get_cot_prompt(instance.problem_statement, target_file, content)
            initial_response = self.llm.generate(cot_prompt, COT_SYSTEM_PROMPT, self.temperature, self.max_tokens)
            initial_patch = self._extract_patch(initial_response)
            
            if not initial_patch:
                return PipelineResult(instance.instance_id, "", instance.patch, initial_response, False, "No initial patch generated")

            # 4. Rubric Evaluation & Refinement
            rubric_prompt = RUBRIC_EVAL_PROMPT.format(
                problem_statement=instance.problem_statement,
                file_path=target_file,
                file_content=content,
                generated_patch=initial_patch,
                rubric=RUBRIC_CHECKLIST
            )
            
            final_response = self.llm.generate(
                prompt=rubric_prompt,
                system_prompt="You are a strict QA engineer.",
                temperature=0.1, # Lower temp for evaluation
                max_tokens=self.max_tokens
            )
            
            final_patch = self._extract_patch(final_response)
            if not final_patch:
                # Fallback to initial if refinement fails to output patch
                final_patch = initial_patch
            
            return PipelineResult(
                instance_id=instance.instance_id,
                generated_patch=final_patch,
                ground_truth_patch=instance.patch,
                raw_response=final_response,
                success=True
            )
            
        except Exception as e:
            return PipelineResult(instance.instance_id, "", instance.patch, "", False, str(e))

    def _get_cot_prompt(self, problem, file_path, content):
        from .cot import COT_FIX_PROMPT
        return COT_FIX_PROMPT.format(
            problem_statement=problem,
            file_path=file_path,
            file_content=content
        )
