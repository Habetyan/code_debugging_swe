"""
Adversarial Pipeline (The Internal Debate).
Generates a solution, debates it, and refines it.
"""
from typing import Optional
from pathlib import Path
from src.data import SWEBenchInstance
from src.llm import LLMProvider
from src.retrieval.source_code import RepoManager
from .cot import CoTPipeline, PipelineResult, COT_SYSTEM_PROMPT

ADVERSARIAL_DEBATE_PROMPT = """# Bug Report
{problem_statement}

# File Content ({file_path})
```python
{file_content}
```

# Task
Fix the bug in '{file_path}' using an Adversarial Debate process.

## Step 0: Localization Verification
Is '{file_path}' definitely the correct file to modify?
Does the bug report stack trace or description point to this file?
If unsure, explain why, but proceed with fixing this file as if it is the culprit.

## Step 1: Propose a Fix
Analyze the bug and propose an initial fix. Explain your reasoning.

## Step 2: Adversarial Critique
Play the role of a harsh code reviewer. 
- Why might this fix fail?
- Are there edge cases?
- Is it idiomatic?
- Does it break existing behavior?
- Is the localization correct?

## Step 3: Refine and Finalize
Based on the critique, improve your fix.
Output the FINAL solution as a Unified Diff.

Output Format:
<localization_check>
...
</localization_check>

<proposal>
...
</proposal>

<critique>
...
</critique>

<final_explanation>
...
</final_explanation>

```diff
--- a/{file_path}
+++ b/{file_path}
@@ ... @@
```
"""

class AdversarialPipeline(CoTPipeline):
    """
    Pipeline that forces an internal debate (Adversarial) before generating the final patch.
    """
    
    def run(self, instance: SWEBenchInstance) -> PipelineResult:
        try:
            repo_path = self.repo_manager.get_repo_path(instance.repo, instance.base_commit)
            
            # 1. Localization (Reuse CoT logic)
            target_file = self._localize_file(instance.problem_statement, repo_path)
            
            if not target_file:
                return PipelineResult(
                    instance_id=instance.instance_id,
                    generated_patch="",
                    ground_truth_patch=instance.patch,
                    raw_response="Localization Failed",
                    success=False,
                    error="Could not identify target file"
                )
            
            # 2. Read File
            file_path = Path(repo_path) / target_file
            content = file_path.read_text(errors='ignore')
            
            # 3. Generate with Adversarial Prompt
            prompt = ADVERSARIAL_DEBATE_PROMPT.format(
                problem_statement=instance.problem_statement,
                file_path=target_file,
                file_content=content
            )
            
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=COT_SYSTEM_PROMPT, 
                temperature=self.temperature,
                max_tokens=self.max_tokens + 1024 
            )
            
            # 4. Extract Patch
            patch = self._extract_patch(response)
            
            return PipelineResult(
                instance_id=instance.instance_id,
                generated_patch=patch,
                ground_truth_patch=instance.patch,
                raw_response=response,
                success=True
            )
            
        except Exception as e:
            return PipelineResult(
                instance_id=instance.instance_id,
                generated_patch="",
                ground_truth_patch=instance.patch,
                raw_response="",
                success=False,
                error=str(e)
            )
