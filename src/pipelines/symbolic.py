"""
Symbolic Trace Pipeline (The Step-by-Step Map).
Traces variable values line-by-line to find logic errors.
"""
from src.data import SWEBenchInstance
from .cot import CoTPipeline, PipelineResult, COT_SYSTEM_PROMPT
from pathlib import Path

SYMBOLIC_TRACE_PROMPT = """# Bug Report
{problem_statement}

# File Content ({file_path})
```python
{file_content}
```

# Task
Fix the bug by performing a Symbolic Trace of the problematic code.

## Step 0: Context Verification
Before tracing, verify: Does this file content match the location validation in the bug report?
If the stack trace mentions line numbers, finding them in this file.

## Step 1: Symbolic Trace
Create a table tracking variable values state-by-state.
Analyze the flow of data through the code relevant to the bug.
Identify EXACTLY where the state becomes invalid.

Format:
| Line | Code | Variables State | Comment |
|------|------|-----------------|---------|
| 10   | x = 5| x=5             | Init    |
...

## Step 2: Fix Plan
Based on the trace, which line is incorrect? How should the logic change?

## Step 3: Patch
Output the Unified Diff.

Output Format:
<context_verification>
...
</context_verification>

<trace>
...
</trace>

<plan>
...
</plan>

```diff
--- a/{file_path}
+++ b/{file_path}
@@ ... @@
```
"""

class SymbolicTracePipeline(CoTPipeline):
    """
    Pipeline that enforces explicit variable state tracing.
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
            prompt = SYMBOLIC_TRACE_PROMPT.format(
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
