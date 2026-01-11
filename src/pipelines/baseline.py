from dataclasses import dataclass
from typing import Optional
from src.data import SWEBenchInstance
from src.llm import LLMProvider


SYSTEM_PROMPT = """You are an expert software engineer specializing in debugging and fixing Python code. 
Your task is to analyze bug reports and generate correct patches to fix the issues.

When generating a patch:
1. Carefully analyze the problem statement and any error traces
2. Identify the root cause of the bug
3. Generate a minimal, focused fix that addresses only the reported issue
4. Output your fix as a unified diff patch

Output format:
```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
 context line
-removed line
+added line
 context line
```
"""

FIX_PROMPT_TEMPLATE = """## Bug Report

{problem_statement}

{hints_section}

## Task

Analyze this bug and generate a patch to fix it. Provide your fix as a unified diff.
Think step-by-step about what might be causing this issue before proposing a fix.
"""


@dataclass
class PipelineResult:
    """Result from running the pipeline on an instance."""
    instance_id: str
    generated_patch: str
    ground_truth_patch: str
    raw_response: str
    success: bool = False
    error: Optional[str] = None


class BaselinePipeline:
    """
    Baseline pipeline for bug fixing using direct LLM prompting.
    No retrieval augmentation - relies solely on LLM's training knowledge.
    """
    
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        self.llm = llm_provider or LLMProvider()
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def run(self, instance: SWEBenchInstance) -> PipelineResult:
        """
        Run the baseline pipeline on a single bug instance.
        
        Args:
            instance: SWE-bench bug instance
            
        Returns:
            PipelineResult with generated patch and metadata
        """
        try:
            # Build the prompt
            hints_section = ""
            if instance.hints_text:
                hints_section = f"## Hints\n\n{instance.hints_text}"
            
            prompt = FIX_PROMPT_TEMPLATE.format(
                problem_statement=instance.problem_statement,
                hints_section=hints_section,
            )
            
            # Generate response
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Extract patch from response
            generated_patch = self._extract_patch(response)
            
            return PipelineResult(
                instance_id=instance.instance_id,
                generated_patch=generated_patch,
                ground_truth_patch=instance.patch,
                raw_response=response,
                success=True,
            )
            
        except Exception as e:
            return PipelineResult(
                instance_id=instance.instance_id,
                generated_patch="",
                ground_truth_patch=instance.patch,
                raw_response="",
                success=False,
                error=str(e),
            )
    
    def _extract_patch(self, response: str) -> str:
        """
        Extract the diff/patch from LLM response.
        Handles various formats and ensures valid patch output.
        """
        lines = response.split('\n')
        in_diff = False
        diff_lines = []
        
        for i, line in enumerate(lines):
            # Start of diff block (markdown code block)
            if line.strip().startswith('```diff') or line.strip() == '```diff':
                in_diff = True
                diff_lines = []  # Reset for new block
                continue
            
            # End of diff block
            if in_diff and line.strip() == '```':
                break
            
            # Collect diff lines
            if in_diff:
                diff_lines.append(line)
            
            # Also handle raw diff without code blocks
            if not in_diff and line.startswith('--- a/'):
                in_diff = True
                diff_lines = [line]
        
        # If no diff found in code blocks, try to find raw diff
        if not diff_lines:
            for i, line in enumerate(lines):
                if line.startswith('--- a/') or line.startswith('--- '):
                    # Found start of diff, collect until end
                    for j in range(i, len(lines)):
                        l = lines[j]
                        if l.startswith('---') or l.startswith('+++') or \
                           l.startswith('@@') or l.startswith(' ') or \
                           l.startswith('-') or l.startswith('+'):
                            diff_lines.append(l)
                        elif l.strip() == '' and diff_lines:
                            # Allow single blank lines within diff
                            if j + 1 < len(lines) and (
                                lines[j+1].startswith(' ') or 
                                lines[j+1].startswith('-') or 
                                lines[j+1].startswith('+') or
                                lines[j+1].startswith('@@')
                            ):
                                diff_lines.append(l)
                            else:
                                break
                        else:
                            break
                    break
        
        # Clean up the diff
        result = self._clean_diff(diff_lines)
        
        # Ensure trailing newline (required by patch command)
        if result and not result.endswith('\n'):
            result += '\n'
        
        return result
    
    def _clean_diff(self, lines: list) -> str:
        """
        Clean and validate diff lines.
        Fixes common LLM issues:
        - Interleaved -/+ lines (should be - first, then +)
        - Missing context lines
        - Malformed hunks
        """
        if not lines:
            return ""
        
        cleaned = []
        current_hunk_lines = []
        in_hunk = False
        
        for line in lines:
            # File headers
            if line.startswith('--- ') or line.startswith('+++ '):
                # Flush current hunk
                if current_hunk_lines:
                    cleaned.extend(self._reorder_hunk(current_hunk_lines))
                    current_hunk_lines = []
                cleaned.append(line)
                in_hunk = False
                continue
            
            # Hunk header
            if line.startswith('@@'):
                # Flush previous hunk
                if current_hunk_lines:
                    cleaned.extend(self._reorder_hunk(current_hunk_lines))
                    current_hunk_lines = []
                cleaned.append(line)
                in_hunk = True
                continue
            
            # Collect hunk lines
            if in_hunk:
                if line.startswith(' ') or line.startswith('-') or line.startswith('+'):
                    current_hunk_lines.append(line)
                elif line == '':
                    current_hunk_lines.append(' ')  # Convert to context
        
        # Flush final hunk
        if current_hunk_lines:
            cleaned.extend(self._reorder_hunk(current_hunk_lines))
        
        return '\n'.join(cleaned)
    
    def _reorder_hunk(self, lines: list) -> list:
        """
        Reorder hunk lines to ensure proper structure:
        - Context lines preserved in order
        - All '-' lines come before '+' lines within each change block
        """
        result = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Context line - just add it
            if line.startswith(' '):
                result.append(line)
                i += 1
                continue
            
            # Change block: collect all - and + lines until context
            minus_lines = []
            plus_lines = []
            
            while i < len(lines) and not lines[i].startswith(' '):
                if lines[i].startswith('-'):
                    minus_lines.append(lines[i])
                elif lines[i].startswith('+'):
                    plus_lines.append(lines[i])
                i += 1
            
            # Add minus lines first, then plus lines
            result.extend(minus_lines)
            result.extend(plus_lines)
        
        return result
    
    def run_multiple(
        self,
        instances: list[SWEBenchInstance],
        attempts_per_instance: int = 1,
    ) -> list[list[PipelineResult]]:
        """
        Run pipeline on multiple instances with multiple attempts each.
        
        Args:
            instances: List of bug instances
            attempts_per_instance: Number of generation attempts per instance
            
        Returns:
            List of result lists (one list of attempts per instance)
        """
        all_results = []
        
        for instance in instances:
            instance_results = []
            for _ in range(attempts_per_instance):
                result = self.run(instance)
                instance_results.append(result)
            all_results.append(instance_results)
        
        return all_results
