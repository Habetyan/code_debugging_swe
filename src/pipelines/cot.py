
import difflib
import re
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

from src.data import SWEBenchInstance
from src.llm import LLMProvider
from src.retrieval.source_code import RepoManager
from src.retrieval.example_retriever import ExampleRetriever
from src.pipelines.baseline import BaselinePipeline, PipelineResult
from src.verification.harness import VerificationHarness

LOCALIZATION_SYSTEM_PROMPT = """You are an expert software engineer specializing in bug localization.
Your task is to identify the SINGLE most likely file that needs to be modified to fix the bug.
Think step-by-step and explain your reasoning before giving the final answer."""

LOCALIZATION_PROMPT_TEMPLATE = """# Bug Report
{problem_statement}

# Repository Files
{file_list}

# Task
Analyze the bug report and identify the file that needs to be modified.

Think step-by-step:
1. What component/module does this bug affect? (e.g., validation, parsing, rendering)
2. What classes, functions, or error messages are mentioned?
3. If there's a stacktrace, which file appears closest to the actual error?
4. Which file in the repository implements that functionality?

Based on your analysis, output the file path:
TARGET_FILE: path/to/file.py
"""

FIX_SYSTEM_PROMPT = """You are an expert software engineer fixing bugs.

IMPORTANT:
1. Use SEARCH/REPLACE blocks to modify the code.
2. The SEARCH block must be an EXACT COPY of the lines in the original file (including indentation).
3. The REPLACE block contains the new code.

FORMAT:
<<<< SEARCH
original code lines
====
new code lines
>>>> REPLACE
"""

RECTIFICATION_SYSTEM_PROMPT = """You are an expert software engineer debugging a failed fix.
Your previous patch was applied, but the tests failed.
Analyze the test output and the original code to fix the logic.

OUTPUT FORMAT:
Provide a NEW SEARCH/REPLACE block to update the file correctly.
"""

RECTIFICATION_PROMPT_TEMPLATE = """# Original Bug
{problem_statement}

# Your Previous Fix
{previous_patch}

# Test Failure Output
{test_output}

# Primary File Content (With your previous fix applied)
```python
{file_content}
```

# Instructions
1. The previous fix failed the tests.
2. Analyze the failure message.
3. Generate a new fix using SEARCH/REPLACE blocks.
"""

# === ADVERSARIAL CRITIQUE PROMPTS ===
ADVERSARIAL_CRITIQUE_PROMPT = """You are a harsh code reviewer. Critique this proposed fix:

## Bug Report
{problem_statement}

## Proposed Fix
{proposed_fix}

## Current File Content
```python
{file_content}
```

Critique the fix by answering:
1. **Why might this fail?** - Edge cases, race conditions, unexpected inputs
2. **Is it idiomatic?** - Does it follow Python best practices?
3. **Does it break behavior?** - Could it affect other functionality?
4. **Is it minimal?** - Does it change more than necessary?

Be harsh and thorough. Output your critique.
"""

REFINE_AFTER_CRITIQUE_PROMPT = """Based on the critique, improve your fix.

## Original Bug
{problem_statement}

## Your Proposed Fix
{proposed_fix}

## Critique
{critique}

## File Content
```python
{file_content}
```

Generate an improved fix using SEARCH/REPLACE blocks.
Address ALL concerns raised in the critique.
"""

# === RUBRIC CHECKLIST PROMPT ===
RUBRIC_CHECKLIST_PROMPT = """Evaluate this patch against quality criteria.

## Bug Report
{problem_statement}

## Generated Patch
```diff
{patch}
```

## Original File
```python
{original_content}
```

Score each criterion (0-10) and explain:

1. **Localization** - Fix applied to correct file/function?
2. **Minimal** - Change is minimal and focused (not over-engineered)?
3. **Correctness** - Directly addresses the reported issue?
4. **Readable** - Code is clean and follows conventions?
5. **Safety** - Avoids introducing new bugs or regressions?
6. **Testable** - Change can be verified with tests?

Output format:
LOCALIZATION: X/10 - explanation
MINIMAL: X/10 - explanation
CORRECTNESS: X/10 - explanation
READABLE: X/10 - explanation
SAFETY: X/10 - explanation
TESTABLE: X/10 - explanation
TOTAL: XX/60
PASS: YES/NO (YES if TOTAL >= 36, i.e. 60%)
"""

FIX_PROMPT_TEMPLATE = """# Bug Report
{problem_statement}

# Relevant Documentation
{doc_context}

# Similar Examples
{examples_content}

# Primary File: {primary_file}
```python
{primary_content}
```

# Instructions
1. Analyze the bug and the file content.
2. Generate a fix using SEARCH/REPLACE blocks.
3. CAREFULLY TRACE the execution flow. Ensure you understand how variables are updated.
"""

@dataclass
class RubricResult:
    """Result of rubric checklist evaluation."""
    scores: dict  # criterion -> score (0-10)
    total: int  # sum of scores (0-60)
    passed: bool  # True if total >= 36 (60%)
    explanation: str  # Full LLM response


class CoTPipeline(BaselinePipeline):
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        repo_manager: Optional[RepoManager] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        max_files_to_show: int = 300,
        num_examples: int = 2,
        exclude_example_ids: Optional[set] = None,
        use_adversarial: bool = False,  
        use_rubric: bool = False,  
        use_repo_embedding: bool = False,  
    ):
        super().__init__(llm_provider, repo_manager, temperature, max_tokens)
        self.max_files_to_show = max_files_to_show
        self.num_examples = num_examples
        self.use_adversarial = use_adversarial
        self.use_rubric = use_rubric
        self.use_repo_embedding = use_repo_embedding

        # Initialize repo file retriever if embedding enabled
        self.repo_retriever = None
        if use_repo_embedding:
            try:
                from src.retrieval.repo_retriever import RepoFileRetriever
                self.repo_retriever = RepoFileRetriever()
                print("[CoT] Repo embedding enabled")
            except Exception as e:
                print(f"Warning: Could not initialize RepoFileRetriever: {e}")
                self.use_repo_embedding = False

        try:
            self.example_retriever = ExampleRetriever(exclude_ids=exclude_example_ids)
        except Exception as e:
            print(f"Warning: Could not initialize ExampleRetriever: {e}")
            self.example_retriever = None
        self.harness = VerificationHarness()

    def _get_python_files(self, repo_path: Path) -> List[str]:
        files = []
        for py in repo_path.rglob("*.py"):
            if "test" not in str(py).lower():
                files.append(str(py.relative_to(repo_path)))
        return sorted(files)[:self.max_files_to_show]

    def _localize_file(self, problem_statement: str, repo_path: Path) -> Optional[str]:
        files = self._get_python_files(repo_path)
        if not files: return None
        prompt = LOCALIZATION_PROMPT_TEMPLATE.format(
            problem_statement=problem_statement[:5000],
            file_list="\n".join(files)
        )
        response = self.llm.generate(prompt, LOCALIZATION_SYSTEM_PROMPT, 0.0, 512)
        # Try multiple patterns for TARGET_FILE extraction
        patterns = [
            r'TARGET_FILE:\s*`?([^`\s]+\.py)`?',  # TARGET_FILE: path or TARGET_FILE: `path`
            r'\*\*TARGET_FILE\*\*:\s*`?([^`\s]+\.py)`?',  # **TARGET_FILE**: path
            r'TARGET FILE:\s*`?([^`\s]+\.py)`?',  # TARGET FILE: path (no underscore)
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        # Fallback: look for any .py file mentioned in last 200 chars
        last_part = response[-200:]
        py_files = re.findall(r'([a-zA-Z_/]+\.py)', last_part)
        if py_files:
            # Check if it's in our file list
            for pf in py_files:
                if pf in files or any(f.endswith(pf) for f in files):
                    return pf
        return files[0] if files else None

    def _localize_file_embedding(self, problem_statement: str, repo_path: Path) -> Optional[str]:
        """Use semantic search over embedded repo files for localization."""
        if not self.repo_retriever:
            return None

        try:
            # Index the repo (uses cache if available)
            self.repo_retriever.index_repo(str(repo_path))

            # Search for relevant files
            results = self.repo_retriever.search(problem_statement, top_k=3)

            if results:
                top_file = results[0][0]
                print(f"  [CoT] Embedding localization found: {top_file}")
                return top_file
        except Exception as e:
            print(f"  [CoT] Embedding localization error: {e}")

        return None



    def _apply_search_replace(self, content: str, response: str) -> str:
        """Apply SEARCH/REPLACE blocks with robust matching."""
        pattern = r"<{4,}\s*SEARCH\s*\n(.*?)\n={4,}\s*\n(.*?)\n>{4,}\s*REPLACE"
        matches = list(re.finditer(pattern, response, re.DOTALL))
        if not matches:
            return content

        for m in matches:
            search_block = m.group(1)
            replace_block = m.group(2)

            # 1. Exact match (best case)
            if search_block in content:
                content = content.replace(search_block, replace_block, 1)
                continue

            # 2. Try stripping both sides
            search_stripped = search_block.strip()
            if search_stripped and search_stripped in content:
                idx = content.find(search_stripped)
                if idx >= 0:
                    content = content[:idx] + replace_block.strip() + content[idx + len(search_stripped):]
                    continue

            # 3. Fuzzy block matching using SequenceMatcher
            if not search_stripped:
                continue
            search_lines = search_block.splitlines()
            if len(search_lines) < 1:
                continue

            content_lines = content.splitlines(keepends=True)
            best_match_idx = -1
            best_match_ratio = 0.0

            # Slide window over content to find best matching block
            for i in range(len(content_lines) - len(search_lines) + 1):
                candidate_block = ''.join(content_lines[i:i + len(search_lines)])
                ratio = difflib.SequenceMatcher(None, search_block, candidate_block).ratio()
                if ratio > best_match_ratio:
                    best_match_ratio = ratio
                    best_match_idx = i
                # Early exit on near-perfect match
                if ratio > 0.98:
                    break

            # Only apply if match quality is high enough (85% threshold)
            if best_match_idx >= 0 and best_match_ratio >= 0.85:
                before = ''.join(content_lines[:best_match_idx])
                after = ''.join(content_lines[best_match_idx + len(search_lines):])
                # Preserve trailing newline consistency
                if replace_block and not replace_block.endswith('\n'):
                    replace_block += '\n'
                content = before + replace_block + after

        return content

    def _rectify_logic_failure(self, instance, previous_patch: str, test_output: str, file_content: str) -> str:
        """
        Self-correcting loop: Ask LLM to fix the logic based on test failures.
        """
        prompt = RECTIFICATION_PROMPT_TEMPLATE.format(
            problem_statement=instance.problem_statement,
            previous_patch=previous_patch,
            test_output=test_output[:5000],
            file_content=file_content[:20000]
        )

        new_response = self.llm.generate(prompt, RECTIFICATION_SYSTEM_PROMPT, self.temperature, self.max_tokens)
        return self._apply_search_replace(file_content, new_response)

    # === ADVERSARIAL CRITIQUE METHODS ===

    def _adversarial_critique(self, problem_statement: str, proposed_fix: str, file_content: str) -> str:
        """Get harsh critique of proposed fix."""
        prompt = ADVERSARIAL_CRITIQUE_PROMPT.format(
            problem_statement=problem_statement[:5000],
            proposed_fix=proposed_fix,
            file_content=file_content[:20000]
        )
        return self.llm.generate(prompt, temperature=0.3, max_tokens=1024)

    def _refine_after_critique(self, problem_statement: str, proposed_fix: str, critique: str, file_content: str) -> str:
        """Refine fix based on critique. Returns LLM response with SEARCH/REPLACE blocks."""
        prompt = REFINE_AFTER_CRITIQUE_PROMPT.format(
            problem_statement=problem_statement[:5000],
            proposed_fix=proposed_fix,
            critique=critique,
            file_content=file_content[:20000]
        )
        return self.llm.generate(prompt, FIX_SYSTEM_PROMPT, self.temperature, self.max_tokens)

    # === RUBRIC CHECKLIST METHODS ===

    def _evaluate_rubric(self, problem_statement: str, patch: str, original_content: str) -> RubricResult:
        """Evaluate patch against quality rubric."""
        prompt = RUBRIC_CHECKLIST_PROMPT.format(
            problem_statement=problem_statement[:5000],
            patch=patch,
            original_content=original_content[:20000]
        )
        response = self.llm.generate(prompt, temperature=0.0, max_tokens=1024)

        # Parse scores from response
        scores = {}
        for criterion in ['LOCALIZATION', 'MINIMAL', 'CORRECTNESS', 'READABLE', 'SAFETY', 'TESTABLE']:
            match = re.search(rf'{criterion}:\s*(\d+)/10', response)
            scores[criterion.lower()] = int(match.group(1)) if match else 5

        total = sum(scores.values())
        passed = total >= 36  # 60% threshold

        return RubricResult(scores=scores, total=total, passed=passed, explanation=response)

    def run(self, instance: SWEBenchInstance) -> PipelineResult:
        repo_path = self.repo_manager.get_repo_path(instance.repo, instance.base_commit)

        # Use embedding-based localization if enabled, otherwise LLM-based
        if self.use_repo_embedding:
            target_file = self._localize_file_embedding(instance.problem_statement, repo_path)
            # Fallback to LLM-based if embedding fails
            if not target_file:
                print("  [CoT] Embedding localization failed, falling back to LLM")
                target_file = self._localize_file(instance.problem_statement, repo_path)
        else:
            target_file = self._localize_file(instance.problem_statement, repo_path)
        
        if not target_file or not (repo_path / target_file).exists():
             return PipelineResult(instance.instance_id, "", instance.patch, "", False, "Localization failed")

        primary_content = (repo_path / target_file).read_text()

        # RAG Context (no per-instance doc extraction to avoid data leakage)
        doc_context = "No documentation available."
        examples_context = ""
        if self.example_retriever:
             exs = self.example_retriever.retrieve(instance.problem_statement, instance.repo, 2)
             examples_context = "\n".join([f"Example: {e['instance_id']}\n{e['patch'][:8000]}" for e in exs])

        prompt = FIX_PROMPT_TEMPLATE.format(
            problem_statement=instance.problem_statement,
            doc_context=doc_context,
            examples_content=examples_context,
            primary_file=target_file,
            primary_content=primary_content[:50000]
        )

        response = self.llm.generate(prompt, FIX_SYSTEM_PROMPT, self.temperature, self.max_tokens)

        # Apply Initial Patch
        new_content = self._apply_search_replace(primary_content, response)

        # === ADVERSARIAL CRITIQUE (if enabled) ===
        if self.use_adversarial and new_content != primary_content:
            print("  [CoT] Running adversarial critique...")
            critique = self._adversarial_critique(
                instance.problem_statement, response, new_content
            )
            print("  [CoT] Refining based on critique...")
            refined_response = self._refine_after_critique(
                instance.problem_statement, response, critique, new_content
            )
            # Apply the refined fix
            refined_content = self._apply_search_replace(primary_content, refined_response)
            if refined_content != primary_content:
                new_content = refined_content
                response = refined_response
                print("  [CoT] Adversarial refinement applied.")
            else:
                print("  [CoT] Adversarial refinement produced no change, keeping original.")

        # Generate Diff
        def get_diff(orig, new, fname):
            diff_lines = difflib.unified_diff(
                orig.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"a/{fname}",
                tofile=f"b/{fname}"
            )
            return "".join(diff_lines)

        patch = get_diff(primary_content, new_content, target_file)

        # ---- VERIFICATION LOOP WITH RE-VERIFICATION ----
        max_rectification_attempts = 2
        verified_success = False

        if patch:
            print(f"  [CoT] Verifying initial patch...")
            success, logs = self.harness.verify_patch_with_logs(instance, patch)

            if success:
                verified_success = True
                print(f"  [CoT] Initial patch verified successfully!")
            else:
                # Rectification loop with re-verification
                current_content = new_content
                current_patch = patch

                for attempt in range(max_rectification_attempts):
                    print(f"  [CoT] Verification failed. Rectification attempt {attempt + 1}/{max_rectification_attempts}...")

                    rectified_content = self._rectify_logic_failure(
                        instance, current_patch, logs, current_content
                    )
                    rectified_patch = get_diff(primary_content, rectified_content, target_file)

                    if not rectified_patch:
                        print(f"  [CoT] Rectification produced empty patch, keeping previous.")
                        break

                    # Re-verify the rectified patch
                    print(f"  [CoT] Re-verifying rectified patch...")
                    success, logs = self.harness.verify_patch_with_logs(instance, rectified_patch)

                    if success:
                        verified_success = True
                        patch = rectified_patch
                        new_content = rectified_content
                        print(f"  [CoT] Rectified patch verified successfully!")
                        break

                    # Update for next iteration
                    current_content = rectified_content
                    current_patch = rectified_patch

                if not verified_success:
                    print(f"  [CoT] All rectification attempts failed. Returning best effort patch.")
                    patch = current_patch

        # === RUBRIC CHECKLIST (if enabled) ===
        if self.use_rubric and patch:
            print("  [CoT] Evaluating against rubric checklist...")
            rubric = self._evaluate_rubric(instance.problem_statement, patch, primary_content)
            print(f"  [CoT] Rubric score: {rubric.total}/60 ({'PASS' if rubric.passed else 'FAIL'})")
            for criterion, score in rubric.scores.items():
                print(f"    - {criterion.capitalize()}: {score}/10")

        return PipelineResult(
            instance_id=instance.instance_id,
            generated_patch=patch,
            ground_truth_patch=instance.patch,
            raw_response=response,
            success=verified_success
        )
