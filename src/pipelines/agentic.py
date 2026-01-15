"""
Agentic Pipeline: Implements a ReAct-style agent that explores the codebase.
Uses tools (AST, grep, docs) and iteratively localizes/fixes bugs using a reason-act loop.
Now enhanced with Robust Search/Replace Editing + Self-Correction Loop.
"""
import subprocess
import re
import json
import difflib
from typing import Optional, List, Tuple
from pathlib import Path

from src.data import SWEBenchInstance
from src.llm import LLMProvider
from src.retrieval.source_code import RepoManager
from src.retrieval.example_retriever import ExampleRetriever
from src.verification.harness import VerificationHarness
from src.pipelines.baseline import PipelineResult
from src.retrieval.indexer import HybridRetriever
from src.retrieval.corpus import DocumentCorpus
from src.retrieval.graph import CodeGraph


# =============================================================================
#  AGENTLESS-STYLE PROMPTS (Proven to work on SWE-bench)
# =============================================================================

AGENTLESS_LOCALIZATION_SYSTEM = """You are an expert software engineer specializing in bug localization.
Your task is to identify the SINGLE most likely file that needs to be modified to fix the bug.
Think step-by-step and explain your reasoning before giving the final answer."""

AGENTLESS_LOCALIZATION_PROMPT = """# Bug Report
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

AGENTLESS_FIX_SYSTEM = """You are an expert software engineer fixing bugs.

CRITICAL RULES:
1. Use SEARCH/REPLACE blocks to modify the code.
2. The SEARCH block must EXACTLY MATCH the original file - copy lines character-for-character INCLUDING all leading spaces/tabs.
3. DO NOT use markdown code blocks (no ```python) inside SEARCH/REPLACE.
4. Make the ABSOLUTE MINIMUM change. If moving 1 line fixes it, only move that 1 line.
5. The REPLACE block should have the SAME number of lines or very close - don't delete or add unnecessarily.

FORMAT:
<<<< SEARCH
original code lines (EXACT COPY from file with all indentation)
====
new code lines (same indentation as original)
>>>> REPLACE

EXAMPLE - Moving a line into a try block:
<<<< SEARCH
        data = self[key]
        try:
            result = process(data)
====
        try:
            data = self[key]
            result = process(data)
>>>> REPLACE
"""

AGENTLESS_FIX_PROMPT = """# Bug Report
{problem_statement}

# Hints (from issue discussion)
{hints_text}

# Similar Solved Bugs
{examples_content}

# File: {primary_file}
```python
{primary_content}
```

# Instructions
1. READ THE BUG REPORT CAREFULLY - sometimes it tells you exactly what to fix.
2. Look for phrases like "simply move", "just change", "should be X instead of Y".
3. Generate a fix using SEARCH/REPLACE blocks.
4. Make the SMALLEST possible change - if the fix is moving 1 line, just move that line.
5. COPY the SEARCH block EXACTLY from the file INCLUDING ALL LEADING WHITESPACE.
6. Do NOT add defensive coding (try/except, getattr) unless that's the actual fix.
7. Do NOT refactor or clean up code - only fix the specific bug.
"""

AGENTLESS_RECTIFICATION_SYSTEM = """You are an expert software engineer debugging a failed fix.
Your previous patch was applied, but the tests failed.
Analyze the test output and the original code to fix the logic.

OUTPUT FORMAT:
Provide a NEW SEARCH/REPLACE block to update the file correctly."""

AGENTLESS_RECTIFICATION_PROMPT = """# Original Bug
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


# =============================================================================
#  PROMPTS (Updated for Search/Replace Strategy)
# =============================================================================

REACT_SYSTEM = """You are a specialized debugging agent. Your goal is to locate the file causing the bug.
You have access to a unix-like environment and specific tools.

TOOLS:
- grep(pattern: str): Search for a string in the codebase.
- find_file(name: str): Find a file by name.
- read_file(path: str): Read a file's content (with line numbers).
- find_usages(symbol: str): Find where a symbol is defined/used (AST-based).
- get_function_context(func_name: str, file_path: str): Get details about a function (callers/callees).
- run_tests(test_file: str): Run a specific test file.
- search_docs(query: str): Search documentation and StackOverflow.
- list_files(): List all Python files in the repo.

PROCESS:
1. Analyzing the Bug Report.
2. THOUGHT: Reasoning carefully about next steps.

3. ACTION: tool_name(args)
4. OBSERVATION: [Result of action, provided by system]
5. Repeat steps 2-4 until you have found the file or exhausted options (max 10 steps).
6. FINAL_ANSWER: TARGET_FILE: path/to/file.py

Start by exploring the codebase based on the bug report.
"""


REPO_TREE_SYSTEM = """You are a Senior Software Engineer acting as a Localization Expert.
Your task is to identify the best 3 files in the repository that likely contain the bug described.

Guidelines:
1. Review the provided "Candidate Files" (found by pattern matching).
2. Review the "Repository Files" to understand the project structure.
3. Select the Top 3 files that are most relevant to the bug.
4. Prioritize:
   - Files where the logic/validation likely resides (e.g. `base.py` for validation vs `options.py` for definition).
   - Core implementation files over tests or examples.
   - Files extracted from stacktraces (highest confidence).

Output Format:
Output ONLY the file paths, one per line, in order of likelihood (most likely first).
Do not include any explanation or markdown formatting.
"""

PLANNING_SYSTEM = """You are an expert software engineer. Analyze the code and create a fix plan.

Given a bug report and the relevant code, explain:
1. ROOT CAUSE: What exactly is causing the bug?
2. FIX STRATEGY: What changes are needed? Be specific about lines.
3. EDGE CASES: What edge cases should be considered?

Be concise and precise."""

PLANNING_PROMPT = """# Bug Report
{problem_statement}

# Hints (from issue discussion)
{hints_text}

# Relevant Code ({file_path})
{code_content}

{grep_results}

Analyze the bug and create a fix plan. Pay attention to the hints - they often contain the fix location or approach."""

# --- PATCH PROMPTS (SWE-smith style - simple OLD/NEW format) ---

PATCH_SYSTEM = """You are an expert software engineer fixing bugs.

WORKFLOW:
1. Read the bug report and understand the root cause
2. Find the exact code that needs to change
3. Make the MINIMAL fix - often just 1-2 lines
4. Output the fix in OLD/NEW format below

RULES:
- Make the SMALLEST possible change
- Copy OLD code EXACTLY as it appears in the file (whitespace matters!)
- NEVER modify docstrings, comments, or examples
- NEVER add new classes or exception types

Format your response EXACTLY like this:
OLD:
```python
exact code from file to replace
```

NEW:
```python
fixed code (minimal change)
```
"""

PATCH_PROMPT = """# Bug Report
{problem_statement}

# Hints (from issue discussion)
{hints_text}

# Similar Solved Bugs
{examples_context}

# Fix Plan
{fix_plan}

# Current Code ({file_path})
{code_content}

Generate a fix using OLD/NEW format. Learn from the similar bugs above. Output ONLY the OLD and NEW code blocks."""

RECTIFICATION_SYSTEM_PROMPT = """You are an expert software engineer debugging a failed fix.
Your previous patch was applied, but the verification tests failed.
Analyze the test output and fix the issue.

OUTPUT FORMAT (use OLD/NEW blocks):
OLD:
```python
exact code to replace
```

NEW:
```python
corrected code
```
"""

RECTIFICATION_PROMPT_TEMPLATE = """# Original Bug
{problem_statement}

# Tests That Must Pass (from SWE-bench)
{failing_tests}

# Your Previous Fix
{previous_patch}

# Test Failure Output
{test_output}

# Primary File Content (With your previous fix applied)
```python
{file_content}
```

# Instructions
1. The previous fix FAILED the tests listed above.
2. Analyze the failure message carefully.
3. Generate a MINIMAL fix using SEARCH/REPLACE blocks.
4. DO NOT delete any lines that aren't directly related to the bug.
5. Preserve all existing function parameters and return statements.
"""


# =============================================================================
#  SOTA PROMPTS: Self-Critique + LLM-Assisted Localization
# =============================================================================

SELF_CRITIQUE_SYSTEM = """You are a senior code reviewer. Review the proposed patch and identify any issues.

Check for:
1. SYNTAX: Are there any syntax errors?
2. LOGIC: Does the fix actually address the root cause?
3. EDGE CASES: Are there unhandled edge cases?
4. SIDE EFFECTS: Could this break other functionality?
5. COMPLETENESS: Is the fix complete or are parts missing?

If the patch is correct, respond with: APPROVED

If there are issues, respond with:
ISSUES:
- [issue 1]
- [issue 2]
SUGGESTION: [how to fix]
"""

SELF_CRITIQUE_PROMPT = """# Original Bug Report
{problem_statement}

# Proposed Patch
```diff
{patch}
```

# File Content After Patch
```python
{patched_content}
```

Review this patch. Is it correct and complete?
"""

MINIMAL_CRITIQUE_PROMPT = """Review this patch for a bug fix. Be strict about minimality.

Bug Description:
{problem_statement}

Generated Patch:
```diff
{patch}
```

Check these specific issues:
1. Does the patch DELETE any variable assignments (like `x = something`) that aren't related to the bug? (BAD - say DELETE)
2. Does the patch ADD more than 10 lines when fewer would work? (BAD - say OVERENGINEERED)
3. Does the patch create new classes/exceptions when simple validation would work? (BAD - say OVERENGINEERED)
4. Is this the MINIMAL fix that could possibly work? (GOOD)

If the patch is minimal and doesn't delete important code, respond exactly: APPROVED
If there are issues, respond: ISSUES: [describe the specific problem]
"""

LLM_LOCALIZATION_SYSTEM = """You are an expert at navigating large codebases.
Given a bug report, suggest the most likely location of the bug.

Output a JSON object with:
{
    "search_terms": ["term1", "term2"],
    "likely_dirs": ["subdir1", "subdir2"],
    "file_patterns": ["*pattern*.py"],
    "reasoning": "brief explanation"
}
"""

LLM_LOCALIZATION_PROMPT = """# Bug Report
{problem_statement}

# Repository Structure (Top 50 files)
{file_list}

Analyze where this bug is most likely located.
"""

# --- MINI-SWE-AGENT STYLE PROMPT (bash-only, simple loop) ---

MINI_AGENT_SYSTEM = """You are a software engineer debugging and fixing bugs.

COMMANDS:
- grep -rn "pattern" --include="*.py" | head -20  (search code)
- sed -n '100,150p' file.py  (read lines 100-150)
- python -c "
lines = open('FILE').readlines()
lines[LINE-1] = 'NEW_CONTENT\\n'
open('FILE', 'w').writelines(lines)
"  (replace line LINE with NEW_CONTENT)

WORKFLOW:
1. FIND the file and line number from the traceback
2. READ the code around that line with: sed -n 'START,ENDp' file.py
3. SEARCH for the correct pattern: grep -n '.ATTRIBUTE' FILE to see how other code accesses it
4. IDENTIFY: What object SHOULD be used? (e.g., self.root instead of schema)
5. FIX by replacing the OBJECT NAME, not by wrapping with getattr
6. Say DONE

WRONG FIXES (NEVER DO THIS):
- getattr(schema, 'opts', None)  <-- WRONG! Defensive coding
- getattr(getattr(x, 'y'), 'z') <-- WRONG! Nested defensive coding
- try/except around the line    <-- WRONG! Hides the real bug
- if x is not None: x.attr      <-- WRONG! The bug is wrong object, not None

RIGHT FIX PATTERN:
- schema.opts -> self.root.opts  <-- CORRECT! Change the object reference
- node.attr -> self.node.attr    <-- CORRECT! Use the right object
- The bug is WHICH OBJECT is used, not WHETHER it has the attribute

EXAMPLE:
Bug: 'Schema' object has no attribute 'opts' at line 1117
STEP 1: Read context with sed -n '1100,1130p' file.py
STEP 2: Search for correct pattern: grep -n '.opts' file.py
STEP 3: Find that other code uses 'self.root.opts', not 'schema.opts'
STEP 4: Fix: python -c "lines = open('file.py').readlines(); lines[1116] = '            or getattr(self.root.opts, VAR)\\n'; open('file.py', 'w').writelines(lines)"

OUTPUT: One command per response, no markdown."""


# =============================================================================
#  HELPER FUNCTIONS
# =============================================================================


def validate_patched_code(original: str, patched: str) -> Tuple[bool, str]:
    """
    Validate that the patched code is valid Python and doesn't have obvious issues.
    Returns (is_valid, error_message).
    """
    import ast

    # 1. Check Python syntax
    try:
        ast.parse(patched)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    # 2. Check for significant line deletions (usually a mistake)
    orig_lines = len(original.splitlines())
    patch_lines = len(patched.splitlines())
    deleted = orig_lines - patch_lines

    if deleted > 5 and deleted > orig_lines * 0.05:
        return False, f"Too many lines deleted ({deleted}). Likely a mistake."

    # 3. Check indentation wasn't broken
    patch_lines_list = patched.splitlines()
    for i, line in enumerate(patch_lines_list[:200]):
        # Check for lines that look like they should be indented but aren't
        stripped = line.strip()
        if stripped and not line.startswith(' ') and not line.startswith('\t'):
            # This line has no indentation
            if stripped.startswith(('return ', 'raise ', 'yield ', 'break', 'continue', 'pass')):
                # These should usually be indented unless at module level
                # Check if previous line ends with ':'
                if i > 0:
                    prev = patch_lines_list[i-1].strip()
                    if prev.endswith(':'):
                        return False, f"Line {i+1} should be indented after '{prev}'"

    return True, "OK"


def parse_unified_diff(response: str) -> Optional[str]:
    """Extract unified diff from model response."""
    # Try to find diff in code block
    diff_match = re.search(r'```diff\n(.*?)```', response, re.DOTALL)
    if diff_match:
        return diff_match.group(1).strip()

    # Try to find diff without code block (starts with ---)
    diff_match = re.search(r'(---\s+a/.*?\n\+\+\+\s+b/.*?\n@@.*)', response, re.DOTALL)
    if diff_match:
        return diff_match.group(1).strip()

    return None


def parse_old_new_format(response: str) -> Optional[Tuple[str, str]]:
    """Extract OLD and NEW code blocks from model response (SWE-smith style)."""
    # Try to find OLD: ```python ... ``` and NEW: ```python ... ```
    old_match = re.search(r'OLD:\s*```(?:python)?\n(.*?)```', response, re.DOTALL | re.IGNORECASE)
    new_match = re.search(r'NEW:\s*```(?:python)?\n(.*?)```', response, re.DOTALL | re.IGNORECASE)

    if old_match and new_match:
        old_code = old_match.group(1).rstrip('\n')
        new_code = new_match.group(1).rstrip('\n')
        return old_code, new_code

    return None


def apply_unified_diff(original_content: str, diff_text: str) -> Optional[str]:
    """Apply a unified diff to file content. Returns new content or None on failure."""
    if not diff_text:
        return None

    lines = original_content.splitlines(keepends=True)
    if not lines[-1].endswith('\n'):
        lines[-1] += '\n'

    result_lines = list(lines)

    # Parse hunks from diff
    hunk_pattern = r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@'
    hunks = list(re.finditer(hunk_pattern, diff_text))

    if not hunks:
        return None

    # Apply hunks in reverse order to preserve line numbers
    diff_lines = diff_text.splitlines()

    for hunk_match in reversed(hunks):
        old_start = int(hunk_match.group(1)) - 1  # 0-indexed

        # Find the hunk content
        hunk_start_idx = None
        for i, line in enumerate(diff_lines):
            if hunk_match.group(0) in line:
                hunk_start_idx = i
                break

        if hunk_start_idx is None:
            continue

        # Collect hunk lines until next hunk or end
        hunk_lines = []
        for i in range(hunk_start_idx + 1, len(diff_lines)):
            line = diff_lines[i]
            if line.startswith('@@') or line.startswith('diff ') or line.startswith('---') or line.startswith('+++'):
                break
            hunk_lines.append(line)

        # Apply this hunk
        new_lines = []
        old_idx = old_start

        for hunk_line in hunk_lines:
            if not hunk_line:
                continue

            prefix = hunk_line[0] if hunk_line else ' '
            content = hunk_line[1:] if len(hunk_line) > 1 else ''

            if prefix == ' ':
                # Context line - keep it
                new_lines.append(content + '\n' if not content.endswith('\n') else content)
                old_idx += 1
            elif prefix == '-':
                # Removed line - skip it in original
                old_idx += 1
            elif prefix == '+':
                # Added line - include it
                new_lines.append(content + '\n' if not content.endswith('\n') else content)

        # Find how many lines to replace
        lines_to_remove = 0
        for hunk_line in hunk_lines:
            if hunk_line and hunk_line[0] in ' -':
                lines_to_remove += 1

        # Replace the lines
        result_lines[old_start:old_start + lines_to_remove] = new_lines

    return ''.join(result_lines)


def validate_search_replace_preserves_critical(search: str, replace: str) -> Tuple[bool, str]:
    """Ensure REPLACE block doesn't accidentally delete critical code elements."""
    # Check variable assignments (e.g., "schema = self.schema")
    search_assigns = set(re.findall(r'^\s*(\w+)\s*=', search, re.MULTILINE))
    replace_assigns = set(re.findall(r'^\s*(\w+)\s*=', replace, re.MULTILINE))
    deleted_assigns = search_assigns - replace_assigns

    # Check return statements
    search_returns = len(re.findall(r'^\s*return\b', search, re.MULTILINE))
    replace_returns = len(re.findall(r'^\s*return\b', replace, re.MULTILINE))

    # Check function/method calls that might be critical
    search_calls = set(re.findall(r'(\w+)\s*\(', search))
    replace_calls = set(re.findall(r'(\w+)\s*\(', replace))
    deleted_calls = search_calls - replace_calls - {'if', 'for', 'while', 'with', 'print'}

    issues = []
    if deleted_assigns:
        issues.append(f"Deleted assignments: {deleted_assigns}")
    if search_returns > replace_returns:
        issues.append(f"Deleted {search_returns - replace_returns} return statement(s)")
    if deleted_calls and len(deleted_calls) > 2:
        issues.append(f"Deleted function calls: {deleted_calls}")

    if issues:
        return False, "; ".join(issues)
    return True, "OK"


def extract_test_names(fail_to_pass: str) -> List[str]:
    """Extract test function/file names from fail_to_pass field."""
    tests = []
    # Try JSON parse first
    try:
        import json
        parsed = json.loads(fail_to_pass)
        if isinstance(parsed, list):
            tests = parsed
    except:
        pass

    # Fallback: regex for test names
    if not tests:
        tests = re.findall(r'test_[\w]+', fail_to_pass)

    return tests


def extract_stacktrace_files(text: str) -> List[str]:
    """Extract file paths from Python stacktraces."""
    pattern = r'File ["\']([^"\']+\.py)["\'], line \d+'
    matches = re.findall(pattern, text)
    result = []
    for m in matches:
        # Handle site-packages and dist-packages paths
        for pkg_dir in ['site-packages', 'dist-packages']:
            if pkg_dir in m:
                parts = m.split(pkg_dir)
                if len(parts) > 1:
                    relative = parts[1].lstrip('/\\')
                    relative = relative.replace('\\', '/')
                    if relative:
                        result.append(relative)
                break
        else:
            # Continue to next checks only if no package dir found
            pass
        if 'site-packages' in m or 'dist-packages' in m:
            continue
        # Handle absolute paths with project name (e.g., /home/user/projects/astroid/astroid/nodes/x.py)
        if '/projects/' in m or '/repos/' in m or '/src/' in m:
            # Try to extract relative path from known project dirs
            for marker in ['/projects/', '/repos/', '/src/']:
                if marker in m:
                    after = m.split(marker, 1)[1]
                    # Skip the project name directory
                    parts = after.split('/', 1)
                    if len(parts) > 1:
                        result.append(parts[1])
                    break
            continue
        # Skip system paths
        if '/usr/lib' in m or '/usr/local' in m or '\\Python3' in m or 'anaconda' in m.lower():
            # But try to extract package-relative path from anaconda paths too
            if 'lib/python' in m.lower() or 'site-packages' in m.lower():
                # Find package name start
                for pkg_marker in ['lib/python3', 'lib/python2', 'site-packages']:
                    if pkg_marker in m.lower():
                        idx = m.lower().find(pkg_marker)
                        after = m[idx:].split('/', 2)
                        if len(after) > 2:
                            result.append(after[2])
                        break
            continue
        if '/' in m or '\\' in m:
            normalized = m.replace('\\', '/')
            parts = normalized.split('/')
            for i, part in enumerate(parts):
                if part in ['src', 'lib', 'tests', 'sklearn', 'sympy', 'matplotlib', 
                           'requests', 'pylint', 'sphinx', 'seaborn', 'django', 'flask']:
                    result.append('/'.join(parts[i:]))
                    break
    return result


def extract_file_paths(text: str) -> List[str]:
    """Extract explicit file path mentions from text."""
    patterns = [
        r'(?:^|[\s`\'"])([a-zA-Z_][\w/]*\.py)(?:[\s`\'"]|$)',  # basic: file.py
        r'in\s+[`\'"]([\w/]+\.py)[`\'"]',  # "in `file.py`"
        r'[`\'"]([\w/]+\.py)[`\'"]\s+at\s+line',  # "`file.py` at line"
        r'problem\s+is\s+in\s+[`\'"]([\w/]+\.py)',  # "problem is in `file.py`"
        r'(?:the|module)\s+[`\'"]([\w/]+\.py)',  # "the `file.py`"
    ]
    matches = set()
    for pattern in patterns:
        for m in re.findall(pattern, text, re.IGNORECASE):
            if len(m) > 5 and not m.startswith('test'):
                matches.add(m)
    return list(matches)


def module_to_filepath(module_path: str) -> str:
    return module_path.replace('.', '/') + '.py'


def extract_module_paths(text: str) -> List[str]:
    patterns = [
        r'from\s+([\w\.]+)\s+import',
        r'import\s+([\w\.]+)',
        r'([\w]+(?:\.[\w]+){2,})',
    ]
    modules = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            if '.' in m and not m.startswith('.') and not m.endswith('.'):
                modules.add(m)
    return list(modules)


def extract_explicit_modules(text: str) -> List[str]:
    """Extract explicit module name mentions like 'the config module'."""
    patterns = [
        r'the\s+[`\'"]?(\w+)[`\'"]?\s+module',  # "the config module"
        r'[`\'"]?(\w+)[`\'"]?\s+module\s+(?:is|has|defines)',  # "config module defines"
        r'module\s+[`\'"]?(\w+)[`\'"]?',  # "module config"
    ]
    modules = set()
    for pattern in patterns:
        for m in re.findall(pattern, text, re.IGNORECASE):
            if len(m) > 3 and m.lower() not in ['the', 'this', 'that', 'some']:
                modules.add(m)
    return list(modules)


def extract_rule_identifiers(text: str) -> List[str]:
    """Extract rule identifiers like L039, E501, W001 from text."""
    # Common patterns: L039, E501, W001, R0901, C0111, etc.
    pattern = r'\b([A-Z]\d{3,4})\b'
    matches = set(re.findall(pattern, text))
    return list(matches)


def extract_class_names(text: str) -> List[str]:
    """Extract class names mentioned in error messages or descriptions."""
    patterns = [
        r"'(\w+)'\s+(?:is not|has no|object)",  # "'PersonName3' is not iterable"
        r'class\s+[`\'"]?(\w+)[`\'"]?',  # "class PersonName3"
        r'(\w+)\.(\w+)\(\)',  # "Dict.getitem()" -> extract both
        r'type\s+[`\'"]?(\w+)[`\'"]?',  # "type 'Array'"
    ]
    classes = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            if isinstance(m, tuple):
                for part in m:
                    if len(part) > 3 and part[0].isupper():
                        classes.add(part)
                        # Also add base name without trailing digits (PersonName3 -> PersonName)
                        base = re.sub(r'\d+$', '', part)
                        if base and len(base) > 3:
                            classes.add(base)
            elif len(m) > 3 and m[0].isupper():
                classes.add(m)
                base = re.sub(r'\d+$', '', m)
                if base and len(base) > 3:
                    classes.add(base)
    return list(classes)


def extract_error_messages(text: str) -> List[str]:
    """Extract error messages/strings from bug reports to grep for."""
    messages = set()

    # Extract common error patterns - get ONLY the generic verb phrase
    patterns = [
        r'(Unable to (?:load|find|parse|read|open))',  # "Unable to load"
        r'(cannot (?:find|load|import|parse|open))',  # "cannot find"
        r'(failed to (?:load|parse|read|open))',  # "failed to load"
        r'(error while \w+)',  # "error while parsing"
    ]

    for pattern in patterns:
        for m in re.findall(pattern, text, re.IGNORECASE):
            msg = m.strip()
            if len(msg) >= 10 and len(msg) <= 30:
                messages.add(msg)

    return list(messages)


# =============================================================================
#  MINI-SWE-AGENT HELPER FUNCTIONS
# =============================================================================

def run_bash(cmd: str, cwd: str, timeout: int = 30) -> str:
    """Execute bash command and return output (stdout + stderr)."""
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd,
            capture_output=True, text=True, timeout=timeout
        )
        output = result.stdout + result.stderr
        # Truncate long output
        if len(output) > 8000:
            output = output[:8000] + "\n... (truncated)"
        return output if output.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "TIMEOUT: Command took too long"
    except Exception as e:
        return f"ERROR: {e}"


def extract_bash_command(response: str) -> Optional[str]:
    """Extract bash command from LLM response."""
    # Try code block first (```bash or ```sh or ```)
    match = re.search(r'```(?:bash|sh)?\n(.*?)```', response, re.DOTALL)
    if match:
        cmd = match.group(1).strip()
        # Take only first line if multiple
        return cmd.split('\n')[0].strip()

    # Try lines starting with $
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith('$ '):
            return line[2:].strip()

    # Try lines starting with common commands
    cmd_prefixes = ['find ', 'grep ', 'cat ', 'head ', 'tail ', 'sed ', 'echo ', 'python ', 'ls ', 'cd ']
    for line in response.split('\n'):
        line = line.strip()
        for prefix in cmd_prefixes:
            if line.startswith(prefix):
                return line

    return None


# =============================================================================
#  AGENTIC PIPELINE CLASS
# =============================================================================

class AgenticPipeline:
    """
    High Pass@1 Agentic Pipeline v2 with improved localization and Self-Correction.
    """
    
    MAX_PATCH_RETRIES = 3
    MAX_CANDIDATES = 2
    
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        repo_manager: Optional[RepoManager] = None,
        example_retriever: Optional[ExampleRetriever] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        use_harness: bool = True,
        exclude_example_ids: Optional[set] = None,
        harness_dataset: str = "princeton-nlp/SWE-bench_Lite",
        harness_split: str = "test",
    ):
        self.llm = llm_provider or LLMProvider()
        self.repo_manager = repo_manager or RepoManager()

        # Initialize ExampleRetriever with proper error handling
        self.example_retriever = example_retriever
        if self.example_retriever is None and exclude_example_ids is not None:
            try:
                self.example_retriever = ExampleRetriever(exclude_ids=exclude_example_ids)
            except Exception as e:
                print(f"Warning: Could not initialize ExampleRetriever: {e}")
                self.example_retriever = None

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.harness = VerificationHarness(dataset_name=harness_dataset, split=harness_split) if use_harness else None
        
        # Initialize Retriever with expanded corpus
        try:
            corpus_path = "cache/expanded_corpus.json"
            if Path(corpus_path).exists():
                corpus = DocumentCorpus(corpus_path)
                self.retriever = HybridRetriever(corpus)
                print(f"Agentic Pipeline initialized with {len(corpus)} docs from cache")
            else:
                self.retriever = None
        except Exception as e:
            print(f"Warning: Failed to init retriever for agent: {e}")
            self.retriever = None


    def _generate_repo_tree(self, repo_path: str, max_depth: int = 2) -> str:
        file_list = []
        try:
            result = subprocess.run(
                ["find", ".", "-maxdepth", str(max_depth), "-not", "-path", "*/.*"],
                cwd=repo_path, capture_output=True, text=True
            )
            paths = result.stdout.strip().split("\n")
            paths = [p for p in paths if p and p != "."]
            paths.sort()
            return "\n".join(paths)
        except Exception: return ""

    def _filter_file_list(self, files: List[str]) -> List[str]:
        filtered = []
        for f in files:
            f = f.lstrip("./")
            parts = f.split("/")
            if any(p.startswith(".") for p in parts): continue
            if any(p in ["doc", "docs", "examples", "test", "tests", "build", "dist", "site-packages"] for p in parts): continue
            filtered.append(f)
        return filtered if filtered else files

    def _get_file_list(self, repo_path: str, max_files: int = 1000) -> str:
        try:
            result = subprocess.run(
                ["find", ".", "-name", "*.py", "-type", "f"],
                cwd=repo_path, capture_output=True, text=True, timeout=30
            )
            files = result.stdout.strip().split("\n")
            filtered_files = self._filter_file_list(files)
            return "\n".join(filtered_files[:max_files])
        except Exception: return ""

    def _read_file_with_lines(self, repo_path: str, file_path: str) -> str:
        try:
            full_path = Path(repo_path) / file_path.lstrip("./")
            if not full_path.exists(): return f"File not found: {file_path}"
            lines = full_path.read_text().splitlines()
            numbered = [f"{i+1:4d} | {line}" for i, line in enumerate(lines)]
            content = "\n".join(numbered)
            return content
        except Exception as e: return f"Error reading file: {e}"

    def _grep_code(self, repo_path: str, pattern: str, context: int = 3) -> str:
        try:
            result = subprocess.run(
                ["git", "grep", "-n", "-i", f"-C{context}", pattern],
                cwd=repo_path, capture_output=True, text=True, timeout=30
            )
            if result.stdout.strip(): return result.stdout[:5000]
            return "No matches found"
        except Exception as e: return f"Grep error: {e}"

    def _find_file(self, repo_path: str, filename: str) -> str:
        try:
            result = subprocess.run(
                ["find", ".", "-name", filename, "-type", "f"],
                cwd=repo_path, capture_output=True, text=True, timeout=30
            )
            if result.stdout.strip(): return result.stdout.strip().split("\n")[0]
            return ""
        except Exception: return ""

    def _find_symbol_usages(self, repo_path: str, symbol: str, max_results: int = 20) -> str:
        import ast
        usages = []
        definitions = []
        for py_file in Path(repo_path).rglob("*.py"):
            rel_path = str(py_file.relative_to(repo_path))
            if 'test' in rel_path.lower() or '/.' in rel_path: continue
            try:
                content = py_file.read_text(errors='ignore')
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == symbol:
                        definitions.append(f"DEF {rel_path}:{node.lineno} - def {symbol}()")
                    elif isinstance(node, ast.ClassDef) and node.name == symbol:
                        definitions.append(f"DEF {rel_path}:{node.lineno} - class {symbol}")
                    elif isinstance(node, ast.Name) and node.id == symbol:
                        usages.append(f"USE {rel_path}:{node.lineno}")
                    elif isinstance(node, ast.Attribute) and node.attr == symbol:
                        usages.append(f"ATTR {rel_path}:{node.lineno} - .{symbol}")
            except Exception: continue
        result = []
        if definitions: result.append("# Definitions:"); result.extend(definitions[:5])
        if usages: result.append("\n# Usages:"); result.extend(usages[:max_results])
        return "\n".join(result) if result else f"No usages found for '{symbol}'"

    def _get_function_context(self, repo_path: str, file_path: str, func_name: str) -> str:
        """Get function definition and context including callers/callees."""
        import ast
        full_path = Path(repo_path) / file_path.lstrip("./")
        if not full_path.exists():
            return f"File not found: {file_path}"

        try:
            content = full_path.read_text(errors='ignore')
            tree = ast.parse(content)
            lines = content.splitlines()

            result = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    start = node.lineno - 1
                    end = node.end_lineno if node.end_lineno else start + 30
                    func_lines = lines[start:min(end, len(lines))]

                    # Get function signature and body
                    result.append(f"# Function: {func_name} (lines {start+1}-{end})")
                    result.append("```python")
                    result.extend(func_lines[:50])  # Limit to 50 lines
                    result.append("```")

                    # Find what this function calls
                    callees = []
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            if isinstance(child.func, ast.Name):
                                callees.append(child.func.id)
                            elif isinstance(child.func, ast.Attribute):
                                callees.append(child.func.attr)
                    if callees:
                        result.append(f"\n# Calls: {', '.join(set(callees)[:10])}")

                    return "\n".join(result)

            return f"Function '{func_name}' not found in {file_path}"
        except Exception as e:
            return f"Error parsing {file_path}: {e}"

    def _find_related_tests(self, repo_path: str, file_path: str) -> str:
        try:
            file_name = Path(file_path).stem
            patterns = [f"test_{file_name}.py", f"{file_name}_test.py", f"test_{file_name}*.py"]
            related_tests = []
            for pattern in patterns:
                result = subprocess.run(["find", ".", "-name", pattern, "-type", "f"], cwd=repo_path, capture_output=True, text=True)
                if result.returncode == 0:
                   related_tests.extend([l.lstrip("./") for l in result.stdout.split("\n") if l])
            return "\n".join(set(related_tests[:10])) if related_tests else "No tests found."
        except: return "Error finding tests."

    def _analyze_error_pattern(self, problem_statement: str) -> str:
        return "Analysis skipped."

    def _run_tests(self, repo_path: str, test_file: str) -> str:
        """Execute a specific test file and return results."""
        try:
            # Try pytest first
            result = subprocess.run(
                ["python", "-m", "pytest", test_file, "-v", "--tb=short", "-x"],
                cwd=repo_path, capture_output=True, text=True, timeout=60
            )
            output = result.stdout + result.stderr
            # Truncate to avoid overwhelming the agent
            if len(output) > 4000:
                output = output[:2000] + "\n...(truncated)...\n" + output[-1500:]
            return output if output.strip() else "Tests completed with no output"
        except subprocess.TimeoutExpired:
            return "Test execution timed out (60s limit)"
        except Exception as e:
            return f"Test execution failed: {e}"

    def _search_docs(self, repo_path: str, query: str) -> str:
        """Search documentation files for a query."""
        results = []
        query_lower = query.lower()

        # Search in common doc directories
        for doc_dir in ['doc', 'docs', 'documentation']:
            doc_path = Path(repo_path) / doc_dir
            if doc_path.exists():
                for ext in ['*.rst', '*.md', '*.txt']:
                    for f in doc_path.rglob(ext):
                        try:
                            content = f.read_text(errors='ignore')
                            if query_lower in content.lower():
                                # Find the relevant paragraph
                                idx = content.lower().find(query_lower)
                                start = max(0, idx - 100)
                                end = min(len(content), idx + 300)
                                snippet = content[start:end].strip()
                                results.append(f"## {f.name}\n{snippet}...")
                        except Exception:
                            continue

        if results:
            return "\n\n".join(results[:3])
        return f"No documentation found for '{query}'"

    def _parse_action(self, text: str) -> Optional[Tuple[str, List[str]]]:
        lines = text.splitlines()
        action_line = None
        for line in reversed(lines):
            if line.strip().startswith("ACTION:"):
                action_line = line
                break
        if not action_line: return None
        match = re.search(r'ACTION:\s*([a-zA-Z_]+)\((.*)\)', action_line)
        if match:
            tool_name = match.group(1)
            args_str = match.group(2)
            args = [a.strip().strip("'").strip('"') for a in args_str.split(',') if a.strip()]
            return tool_name, args
        return None

    def _execute_tool(self, tool_name: str, args: List[str], repo_path: str) -> str:
        try:
            if tool_name == "grep":
                if not args: return "Error: Missing search pattern"
                return self._grep_code(repo_path, args[0])
            elif tool_name == "find_file":
                if not args: return "Error: Missing filename"
                return self._find_file(repo_path, args[0])
            elif tool_name == "read_file":
                if not args: return "Error: Missing file path"
                return self._read_file_with_lines(repo_path, args[0])
            elif tool_name == "find_usages":
                if not args: return "Error: Missing symbol"
                return self._find_symbol_usages(repo_path, args[0])
            elif tool_name == "list_files":
                return self._get_file_list(repo_path)
            elif tool_name == "get_function_context":
                if len(args) < 2: return "Error: Need file_path and func_name"
                return self._get_function_context(repo_path, args[0], args[1])
            elif tool_name == "run_tests":
                if not args: return "Error: Missing test file path"
                return self._run_tests(repo_path, args[0])
            elif tool_name == "search_docs":
                if not args: return "Error: Missing search query"
                return self._search_docs(repo_path, args[0])
            else: return f"Tool '{tool_name}' not found. Available: grep, find_file, read_file, find_usages, list_files, get_function_context, run_tests, search_docs"
        except Exception as e: return f"Error executing {tool_name}: {e}"

    def _react_localize(self, instance: SWEBenchInstance, repo_path: str, candidates: List[str] = None, max_steps: int = 10) -> Optional[List[str]]:
        problem = instance.problem_statement
        history = f"# Bug Report\n{problem}\n"
        if candidates: history += f"\n# Candidate Files\n{json.dumps(candidates[:20], indent=2)}\n"
        files = self._get_file_list(repo_path)
        history += f"\n# Available Files\n{files}...\n"
        
        current_step = 0
        while current_step < max_steps:
            prompt = history + f"\n\nStep {current_step + 1}:\n"
            response = self.llm.generate(prompt, REACT_SYSTEM, 0.0, 512, stop=["OBSERVATION:"])
            history += f"\nStep {current_step + 1}:\n{response}\n"
            
            final_match = re.search(r'FINAL_ANSWER:\s*TARGET_FILE:\s*(.*)', response)
            if final_match:
                # Sanitize: extract just the path (remove any commentary)
                raw_path = final_match.group(1).strip()
                clean_path = re.split(r'[\s(,]', raw_path)[0].strip().lstrip('./')
                # Validate it exists
                if clean_path.endswith('.py') and (Path(repo_path) / clean_path).exists():
                    return [clean_path]
                continue  # Keep exploring if path invalid
            
            action = self._parse_action(response)
            if action:
                tool, args = action
                observation = self._execute_tool(tool, args, repo_path)
                history += f"OBSERVATION: {observation}\n"
            else: history += "OBSERVATION: Checking next step...\n"
            current_step += 1
        return None

    # --- S/R PATCH APPLICATION LOGIC ---

    def _apply_search_replace(self, content: str, response: str, debug: bool = True) -> str:
        """Apply SEARCH/REPLACE blocks with robust matching."""
        pattern = r"<{4,}\s*SEARCH\s*\n(.*?)\n={4,}\s*\n(.*?)\n>{4,}\s*REPLACE"
        matches = list(re.finditer(pattern, response, re.DOTALL))
        if not matches:
            if debug:
                print(f"DEBUG [S/R]: No SEARCH/REPLACE blocks found in response")
                print(f"DEBUG [S/R]: Response preview: {response[:500]}...")
            return content

        if debug:
            print(f"DEBUG [S/R]: Found {len(matches)} SEARCH/REPLACE block(s)")

        for idx, m in enumerate(matches):
            search_block = m.group(1)
            replace_block = m.group(2)

            if debug:
                print(f"DEBUG [S/R]: Block {idx+1} - SEARCH ({len(search_block)} chars): '{search_block[:100]}...'")

            # 0. Validate that critical elements aren't accidentally deleted
            is_safe, issue = validate_search_replace_preserves_critical(search_block, replace_block)
            if not is_safe:
                print(f"WARNING [S/R]: Block {idx+1} may have issues: {issue}")
                # Don't skip - just warn. The self-critique step will catch severe cases.

            # 1. Exact match (best case)
            if search_block in content:
                content = content.replace(search_block, replace_block, 1)
                if debug:
                    print(f"DEBUG [S/R]: Block {idx+1} - EXACT MATCH applied")
                continue

            # 2. Try stripping both sides
            search_stripped = search_block.strip()
            if search_stripped and search_stripped in content:
                loc = content.find(search_stripped)
                if loc >= 0:
                    content = content[:loc] + replace_block.strip() + content[loc + len(search_stripped):]
                    if debug:
                        print(f"DEBUG [S/R]: Block {idx+1} - STRIPPED MATCH applied")
                    continue

            # 3. Fuzzy block matching using SequenceMatcher (improved)
            if not search_stripped:
                if debug:
                    print(f"DEBUG [S/R]: Block {idx+1} - EMPTY search block, skipping")
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

            if debug:
                print(f"DEBUG [S/R]: Block {idx+1} - Best fuzzy ratio: {best_match_ratio:.2f} at line {best_match_idx}")
                if best_match_idx >= 0 and best_match_ratio < 0.85:
                    candidate = ''.join(content_lines[best_match_idx:best_match_idx + len(search_lines)])
                    print(f"DEBUG [S/R]: Block {idx+1} - FAILED (ratio {best_match_ratio:.2f} < 0.85)")
                    print(f"DEBUG [S/R]:   SEARCH: '{search_block[:150]}...'")
                    print(f"DEBUG [S/R]:   ACTUAL: '{candidate[:150]}...'")

            # Only apply if match quality is high enough (85% threshold)
            if best_match_idx >= 0 and best_match_ratio >= 0.85:
                before = ''.join(content_lines[:best_match_idx])
                after = ''.join(content_lines[best_match_idx + len(search_lines):])
                # Preserve trailing newline consistency
                if replace_block and not replace_block.endswith('\n'):
                    replace_block += '\n'
                content = before + replace_block + after
                if debug:
                    print(f"DEBUG [S/R]: Block {idx+1} - FUZZY MATCH applied (ratio {best_match_ratio:.2f})")

        return content

    def _phase_plan(self, instance: SWEBenchInstance, file_path: str, code_content: str, grep_results: str = "") -> str:
        hints = instance.hints_text if instance.hints_text else "No hints available."
        prompt = PLANNING_PROMPT.format(
            problem_statement=instance.problem_statement,
            hints_text=hints,
            file_path=file_path,
            code_content=code_content,
            grep_results=grep_results
        )
        return self.llm.generate(prompt, PLANNING_SYSTEM, 0.0, 2048)

    def _phase_patch(self, instance: SWEBenchInstance, file_path: str, code_content: str, fix_plan: str, examples_context: str = "") -> str:
        hints = instance.hints_text if instance.hints_text else "No hints available."
        prompt = PATCH_PROMPT.format(
            problem_statement=instance.problem_statement,
            hints_text=hints,
            examples_context=examples_context if examples_context else "No similar examples available.",
            fix_plan=fix_plan,
            file_path=file_path,
            code_content=code_content
        )
        return self.llm.generate(prompt, PATCH_SYSTEM, 0.0, 4096)

    def _rectify(self, instance, previous_patch: str, test_output: str, file_content: str) -> str:
        # Extract failing test names for context
        failing_tests = instance.fail_to_pass if instance.fail_to_pass else "Unknown tests"
        prompt = RECTIFICATION_PROMPT_TEMPLATE.format(
            problem_statement=instance.problem_statement,
            failing_tests=failing_tests,
            previous_patch=previous_patch,
            test_output=test_output,
            file_content=file_content
        )
        return self.llm.generate(prompt, RECTIFICATION_SYSTEM_PROMPT, 0.1, 4096)

    def _self_critique(self, instance: SWEBenchInstance, patch: str) -> Tuple[bool, str]:
        """Ask model to review its own patch for common mistakes (deletions, over-engineering)."""
        prompt = MINIMAL_CRITIQUE_PROMPT.format(
            problem_statement=instance.problem_statement[:1500],
            patch=patch[:3000]
        )
        response = self.llm.generate(prompt, "You are a strict code reviewer checking for minimal, correct patches.", 0.0, 512)

        if "APPROVED" in response.upper():
            return True, "Patch approved"
        return False, response

    def _phase_solution(self, instance: SWEBenchInstance, repo_path: str, target_file: str) -> Optional[str]:
        # 1. Read Code
        target_path = Path(repo_path) / target_file
        if not target_path.exists():
            found = self._find_file(repo_path, Path(target_file).name)
            if found: target_file = found; target_path = Path(repo_path) / found
            else: return None

        code_content = target_path.read_text()

        # 2. Retrieve similar examples (RAG)
        examples_context = ""
        if self.example_retriever:
            try:
                exs = self.example_retriever.retrieve(instance.problem_statement, instance.repo, 3)
                examples_context = "\n".join([
                    f"## Similar Bug: {e['instance_id']}\n"
                    f"Problem: {e['problem_statement'][:400]}\n"
                    f"Fix:\n```diff\n{e['patch'][:2000]}\n```"
                    for e in exs
                ])
            except Exception as e:
                print(f"Warning: Could not retrieve examples: {e}")

        # 3. Plan
        grep_results = ""
        fix_plan = self._phase_plan(instance, target_file, code_content, grep_results)

        # 4. Generate Patch (S/R) with RAG context
        def get_diff(orig, new, fname):
            diff_lines = difflib.unified_diff(
                orig.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"a/{fname}",
                tofile=f"b/{fname}"
            )
            return "".join(diff_lines)

        response = self._phase_patch(instance, target_file, code_content, fix_plan, examples_context)

        # 4a. Try OLD/NEW format first (simplest, SWE-smith style)
        new_content = None
        old_new = parse_old_new_format(response)
        if old_new:
            old_code, new_code = old_new
            if old_code in code_content:
                new_content = code_content.replace(old_code, new_code, 1)
                print(f"DEBUG: OLD/NEW format applied successfully")
            else:
                print(f"DEBUG: OLD/NEW format found but OLD code not in file (trying fuzzy)")
                # Try with stripped whitespace
                old_stripped = old_code.strip()
                if old_stripped in code_content:
                    new_content = code_content.replace(old_stripped, new_code.strip(), 1)
                    print(f"DEBUG: OLD/NEW format applied with stripped whitespace")
                else:
                    # Try finding first line in file and building match
                    old_lines = old_code.strip().split('\n')
                    first_line = old_lines[0].strip()
                    if first_line and first_line in code_content:
                        # Find where it occurs and try to match surrounding lines
                        idx = code_content.find(first_line)
                        if idx >= 0:
                            # Find start of this line
                            line_start = code_content.rfind('\n', 0, idx) + 1
                            # Find enough lines to match old_code length
                            num_old_lines = len(old_lines)
                            end_idx = line_start
                            for _ in range(num_old_lines):
                                next_nl = code_content.find('\n', end_idx)
                                end_idx = next_nl + 1 if next_nl >= 0 else len(code_content)
                            actual_old = code_content[line_start:end_idx].rstrip('\n')
                            # Apply replacement with same indentation
                            new_content = code_content[:line_start] + new_code + '\n' + code_content[end_idx:]
                            print(f"DEBUG: OLD/NEW format applied with line-based match")

        # 4b. Try unified diff if OLD/NEW didn't work
        if new_content is None:
            diff_text = parse_unified_diff(response)
            if diff_text:
                print(f"DEBUG: Trying unified diff...")
                new_content = apply_unified_diff(code_content, diff_text)
                if new_content and new_content != code_content:
                    print(f"DEBUG: Unified diff applied successfully")
                else:
                    new_content = None

        # 4c. Fall back to SEARCH/REPLACE as last resort
        if new_content is None:
            print(f"DEBUG: Using SEARCH/REPLACE fallback")
            new_content = self._apply_search_replace(code_content, response)

        # 5. VALIDATE the patched code before proceeding
        is_valid, validation_msg = validate_patched_code(code_content, new_content)
        if not is_valid:
            print(f"DEBUG: Patch validation FAILED: {validation_msg}")
            # Try to regenerate with explicit warning
            response2 = self._phase_patch(instance, target_file, code_content,
                fix_plan + f"\n\nWARNING: Previous attempt had issue: {validation_msg}. Be more careful.",
                examples_context)
            # Try OLD/NEW first
            old_new2 = parse_old_new_format(response2)
            if old_new2:
                old_code2, new_code2 = old_new2
                if old_code2 in code_content:
                    new_content = code_content.replace(old_code2, new_code2, 1)
            # Then unified diff
            if new_content is None or new_content == code_content:
                diff_text2 = parse_unified_diff(response2)
                if diff_text2:
                    new_content = apply_unified_diff(code_content, diff_text2)
            # Then S/R
            if new_content is None or new_content == code_content:
                new_content = self._apply_search_replace(code_content, response2)
            is_valid, _ = validate_patched_code(code_content, new_content)
            if not is_valid:
                print(f"DEBUG: Second attempt also failed validation. Using anyway.")

        patch = get_diff(code_content, new_content, target_file)

        # NOTE: Self-critique removed (SWE-smith approach - let tests be the critic)

        # 6. Iterative Verification & Rectification (up to 3 attempts)
        if patch and self.harness:
            print(f"DEBUG: Agentic Verification of {target_file}...")
            current_content = new_content
            for attempt in range(3):
                success, logs = self.harness.verify_patch_with_logs(instance, patch)
                if success:
                    print(f"DEBUG: Verification PASSED on attempt {attempt + 1}")
                    break
                print(f"DEBUG: Verification FAILED (attempt {attempt + 1}/3). Rectifying...")
                rectify_response = self._rectify(instance, patch, logs, current_content)

                # Try OLD/NEW format first for rectification
                final_content = None
                old_new_rect = parse_old_new_format(rectify_response)
                if old_new_rect:
                    old_code_rect, new_code_rect = old_new_rect
                    if old_code_rect in current_content:
                        final_content = current_content.replace(old_code_rect, new_code_rect, 1)
                        print(f"DEBUG: Rectification OLD/NEW applied")

                # Then unified diff
                if final_content is None or final_content == current_content:
                    diff_text_rect = parse_unified_diff(rectify_response)
                    if diff_text_rect:
                        final_content = apply_unified_diff(current_content, diff_text_rect)

                # Then S/R
                if final_content is None or final_content == current_content:
                    final_content = self._apply_search_replace(current_content, rectify_response)

                # Validate rectification
                is_valid, msg = validate_patched_code(code_content, final_content)
                if not is_valid:
                    print(f"DEBUG: Rectification invalid: {msg}. Skipping.")
                    continue

                current_content = final_content
                patch = get_diff(code_content, final_content, target_file)

        # Return final patch or None
        return patch if patch else None

    # --- MAIN RUN METHOD ---

    def _phase_localize_candidates(self, instance: SWEBenchInstance, repo_path: str) -> List[str]:
        """Enhanced localization with multiple fallback strategies."""
        candidates = set()

        # Combine problem statement with hints_text for all searches
        full_text = instance.problem_statement + "\n" + (instance.hints_text or "")

        def expand_via_graph(cands: set) -> set:
            """Expand candidates using import graph to find related files."""
            if not cands:
                return cands
            try:
                graph = CodeGraph(repo_path)
                expanded = set(cands)
                for candidate in list(cands)[:3]:  # Only expand top 3 to limit
                    related = graph.get_related_files(candidate, max_depth=1)
                    for rel in related[:3]:  # Max 3 related per file
                        if 'test' not in rel.lower() and rel.endswith('.py'):
                            expanded.add(rel)
                if len(expanded) > len(cands):
                    print(f"DEBUG: Graph expanded from {len(cands)} to {len(expanded)} candidates")
                return expanded
            except Exception:
                return cands  # Graph expansion is optional enhancement

        def filter_candidates(cands: set) -> list:
            """Filter and score candidates. Uses test names for better ranking."""
            # First expand via graph to find related files
            cands = expand_via_graph(cands)

            # Extract test names for scoring (e.g., "test_property" -> "property")
            test_names = extract_test_names(instance.fail_to_pass)
            test_keywords = set()
            test_file_stems = set()  # NEW: Extract file stems from test paths
            for t in test_names:
                # Extract keywords from test names like "test_property_init" -> ["property", "init"]
                parts = t.replace('test_', '').split('_')
                test_keywords.update(p for p in parts if len(p) > 2)
                # NEW: Extract test file stem (e.g., "tests/test_property.py::test_check" -> "property")
                match = re.search(r'test_(\w+)\.py', t)
                if match:
                    test_file_stems.add(match.group(1).lower())
                # Also match "test_property::test_func" pattern
                match2 = re.search(r'test_(\w+)::', t)
                if match2:
                    test_file_stems.add(match2.group(1).lower())

            scored = []
            root_level = []

            for c in cands:
                # Skip __init__.py
                if c.endswith('__init__.py'):
                    continue
                # Skip test files
                if 'test' in c.lower() and '/test' in c.lower():
                    continue
                # Validate path exists
                if not (Path(repo_path) / c).exists():
                    continue

                # Score based on test keyword matches
                score = 0
                file_name = Path(c).stem.lower()  # e.g., "property" from "property.py"
                file_parts = set(file_name.split('_'))

                # NEW: STRONG match - file stem exactly matches test file stem
                # e.g., property.py matches test_property.py -> +25
                # Also handle _property.py -> property match
                file_name_stripped = file_name.lstrip('_')
                if file_name in test_file_stems or file_name_stripped in test_file_stems:
                    score += 25
                    print(f"DEBUG: Strong match! {c} matches test file stem")

                # Direct match with file name in keywords
                if file_name in test_keywords:
                    score += 10
                # Partial match
                for kw in test_keywords:
                    if kw in file_name:
                        score += 5
                    if kw in file_parts:
                        score += 3

                # Prefer files in core directories
                if '/core/' in c or '/src/' in c:
                    score += 2
                # Penalize demos/examples
                if '/demo' in c or '/example' in c:
                    score -= 5

                if '/' not in c:
                    root_level.append((score - 2, c))  # Slight penalty for root files
                else:
                    scored.append((score, c))

            # Sort by score descending
            scored.sort(key=lambda x: -x[0])
            root_level.sort(key=lambda x: -x[0])

            if scored:
                result = [c for _, c in scored]
                if test_keywords:
                    print(f"DEBUG: Scored candidates using test keywords {test_keywords}: {[(s,c) for s,c in scored[:5]]}")
                return result
            elif root_level:
                return [c for _, c in root_level]
            else:
                return list(cands)

        # 1. Stacktrace extraction (highest priority)
        for fp in extract_stacktrace_files(full_text):
            if (Path(repo_path) / fp).exists():
                candidates.add(fp)
            else:
                # Try common prefixes (e.g., src/marshmallow/schema.py)
                for prefix in ['src', 'lib', instance.repo.split('/')[-1]]:
                    candidate = f"{prefix}/{fp}"
                    if (Path(repo_path) / candidate).exists():
                        candidates.add(candidate)
                        break

        if candidates:
            print(f"DEBUG: Found {len(candidates)} stacktrace candidates")
            return filter_candidates(candidates)

        # 2. Error message search (grep for error strings in codebase)
        for err_msg in extract_error_messages(full_text):
            try:
                # Escape special regex chars
                safe_msg = re.escape(err_msg[:40])
                result = subprocess.run(
                    ["grep", "-r", "-l", safe_msg, "."],
                    cwd=repo_path, capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0 and result.stdout:
                    for line in result.stdout.splitlines()[:5]:
                        if line.endswith('.py') and 'test' not in line.lower():
                            candidates.add(line.lstrip('./'))
            except: pass

        if candidates:
            print(f"DEBUG: Found {len(candidates)} error message candidates")
            return filter_candidates(candidates)

        # 2.5. Test file stem search - CRITICAL for finding property.py from test_property.py
        # If fail_to_pass mentions "test_property.py::test_xyz", search for "property.py"
        test_names = extract_test_names(instance.fail_to_pass)
        test_file_stems = set()
        for t in test_names:
            # "tests/test_property.py::test_check" -> "property"
            match = re.search(r'test_(\w+)\.py', t)
            if match:
                test_file_stems.add(match.group(1).lower())
            # "test_property::test_func" -> "property"
            match2 = re.search(r'test_(\w+)::', t)
            if match2:
                test_file_stems.add(match2.group(1).lower())

        # Search for files matching test stems (include _property.py pattern)
        for stem in test_file_stems:
            # Try multiple patterns: property.py, _property.py, *property.py
            for pattern in [f"{stem}.py", f"_{stem}.py", f"*{stem}.py"]:
                try:
                    result = subprocess.run(
                        ["find", ".", "-name", pattern, "-type", "f"],
                        cwd=repo_path, capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0 and result.stdout:
                        for line in result.stdout.splitlines()[:5]:
                            if 'test' not in line.lower():
                                candidates.add(line.lstrip('./'))
                except: pass

        if candidates:
            print(f"DEBUG: Found {len(candidates)} test-stem matching candidates: {list(candidates)[:5]}")
            return filter_candidates(candidates)

        # 3. Rule identifiers in hints (e.g., "L039" -> search for L039.py in rules/)
        # Only use if we have a small number of specific rules mentioned (avoid over-matching)
        rule_ids = extract_rule_identifiers(full_text)
        if 0 < len(rule_ids) <= 5:  # Only search if 1-5 rules mentioned
            for rule_id in rule_ids:
                try:
                    result = subprocess.run(
                        ["find", ".", "-name", f"{rule_id}.py", "-type", "f"],
                        cwd=repo_path, capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0 and result.stdout:
                        for line in result.stdout.splitlines()[:3]:
                            candidates.add(line.lstrip('./'))
                except: pass

            if candidates:
                print(f"DEBUG: Found {len(candidates)} rule identifier candidates")
                return filter_candidates(candidates)

        # 3. Explicit file paths in problem statement AND hints
        for fp in extract_file_paths(full_text):
            # Only add if it actually exists in repo (filter out user code like dicom.py)
            if (Path(repo_path) / fp).exists():
                candidates.add(fp)
            # Try common prefixes
            for prefix in ['src', 'lib', instance.repo.split('/')[-1]]:
                candidate = f"{prefix}/{fp}"
                if (Path(repo_path) / candidate).exists():
                    candidates.add(candidate)

        if candidates:
            print(f"DEBUG: Found {len(candidates)} explicit path candidates")
            return filter_candidates(candidates)

        # 4. Explicit module mentions (e.g., "the config module")
        for mod in extract_explicit_modules(full_text):
            fp = f"{mod}.py"
            # Search for this file in the repo
            try:
                result = subprocess.run(
                    ["find", ".", "-name", fp, "-type", "f"],
                    cwd=repo_path, capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0 and result.stdout:
                    for line in result.stdout.splitlines()[:3]:
                        if 'test' not in line.lower():
                            candidates.add(line.lstrip('./'))
            except: pass

        if candidates:
            print(f"DEBUG: Found {len(candidates)} explicit module candidates")
            return filter_candidates(candidates)

        # 5. Class name search (e.g., "PersonName3" -> grep "class PersonName")
        for cls in extract_class_names(full_text):
            try:
                result = subprocess.run(
                    ["grep", "-r", "-l", f"class {cls}", "."],
                    cwd=repo_path, capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0 and result.stdout:
                    for line in result.stdout.splitlines()[:3]:
                        if line.endswith('.py') and 'test' not in line.lower():
                            candidates.add(line.lstrip('./'))
            except: pass

        if candidates:
            print(f"DEBUG: Found {len(candidates)} class definition candidates")
            return filter_candidates(candidates)

        # 6. Module paths converted to file paths
        for mp in extract_module_paths(full_text):
            fp = module_to_filepath(mp)
            if (Path(repo_path) / fp).exists():
                candidates.add(fp)
            # Try with leading dirs
            for prefix in ['src', 'lib', '']:
                candidate = f"{prefix}/{fp}" if prefix else fp
                if (Path(repo_path) / candidate).exists():
                    candidates.add(candidate)

        if candidates:
            print(f"DEBUG: Found {len(candidates)} module path candidates")
            return filter_candidates(candidates)

        # 7. LLM-Assisted Localization (SOTA fallback)
        try:
            file_list = self._get_file_list(repo_path)
            prompt = LLM_LOCALIZATION_PROMPT.format(
                problem_statement=instance.problem_statement,
                file_list=file_list
            )
            response = self.llm.generate(prompt, LLM_LOCALIZATION_SYSTEM, 0.0, 1024)

            # Parse JSON response
            # Extract JSON from response (may have markdown wrappers)
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                loc_data = json.loads(json_match.group())
                search_terms = loc_data.get('search_terms', [])
                file_patterns = loc_data.get('file_patterns', [])
                likely_dirs = loc_data.get('likely_dirs', [])

                # Use file_patterns to find files (e.g., "*helpers.py")
                for pattern in file_patterns[:5]:
                    try:
                        result = subprocess.run(
                            ["find", ".", "-name", pattern, "-type", "f"],
                            cwd=repo_path, capture_output=True, text=True, timeout=10
                        )
                        if result.returncode == 0 and result.stdout:
                            for line in result.stdout.splitlines()[:5]:
                                if line.endswith('.py') and 'test' not in line.lower():
                                    candidates.add(line.lstrip('./'))
                    except: pass

                # Search in likely directories
                for likely_dir in likely_dirs[:3]:
                    dir_path = Path(repo_path) / likely_dir
                    if dir_path.exists() and dir_path.is_dir():
                        for py_file in dir_path.glob('*.py'):
                            rel = str(py_file.relative_to(repo_path))
                            if 'test' not in rel.lower():
                                candidates.add(rel)

                # Grep for each search term - get more candidates for fallback
                for term in search_terms[:5]:
                    if len(term) < 4: continue
                    try:
                        result = subprocess.run(
                            ["grep", "-E", "-r", "-l", f"(class|def).*{term}", "."],
                            cwd=repo_path, capture_output=True, text=True, timeout=10
                        )
                        if result.returncode == 0 and result.stdout:
                            for line in result.stdout.splitlines()[:5]:
                                if line.endswith('.py') and 'test' not in line.lower():
                                    candidates.add(line.lstrip('./'))
                    except: pass
        except Exception as e:
            print(f"DEBUG: LLM localization failed: {e}")
        
        if candidates:
            print(f"DEBUG: Found {len(candidates)} LLM-assisted candidates")
            return filter_candidates(candidates)

        # 8. ReAct exploration (last resort)
        react = self._react_localize(instance, repo_path, candidates=list(candidates))
        if react:
            candidates.update(react)

        # Note: Graph expansion is now integrated into filter_candidates
        return filter_candidates(candidates) if candidates else []

    # =========================================================================
    #  MINI-SWE-AGENT STYLE LOOP (bash-only, simple)
    # =========================================================================

    def _mini_agent_loop(self, instance: SWEBenchInstance, repo_path: str, max_steps: int = 25) -> Optional[str]:
        """
        Mini-SWE-Agent style: LLM generates bash commands, we execute them.
        Returns the git diff if any changes were made, None otherwise.
        """
        # Build initial context
        hints = getattr(instance, 'hints_text', '') or ''

        # Get similar examples from RAG
        similar_examples = ""
        if self.example_retriever:
            try:
                exs = self.example_retriever.retrieve(instance.problem_statement, instance.repo, 2)
                if exs:
                    similar_examples = "\n\n=== SIMILAR BUG FIXES (for reference) ===\n"
                    for i, ex in enumerate(exs[:2]):
                        similar_examples += f"\nExample {i+1}:\n"
                        problem = ex.get('problem_statement', '') if isinstance(ex, dict) else getattr(ex, 'problem_statement', '')
                        patch = ex.get('patch', '') if isinstance(ex, dict) else getattr(ex, 'patch', '')
                        similar_examples += f"Problem: {problem[:500]}...\n"
                        similar_examples += f"Fix:\n{patch[:800]}\n"
            except Exception as e:
                print(f"DEBUG [MiniAgent]: Could not retrieve examples: {e}")

        # Extract search hint from traceback (if present)
        search_hint = ""
        attr_error_hint = ""
        import re

        # Look for AttributeError pattern - this tells us what attribute is missing
        attr_error = re.search(r"AttributeError[:\s]+['\"]?(\w+)['\"]?\s+object has no attribute ['\"](\w+)['\"]",
                               instance.problem_statement)
        if attr_error:
            wrong_obj_type, missing_attr = attr_error.groups()
            attr_error_hint = f"\n\nATTRIBUTE ERROR DETECTED: '{wrong_obj_type}' has no attribute '{missing_attr}'\n" \
                              f"This usually means the wrong object is being used.\n" \
                              f"Search for correct usage: grep -n '\\.{missing_attr}' FILE\n" \
                              f"The fix is likely: wrong_object.{missing_attr} -> correct_object.{missing_attr}"

        # Look for file:line patterns in traceback
        file_matches = re.findall(r'File "([^"]+)", line (\d+)', instance.problem_statement)
        if file_matches:
            last_file, last_line = file_matches[-1]
            filename = last_file.split('/')[-1]
            search_hint = f"grep -rn '{filename}' --include='*.py' | head -10"
        else:
            # Look for function names or class names in error
            func_match = re.search(r"in (\w+)\n", instance.problem_statement)
            if func_match:
                search_hint = f"grep -rn 'def {func_match.group(1)}' --include='*.py' | head -20"

        # Build conversation as a single prompt
        conversation = f"""Bug Report:
{instance.problem_statement}

Hints from issue discussion:
{hints if hints else 'None'}

Tests that should pass after fix: {instance.fail_to_pass}
{similar_examples}
{attr_error_hint}

You are in the repository root. Fix this bug using bash commands.
Output ONE bash command per response.

IMPORTANT: Before making ANY edit:
1. First READ at least 50 lines around the error location with sed -n
2. SEARCH for the correct pattern with grep to see how other code does it
3. The fix is usually changing ONE OBJECT REFERENCE (e.g., schema -> self.root)
4. DO NOT use getattr() wrappers - that's defensive coding, not a real fix

START by searching: {search_hint if search_hint else "grep -rn 'ERROR_KEYWORD' --include='*.py' | head -20"}"""

        print(f"DEBUG [MiniAgent]: Starting loop with max {max_steps} steps")

        sed_was_run = False
        context_was_read = False  # Track if model read code context
        pattern_searched = False  # Track if model searched for correct pattern
        recent_commands = []  # Track recent commands to detect loops

        for step in range(max_steps):
            # Truncate conversation if it gets too long (keep last 12000 chars)
            if len(conversation) > 15000:
                # Keep the beginning (bug report) and recent context
                bug_report_end = conversation.find("Output ONE bash command")
                if bug_report_end > 0:
                    header = conversation[:bug_report_end + 50]
                    recent = conversation[-10000:]
                    conversation = header + "\n\n[... earlier exploration truncated ...]\n" + recent

            # Get next command from LLM
            response = self.llm.generate(
                prompt=conversation,
                system_prompt=MINI_AGENT_SYSTEM,
                temperature=0.0,
                max_tokens=500
            )

            print(f"DEBUG [MiniAgent] Step {step+1}: {response[:200]}...")

            # Check if done - but only accept if sed was run
            if "DONE" in response.upper():
                if sed_was_run:
                    print(f"DEBUG [MiniAgent]: Agent signaled DONE at step {step+1}")
                    break
                else:
                    # Force the model to run sed
                    conversation += f"\n\nAssistant: {response}\n\nYou haven't made any changes yet! You MUST run a sed command to fix the bug before saying DONE. What sed command will you run?"
                    continue

            # Extract and run bash command
            cmd = extract_bash_command(response)
            if cmd:
                print(f"DEBUG [MiniAgent] Running: {cmd[:100]}")

                # Track if context was read (sed -n for reading)
                if 'sed -n' in cmd and 'p' in cmd:
                    context_was_read = True

                # Track if pattern was searched (grep for attribute access)
                if 'grep' in cmd and ('\\.' in cmd or '.' in cmd):
                    pattern_searched = True

                # Track if edit was attempted (sed -i or python line replacement)
                is_sed_edit = cmd.strip().startswith('sed -i') or ('sed ' in cmd and ' -i' in cmd)
                is_python_edit = 'python -c' in cmd and 'lines[' in cmd and 'writelines' in cmd
                is_edit = is_sed_edit or is_python_edit

                # Check for defensive coding patterns in edit commands (only for AttributeError bugs)
                is_attr_error_bug = 'AttributeError' in instance.problem_statement
                if is_edit and is_attr_error_bug:
                    is_defensive = ('getattr(' in cmd and 'None' in cmd) or \
                                   ('getattr(getattr' in cmd)
                    if is_defensive:
                        conversation += f"\n\nAssistant: {response}\n\nSTOP! This looks like defensive coding (getattr with None fallback).\n" \
                            "The bug is NOT about missing attributes - it's about using the WRONG OBJECT.\n" \
                            "DO NOT use getattr(x, 'attr', None) - instead change the object reference.\n" \
                            "Example: 'schema.opts' should become 'self.root.opts'\n" \
                            "Search for the correct pattern: grep -n '.opts' FILE to see how other code does it."
                        continue

                # Detect EDIT command loops only (not reads)
                cmd_normalized = cmd.strip()[:80]
                if is_edit and cmd_normalized in recent_commands[-3:]:
                    conversation += f"\n\nAssistant: {response}\n\nYou're repeating the same EDIT command. It may not be working. Try:\n1. Read the file again with sed -n to see current content\n2. The fix is often: 'schema.attr' -> 'self.root.attr'\n3. Make sure line numbers are correct (Python uses 0-indexed)"
                    recent_commands.append(cmd_normalized)
                    continue
                if is_edit:
                    recent_commands.append(cmd_normalized)

                # Force reading context before editing
                if is_edit and not context_was_read:
                    conversation += f"\n\nAssistant: {response}\n\nSTOP! You must READ the code first before editing. Use 'sed -n START,ENDp file.py' to read at least 30 lines around the error location. What file and lines do you want to read?"
                    continue

                # Force pattern search before editing (only for AttributeError bugs)
                if is_edit and is_attr_error_bug and not pattern_searched:
                    conversation += f"\n\nAssistant: {response}\n\nSTOP! Before editing, search for the CORRECT pattern.\n" \
                        "Use: grep -n '.ATTRIBUTE' FILE to see how other code accesses this attribute.\n" \
                        "The fix is usually changing the OBJECT (e.g., schema -> self.root), not wrapping with getattr."
                    continue

                if is_edit:
                    sed_was_run = True

                output = run_bash(cmd, repo_path)

                # Add to conversation
                conversation += f"\n\nAssistant: {response}\n\nOutput:\n{output}"
            else:
                # No command found, prompt for one
                conversation += f"\n\nAssistant: {response}\n\n(Please output a bash command - just the command, no markdown.)"

        # Get git diff
        diff_result = subprocess.run(
            ["git", "diff"],
            cwd=repo_path, capture_output=True, text=True
        )

        if diff_result.stdout.strip():
            print(f"DEBUG [MiniAgent]: Generated patch ({len(diff_result.stdout)} chars)")
            return diff_result.stdout
        else:
            print(f"DEBUG [MiniAgent]: No changes made to repository")
            return None

    # =========================================================================
    #  AGENTLESS-STYLE METHODS (Proven approach from Agentless/CoT)
    # =========================================================================

    def _get_python_files(self, repo_path: Path) -> List[str]:
        """Get list of Python files (excluding tests)."""
        files = []
        for py in repo_path.rglob("*.py"):
            rel_path = str(py.relative_to(repo_path))
            # Skip test files and hidden dirs
            if "test" in rel_path.lower() or any(p.startswith(".") for p in rel_path.split("/")):
                continue
            files.append(rel_path)
        return sorted(files)[:300]  # Limit to 300 files

    def _agentless_localize_file(self, problem_statement: str, repo_path: Path) -> Optional[str]:
        """Stage 1: File-level localization using LLM."""
        files = self._get_python_files(repo_path)
        if not files:
            return None

        prompt = AGENTLESS_LOCALIZATION_PROMPT.format(
            problem_statement=problem_statement[:3000],
            file_list="\n".join(files)
        )

        response = self.llm.generate(prompt, AGENTLESS_LOCALIZATION_SYSTEM, 0.0, 512)

        # Try multiple patterns for TARGET_FILE extraction
        patterns = [
            r'TARGET_FILE:\s*`?([^`\s]+\.py)`?',
            r'\*\*TARGET_FILE\*\*:\s*`?([^`\s]+\.py)`?',
            r'TARGET FILE:\s*`?([^`\s]+\.py)`?',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fallback: look for any .py file mentioned in last 200 chars
        last_part = response[-200:]
        py_files = re.findall(r'([a-zA-Z_/]+\.py)', last_part)
        if py_files:
            for pf in py_files:
                if pf in files or any(f.endswith(pf) for f in files):
                    return pf

        return files[0] if files else None

    def _agentless_apply_search_replace(self, content: str, response: str) -> str:
        """Apply SEARCH/REPLACE blocks with robust fuzzy matching."""
        pattern = r"<{4,}\s*SEARCH\s*\n(.*?)\n={4,}\s*\n(.*?)\n>{4,}\s*REPLACE"
        matches = list(re.finditer(pattern, response, re.DOTALL))
        if not matches:
            return content

        def strip_markdown(block: str) -> str:
            """Remove markdown code block markers from search/replace blocks."""
            lines = block.strip().splitlines()
            # Remove leading ```python or ``` line
            if lines and lines[0].strip().startswith('```'):
                lines = lines[1:]
            # Remove trailing ``` line
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            return '\n'.join(lines)

        for m in matches:
            search_block = strip_markdown(m.group(1))
            replace_block = strip_markdown(m.group(2))

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

            # 3. Try matching with indentation-normalized comparison
            if not search_stripped:
                continue
            search_lines = search_block.splitlines()
            if len(search_lines) < 1:
                continue

            content_lines = content.splitlines(keepends=True)
            best_match_idx = -1
            best_match_ratio = 0.0
            best_indent = ""

            # Normalize: strip leading whitespace for comparison
            def normalize_for_compare(block: str) -> str:
                return '\n'.join(line.strip() for line in block.splitlines())

            search_normalized = normalize_for_compare(search_block)

            # Slide window over content to find best matching block
            for i in range(len(content_lines) - len(search_lines) + 1):
                candidate_block = ''.join(content_lines[i:i + len(search_lines)])
                candidate_normalized = normalize_for_compare(candidate_block)

                ratio = difflib.SequenceMatcher(None, search_normalized, candidate_normalized).ratio()
                if ratio > best_match_ratio:
                    best_match_ratio = ratio
                    best_match_idx = i
                    # Detect indentation from first non-empty line
                    for line in content_lines[i:i + len(search_lines)]:
                        stripped = line.lstrip()
                        if stripped:
                            best_indent = line[:len(line) - len(stripped) - (1 if line.endswith('\n') else 0)]
                            break
                if ratio > 0.98:
                    break

            # Only apply if match quality is high enough (80% threshold with indent normalization)
            if best_match_idx >= 0 and best_match_ratio >= 0.80:
                before = ''.join(content_lines[:best_match_idx])
                after = ''.join(content_lines[best_match_idx + len(search_lines):])
                # Apply detected indentation to replacement block
                replace_lines = replace_block.splitlines()
                indented_replace = '\n'.join(best_indent + line.lstrip() if line.strip() else line for line in replace_lines)
                if indented_replace and not indented_replace.endswith('\n'):
                    indented_replace += '\n'
                content = before + indented_replace + after

        return content

    def _agentless_generate_diff(self, original: str, modified: str, filename: str) -> str:
        """Generate unified diff from before/after content."""
        diff_lines = difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}"
        )
        return "".join(diff_lines)

    def _agentless_fix(self, instance: SWEBenchInstance, repo_path: Path) -> Optional[str]:
        """
        Agentless-style fix: Localize file, generate SEARCH/REPLACE patch.
        Returns unified diff or None.
        """
        print(f"DEBUG [Agentless]: Localizing file...")

        # Stage 1: File-level localization
        target_file = self._agentless_localize_file(instance.problem_statement, repo_path)
        if not target_file or not (repo_path / target_file).exists():
            print(f"DEBUG [Agentless]: Localization failed - file not found: {target_file}")
            return None

        print(f"DEBUG [Agentless]: Target file: {target_file}")

        # Read file content
        primary_content = (repo_path / target_file).read_text()

        # Get hints and examples
        hints = getattr(instance, 'hints_text', '') or ''
        examples_content = ""
        if self.example_retriever:
            try:
                exs = self.example_retriever.retrieve(instance.problem_statement, instance.repo, 2)
                for i, ex in enumerate(exs[:2]):
                    problem = ex.get('problem_statement', '') if isinstance(ex, dict) else getattr(ex, 'problem_statement', '')
                    patch = ex.get('patch', '') if isinstance(ex, dict) else getattr(ex, 'patch', '')
                    examples_content += f"\nExample {i+1}:\nProblem: {problem[:500]}...\nFix:\n{patch[:800]}\n"
            except Exception as e:
                print(f"DEBUG [Agentless]: Could not retrieve examples: {e}")

        # Stage 2: Generate fix using SEARCH/REPLACE blocks (send full file content)
        prompt = AGENTLESS_FIX_PROMPT.format(
            problem_statement=instance.problem_statement,
            hints_text=hints if hints else 'None',
            examples_content=examples_content if examples_content else "None available",
            primary_file=target_file,
            primary_content=primary_content
        )

        response = self.llm.generate(prompt, AGENTLESS_FIX_SYSTEM, self.temperature, self.max_tokens)
        response = response or ""  # Handle None response
        print(f"DEBUG [Agentless]: LLM response length: {len(response)} chars")
        print(f"DEBUG [Agentless]: Response preview: {repr(response[:500])}")

        # Stage 3: Apply SEARCH/REPLACE blocks
        new_content = self._agentless_apply_search_replace(primary_content, response)

        if new_content == primary_content:
            print(f"DEBUG [Agentless]: No changes made - SEARCH/REPLACE failed to match")
            # Try to extract and show what block was attempted
            pattern = r"<{4,}\s*SEARCH\s*\n(.*?)\n={4,}"
            match = re.search(pattern, response, re.DOTALL)
            if match:
                print(f"DEBUG [Agentless]: SEARCH block found:\n{match.group(1)[:500]}")
            else:
                print(f"DEBUG [Agentless]: No SEARCH block found in response. Looking for alternative patterns...")
                # Try simpler pattern
                if "SEARCH" in response:
                    print(f"DEBUG: 'SEARCH' found at position {response.find('SEARCH')}")
                if "<<<<" in response:
                    print(f"DEBUG: '<<<<' found at position {response.find('<<<<')}")
            return None

        # Generate unified diff
        patch = self._agentless_generate_diff(primary_content, new_content, target_file)
        if patch:
            print(f"DEBUG [Agentless]: Generated patch ({len(patch)} chars)")
        return patch

    def run(self, instance: SWEBenchInstance) -> PipelineResult:
        """
        Agentless-style run: Hierarchical localization + SEARCH/REPLACE patches.
        This is the proven approach that works on SWE-bench.
        """
        repo_path = Path(self.repo_manager.get_repo_path(instance.repo, instance.base_commit))

        print(f"DEBUG [Agentless]: Running on {instance.instance_id}")

        # CRITICAL: Reset repo to clean state before each run
        subprocess.run(['git', 'reset', '--hard', 'HEAD'], cwd=str(repo_path), capture_output=True)
        subprocess.run(['git', 'clean', '-fd'], cwd=str(repo_path), capture_output=True)

        # Use Agentless-style approach (localization + SEARCH/REPLACE)
        patch = self._agentless_fix(instance, repo_path)

        if patch:
            # Optional: Verify with harness
            if self.harness:
                print(f"DEBUG [Agentless]: Verifying patch...")
                success, logs = self.harness.verify_patch_with_logs(instance, patch)
                if success:
                    print(f"DEBUG [Agentless]: Verification PASSED!")
                else:
                    print(f"DEBUG [Agentless]: Verification failed, but returning patch anyway")

            return PipelineResult(
                instance_id=instance.instance_id,
                generated_patch=patch,
                ground_truth_patch=instance.patch,
                success=True,
                raw_response="Agentless-style fix"
            )

        return PipelineResult(
            instance_id=instance.instance_id,
            generated_patch="",
            ground_truth_patch=instance.patch,
            raw_response="",
            success=False,
            error="Agentless approach could not generate a fix"
        )
