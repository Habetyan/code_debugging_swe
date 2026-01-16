"""
Pass@1 Compatible Patch Validation

This module provides static validation checks for generated patches that
do NOT require test execution, maintaining pass@1 compliance.

Checks included:
1. Diff quality (size, deletions, keyword relevance)
2. Syntax validation (ast.parse)
3. LLM self-critique (code review, no execution)

These checks help improve patch quality while remaining pass@1 compatible
because they don't use test execution feedback.
"""
import ast
import re
from typing import List, Tuple, Optional, Callable


# =============================================================================
#  LLM SELF-CRITIQUE PROMPTS
# =============================================================================

SELF_CRITIQUE_SYSTEM = """You are a strict code reviewer. Review patches for obvious bugs and issues.
Do NOT execute code - analyze it statically.
Be concise and precise in your feedback."""

SELF_CRITIQUE_PROMPT = """Review this patch for a bug fix. Check for obvious issues.

## Bug Description
{problem_statement}

## Generated Patch
```diff
{patch}
```

## Review Checklist
1. SYNTAX: Any obvious syntax errors in the new code?
2. LOGIC: Does the fix address the root cause described in the bug?
3. MINIMAL: Is this the smallest possible fix? (no unnecessary changes)
4. COMPLETE: Is anything missing from the fix?
5. SIDE EFFECTS: Could this change break other functionality?

If the patch looks correct and addresses the bug, respond exactly: APPROVED

If there are issues, respond with:
ISSUES:
- [issue 1]
- [issue 2]
"""

REGENERATE_PROMPT = """The previous patch had issues. Generate a better fix.

## Bug Description
{problem_statement}

## Previous Patch Issues
{issues}

## File Content
```python
{file_content}
```

Generate a corrected fix using SEARCH/REPLACE format:
<<<< SEARCH
exact original code
====
fixed code
>>>> REPLACE
"""


# =============================================================================
#  DIFF QUALITY VALIDATION
# =============================================================================

def extract_patch_keywords(problem_statement: str) -> List[str]:
    """
    Extract likely relevant identifiers from problem statement.
    Used to check if patch touches relevant code.
    """
    keywords = set()

    # Function/method names (snake_case with parentheses)
    funcs = re.findall(r'\b([a-z_][a-z0-9_]{3,})\s*\(', problem_statement)
    keywords.update(funcs)

    # Class names (CamelCase)
    classes = re.findall(r'\b([A-Z][a-zA-Z0-9]{2,})\b', problem_statement)
    keywords.update(classes)

    # Backtick identifiers (commonly used in bug reports)
    backticks = re.findall(r'`([a-zA-Z_][a-zA-Z0-9_]*)`', problem_statement)
    keywords.update(backticks)

    # Attribute access patterns (e.g., .some_attr)
    attrs = re.findall(r'\.([a-z_][a-z0-9_]{2,})', problem_statement)
    keywords.update(attrs)

    # Filter out common words that aren't identifiers
    stopwords = {
        'none', 'true', 'false', 'self', 'return', 'import', 'from',
        'class', 'def', 'and', 'or', 'not', 'the', 'this', 'that',
        'with', 'for', 'while', 'try', 'except', 'raise', 'pass',
        'print', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple'
    }
    keywords = {kw for kw in keywords if kw.lower() not in stopwords and len(kw) > 2}

    return list(keywords)[:15]  # Top 15 keywords


def validate_diff_quality(patch: str, problem_statement: str) -> Tuple[bool, str]:
    """
    Validate patch quality using static heuristics.
    NO test execution - purely analyzing the diff structure.

    Returns:
        (is_valid, message)
    """
    if not patch or not patch.strip():
        return False, "Empty patch"

    lines = patch.split('\n')

    # Count added/removed lines (excluding diff headers)
    added = len([l for l in lines if l.startswith('+') and not l.startswith('+++')])
    removed = len([l for l in lines if l.startswith('-') and not l.startswith('---')])

    # Check 1: Patch is not too large (probably wrong file or over-engineered)
    if added + removed > 100:
        return False, f"Patch too large ({added} additions, {removed} deletions) - likely wrong approach"

    # Check 2: Patch has actual changes
    if added == 0 and removed == 0:
        return False, "No actual changes in patch"

    # Check 3: Patch is not just deletions (suspicious - usually wrong)
    if removed > 10 and added == 0:
        return False, f"Patch only deletes {removed} lines with no additions - suspicious"

    # Check 4: Massive deletions without proportional additions (likely mistake)
    if removed > 20 and added < removed * 0.3:
        return False, f"Too many deletions ({removed}) compared to additions ({added}) - likely mistake"

    # Check 5: Patch touches relevant code (keywords from problem statement)
    keywords = extract_patch_keywords(problem_statement)
    if keywords:
        patch_lower = patch.lower()
        keyword_hits = sum(1 for kw in keywords if kw.lower() in patch_lower)

        # If we have many keywords but none appear in patch, suspicious
        if len(keywords) >= 3 and keyword_hits == 0:
            return False, f"Patch doesn't reference any mentioned symbols: {keywords[:5]}"

    # Check 6: File path exists in diff header
    if '--- a/' not in patch and '--- ' not in patch:
        return False, "Patch missing file header (--- a/path)"

    return True, "OK"


# =============================================================================
#  SYNTAX VALIDATION
# =============================================================================

def validate_syntax(patched_content: str) -> Tuple[bool, str]:
    """
    Validate Python syntax using ast.parse.
    NO execution - just parsing.

    Returns:
        (is_valid, message)
    """
    try:
        ast.parse(patched_content)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Parse error: {str(e)}"


def validate_no_mass_deletion(original: str, patched: str) -> Tuple[bool, str]:
    """
    Check that we didn't accidentally delete large portions of code.

    Returns:
        (is_valid, message)
    """
    orig_lines = len(original.splitlines())
    patch_lines = len(patched.splitlines())
    deleted = orig_lines - patch_lines

    # Allow up to 5 lines or 5% deletion
    if deleted > 5 and deleted > orig_lines * 0.05:
        return False, f"Too many lines deleted ({deleted} of {orig_lines})"

    return True, "OK"


def validate_preserved_structure(original: str, patched: str) -> Tuple[bool, str]:
    """
    Check that important code structures weren't deleted.
    Uses AST comparison - NO execution.

    Returns:
        (is_valid, message)
    """
    try:
        orig_tree = ast.parse(original)
        patch_tree = ast.parse(patched)
    except SyntaxError:
        # If we can't parse, let syntax validation catch it
        return True, "OK (skipped - parse error)"

    # Check functions weren't deleted
    orig_funcs = {n.name for n in ast.walk(orig_tree) if isinstance(n, ast.FunctionDef)}
    patch_funcs = {n.name for n in ast.walk(patch_tree) if isinstance(n, ast.FunctionDef)}
    deleted_funcs = orig_funcs - patch_funcs

    if deleted_funcs:
        return False, f"Deleted functions: {deleted_funcs}"

    # Check classes weren't deleted
    orig_classes = {n.name for n in ast.walk(orig_tree) if isinstance(n, ast.ClassDef)}
    patch_classes = {n.name for n in ast.walk(patch_tree) if isinstance(n, ast.ClassDef)}
    deleted_classes = orig_classes - patch_classes

    if deleted_classes:
        return False, f"Deleted classes: {deleted_classes}"

    return True, "OK"


# =============================================================================
#  LLM SELF-CRITIQUE
# =============================================================================

def llm_self_critique(
    patch: str,
    problem_statement: str,
    llm_generate: Callable[[str, str, float, int], str]
) -> Tuple[bool, str]:
    """
    Ask LLM to review its own patch for obvious issues.
    NO test execution - pure code review.

    Args:
        patch: The generated patch (unified diff format)
        problem_statement: The bug description
        llm_generate: Function to call LLM (prompt, system, temp, max_tokens) -> response

    Returns:
        (approved, feedback)
    """
    prompt = SELF_CRITIQUE_PROMPT.format(
        problem_statement=problem_statement[:2500],
        patch=patch[:4000]
    )

    try:
        response = llm_generate(prompt, SELF_CRITIQUE_SYSTEM, 0.0, 600)

        if response and "APPROVED" in response.upper():
            return True, "LLM approved patch"

        # Extract issues if present
        if "ISSUES" in response.upper():
            return False, response

        # Ambiguous response - default to approved
        return True, "LLM response ambiguous, defaulting to approved"

    except Exception as e:
        # On LLM error, don't block - just warn
        return True, f"LLM critique skipped: {str(e)}"


# =============================================================================
#  COMBINED PASS@1 VALIDATOR
# =============================================================================

def validate_patch_pass1(
    patch: str,
    original_content: str,
    patched_content: str,
    problem_statement: str,
    llm_generate: Optional[Callable[[str, str, float, int], str]] = None,
    skip_llm_critique: bool = False
) -> Tuple[bool, List[str]]:
    """
    Full pass@1 compatible validation pipeline.
    NO test execution - only static analysis and LLM review.

    Args:
        patch: The generated patch (unified diff)
        original_content: Original file content
        patched_content: Content after applying patch
        problem_statement: The bug description
        llm_generate: Optional LLM function for self-critique
        skip_llm_critique: If True, skip the LLM review step

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []

    # 1. Diff quality check (heuristics)
    valid, msg = validate_diff_quality(patch, problem_statement)
    if not valid:
        issues.append(f"Diff quality: {msg}")

    # 2. Syntax validation
    valid, msg = validate_syntax(patched_content)
    if not valid:
        issues.append(f"Syntax: {msg}")
        # If syntax fails, skip further checks that need parsed AST
        return False, issues

    # 3. Mass deletion check
    valid, msg = validate_no_mass_deletion(original_content, patched_content)
    if not valid:
        issues.append(f"Structure: {msg}")

    # 4. Preserved structure check (AST-based)
    valid, msg = validate_preserved_structure(original_content, patched_content)
    if not valid:
        issues.append(f"Structure: {msg}")

    # 5. LLM self-critique (optional, but recommended)
    if llm_generate and not skip_llm_critique and not issues:
        # Only run LLM critique if basic checks passed
        approved, feedback = llm_self_critique(patch, problem_statement, llm_generate)
        if not approved:
            issues.append(f"LLM review: {feedback}")

    return len(issues) == 0, issues


# =============================================================================
#  UTILITY: APPLY PATCH (for validation)
# =============================================================================

def try_apply_patch(original: str, patch: str) -> Tuple[Optional[str], str]:
    """
    Try to apply a unified diff patch to content.
    Returns (patched_content, error_message).

    This is a simple implementation - for complex patches,
    the actual application happens in the pipeline.
    """
    # This is a placeholder - the actual patch application
    # is done in the pipeline code. This is just for standalone testing.
    return None, "Use pipeline's patch application"
