"""
Pass@1 Compatible Patch Validation

This module provides static validation checks for generated patches that
do NOT require test execution, maintaining pass@1 compliance.

Checks included:
1. Diff quality (size, deletions, keyword relevance)
2. Syntax validation (ast.parse)
3. Balanced brackets/parentheses/braces
4. Mass deletion detection
5. Preserved code structure (functions/classes not deleted)
6. Debug code detection (print, pdb, breakpoint)
7. Defensive coding patterns (getattr with None, bare except)
8. Indentation consistency (no mixed tabs/spaces)
9. Import preservation
10. Control flow removal (return/continue/break/raise)
11. Parameter removal detection
12. LLM self-critique (code review, no execution)

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
#  ADDITIONAL HEURISTIC CHECKS
# =============================================================================

def check_debug_code_added(patch: str) -> Tuple[bool, str]:
    """
    Check if patch adds debug/print statements that shouldn't be in production.
    Returns (is_ok, message).
    """
    # Get added lines only
    added_lines = [l[1:] for l in patch.split('\n') if l.startswith('+') and not l.startswith('+++')]

    debug_patterns = [
        (r'\bprint\s*\(', 'print statement'),
        (r'\bpdb\.set_trace\s*\(', 'pdb debugger'),
        (r'\bbreakpoint\s*\(', 'breakpoint'),
        (r'\blogging\.debug\s*\(', 'debug logging'),
        (r'#\s*DEBUG', 'DEBUG comment'),
        (r'#\s*TODO', 'TODO comment'),
        (r'#\s*FIXME', 'FIXME comment'),
        (r'#\s*HACK', 'HACK comment'),
    ]

    for line in added_lines:
        for pattern, name in debug_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return False, f"Adds {name}: {line.strip()[:50]}"

    return True, "OK"


def check_defensive_coding(patch: str) -> Tuple[bool, str]:
    """
    Check for excessive defensive coding that might hide bugs.
    Returns (is_ok, message).
    """
    added_lines = [l[1:] for l in patch.split('\n') if l.startswith('+') and not l.startswith('+++')]

    suspicious_patterns = [
        # getattr with None fallback - often hides the real issue
        (r'getattr\s*\([^,]+,\s*[^,]+,\s*None\s*\)', 'getattr(..., None) - may hide real issue'),
        # Broad exception catching
        (r'except\s*:\s*$', 'bare except - catches too much'),
        # Silent pass in except
        (r'except.*:\s*\n\s*pass', 'silent exception handling'),
    ]

    suspicious_count = 0
    for line in added_lines:
        for pattern, name in suspicious_patterns:
            if re.search(pattern, line):
                suspicious_count += 1
                if suspicious_count >= 2:
                    return False, f"Multiple defensive patterns detected: {name}"

    return True, "OK"


def check_indentation_consistency(patched_content: str) -> Tuple[bool, str]:
    """
    Check for mixed tabs/spaces and inconsistent indentation.
    Returns (is_ok, message).
    """
    lines = patched_content.split('\n')
    has_tabs = False
    has_spaces = False

    for line in lines:
        if line.startswith('\t'):
            has_tabs = True
        elif line.startswith('    '):
            has_spaces = True

    if has_tabs and has_spaces:
        return False, "Mixed tabs and spaces for indentation"

    return True, "OK"


def check_import_issues(original: str, patched: str) -> Tuple[bool, str]:
    """
    Check that we didn't accidentally break imports.
    Returns (is_ok, message).
    """
    try:
        orig_tree = ast.parse(original)
        patch_tree = ast.parse(patched)
    except SyntaxError:
        return True, "OK (skipped - parse error)"

    # Get original imports
    orig_imports = set()
    for node in ast.walk(orig_tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                orig_imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                orig_imports.add(node.module)

    # Get patched imports
    patch_imports = set()
    for node in ast.walk(patch_tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                patch_imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                patch_imports.add(node.module)

    # Check if we removed important imports
    removed = orig_imports - patch_imports
    if removed:
        # Filter out common false positives
        important_removed = [i for i in removed if not i.startswith('_')]
        if important_removed:
            return False, f"Removed imports: {important_removed}"

    return True, "OK"


def check_balanced_brackets(patched_content: str) -> Tuple[bool, str]:
    """
    Quick check for balanced parentheses, brackets, braces.
    Returns (is_ok, message).
    """
    # Count brackets (simple heuristic, not perfect due to strings)
    open_parens = patched_content.count('(')
    close_parens = patched_content.count(')')
    open_brackets = patched_content.count('[')
    close_brackets = patched_content.count(']')
    open_braces = patched_content.count('{')
    close_braces = patched_content.count('}')

    if open_parens != close_parens:
        diff = open_parens - close_parens
        return False, f"Unbalanced parentheses ({diff:+d})"

    if open_brackets != close_brackets:
        diff = open_brackets - close_brackets
        return False, f"Unbalanced brackets ({diff:+d})"

    if open_braces != close_braces:
        diff = open_braces - close_braces
        return False, f"Unbalanced braces ({diff:+d})"

    return True, "OK"


def check_string_literals(patch: str) -> Tuple[bool, str]:
    """
    Check for common string issues in added code.
    Returns (is_ok, message).
    """
    added_lines = [l[1:] for l in patch.split('\n') if l.startswith('+') and not l.startswith('+++')]

    for line in added_lines:
        # Check for unmatched quotes (simple heuristic)
        single_quotes = line.count("'") - line.count("\\'") - line.count("'''") * 3
        double_quotes = line.count('"') - line.count('\\"') - line.count('"""') * 3

        # Very rough check - odd number of quotes might indicate issue
        # Skip if line contains triple quotes (docstrings)
        if "'''" not in line and '"""' not in line:
            if single_quotes % 2 != 0 and double_quotes % 2 != 0:
                return False, f"Possibly unmatched quotes: {line.strip()[:50]}"

    return True, "OK"


def check_control_flow_removal(patch: str) -> Tuple[bool, str]:
    """
    Check if patch removes control flow statements (return, continue, break, raise).
    These are usually critical and shouldn't be removed without replacement.
    Returns (is_ok, message).
    """
    removed_lines = [l[1:].strip() for l in patch.split('\n')
                     if l.startswith('-') and not l.startswith('---')]
    added_lines = [l[1:].strip() for l in patch.split('\n')
                   if l.startswith('+') and not l.startswith('+++')]

    # Keywords that are critical control flow
    control_keywords = ['return', 'continue', 'break', 'raise']

    for keyword in control_keywords:
        # Count occurrences in removed vs added
        removed_count = sum(1 for line in removed_lines
                           if re.search(rf'\b{keyword}\b', line))
        added_count = sum(1 for line in added_lines
                         if re.search(rf'\b{keyword}\b', line))

        # If we removed more than we added, that's suspicious
        if removed_count > added_count:
            return False, f"Removed {removed_count - added_count} '{keyword}' statement(s) without replacement"

    return True, "OK"


def check_parameter_removal(original: str, patched: str) -> Tuple[bool, str]:
    """
    Check if function parameters were removed.
    Returns (is_ok, message).
    """
    try:
        orig_tree = ast.parse(original)
        patch_tree = ast.parse(patched)
    except SyntaxError:
        return True, "OK (skipped - parse error)"

    # Get function signatures
    def get_func_params(tree):
        funcs = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                params = [arg.arg for arg in node.args.args]
                funcs[node.name] = set(params)
        return funcs

    orig_funcs = get_func_params(orig_tree)
    patch_funcs = get_func_params(patch_tree)

    # Check each function that exists in both
    for func_name in orig_funcs:
        if func_name in patch_funcs:
            removed_params = orig_funcs[func_name] - patch_funcs[func_name]
            if removed_params:
                return False, f"Function '{func_name}' lost parameters: {removed_params}"

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
        problem_statement=problem_statement[:5000],
        patch=patch[:8000]
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

    # 2. Syntax validation (CRITICAL - fail fast)
    valid, msg = validate_syntax(patched_content)
    if not valid:
        issues.append(f"Syntax: {msg}")
        # If syntax fails, skip further checks that need parsed AST
        return False, issues

    # 3. Balanced brackets check (before AST checks)
    valid, msg = check_balanced_brackets(patched_content)
    if not valid:
        issues.append(f"Brackets: {msg}")

    # 4. Mass deletion check
    valid, msg = validate_no_mass_deletion(original_content, patched_content)
    if not valid:
        issues.append(f"Structure: {msg}")

    # 5. Preserved structure check (AST-based)
    valid, msg = validate_preserved_structure(original_content, patched_content)
    if not valid:
        issues.append(f"Structure: {msg}")

    # 6. Check for debug code added
    valid, msg = check_debug_code_added(patch)
    if not valid:
        issues.append(f"Debug code: {msg}")

    # 7. Check for defensive coding patterns
    valid, msg = check_defensive_coding(patch)
    if not valid:
        issues.append(f"Defensive coding: {msg}")

    # 8. Check indentation consistency
    valid, msg = check_indentation_consistency(patched_content)
    if not valid:
        issues.append(f"Indentation: {msg}")

    # 9. Check import issues
    valid, msg = check_import_issues(original_content, patched_content)
    if not valid:
        issues.append(f"Imports: {msg}")

    # 10. Check control flow removal (return/continue/break/raise)
    valid, msg = check_control_flow_removal(patch)
    if not valid:
        issues.append(f"Control flow: {msg}")

    # 11. Check parameter removal
    valid, msg = check_parameter_removal(original_content, patched_content)
    if not valid:
        issues.append(f"Parameters: {msg}")

    # 13. LLM self-critique (optional, but recommended)
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
