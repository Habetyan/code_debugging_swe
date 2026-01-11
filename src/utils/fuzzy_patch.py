"""
Fuzzy Patch Application

Applies patches even when context lines don't match exactly.
Uses fuzzy string matching to find the target location.
"""

import re
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PatchHunk:
    """A single hunk from a unified diff."""
    file_path: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    context_lines: list[str]  # Lines starting with ' '
    removed_lines: list[str]  # Lines starting with '-'
    added_lines: list[str]    # Lines starting with '+'


def parse_unified_diff(patch: str) -> list[PatchHunk]:
    """Parse a unified diff into hunks."""
    hunks = []
    lines = patch.split('\n')
    
    current_file = None
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # File header
        if line.startswith('--- a/'):
            current_file = line[6:]
            i += 1
            continue
        
        if line.startswith('+++ b/'):
            i += 1
            continue
        
        # Hunk header
        if line.startswith('@@'):
            # Parse @@ -old_start,old_count +new_start,new_count @@
            match = re.match(r'@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@', line)
            if match:
                old_start = int(match.group(1))
                old_count = int(match.group(2) or 1)
                new_start = int(match.group(3))
                new_count = int(match.group(4) or 1)
                
                # Collect hunk lines
                context = []
                removed = []
                added = []
                
                i += 1
                while i < len(lines) and lines[i] and not lines[i].startswith('@@') and not lines[i].startswith('---'):
                    hline = lines[i]
                    if hline.startswith(' '):
                        context.append(hline[1:])
                    elif hline.startswith('-'):
                        removed.append(hline[1:])
                    elif hline.startswith('+'):
                        added.append(hline[1:])
                    i += 1
                
                hunks.append(PatchHunk(
                    file_path=current_file or 'unknown.py',
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    context_lines=context,
                    removed_lines=removed,
                    added_lines=added,
                ))
                continue
        
        i += 1
    
    return hunks


def fuzzy_find_location(content: str, search_lines: list[str], threshold: float = 0.8) -> Optional[int]:
    """
    Find the location of search_lines in content using fuzzy matching.
    Returns the character offset or None if not found.
    """
    content_lines = content.split('\n')
    search_text = '\n'.join(search_lines)
    
    best_ratio = 0
    best_start = None
    
    # Slide window through file
    for i in range(len(content_lines) - len(search_lines) + 1):
        window = '\n'.join(content_lines[i:i + len(search_lines)])
        ratio = difflib.SequenceMatcher(None, search_text, window).ratio()
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i
    
    if best_ratio >= threshold:
        # Convert line index to character offset
        offset = sum(len(line) + 1 for line in content_lines[:best_start])
        return offset
    
    return None


def apply_hunk_fuzzy(content: str, hunk: PatchHunk, threshold: float = 0.7) -> tuple[str, bool]:
    """
    Apply a single hunk to content using fuzzy matching.
    Returns (new_content, success).
    """
    content_lines = content.split('\n')
    
    # Build search pattern from removed lines or context
    search_lines = hunk.removed_lines if hunk.removed_lines else hunk.context_lines[:3]
    
    # Handle addition-only hunks (no removed lines, no context)
    if not search_lines:
        # Use line number from hunk header for insertion
        insert_line = max(0, hunk.old_start - 1)
        if insert_line <= len(content_lines):
            new_lines = (
                content_lines[:insert_line] +
                hunk.added_lines +
                content_lines[insert_line:]
            )
            return '\n'.join(new_lines), True
        return content, False
    
    search_text = '\n'.join(search_lines)
    
    # Find best match
    best_ratio = 0
    best_start = None
    
    for i in range(len(content_lines)):
        window_end = min(i + len(search_lines), len(content_lines))
        window = content_lines[i:window_end]
        window_text = '\n'.join(window)
        
        ratio = difflib.SequenceMatcher(None, search_text, window_text).ratio()
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i
    
    if best_ratio < threshold or best_start is None:
        return content, False
    
    # Apply the change
    new_lines = (
        content_lines[:best_start] +
        hunk.added_lines +
        content_lines[best_start + len(search_lines):]
    )
    
    return '\n'.join(new_lines), True


def resolve_file_path(repo_path: Path, file_path_str: str) -> Optional[Path]:
    """
    Resolve a file path from a patch, handling potential hallucinations.
    Strategies:
    1. Exact match
    2. Basename match (e.g. label.py -> sklearn/preprocessing/label.py)
    3. Fuzzy name match (e.g. label.py -> _label.py)
    """
    repo = Path(repo_path)
    fpath = repo / file_path_str
    
    # 1. Exact match
    if fpath.exists():
        return fpath
        
    # 2. Basename match
    basename = Path(file_path_str).name
    matches = list(repo.rglob(f"**/{basename}"))
    if matches:
        # Prefer shortest path or one that matches parent dirs
        return matches[0]
        
    # 3. Fuzzy name match (Levenshtein on basename)
    # Get all python files
    all_files = list(repo.rglob("*.py"))
    best_match = None
    best_ratio = 0.0
    
    for cand in all_files:
        ratio = difflib.SequenceMatcher(None, basename, cand.name).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = cand
            
    if best_ratio > 0.8: # High confidence threshold
        return best_match
        
    return None


def apply_patch_fuzzy(patch: str, repo_path: str, threshold: float = 0.7) -> tuple[bool, str, list[str]]:
    """
    Apply a patch to a repository using fuzzy matching.
    
    Returns:
        (success, message, files_modified)
    """
    hunks = parse_unified_diff(patch)
    
    if not hunks:
        return False, "No hunks found in patch", []
    
    files_modified = []
    messages = []
    
    for hunk in hunks:
        file_path = resolve_file_path(Path(repo_path), hunk.file_path)
        
        if not file_path:
            messages.append(f"File not found (and no fuzzy match): {hunk.file_path}")
            continue
            
        content = file_path.read_text()
        new_content, success = apply_hunk_fuzzy(content, hunk, threshold)
        
        if success:
            file_path.write_text(new_content)
            files_modified.append(str(file_path.relative_to(repo_path)))
            messages.append(f"Applied hunk to {file_path.relative_to(repo_path)} (orig: {hunk.file_path})")
        else:
            messages.append(f"Failed to apply hunk to {hunk.file_path}")
    
    all_success = len(files_modified) > 0
    return all_success, '\n'.join(messages), files_modified


def test_patch_fuzzy(patch: str, repo_path: str, threshold: float = 0.7) -> tuple[bool, str]:
    """
    Test if a patch can be applied using fuzzy matching (without modifying files).
    """
    hunks = parse_unified_diff(patch)
    
    if not hunks:
        return False, "No hunks found"
    
    for hunk in hunks:
        file_path = resolve_file_path(Path(repo_path), hunk.file_path)
        
        if not file_path:
            return False, f"File not found: {hunk.file_path}"
        
        content = file_path.read_text()
        _, success = apply_hunk_fuzzy(content, hunk, threshold)
        
        if not success:
            return False, f"Cannot match hunk in {file_path.name}"
    
    return True, "All hunks can be applied"
