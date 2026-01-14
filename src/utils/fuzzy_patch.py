"""
Fuzzy Patching Utility: Applies patches with fuzzy matching.
Allows applying patches even when line numbers shifted or context slightly changed.
"""
import re
import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple

@dataclass
class HunkLine:
    type: str  # ' ', '-', '+'
    content: str

@dataclass
class PatchHunk:
    file_path: str
    lines: List[HunkLine]
    
    @property
    def source_block(self) -> str:
        """The specific block of code expected to be in the original file."""
        return '\n'.join(line.content for line in self.lines if line.type in (' ', '-'))
    
    @property
    def target_block(self) -> str:
        """The block of code that should replace the source block."""
        return '\n'.join(line.content for line in self.lines if line.type in (' ', '+'))


def parse_unified_diff(patch: str) -> List[PatchHunk]:
    """Parse a unified diff into hunks, preserving line order."""
    hunks = []
    lines = patch.split('\n')
    current_file = None
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        if line.startswith('--- a/'):
            current_file = line[6:]
            i += 1
            continue
        if line.startswith('+++ b/'):
            i += 1
            continue
            
        if line.startswith('@@'):
            hunk_lines = []
            i += 1
            while i < len(lines):
                line = lines[i]
                if line.startswith('@@') or line.startswith('---'):
                    break
                
                if line.startswith(' ') or line.startswith('-') or line.startswith('+'):
                    hunk_lines.append(HunkLine(line[0], line[1:]))
                elif not line.strip():
                    # Empty line in patch, usually context
                    hunk_lines.append(HunkLine(' ', ''))
                else:
                    # Look ahead/break on unknown formats
                    break
                i += 1
                
            if hunk_lines:
                hunks.append(PatchHunk(
                    file_path=current_file or 'unknown.py',
                    lines=hunk_lines
                ))
            continue
            
        i += 1
        
    return hunks


def fuzzy_find(content_lines: List[str], search_lines: List[str], threshold: float = 0.8) -> Tuple[Optional[int], float]:
    """Find the best matching location for search_lines in content_lines."""
    if not search_lines:
        return None, 1.0
        
    best_ratio = 0.0
    best_idx = None
    
    # Simple optimization: content usually contains explicit unique logic
    search_text = '\n'.join(search_lines)
    n_search = len(search_lines)
    
    for i in range(len(content_lines) - n_search + 1):
        window = '\n'.join(content_lines[i : i + n_search])
        ratio = difflib.SequenceMatcher(None, search_text, window).ratio()
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i
            
            if ratio > 0.99: # Perfect match optimization
                break
                
    return best_idx, best_ratio


def apply_hunk_fuzzy(content: str, hunk: PatchHunk, threshold: float = 0.85) -> Tuple[str, bool]:
    """Apply a single hunk to content using fuzzy matching."""
    content_lines = content.split('\n')
    
    # 1. Build source block (lines to look for)
    source_lines = [line.content for line in hunk.lines if line.type in (' ', '-')]
    
    # 2. Build target block (lines to replace with)
    target_lines = [line.content for line in hunk.lines if line.type in (' ', '+')]
    
    # 3. Handle specific case: Addition only (no context)
    if not source_lines:
        # Append to end if no context (rare for valid unified diffs)
        return content + '\n' + '\n'.join(target_lines), True

    # 4. Find location
    match_idx, ratio = fuzzy_find(content_lines, source_lines, threshold)
    
    if match_idx is not None and ratio >= threshold:
        # 5. Apply replacement
        new_lines = (
            content_lines[:match_idx] +
            target_lines +
            content_lines[match_idx + len(source_lines):]
        )
        return '\n'.join(new_lines), True
        
    return content, False


def resolve_file_path(repo_path: Path, file_path_str: str) -> Optional[Path]:
    """Resolve file path with fuzzy fallback."""
    repo = Path(repo_path)
    fpath = repo / file_path_str
    
    if fpath.exists():
        return fpath
        
    # Basename match
    basename = Path(file_path_str).name
    matches = list(repo.rglob(f"**/{basename}"))
    if matches:
        return matches[0]
        
    return None


def apply_patch_fuzzy(patch: str, repo_path: str, threshold: float = 0.85) -> Tuple[bool, str, List[str]]:
    """Apply a patch to a repository."""
    hunks = parse_unified_diff(patch)
    if not hunks:
        return False, "No hunks parsed", []
        
    files_modified = []
    messages = []
    all_success = True
    
    # Group hunks by file to apply sequentially
    from collections import defaultdict
    hunks_by_file = defaultdict(list)
    for hunk in hunks:
        hunks_by_file[hunk.file_path].append(hunk)
        
    for file_path_str, file_hunks in hunks_by_file.items():
        file_path = resolve_file_path(Path(repo_path), file_path_str)
        
        if not file_path:
            messages.append(f"File not found: {file_path_str}")
            all_success = False
            continue
            
        content = file_path.read_text()
        original_content = content
        file_success = True
        
        for hunk in file_hunks:
            content, success = apply_hunk_fuzzy(content, hunk, threshold)
            if not success:
                file_success = False
                messages.append(f"Failed hunk in {file_path_str}")
                break
        
        if file_success and content != original_content:
            file_path.write_text(content)
            files_modified.append(str(file_path.relative_to(repo_path)))
        elif not file_success:
            all_success = False
            
    return all_success, '\n'.join(messages), files_modified
