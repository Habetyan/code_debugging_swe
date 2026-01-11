import re
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional
import os


class RepoManager:
    """
    Manages repository cloning and source file extraction.
    Uses a cache directory to avoid re-cloning.
    """
    
    def __init__(self, cache_dir: str = "repo_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_repo_path(self, repo: str, commit: str) -> Optional[Path]:
        """
        Get path to a cloned repo at a specific commit.
        Clones if not cached.
        """
        # Create safe directory name
        safe_name = repo.replace("/", "__") + f"__{commit[:8]}"
        repo_path = self.cache_dir / safe_name
        
        if repo_path.exists():
            return repo_path
        
        # Clone the repo
        success = self._clone_repo(repo, commit, repo_path)
        if success:
            return repo_path
        return None
    
    def _clone_repo(self, repo: str, commit: str, dest: Path) -> bool:
        """Clone a repository and checkout specific commit."""
        try:
            print(f"Cloning {repo} at {commit[:8]}...")
            
            # Clone with minimal history
            subprocess.run(
                ["git", "clone", "--depth", "50", 
                 f"https://github.com/{repo}.git", str(dest)],
                check=True,
                capture_output=True,
                timeout=180,
            )
            
            # Try to checkout the specific commit
            result = subprocess.run(
                ["git", "checkout", commit],
                cwd=str(dest),
                capture_output=True,
            )
            
            if result.returncode != 0:
                # Commit not in shallow clone, fetch it
                subprocess.run(
                    ["git", "fetch", "--depth", "100", "origin", commit],
                    cwd=str(dest),
                    capture_output=True,
                )
                subprocess.run(
                    ["git", "checkout", commit],
                    cwd=str(dest),
                    capture_output=True,
                )
            
            print(f"✓ Cloned {repo}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to clone {repo}: {e}")
            if dest.exists():
                shutil.rmtree(dest)
            return False
    
    def get_file_content(self, repo_path: Path, file_path: str) -> Optional[str]:
        """Read content of a file from the repository."""
        full_path = repo_path / file_path
        
        if not full_path.exists():
            return None
        
        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except Exception:
            return None
    
    def cleanup(self, repo: str, commit: str):
        """Remove a cached repository."""
        safe_name = repo.replace("/", "__") + f"__{commit[:8]}"
        repo_path = self.cache_dir / safe_name
        if repo_path.exists():
            shutil.rmtree(repo_path)


def extract_file_paths(text: str) -> list[str]:
    """
    Extract potential file paths from problem statement or patch.
    
    Looks for patterns like:
    - path/to/file.py
    - `path/to/file.py`
    - in file.py
    """
    paths = set()
    
    # Pattern 1: Paths in backticks
    backtick_pattern = r'`([a-zA-Z0-9_/\-\.]+\.py)`'
    paths.update(re.findall(backtick_pattern, text))
    
    # Pattern 2: Explicit file paths
    path_pattern = r'(?:^|\s)([a-zA-Z0-9_]+(?:/[a-zA-Z0-9_]+)*\.py)(?:\s|$|:|\))'
    paths.update(re.findall(path_pattern, text, re.MULTILINE))
    
    # Pattern 3: "in file.py" or "file file.py"
    in_file_pattern = r'(?:in|file|from|at)\s+([a-zA-Z0-9_/\-\.]+\.py)'
    paths.update(re.findall(in_file_pattern, text, re.IGNORECASE))
    
    # Pattern 4: Module paths (like sympy.physics.units)
    module_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)'
    for match in re.findall(module_pattern, text):
        # Convert module path to file path
        if not match.endswith('.py'):
            path = match.replace('.', '/') + '.py'
            paths.add(path)
    
    return list(paths)


def find_file_in_repo(repo_path: Path, filename: str) -> Optional[str]:
    """Find a file in the repository by name or partial path."""
    # Try exact path first
    if (repo_path / filename).exists():
        return filename
    
    # Search for file
    basename = Path(filename).name
    for path in repo_path.rglob(f"**/{basename}"):
        if path.is_file():
            return str(path.relative_to(repo_path))
    
    # Try with common prefixes
    prefixes = ["src/", "lib/", ""]
    for prefix in prefixes:
        test_path = repo_path / prefix / filename
        if test_path.exists():
            return prefix + filename
    
    return None


def get_source_context(
    repo: str,
    commit: str,
    problem_statement: str,
    patch: str = "",
    max_chars: int = 8000,
    repo_manager: Optional[RepoManager] = None
) -> tuple[str, list[str]]:
    """
    Get source code context for a bug instance.
    
    Returns:
        (source_context, list_of_files_included)
    """
    if repo_manager is None:
        repo_manager = RepoManager()
    
    # Clone/get repo
    repo_path = repo_manager.get_repo_path(repo, commit)
    if repo_path is None:
        return "", []
    
    # Extract potential file paths
    all_text = problem_statement + "\n" + patch
    file_paths = extract_file_paths(all_text)
    
    # Find and read files
    context_parts = []
    files_included = []
    total_chars = 0
    
    for path in file_paths:
        # Try to find the file
        actual_path = find_file_in_repo(repo_path, path)
        if actual_path is None:
            continue
        
        content = repo_manager.get_file_content(repo_path, actual_path)
        if content is None:
            continue
        
        # Truncate if needed
        if len(content) > max_chars // 2:
            content = content[:max_chars // 2] + "\n# ... (truncated)"
        
        if total_chars + len(content) > max_chars:
            break
        
        context_parts.append(f"## File: {actual_path}\n```python\n{content}\n```")
        files_included.append(actual_path)
        total_chars += len(content)
    
    return "\n\n".join(context_parts), files_included
