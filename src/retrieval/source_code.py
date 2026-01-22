import re
import subprocess
import shutil
from pathlib import Path
from typing import Optional


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
            
            print(f"[OK] Cloned {repo}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to clone {repo}: {e}")
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




