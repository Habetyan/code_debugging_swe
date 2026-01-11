import ast
import os
from pathlib import Path
from typing import Set, List

class CodeGraph:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)

    def get_related_files(self, start_file: str, max_depth: int = 1) -> List[str]:
        """
        Find files related to the start_file via imports.
        """
        visited = set()
        queue = [(start_file, 0)]
        related_files = []

        while queue:
            current_file, depth = queue.pop(0)
            if current_file in visited:
                continue
            visited.add(current_file)
            
            if current_file != start_file:
                related_files.append(current_file)

            if depth >= max_depth:
                continue

            # Parse file to find imports
            try:
                full_path = self.repo_path / current_file
                if not full_path.exists():
                    continue
                    
                content = full_path.read_text()
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        imported_files = self._resolve_import(node, current_file)
                        for f in imported_files:
                            if f not in visited:
                                queue.append((f, depth + 1))
            except Exception as e:
                print(f"Error parsing {current_file}: {e}")
                continue
                
        return related_files

    def _resolve_import(self, node: ast.AST, current_file: str) -> Set[str]:
        """
        Resolve an AST import node to potential file paths.
        """
        resolved = set()
        current_dir = (self.repo_path / current_file).parent
        
        if isinstance(node, ast.ImportFrom):
            # from . import module
            # from .module import submodule
            if node.level > 0:
                # Relative import
                # level 1 = ., level 2 = ..
                base_dir = current_dir
                for _ in range(node.level - 1):
                    base_dir = base_dir.parent
                
                if node.module:
                    module_path = node.module.replace('.', '/')
                    # Check for .py file
                    candidate = base_dir / f"{module_path}.py"
                    if candidate.exists():
                        resolved.add(str(candidate.relative_to(self.repo_path)))
                    # Check for package dir
                    candidate_dir = base_dir / module_path / "__init__.py"
                    if candidate_dir.exists():
                        resolved.add(str(candidate_dir.relative_to(self.repo_path)))
            else:
                # Absolute import (within the repo)
                # e.g. sklearn.utils
                if node.module:
                    parts = node.module.split('.')
                    # Try to match top-level directories in repo
                    # This is heuristic; assumes repo root is python path root or similar
                    candidate = self.repo_path.joinpath(*parts).with_suffix('.py')
                    if candidate.exists():
                        resolved.add(str(candidate.relative_to(self.repo_path)))
                    
                    candidate_dir = self.repo_path.joinpath(*parts) / "__init__.py"
                    if candidate_dir.exists():
                        resolved.add(str(candidate_dir.relative_to(self.repo_path)))

        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split('.')
                candidate = self.repo_path.joinpath(*parts).with_suffix('.py')
                if candidate.exists():
                    resolved.add(str(candidate.relative_to(self.repo_path)))
                
                candidate_dir = self.repo_path.joinpath(*parts) / "__init__.py"
                if candidate_dir.exists():
                    resolved.add(str(candidate_dir.relative_to(self.repo_path)))
                    
        return resolved
