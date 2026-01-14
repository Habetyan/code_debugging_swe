"""
Code Graph: analysis of code structure.
Provides tools to navigate call graphs and file relationships.
"""
import ast
import subprocess
from pathlib import Path
from typing import Set, List, Dict

class CodeGraph:
    """
    Analyzes Python code repositories to build a graph of file relationships
    (via imports) and function call graphs.
    """
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
                # print(f"Error parsing {current_file}: {e}")
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
                base_dir = current_dir
                for _ in range(node.level - 1):
                    base_dir = base_dir.parent
                
                if node.module:
                    module_path = node.module.replace('.', '/')
                    # Check for .py file
                    candidate = base_dir / f"{module_path}.py"
                    if candidate.exists():
                        try:
                            resolved.add(str(candidate.relative_to(self.repo_path)))
                        except ValueError: pass
                    # Check for package dir
                    candidate_dir = base_dir / module_path / "__init__.py"
                    if candidate_dir.exists():
                        try:
                            resolved.add(str(candidate_dir.relative_to(self.repo_path)))
                        except ValueError: pass
            else:
                # Absolute import
                if node.module:
                    parts = node.module.split('.')
                    # Try to match top-level directories in repo
                    candidate = self.repo_path.joinpath(*parts).with_suffix('.py')
                    if candidate.exists():
                        try:
                            resolved.add(str(candidate.relative_to(self.repo_path)))
                        except ValueError: pass
                    
                    candidate_dir = self.repo_path.joinpath(*parts) / "__init__.py"
                    if candidate_dir.exists():
                        try:
                            resolved.add(str(candidate_dir.relative_to(self.repo_path)))
                        except ValueError: pass

        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split('.')
                candidate = self.repo_path.joinpath(*parts).with_suffix('.py')
                if candidate.exists():
                    try:
                        resolved.add(str(candidate.relative_to(self.repo_path)))
                    except ValueError: pass
                
                candidate_dir = self.repo_path.joinpath(*parts) / "__init__.py"
                if candidate_dir.exists():
                    try:
                        resolved.add(str(candidate_dir.relative_to(self.repo_path)))
                    except ValueError: pass
                    
        return resolved

    def get_callees(self, file_path: str, func_name: str) -> Set[str]:
        """Find functions called by func_name in the given file."""
        callees = set()
        try:
            full_path = self.repo_path / file_path
            if not full_path.exists():
                return callees
            
            content = full_path.read_text(errors='ignore')
            tree = ast.parse(content)
            
            # Find the function definition
            target_node = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    target_node = node
                    break
            
            if target_node:
                # Walk the function body
                for node in ast.walk(target_node):
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            callees.add(node.func.id)
                        elif isinstance(node.func, ast.Attribute):
                            callees.add(node.func.attr)
        except Exception:
            pass
        return callees

    def get_callers(self, func_name: str) -> List[Dict]:
        """
        Find functions that call func_name.
        Returns list of {"file": str, "line": int, "caller": str}
        """
        callers = []
        
        # 1. Grep for potential usage (fast filter)
        try:
            grep_cmd = ["grep", "-r", "-l", func_name, "."]
            result = subprocess.run(
                grep_cmd, 
                cwd=self.repo_path, 
                capture_output=True, 
                text=True,
                timeout=10
            )
            candidate_files = [f for f in result.stdout.splitlines() if f.endswith(".py")]
        except Exception:
            return []
            
        # 2. Parse candidates to confirm usage
        for rel_path in candidate_files:
            if rel_path.startswith("./"):
                rel_path = rel_path[2:]
            
            try:
                full_path = self.repo_path / rel_path
                content = full_path.read_text(errors='ignore')
                tree = ast.parse(content)

                current_function = "global"
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Enter function
                        current_function = node.name
                        # Check calls inside this function
                        for child in ast.walk(node):
                            if isinstance(child, ast.Call):
                                is_call = False
                                if isinstance(child.func, ast.Name) and child.func.id == func_name:
                                    is_call = True
                                elif isinstance(child.func, ast.Attribute) and child.func.attr == func_name:
                                    is_call = True
                                
                                if is_call:
                                    callers.append({
                                        "file": rel_path,
                                        "line": child.lineno,
                                        "caller": current_function
                                    })
                        
                        # Reset to global (simplification - doesn't handle nested functions perfectly)
                        current_function = "global"
                        
            except Exception:
                continue
                
        return callers
