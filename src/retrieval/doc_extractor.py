import ast
import os
from typing import List
from .corpus import Document

class DocExtractor:
    """Extracts documentation from Python source code."""
    
    def extract_from_repo(self, repo_path: str, repo_name: str) -> List[Document]:
        documents = []
        
        for root, _, files in os.walk(repo_path):
            for file in files:
                if not file.endswith('.py'):
                    continue
                    
                file_path = os.path.join(root, file)
                try:
                    rel_path = os.path.relpath(file_path, repo_path)
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        source = f.read()
                        
                    tree = ast.parse(source)
                    
                    # Module docstring
                    module_doc = ast.get_docstring(tree)
                    if module_doc:
                        documents.append(Document(
                            doc_id=f"{repo_name}-{rel_path}-module",
                            title=f"Module {rel_path}",
                            content=f"Module: {rel_path}\n\n{module_doc}",
                            source="source-code",
                            library=repo_name,
                            doc_type="documentation"
                        ))
                        
                    # Classes and Functions
                    for node in tree.body:
                        if isinstance(node, ast.ClassDef):
                            self._process_class(node, rel_path, repo_name, documents)
                        elif isinstance(node, ast.FunctionDef):
                            self._process_function(node, rel_path, repo_name, documents)
                            
                except Exception as e:
                    # print(f"Error processing {file_path}: {e}")
                    pass
                    
        return documents

    def _process_class(self, node: ast.ClassDef, file_path: str, library: str, documents: List[Document]):
        """Processing a class definition."""
        doc = ast.get_docstring(node)
        if doc:
            documents.append(Document(
                doc_id=f"{library}-{file_path}-{node.name}",
                title=f"Class {node.name} ({file_path})",
                content=f"Class: {node.name}\nFile: {file_path}\n\n{doc}",
                source="source-code",
                library=library,
                doc_type="documentation"
            ))
            
        # Methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                self._process_function(item, file_path, library, documents, class_name=node.name)

    def _process_function(self, node: ast.FunctionDef, file_path: str, library: str, documents: List[Document], class_name: str = None):
        """Processing a function definition."""
        doc = ast.get_docstring(node)
        if doc:
            name = f"{class_name}.{node.name}" if class_name else node.name
            documents.append(Document(
                doc_id=f"{library}-{file_path}-{name}",
                title=f"Function {name} ({file_path})",
                content=f"Function: {name}\nFile: {file_path}\nSignature: {self._get_args(node)}\n\n{doc}",
                source="source-code",
                library=library,
                doc_type="documentation"
            ))

    def _get_args(self, node: ast.FunctionDef) -> str:
        """Helper to get function arguments string."""
        # Simplified reconstruction
        args = [a.arg for a in node.args.args]
        return f"({', '.join(args)})"
