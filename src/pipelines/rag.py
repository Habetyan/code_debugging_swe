from typing import Optional
from pathlib import Path
import subprocess
import re

from src.data import SWEBenchInstance
from src.llm import LLMProvider
from src.retrieval import HybridRetriever
from src.retrieval.source_code import RepoManager, extract_file_paths, get_source_context
from src.retrieval.graph import CodeGraph
from src.retrieval.corpus import create_sample_corpus
from .baseline import BaselinePipeline, PipelineResult


RAG_SYSTEM_PROMPT = """You are an expert software engineer fixing bugs in Python code.
You have access to:
1. The buggy code
2. Related code files (dependencies)
3. Relevant documentation

Your task is to analyze the code and dependencies to fix the bug described.

OUTPUT FORMAT:
Please provide a Unified Diff (git diff) to fix the issue.
Start with:
```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ ... @@
```
"""

RAG_PROMPT_TEMPLATE = """# Bug Report
{problem_statement}

# Documentation Context
{doc_context}

# Source Code Context

## Primary File: {primary_file}
```python
{primary_content}
```

## Related Files (Dependencies)
{related_content}

# Instructions
Fix the bug in '{primary_file}'. 
Use the related files and documentation to understand the context.
Output your fix as a Unified Diff.
"""


class RAGPipeline(BaselinePipeline):
    """
    Advanced RAG Pipeline with:
    1. Source Code Retrieval (Cloning & Finding Files)
    2. Symbol Search Fallback (Grep)
    3. Code Graph Expansion (Imports/Dependencies)
    4. Documentation Retrieval
    """
    
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        repo_manager: Optional[RepoManager] = None,
        retriever: Optional[HybridRetriever] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        include_docs: bool = True,
    ):
        super().__init__(llm_provider, temperature, max_tokens)
        
        self.repo_manager = repo_manager or RepoManager()
        self.include_docs = include_docs
        
        if include_docs:
            if retriever is None:
                corpus = create_sample_corpus()
                self.retriever = HybridRetriever(corpus)
            else:
                self.retriever = retriever
        else:
            self.retriever = None

    def run(self, instance: SWEBenchInstance) -> PipelineResult:
        # 1. Clone Repo
        repo_path = self.repo_manager.get_repo_path(instance.repo, instance.base_commit)
        
        # 2. Identify Primary File
        # Try to find file paths in the problem statement
        file_paths = extract_file_paths(instance.problem_statement)
        
        primary_file = None
        if file_paths:
            # Check which exists
            for fp in file_paths:
                if (Path(repo_path) / fp).exists():
                    primary_file = fp
                    break
        
        if not primary_file:
            # Fallback: Search for class names in problem statement
            # Look for CamelCase words that might be classes
            potential_classes = re.findall(r'\\b[A-Z][a-zA-Z0-9]+\\b', instance.problem_statement)
            
            for cls_name in potential_classes:
                if len(cls_name) < 4: continue # Skip short ones
                
                # Grep for "class ClsName"
                try:
                    result = subprocess.run(
                        ["grep", "-r", "-l", f"class {cls_name}", "."],
                        cwd=str(repo_path),
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0 and result.stdout:
                        # Pick first match that is a python file
                        for line in result.stdout.splitlines():
                            if line.endswith('.py') and 'test' not in line:
                                primary_file = line
                                break
                        if primary_file:
                            break
                except Exception:
                    pass

        primary_content = ""
        related_content = ""
        
        if primary_file:
            try:
                primary_content = (Path(repo_path) / primary_file).read_text()
            except Exception:
                primary_content = "# Error reading file"
            
            # 3. Graph Expansion
            try:
                graph = CodeGraph(repo_path)
                related_files = graph.get_related_files(primary_file, max_depth=1)
                
                for rf in related_files[:3]: # Limit to top 3 related files
                    try:
                        content = (Path(repo_path) / rf).read_text()
                        # Truncate if too long
                        if len(content) > 2000:
                            content = content[:2000] + "\\n... (truncated)"
                        related_content += f"\\n### {rf}\\n```python\\n{content}\\n```\\n"
                    except:
                        pass
            except Exception:
                pass # Graph analysis failed, proceed without it
        else:
            primary_content = "# Could not identify primary file from problem statement"

        # 4. Retrieve Docs
        doc_context = ""
        if self.retriever and self.include_docs:
            results = self.retriever.search(instance.problem_statement, top_k=3)
            doc_context = self.retriever.format_context(results)

        # 5. Build Prompt
        prompt = RAG_PROMPT_TEMPLATE.format(
            problem_statement=instance.problem_statement,
            doc_context=doc_context,
            primary_file=primary_file or "unknown",
            primary_content=primary_content,
            related_content=related_content
        )

        # 6. Generate
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=RAG_SYSTEM_PROMPT,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # 7. Extract Patch
        # We use the base class method which looks for ```diff blocks
        generated_patch = self._extract_patch(response)
        
        return PipelineResult(
            instance_id=instance.instance_id,
            generated_patch=generated_patch,
            ground_truth_patch=instance.patch,
            raw_response=response,
            success=True
        )
