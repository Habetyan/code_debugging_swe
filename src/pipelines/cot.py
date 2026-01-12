
from typing import Optional, Tuple, List
from pathlib import Path
import subprocess
import re

from src.data import SWEBenchInstance
from src.llm import LLMProvider
from src.retrieval.source_code import RepoManager
from src.retrieval.example_retriever import ExampleRetriever
from src.retrieval.indexer import HybridRetriever
from src.retrieval.corpus import DocumentCorpus, Document
from src.pipelines.baseline import BaselinePipeline, PipelineResult


# Stage 1: Localization Prompt
LOCALIZATION_SYSTEM_PROMPT = """You are an expert software engineer. Your task is to identify which file needs to be modified to fix a bug.

Think step by step:
1. What is the bug about? What component/feature is affected?
2. What kind of code would implement this feature?
3. Based on the file list, which file is most likely to contain the bug?

Be precise. Output ONLY the file path at the end in the format:
TARGET_FILE: path/to/file.py
"""

LOCALIZATION_PROMPT_TEMPLATE = """# Bug Report
{problem_statement}

# Repository Structure
Here are the Python files in this repository (showing first 200 most relevant):

{file_list}

# Task
Analyze the bug report and identify which file needs to be modified.
Think through your reasoning step by step, then output the target file.

Remember to end with:
TARGET_FILE: path/to/file.py
"""

# Stage 2: Fix Generation Prompt (now with RAG context)
FIX_SYSTEM_PROMPT = """You are an expert software engineer fixing bugs in Python code.

IMPORTANT RULES:
1. ONLY modify the file specified in the "Primary File" section.
2. Do NOT invent new files or modify unrelated files.
3. Ensure your patch is syntactically correct Python.
4. Keep changes minimal - fix only the bug, don't refactor.
5. Learn from the similar solved examples provided.
6. CRITICAL: Analyze the control flow carefully. Do NOT break existing variable updates or logic unless necessary.
7. Verify your fix ensures the reported bug is solved while maintaining existing functionality.

OUTPUT FORMAT:
Provide a Unified Diff (git diff) to fix the issue.
The file path in the diff MUST match the Primary File path exactly.
```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ ... @@
```
"""

FIX_PROMPT_TEMPLATE = """# Bug Report
{problem_statement}

# Similar Solved Bugs (Learn from these examples)
{examples_content}

# Relevant Documentation
{doc_context}

# Primary File: {primary_file}
```python
{primary_content}
```

# Instructions
1. Study the similar solved bugs above to understand the fix patterns.
2. Analyze the bug in the Primary File. 
3. CAREFULLY TRACE the execution flow. Ensure you understand how variables (like indices/counters) are updated.
4. Generate a minimal fix.
5. Output a syntactically correct Unified Diff.
"""


def validate_python_syntax(code: str) -> Tuple[bool, str]:
    """Validate Python syntax by attempting to compile."""
    try:
        compile(code, '<string>', 'exec')
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


class CoTPipeline(BaselinePipeline):
    """
    Chain-of-Thought + RAG Pipeline:
    1. File Localization with CoT reasoning (LLM-based)
    2. Few-shot examples from similar bugs (RAG)
    3. Repository documentation (RAG)
    4. Fix generation with rich context
    """
    
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        repo_manager: Optional[RepoManager] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_files_to_show: int = 200,
        num_examples: int = 2,
        include_docs: bool = True,
    ):
        super().__init__(llm_provider, temperature, max_tokens)
        self.repo_manager = repo_manager or RepoManager()
        self.max_files_to_show = max_files_to_show
        self.num_examples = num_examples
        self.include_docs = include_docs
        
        # Initialize ExampleRetriever for few-shot learning
        try:
            self.example_retriever = ExampleRetriever()
        except Exception as e:
            print(f"Warning: Could not initialize ExampleRetriever: {e}")
            self.example_retriever = None
        
        # Doc retriever will be initialized per-repo
        self.doc_retriever = None
    
    def _get_python_files(self, repo_path: str) -> List[str]:
        """Get list of Python files in the repository."""
        repo_path = Path(repo_path)
        files = []
        
        try:
            result = subprocess.run(
                ["find", ".", "-name", "*.py", "-type", "f"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    fp = line.lstrip('./')
                    fp_lower = fp.lower()
                    if any(x in fp_lower for x in ['test', 'example', 'doc/', 'docs/', '__pycache__', 'conftest']):
                        continue
                    files.append(fp)
        except Exception:
            for py_file in repo_path.rglob("*.py"):
                rel_path = str(py_file.relative_to(repo_path))
                if 'test' not in rel_path.lower():
                    files.append(rel_path)
        
        files.sort(key=lambda x: (x.count('/'), x))
        return files[:self.max_files_to_show]
    
    def _localize_file(self, problem_statement: str, repo_path: str) -> Optional[str]:
        """Stage 1: Use LLM with Chain-of-Thought to identify the target file."""
        files = self._get_python_files(repo_path)
        if not files:
            return None
        
        file_list = "\n".join(files)
        
        prompt = LOCALIZATION_PROMPT_TEMPLATE.format(
            problem_statement=problem_statement[:4000],
            file_list=file_list
        )
        
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=LOCALIZATION_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=1024
        )
        
        match = re.search(r'TARGET_FILE:\s*(.+\.py)', response)
        if match:
            target_file = match.group(1).strip()
            if (Path(repo_path) / target_file).exists():
                return target_file
            for f in files:
                if f.endswith(target_file) or target_file.endswith(f):
                    return f
        
        for f in files:
            if f in response:
                return f
        
        return None
    
    def _extract_repo_docs(self, repo_path: str) -> list:
        """Extract documentation files from a repository."""
        repo_path = Path(repo_path)
        docs = []
        
        doc_dirs = ['doc', 'docs', 'documentation']
        doc_extensions = ['.rst', '.md', '.txt']
        
        for doc_dir in doc_dirs:
            dir_path = repo_path / doc_dir
            if dir_path.exists() and dir_path.is_dir():
                for ext in doc_extensions:
                    for doc_file in list(dir_path.rglob(f'*{ext}'))[:50]:
                        try:
                            content = doc_file.read_text(errors='ignore')[:2000]
                            docs.append(Document(
                                content=content,
                                title=doc_file.name,
                                library=repo_path.name,
                                source=str(doc_file.relative_to(repo_path))
                            ))
                        except Exception:
                            continue
        
        return docs
    
    def _get_doc_context(self, problem_statement: str, repo_path: str) -> str:
        """Retrieve relevant documentation for the problem."""
        if not self.include_docs:
            return "No documentation available."
        
        repo_docs = self._extract_repo_docs(repo_path)
        if not repo_docs:
            return "No documentation found in repository."
        
        # Build a corpus and index
        corpus = DocumentCorpus()
        for doc in repo_docs:
            corpus.add(doc)
        
        try:
            retriever = HybridRetriever(corpus)
            results = retriever.search(problem_statement, top_k=3)
            
            doc_parts = []
            for doc, score in results:
                doc_parts.append(f"## {doc.title}\n{doc.content[:500]}...")
            
            return "\n\n".join(doc_parts) if doc_parts else "No relevant documentation found."
        except Exception as e:
            return f"Documentation retrieval failed: {e}"
    
    def _get_examples_context(self, problem_statement: str, repo: str) -> str:
        """Retrieve similar solved bug examples."""
        if not self.example_retriever:
            return "No examples available."
        
        try:
            examples = self.example_retriever.retrieve(
                problem_statement=problem_statement,
                repo=repo,
                top_k=self.num_examples
            )
            
            if not examples:
                return "No similar solved bugs found."
            
            parts = []
            for i, ex in enumerate(examples, 1):
                parts.append(f"""## Example {i}: {ex['instance_id']}
**Problem:** {ex['problem_statement'][:300]}...

**Solution (Patch):**
```diff
{ex['patch'][:800]}
```
""")
            
            return "\n".join(parts)
        except Exception as e:
            return f"Example retrieval failed: {e}"
    
    def _generate_fix(self, problem_statement: str, primary_file: str, 
                      primary_content: str, examples_content: str, doc_context: str) -> str:
        """Stage 2: Generate the fix with RAG context."""
        prompt = FIX_PROMPT_TEMPLATE.format(
            problem_statement=problem_statement,
            examples_content=examples_content,
            doc_context=doc_context,
            primary_file=primary_file,
            primary_content=primary_content[:30000]  # Increase limit to 30k
        )
        
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=FIX_SYSTEM_PROMPT + "\n\nCRITICAL: Ensure your diff produces valid Python syntax. Do not leave 'try' blocks without 'except'/'finally'.",
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response
    
    def run(self, instance: SWEBenchInstance) -> PipelineResult:
        """Run the CoT + RAG pipeline."""
        
        # 1. Clone Repo
        repo_path = self.repo_manager.get_repo_path(instance.repo, instance.base_commit)
        
        # 2. Stage 1: Localization with CoT
        primary_file = self._localize_file(instance.problem_statement, repo_path)
        
        if not primary_file:
            return PipelineResult(
                instance_id=instance.instance_id,
                generated_patch="",
                ground_truth_patch=instance.patch,
                raw_response="Could not localize target file",
                success=False,
                error="Localization failed"
            )
        
        # 3. Read primary file content
        try:
            primary_content = (Path(repo_path) / primary_file).read_text()
        except Exception as e:
            return PipelineResult(
                instance_id=instance.instance_id,
                generated_patch="",
                ground_truth_patch=instance.patch,
                raw_response=str(e),
                success=False,
                error=f"Could not read file: {primary_file}"
            )
        
        # 4. Get RAG context
        examples_content = self._get_examples_context(instance.problem_statement, instance.repo)
        doc_context = self._get_doc_context(instance.problem_statement, repo_path)
        
        # 5. Stage 2: Generate Fix with RAG context
        response = self._generate_fix(
            instance.problem_statement, 
            primary_file, 
            primary_content,
            examples_content,
            doc_context
        )
        
        # 6. Extract patch
        generated_patch = self._extract_patch(response)
        
        return PipelineResult(
            instance_id=instance.instance_id,
            generated_patch=generated_patch,
            ground_truth_patch=instance.patch,
            raw_response=response,
            success=True
        )
