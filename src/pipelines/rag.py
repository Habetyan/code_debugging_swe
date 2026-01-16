"""
RAG Pipeline: Retrieval-Augmented Generation.
Injects Source Code + Documentation + Solved Examples into the prompt context to help LLM generate patches.
"""
from typing import Optional, Tuple, List
from pathlib import Path
import subprocess
import re
import tempfile
import shutil

from src.data import SWEBenchInstance
from src.llm import LLMProvider
from src.retrieval import HybridRetriever
from src.retrieval.source_code import RepoManager, extract_file_paths
from src.retrieval.graph import CodeGraph
from src.retrieval.corpus import DocumentCorpus
from src.retrieval.example_retriever import ExampleRetriever
from .baseline import BaselinePipeline, PipelineResult


RAG_SYSTEM_PROMPT = """You are an expert software engineer fixing bugs in Python code.

IMPORTANT RULES:
1. ONLY modify the file specified in the "Primary File" section.
2. Do NOT invent new files or modify unrelated files.
3. Ensure your patch is syntactically correct Python.
4. Keep changes minimal - fix only the bug, don't refactor.

OUTPUT FORMAT:
Provide a Unified Diff (git diff) to fix the issue.
The file path in the diff MUST match the Primary File path exactly.
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

# Similar Solved Examples
{examples_content}

# Instructions
1. Fix the bug in '{primary_file}' ONLY.
2. Do NOT modify any other file.
3. Output a syntactically correct Unified Diff.
"""


def extract_stacktrace_files(text: str) -> List[str]:
    """Extract file paths from Python stacktraces."""
    # Pattern: File "/path/to/file.py", line N
    pattern = r'File ["\']([^"\']+\.py)["\'], line \d+'
    matches = re.findall(pattern, text)
    
    result = []
    for m in matches:
        # If it's a site-packages path, extract the relative part AFTER site-packages
        if 'site-packages' in m:
            parts = m.split('site-packages')
            if len(parts) > 1:
                # Get the part after site-packages and strip leading slashes
                relative = parts[1].lstrip('/\\')
                # Normalize backslashes to forward slashes
                relative = relative.replace('\\', '/')
                if relative:
                    result.append(relative)
            continue
        
        # Skip stdlib paths
        if '/usr/lib' in m or '/usr/local' in m or '\\Python3' in m:
            continue
            
        # For other paths, try to extract relative part
        if '/' in m or '\\' in m:
            # Normalize to forward slashes
            normalized = m.replace('\\', '/')
            parts = normalized.split('/')
            # Find repo-like starting points
            for i, part in enumerate(parts):
                if part in ['src', 'lib', 'tests', 'sklearn', 'sympy', 'matplotlib', 
                           'requests', 'pylint', 'sphinx', 'seaborn', 'django', 'flask']:
                    result.append('/'.join(parts[i:]))
                    break
    
    return result



def module_to_filepath(module_path: str) -> str:
    """Convert Python module path to file path.
    
    Example: sklearn.utils.multiclass -> sklearn/utils/multiclass.py
    """
    return module_path.replace('.', '/') + '.py'


def extract_module_paths(text: str) -> List[str]:
    """Extract Python module paths from text.
    
    Looks for patterns like:
    - from sklearn.utils.multiclass import ...
    - import sympy.physics.units
    - sklearn.preprocessing._label
    """
    patterns = [
        r'from\s+([\w\.]+)\s+import',  # from module.path import
        r'import\s+([\w\.]+)',          # import module.path
        r'([\w]+(?:\.[\w]+){2,})',      # any dotted path with 3+ parts
    ]
    
    modules = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            # Skip if it's a method call or attribute access
            if m.startswith('.') or m.endswith('.'):
                continue
            # Only consider paths with at least 2 parts
            if '.' in m:
                modules.add(m)
    
    return list(modules)


def validate_python_syntax(code: str) -> Tuple[bool, str]:
    """Validate Python syntax by attempting to compile."""
    try:
        compile(code, '<string>', 'exec')
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


class RAGPipeline(BaselinePipeline):
    """
    Advanced RAG Pipeline with:
    1. Stacktrace Parsing for file identification
    2. Module Path to File conversion
    3. Symbol Search Fallback (Functions and Classes)
    4. Code Graph Expansion
    5. Syntax Validation
    """
    
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        repo_manager: Optional[RepoManager] = None,
        retriever: Optional[HybridRetriever] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        include_docs: bool = True,
        max_syntax_retries: int = 2,
        exclude_example_ids: Optional[set] = None,
    ):
        super().__init__(
            llm_provider=llm_provider,
            repo_manager=repo_manager,
            temperature=temperature,
            max_tokens=max_tokens
        )

        self.repo_manager = repo_manager or RepoManager()
        self.include_docs = include_docs
        self.max_syntax_retries = max_syntax_retries

        if include_docs:
            if retriever is None:
                # Load pre-built corpus only (no per-instance extraction to avoid data leakage)
                corpus_path = Path("cache/expanded_corpus.json")
                if corpus_path.exists():
                    corpus = DocumentCorpus(corpus_path)
                    self.retriever = HybridRetriever(corpus)
                    print(f"Loaded pre-built corpus from {corpus_path}")
                else:
                    # No corpus available - disable doc retrieval
                    print("Warning: No pre-built corpus found. Doc retrieval disabled.")
                    self.retriever = None
            else:
                self.retriever = retriever

            # Initialize Example Retriever (Few-Shot RAG)
            try:
                self.example_retriever = ExampleRetriever(exclude_ids=exclude_example_ids)
            except Exception as e:
                print(f"Warning: Could not init ExampleRetriever: {e}")
                self.example_retriever = None
        else:
            self.retriever = None
            self.example_retriever = None
    
    def _extract_repo_docs(self, repo_path: str) -> list:
        """Extract documentation files from a repository."""
        from src.retrieval.corpus import Document
        
        repo_path = Path(repo_path)
        docs = []
        
        # Common doc directories
        doc_dirs = ['doc', 'docs', 'documentation', 'examples']
        
        # Common doc file extensions
        doc_extensions = ['.rst', '.md', '.txt']
        
        for doc_dir in doc_dirs:
            dir_path = repo_path / doc_dir
            if dir_path.exists() and dir_path.is_dir():
                # Find all doc files
                for ext in doc_extensions:
                    for doc_file in dir_path.rglob(f'*{ext}'):
                        try:
                            content = doc_file.read_text(encoding='utf-8', errors='ignore')
                            # Skip very large files
                            if len(content) > 50000:
                                content = content[:50000]
                            
                            # Create document
                            doc = Document(
                                doc_id=str(doc_file.relative_to(repo_path)),
                                title=doc_file.stem,
                                content=content,
                                source=str(repo_path.name),
                                library=str(repo_path.name).split('__')[0],
                                doc_type='documentation'
                            )
                            docs.append(doc)
                            
                            # Limit to 50 docs to avoid indexing overhead
                            if len(docs) >= 50:
                                return docs
                        except Exception:
                            continue
        
        return docs

    def _find_primary_file(self, problem_statement: str, repo_path: str) -> Optional[str]:
        """Find the primary file to edit using multiple strategies.
        
        Priority order (most to least reliable):
        1. Stacktrace files - shows exact error location
        2. Explicit file paths in text
        3. Module paths converted to files
        4. Class name grep
        5. Function name grep
        """
        
        # Strategy 1: Parse stacktraces (HIGHEST PRIORITY - shows error location)
        stacktrace_files = extract_stacktrace_files(problem_statement)
        for fp in stacktrace_files:
            # Try exact match first
            if (Path(repo_path) / fp).exists():
                return fp
            # Try without leading path segment
            if '/' in fp:
                short = fp.split('/', 1)[1]
                if (Path(repo_path) / short).exists():
                    return short
        
        # Strategy 2: Extract explicit file paths (e.g., "in file foo/bar.py")
        file_paths = extract_file_paths(problem_statement)
        for fp in file_paths:
            if (Path(repo_path) / fp).exists():
                return fp
        
        # Strategy 3: Convert module paths to file paths
        module_paths = extract_module_paths(problem_statement)
        for mp in module_paths:
            fp = module_to_filepath(mp)
            if (Path(repo_path) / fp).exists():
                return fp
            # Try with leading dirs
            for prefix in ['src', 'lib', '']:
                candidate = f"{prefix}/{fp}" if prefix else fp
                if (Path(repo_path) / candidate).exists():
                    return candidate
        
        # Strategy 4: Search for class names
        class_names = re.findall(r'\b[A-Z][a-zA-Z0-9]+\b', problem_statement)
        for cls_name in class_names:
            if len(cls_name) < 4:
                continue
            try:
                result = subprocess.run(
                    ["grep", "-r", "-l", f"class {cls_name}", "."],
                    cwd=str(repo_path),
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0 and result.stdout:
                    for line in result.stdout.splitlines():
                        if line.endswith('.py') and 'test' not in line.lower():
                            return line.lstrip('./')
            except Exception:
                pass
        
        # Strategy 5: Search for function names (snake_case)
        func_names = re.findall(r'\b([a-z_][a-z0-9_]+)\s*\(', problem_statement)
        for func_name in func_names:
            if len(func_name) < 5 or func_name in ['print', 'range', 'list', 'dict', 'set', 'str', 'int', 'float']:
                continue
            try:
                result = subprocess.run(
                    ["grep", "-r", "-l", f"def {func_name}", "."],
                    cwd=str(repo_path),
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0 and result.stdout:
                    for line in result.stdout.splitlines():
                        if line.endswith('.py') and 'test' not in line.lower():
                            return line.lstrip('./')
            except Exception:
                pass
        
        # Strategy 6: Search for identifiers in backticks (e.g., `__isnull`, `MyClass`)
        # These are often class/method names mentioned in feature requests
        backtick_ids = re.findall(r'`([A-Za-z_][A-Za-z0-9_]*)`', problem_statement)
        for identifier in backtick_ids:
            if len(identifier) < 4:
                continue
            try:
                # Search for class or def with this name
                result = subprocess.run(
                    ["grep", "-r", "-l", f"class {identifier}\\|def {identifier}", "."],
                    cwd=str(repo_path),
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0 and result.stdout:
                    for line in result.stdout.splitlines():
                        if line.endswith('.py') and 'test' not in line.lower():
                            return line.lstrip('./')
            except Exception:
                pass
        
        return None

    def _validate_patch_syntax(self, patch: str, repo_path: str, primary_file: str) -> Tuple[bool, str]:
        """Apply patch to a temp copy and validate syntax."""
        if not patch.strip() or not primary_file:
            return False, "Empty patch or no primary file"
        
        try:
            # Create temp directory
            with tempfile.TemporaryDirectory() as tmpdir:
                # Copy the file
                src = Path(repo_path) / primary_file
                if not src.exists():
                    return False, f"File {primary_file} not found"
                
                dst = Path(tmpdir) / "test.py"
                shutil.copy(src, dst)
                
                # Apply patch
                proc = subprocess.run(
                    ["patch", "-p1", "--no-backup-if-mismatch"],
                    input=patch,
                    cwd=tmpdir,
                    capture_output=True,
                    text=True
                )
                
                if proc.returncode != 0:
                    return False, f"Patch failed to apply: {proc.stderr}"
                
                # Validate syntax
                content = dst.read_text()
                valid, error = validate_python_syntax(content)
                return valid, error
                
        except Exception as e:
            return False, str(e)

    def run(self, instance: SWEBenchInstance) -> PipelineResult:
        """
        Executes the RAG pipeline for a given SWEBench instance.

        Args:
            instance (SWEBenchInstance): The SWEBench instance containing problem details.

        Returns:
            PipelineResult: The result of the pipeline execution, including the generated patch.
        """
        # 1. Cloning Repo
        repo_path = self.repo_manager.get_repo_path(instance.repo, instance.base_commit)

        # Check if clone failed
        if repo_path is None:
            print(f"[ERROR] Failed to clone repo for {instance.instance_id}")
            return PipelineResult(
                instance_id=instance.instance_id,
                generated_patch="",
                ground_truth_patch=instance.patch,
                raw_response="ERROR: Failed to clone repository",
                success=False
            )

        # 2. Identify Primary File (using multiple strategies)
        primary_file = self._find_primary_file(instance.problem_statement, repo_path)
        
        primary_content = ""
        related_content = ""
        
        if primary_file:
            try:
                primary_content = (Path(repo_path) / primary_file).read_text()
                # Truncate if very long
                if len(primary_content) > 10000:
                    primary_content = primary_content[:10000] + "\n... (truncated)"
            except Exception:
                primary_content = "# Error reading file"
            
            # 3. Graph Expansion
            try:
                graph = CodeGraph(repo_path)
                related_files = graph.get_related_files(primary_file, max_depth=1)
                
                for rf in related_files[:1]:  # Reduced from 3 to 1 to save context
                    try:
                        content = (Path(repo_path) / rf).read_text()
                        if len(content) > 2000:
                            content = content[:2000] + "\n... (truncated)"
                        related_content += f"\n### {rf}\n```python\n{content}\n```\n"
                    except:
                        pass
            except Exception:
                pass
        else:
            primary_content = "# Could not identify primary file from problem statement"
            primary_file = "unknown"

        # 4. Retrieve Docs
        doc_context = ""
        if self.retriever and self.include_docs:
            results = self.retriever.search(instance.problem_statement, top_k=1)  # Reduced from 3 to 1 to save context
            doc_context = self.retriever.format_context(results)

        # 4.5 Retrieve Similar Examples
        examples_content = "No similar examples found."
        if self.example_retriever:
            try:
                examples = self.example_retriever.retrieve(instance.problem_statement, instance.repo, k=1)  # Reduced from 2 to 1 to save context
                if examples:
                    examples_content = ""
                    for i, ex in enumerate(examples):
                        examples_content += f"\n## Example {i+1} (from {ex['repo']})\n"
                        examples_content += f"### Problem\n{ex['problem_statement'][:500]}...\n"
                        examples_content += f"### Fix\n```diff\n{ex['patch']}\n```\n"
            except Exception as e:
                print(f"Error retrieving examples: {e}")

        # 5. Build Prompt
        prompt = RAG_PROMPT_TEMPLATE.format(
            problem_statement=instance.problem_statement,
            doc_context=doc_context,
            primary_file=primary_file,
            primary_content=primary_content,
            related_content=related_content,
            examples_content=examples_content
        )

        # 6. Generate with Syntax Validation Loop
        generated_patch = ""
        syntax_error = ""
        response = ""

        for attempt in range(self.max_syntax_retries + 1):
            # Add syntax error to prompt if retrying
            current_prompt = prompt
            if syntax_error:
                current_prompt += f"\n\n# PREVIOUS ATTEMPT FAILED\nYour previous patch had a syntax error:\n{syntax_error}\n\nPlease fix the syntax error and try again."
            
            response = self.llm.generate(
                prompt=current_prompt,
                system_prompt=RAG_SYSTEM_PROMPT,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            generated_patch = self._extract_patch(response)
            
            if not generated_patch:
                break  # No patch generated, skip validation
            
            # Validate syntax
            if primary_file and primary_file != "unknown":
                valid, error = self._validate_patch_syntax(generated_patch, repo_path, primary_file)
                if valid:
                    break
                else:
                    syntax_error = error
                    if attempt < self.max_syntax_retries:
                        continue  # Retry
            else:
                break  # Can't validate without knowing the file
        
        return PipelineResult(
            instance_id=instance.instance_id,
            generated_patch=generated_patch,
            ground_truth_patch=instance.patch,
            raw_response=response,
            success=True
        )
