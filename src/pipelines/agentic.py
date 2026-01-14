"""
Agentic Pipeline: Implements a ReAct-style agent that explores the codebase.
Uses tools (AST, grep, docs) and iteratively localizes/fixes bugs using a reason-act loop.
Now enhanced with Robust Search/Replace Editing + Self-Correction Loop.
"""
import subprocess
import re
import json
import difflib
from typing import Optional, List, Tuple
from pathlib import Path

from src.data import SWEBenchInstance
from src.llm import LLMProvider
from src.retrieval.source_code import RepoManager
from src.retrieval.example_retriever import ExampleRetriever
from src.verification.harness import VerificationHarness
from src.pipelines.baseline import PipelineResult
from src.retrieval.indexer import HybridRetriever
from src.retrieval.corpus import DocumentCorpus


# =============================================================================
#  PROMPTS (Updated for Search/Replace Strategy)
# =============================================================================

REACT_SYSTEM = """You are a specialized debugging agent. Your goal is to locate the file causing the bug.
You have access to a unix-like environment and specific tools.

TOOLS:
- grep(pattern: str): Search for a string in the codebase.
- find_file(name: str): Find a file by name.
- read_file(path: str): Read a file's content (with line numbers).
- find_usages(symbol: str): Find where a symbol is defined/used (AST-based).
- get_function_context(func_name: str, file_path: str): Get details about a function (callers/callees).
- run_tests(test_file: str): Run a specific test file.
- search_docs(query: str): Search documentation and StackOverflow.
- list_files(): List all Python files in the repo.

PROCESS:
1. Analyzing the Bug Report.
2. THOUGHT: Reasoning carefully about next steps.

3. ACTION: tool_name(args)
4. OBSERVATION: [Result of action, provided by system]
5. Repeat steps 2-4 until you have found the file or exhausted options (max 10 steps).
6. FINAL_ANSWER: TARGET_FILE: path/to/file.py

Start by exploring the codebase based on the bug report.
"""


REPO_TREE_SYSTEM = """You are a Senior Software Engineer acting as a Localization Expert.
Your task is to identify the best 3 files in the repository that likely contain the bug described.

Guidelines:
1. Review the provided "Candidate Files" (found by pattern matching).
2. Review the "Repository Files" to understand the project structure.
3. Select the Top 3 files that are most relevant to the bug.
4. Prioritize:
   - Files where the logic/validation likely resides (e.g. `base.py` for validation vs `options.py` for definition).
   - Core implementation files over tests or examples.
   - Files extracted from stacktraces (highest confidence).

Output Format:
Output ONLY the file paths, one per line, in order of likelihood (most likely first).
Do not include any explanation or markdown formatting.
"""

PLANNING_SYSTEM = """You are an expert software engineer. Analyze the code and create a fix plan.

Given a bug report and the relevant code, explain:
1. ROOT CAUSE: What exactly is causing the bug?
2. FIX STRATEGY: What changes are needed? Be specific about lines.
3. EDGE CASES: What edge cases should be considered?

Be concise and precise."""

PLANNING_PROMPT = """# Bug Report
{problem_statement}

# Relevant Code ({file_path})
{code_content}

{grep_results}

Analyze the bug and create a fix plan."""

# --- UPDATED PATCH PROMPTS (S/R) ---

PATCH_SYSTEM = """You are an expert software engineer fixing bugs.

IMPORTANT:
1. Use SEARCH/REPLACE blocks to modify the code.
2. The SEARCH block must be an EXACT COPY of the lines in the original file (including indentation).
3. The REPLACE block contains the new code.

FORMAT:
<<<< SEARCH
original code lines
====
new code lines
>>>> REPLACE
"""

PATCH_PROMPT = """# Bug Report
{problem_statement}

# Fix Plan
{fix_plan}

# Current Code ({file_path})
{code_content}

Generate a fix using SEARCH/REPLACE blocks. Output ONLY the blocks."""

RECTIFICATION_SYSTEM_PROMPT = """You are an expert software engineer debugging a failed fix.
Your previous patch was applied, but the verification tests failed.
Analyze the test output and the original code to fix the logic.

OUTPUT FORMAT:
Provide a NEW SEARCH/REPLACE block to update the file correctly.
"""

RECTIFICATION_PROMPT_TEMPLATE = """# Original Bug
{problem_statement}

# Your Previous Fix
{previous_patch}

# Test Failure Output
{test_output}

# Primary File Content (With your previous fix applied)
```python
{file_content}
```

# Instructions
1. The previous fix failed the tests.
2. Analyze the failure message.
3. Generate a new fix using SEARCH/REPLACE blocks.
"""


# =============================================================================
#  SOTA PROMPTS: Self-Critique + LLM-Assisted Localization
# =============================================================================

SELF_CRITIQUE_SYSTEM = """You are a senior code reviewer. Review the proposed patch and identify any issues.

Check for:
1. SYNTAX: Are there any syntax errors?
2. LOGIC: Does the fix actually address the root cause?
3. EDGE CASES: Are there unhandled edge cases?
4. SIDE EFFECTS: Could this break other functionality?
5. COMPLETENESS: Is the fix complete or are parts missing?

If the patch is correct, respond with: APPROVED

If there are issues, respond with:
ISSUES:
- [issue 1]
- [issue 2]
SUGGESTION: [how to fix]
"""

SELF_CRITIQUE_PROMPT = """# Original Bug Report
{problem_statement}

# Proposed Patch
```diff
{patch}
```

# File Content After Patch
```python
{patched_content}
```

Review this patch. Is it correct and complete?
"""

LLM_LOCALIZATION_SYSTEM = """You are an expert at navigating large codebases.
Given a bug report, suggest the most likely location of the bug.

Output a JSON object with:
{
    "search_terms": ["term1", "term2"],
    "likely_dirs": ["subdir1", "subdir2"],
    "file_patterns": ["*pattern*.py"],
    "reasoning": "brief explanation"
}
"""

LLM_LOCALIZATION_PROMPT = """# Bug Report
{problem_statement}

# Repository Structure (Top 50 files)
{file_list}

Analyze where this bug is most likely located.
"""


# =============================================================================
#  HELPER FUNCTIONS
# =============================================================================


def extract_stacktrace_files(text: str) -> List[str]:
    """Extract file paths from Python stacktraces."""
    pattern = r'File ["\']([^"\']+\.py)["\'], line \d+'
    matches = re.findall(pattern, text)
    result = []
    for m in matches:
        if 'site-packages' in m:
            parts = m.split('site-packages')
            if len(parts) > 1:
                relative = parts[1].lstrip('/\\')
                relative = relative.replace('\\', '/')
                if relative:
                    result.append(relative)
            continue
        if '/usr/lib' in m or '/usr/local' in m or '\\Python3' in m:
            continue
        if '/' in m or '\\' in m:
            normalized = m.replace('\\', '/')
            parts = normalized.split('/')
            for i, part in enumerate(parts):
                if part in ['src', 'lib', 'tests', 'sklearn', 'sympy', 'matplotlib', 
                           'requests', 'pylint', 'sphinx', 'seaborn', 'django', 'flask']:
                    result.append('/'.join(parts[i:]))
                    break
    return result


def extract_file_paths(text: str) -> List[str]:
    pattern = r'(?:^|[\s`\'"])([a-zA-Z_][\w/]*\.py)(?:[\s`\'"]|$)'
    matches = re.findall(pattern, text)
    return [m for m in matches if len(m) > 5]


def module_to_filepath(module_path: str) -> str:
    return module_path.replace('.', '/') + '.py'


def extract_module_paths(text: str) -> List[str]:
    patterns = [
        r'from\s+([\w\.]+)\s+import',
        r'import\s+([\w\.]+)',
        r'([\w]+(?:\.[\w]+){2,})',
    ]
    modules = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            if '.' in m and not m.startswith('.') and not m.endswith('.'):
                modules.add(m)
    return list(modules)


# =============================================================================
#  AGENTIC PIPELINE CLASS
# =============================================================================

class AgenticPipeline:
    """
    High Pass@1 Agentic Pipeline v2 with improved localization and Self-Correction.
    """
    
    MAX_PATCH_RETRIES = 3
    MAX_CANDIDATES = 2
    
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        repo_manager: Optional[RepoManager] = None,
        example_retriever: Optional[ExampleRetriever] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        use_harness: bool = True,
        exclude_example_ids: Optional[set] = None,
    ):
        self.llm = llm_provider or LLMProvider()
        self.repo_manager = repo_manager or RepoManager()

        # Initialize ExampleRetriever with proper error handling
        self.example_retriever = example_retriever
        if self.example_retriever is None and exclude_example_ids is not None:
            try:
                self.example_retriever = ExampleRetriever(exclude_ids=exclude_example_ids)
            except Exception as e:
                print(f"Warning: Could not initialize ExampleRetriever: {e}")
                self.example_retriever = None

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.harness = VerificationHarness() if use_harness else None
        
        # Initialize Retriever with expanded corpus
        try:
            corpus_path = "cache/expanded_corpus.json"
            if Path(corpus_path).exists():
                corpus = DocumentCorpus(corpus_path)
                self.retriever = HybridRetriever(corpus)
                print(f"Agentic Pipeline initialized with {len(corpus)} docs from cache")
            else:
                self.retriever = None
        except Exception as e:
            print(f"Warning: Failed to init retriever for agent: {e}")
            self.retriever = None


    def _generate_repo_tree(self, repo_path: str, max_depth: int = 2) -> str:
        file_list = []
        try:
            result = subprocess.run(
                ["find", ".", "-maxdepth", str(max_depth), "-not", "-path", "*/.*"],
                cwd=repo_path, capture_output=True, text=True
            )
            paths = result.stdout.strip().split("\n")
            paths = [p for p in paths if p and p != "."]
            paths.sort()
            return "\n".join(paths)
        except Exception: return ""

    def _filter_file_list(self, files: List[str]) -> List[str]:
        filtered = []
        for f in files:
            f = f.lstrip("./")
            parts = f.split("/")
            if any(p.startswith(".") for p in parts): continue
            if any(p in ["doc", "docs", "examples", "test", "tests", "build", "dist", "site-packages"] for p in parts): continue
            filtered.append(f)
        return filtered if filtered else files

    def _get_file_list(self, repo_path: str, max_files: int = 1000) -> str:
        try:
            result = subprocess.run(
                ["find", ".", "-name", "*.py", "-type", "f"],
                cwd=repo_path, capture_output=True, text=True, timeout=30
            )
            files = result.stdout.strip().split("\n")
            filtered_files = self._filter_file_list(files)
            return "\n".join(filtered_files[:max_files])
        except Exception: return ""

    def _read_file_with_lines(self, repo_path: str, file_path: str) -> str:
        try:
            full_path = Path(repo_path) / file_path.lstrip("./")
            if not full_path.exists(): return f"File not found: {file_path}"
            lines = full_path.read_text().splitlines()
            numbered = [f"{i+1:4d} | {line}" for i, line in enumerate(lines)]
            content = "\n".join(numbered)
            if len(content) > 50000: content = content[:50000] + "\n... (truncated)"
            return content
        except Exception as e: return f"Error reading file: {e}"

    def _grep_code(self, repo_path: str, pattern: str, context: int = 3) -> str:
        try:
            result = subprocess.run(
                ["git", "grep", "-n", "-i", f"-C{context}", pattern],
                cwd=repo_path, capture_output=True, text=True, timeout=30
            )
            if result.stdout.strip(): return result.stdout[:5000]
            return "No matches found"
        except Exception as e: return f"Grep error: {e}"

    def _find_file(self, repo_path: str, filename: str) -> str:
        try:
            result = subprocess.run(
                ["find", ".", "-name", filename, "-type", "f"],
                cwd=repo_path, capture_output=True, text=True, timeout=30
            )
            if result.stdout.strip(): return result.stdout.strip().split("\n")[0]
            return ""
        except Exception: return ""

    def _find_symbol_usages(self, repo_path: str, symbol: str, max_results: int = 20) -> str:
        import ast
        usages = []
        definitions = []
        for py_file in Path(repo_path).rglob("*.py"):
            rel_path = str(py_file.relative_to(repo_path))
            if 'test' in rel_path.lower() or '/.' in rel_path: continue
            try:
                content = py_file.read_text(errors='ignore')
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == symbol:
                        definitions.append(f"DEF {rel_path}:{node.lineno} - def {symbol}()")
                    elif isinstance(node, ast.ClassDef) and node.name == symbol:
                        definitions.append(f"DEF {rel_path}:{node.lineno} - class {symbol}")
                    elif isinstance(node, ast.Name) and node.id == symbol:
                        usages.append(f"USE {rel_path}:{node.lineno}")
                    elif isinstance(node, ast.Attribute) and node.attr == symbol:
                        usages.append(f"ATTR {rel_path}:{node.lineno} - .{symbol}")
            except Exception: continue
        result = []
        if definitions: result.append("# Definitions:"); result.extend(definitions[:5])
        if usages: result.append("\n# Usages:"); result.extend(usages[:max_results])
        return "\n".join(result) if result else f"No usages found for '{symbol}'"

    def _get_function_context(self, repo_path: str, file_path: str, func_name: str) -> str:
        return ""

    def _find_related_tests(self, repo_path: str, file_path: str) -> str:
        try:
            file_name = Path(file_path).stem
            patterns = [f"test_{file_name}.py", f"{file_name}_test.py", f"test_{file_name}*.py"]
            related_tests = []
            for pattern in patterns:
                result = subprocess.run(["find", ".", "-name", pattern, "-type", "f"], cwd=repo_path, capture_output=True, text=True)
                if result.returncode == 0:
                   related_tests.extend([l.lstrip("./") for l in result.stdout.split("\n") if l])
            return "\n".join(set(related_tests[:10])) if related_tests else "No tests found."
        except: return "Error finding tests."

    def _analyze_error_pattern(self, problem_statement: str) -> str:
        return "Analysis skipped."

    def _parse_action(self, text: str) -> Optional[Tuple[str, List[str]]]:
        lines = text.splitlines()
        action_line = None
        for line in reversed(lines):
            if line.strip().startswith("ACTION:"):
                action_line = line
                break
        if not action_line: return None
        match = re.search(r'ACTION:\s*([a-zA-Z_]+)\((.*)\)', action_line)
        if match:
            tool_name = match.group(1)
            args_str = match.group(2)
            args = [a.strip().strip("'").strip('"') for a in args_str.split(',') if a.strip()]
            return tool_name, args
        return None

    def _execute_tool(self, tool_name: str, args: List[str], repo_path: str) -> str:
        try:
            if tool_name == "grep": return self._grep_code(repo_path, args[0])
            elif tool_name == "find_file": return self._find_file(repo_path, args[0])
            elif tool_name == "read_file": 
                if not args: return "Error: Missing file path"
                return self._read_file_with_lines(repo_path, args[0])
            elif tool_name == "find_usages":
                if not args: return "Error: Missing symbol"
                return self._find_symbol_usages(repo_path, args[0])
            elif tool_name == "list_files": return self._get_file_list(repo_path)
            else: return f"Tool '{tool_name}' not found."
        except Exception as e: return f"Error executing {tool_name}: {e}"

    def _react_localize(self, instance: SWEBenchInstance, repo_path: str, candidates: List[str] = None, max_steps: int = 6) -> Optional[List[str]]:
        problem = instance.problem_statement
        history = f"# Bug Report\n{problem[:4000]}\n"
        if candidates: history += f"\n# Candidate Files\n{json.dumps(candidates[:20], indent=2)}\n"
        files = self._get_file_list(repo_path)
        history += f"\n# Available Files\n{files[:5000]}...\n"
        
        current_step = 0
        while current_step < max_steps:
            prompt = history + f"\n\nStep {current_step + 1}:\n"
            response = self.llm.generate(prompt, REACT_SYSTEM, 0.0, 512, stop=["OBSERVATION:"])
            history += f"\nStep {current_step + 1}:\n{response}\n"
            
            final_match = re.search(r'FINAL_ANSWER:\s*TARGET_FILE:\s*(.*)', response)
            if final_match: return [final_match.group(1).strip()]
            
            action = self._parse_action(response)
            if action:
                tool, args = action
                observation = self._execute_tool(tool, args, repo_path)
                history += f"OBSERVATION: {observation}\n"
            else: history += "OBSERVATION: Checking next step...\n"
            current_step += 1
        return None

    # --- S/R PATCH APPLICATION LOGIC ---

    def _apply_search_replace(self, content: str, response: str) -> str:
        """Apply SEARCH/REPLACE blocks with robust matching."""
        pattern = r"<{4,}\s*SEARCH\s*\n(.*?)\n={4,}\s*\n(.*?)\n>{4,}\s*REPLACE"
        matches = list(re.finditer(pattern, response, re.DOTALL))
        if not matches:
            return content

        for m in matches:
            search_block = m.group(1)
            replace_block = m.group(2)

            # 1. Exact match (best case)
            if search_block in content:
                content = content.replace(search_block, replace_block, 1)
                continue

            # 2. Try stripping both sides
            search_stripped = search_block.strip()
            if search_stripped and search_stripped in content:
                idx = content.find(search_stripped)
                if idx >= 0:
                    content = content[:idx] + replace_block.strip() + content[idx + len(search_stripped):]
                    continue

            # 3. Fuzzy block matching using SequenceMatcher (improved)
            if not search_stripped:
                continue
            search_lines = search_block.splitlines()
            if len(search_lines) < 1:
                continue

            content_lines = content.splitlines(keepends=True)
            best_match_idx = -1
            best_match_ratio = 0.0

            # Slide window over content to find best matching block
            for i in range(len(content_lines) - len(search_lines) + 1):
                candidate_block = ''.join(content_lines[i:i + len(search_lines)])
                ratio = difflib.SequenceMatcher(None, search_block, candidate_block).ratio()
                if ratio > best_match_ratio:
                    best_match_ratio = ratio
                    best_match_idx = i
                # Early exit on near-perfect match
                if ratio > 0.98:
                    break

            # Only apply if match quality is high enough (85% threshold)
            if best_match_idx >= 0 and best_match_ratio >= 0.85:
                before = ''.join(content_lines[:best_match_idx])
                after = ''.join(content_lines[best_match_idx + len(search_lines):])
                # Preserve trailing newline consistency
                if replace_block and not replace_block.endswith('\n'):
                    replace_block += '\n'
                content = before + replace_block + after

        return content

    def _phase_plan(self, instance: SWEBenchInstance, file_path: str, code_content: str, grep_results: str = "") -> str:
        prompt = PLANNING_PROMPT.format(
            problem_statement=instance.problem_statement[:4000],
            file_path=file_path,
            code_content=code_content[:50000],
            grep_results=grep_results
        )
        return self.llm.generate(prompt, PLANNING_SYSTEM, 0.0, 2048)

    def _phase_patch(self, instance: SWEBenchInstance, file_path: str, code_content: str, fix_plan: str) -> str:
        prompt = PATCH_PROMPT.format(
            problem_statement=instance.problem_statement[:4000],
            fix_plan=fix_plan[:2000],
            file_path=file_path,
            code_content=code_content[:50000]  # Increased for complex files
        )
        return self.llm.generate(prompt, PATCH_SYSTEM, 0.0, 4096)

    def _rectify(self, instance, previous_patch: str, test_output: str, file_content: str) -> str:
        prompt = RECTIFICATION_PROMPT_TEMPLATE.format(
            problem_statement=instance.problem_statement,
            previous_patch=previous_patch,
            test_output=test_output[:5000],
            file_content=file_content[:50000]
        )
        return self.llm.generate(prompt, RECTIFICATION_SYSTEM_PROMPT, 0.1, 4096)

    def _phase_solution(self, instance: SWEBenchInstance, repo_path: str, target_file: str) -> Optional[str]:
        # 1. Read Code
        target_path = Path(repo_path) / target_file
        if not target_path.exists():
            found = self._find_file(repo_path, Path(target_file).name)
            if found: target_file = found; target_path = Path(repo_path) / found
            else: return None
            
        code_content = target_path.read_text()
        
        # 2. Plan
        grep_results = "" # Skipped robust grep for now
        fix_plan = self._phase_plan(instance, target_file, code_content, grep_results)
        
        # 3. Generate Patch (S/R)
        response = self._phase_patch(instance, target_file, code_content, fix_plan)
        new_content = self._apply_search_replace(code_content, response)
        
        # 4. Create Diff
        def get_diff(orig, new, fname):
            diff_lines = difflib.unified_diff(
                orig.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"a/{fname}",
                tofile=f"b/{fname}"
            )
            return "".join(diff_lines)
            
        patch = get_diff(code_content, new_content, target_file)
        
        # 5. Iterative Verification & Rectification
        if patch and self.harness:
            print(f"DEBUG: Agentic Verification of {target_file}...")
            success, logs = self.harness.verify_patch_with_logs(instance, patch)
            if not success:
                 print(f"DEBUG: Verification Failed. Logic Error likely. Rectifying...")
                 rectify_response = self._rectify(instance, patch, logs, new_content)
                 final_content = self._apply_search_replace(new_content, rectify_response)
                 patch = get_diff(code_content, final_content, target_file)
        
        # Return final patch or None
        return patch if patch else None

    # --- MAIN RUN METHOD ---

    def _phase_localize_candidates(self, instance: SWEBenchInstance, repo_path: str) -> List[str]:
        """Enhanced localization with multiple fallback strategies."""
        candidates = set()
        
        # 1. Stacktrace extraction (highest priority)
        for fp in extract_stacktrace_files(instance.problem_statement):
            if (Path(repo_path) / fp).exists(): 
                candidates.add(fp)
        
        if candidates:
            print(f"DEBUG: Found {len(candidates)} stacktrace candidates")
            return list(candidates)
        
        # 2. Explicit file paths in problem statement
        for fp in extract_file_paths(instance.problem_statement):
            if (Path(repo_path) / fp).exists():
                candidates.add(fp)
        
        if candidates:
            print(f"DEBUG: Found {len(candidates)} explicit path candidates")
            return list(candidates)
        
        # 3. Module paths converted to file paths
        for mp in extract_module_paths(instance.problem_statement):
            fp = module_to_filepath(mp)
            if (Path(repo_path) / fp).exists():
                candidates.add(fp)
            # Try with leading dirs
            for prefix in ['src', 'lib', '']:
                candidate = f"{prefix}/{fp}" if prefix else fp
                if (Path(repo_path) / candidate).exists():
                    candidates.add(candidate)
        
        if candidates:
            print(f"DEBUG: Found {len(candidates)} module path candidates")
            return list(candidates)
        
        # 4. LLM-Assisted Localization (SOTA fallback)
        try:
            file_list = self._get_file_list(repo_path)
            prompt = LLM_LOCALIZATION_PROMPT.format(
                problem_statement=instance.problem_statement[:3000],
                file_list=file_list[:4000]
            )
            response = self.llm.generate(prompt, LLM_LOCALIZATION_SYSTEM, 0.0, 1024)
            
            # Parse JSON response
            # Extract JSON from response (may have markdown wrappers)
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                loc_data = json.loads(json_match.group())
                search_terms = loc_data.get('search_terms', [])
                
                # Grep for each search term
                for term in search_terms[:3]:
                    if len(term) < 4: continue
                    try:
                        result = subprocess.run(
                            ["grep", "-r", "-l", f"(class|def) {term}", "."],
                            cwd=repo_path, capture_output=True, text=True, timeout=10
                        )
                        if result.returncode == 0 and result.stdout:
                            for line in result.stdout.splitlines()[:3]:
                                if line.endswith('.py') and 'test' not in line.lower():
                                    candidates.add(line.lstrip('./'))
                    except: pass
        except Exception as e:
            print(f"DEBUG: LLM localization failed: {e}")
        
        if candidates:
            print(f"DEBUG: Found {len(candidates)} LLM-assisted candidates")
            return list(candidates)
            
        # 5. ReAct exploration (last resort)
        react = self._react_localize(instance, repo_path, candidates=list(candidates))
        if react: 
            candidates.update(react)
            
        return list(candidates)


    def run(self, instance: SWEBenchInstance) -> PipelineResult:
        repo_path = self.repo_manager.get_repo_path(instance.repo, instance.base_commit)
        
        # 1. Localize
        candidates = self._phase_localize_candidates(instance, repo_path)
        if not candidates:
            return PipelineResult(
                instance_id=instance.instance_id,
                generated_patch="",
                ground_truth_patch=instance.patch,
                raw_response="",
                success=False,
                error="Localization failed"
            )
            
        print(f"DEBUG: Candidates: {candidates}")
        
        last_error = ""
        for i, target_file in enumerate(candidates[:2]): 
            print(f"DEBUG: Trying candidate {target_file}")
            patch = self._phase_solution(instance, repo_path, target_file)
            
            if patch:
                return PipelineResult(
                    instance_id=instance.instance_id,
                    generated_patch=patch,
                    ground_truth_patch=instance.patch,
                    success=True,
                    raw_response=f"Fixed {target_file}"
                )
            else:
                last_error = f"Failed to generate patch for {target_file}"
                
        return PipelineResult(
            instance_id=instance.instance_id,
            generated_patch="",
            ground_truth_patch=instance.patch,
            raw_response="",
            success=False,
            error=last_error
        )
