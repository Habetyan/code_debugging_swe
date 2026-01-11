"""
Search-and-Replace Pipeline

Uses a simpler output format that avoids patch alignment issues.
LLM outputs SEARCH/REPLACE blocks instead of unified diffs.
"""

import re
from dataclasses import dataclass
from typing import Optional
from src.data import SWEBenchInstance
from src.llm import LLMProvider
from src.retrieval import HybridRetriever
from src.retrieval.corpus import create_sample_corpus
from src.retrieval.source_code import RepoManager, get_source_context
from .baseline import BaselinePipeline, PipelineResult


SEARCH_REPLACE_SYSTEM = """You are an expert software engineer fixing bugs in Python code.

OUTPUT FORMAT - You MUST use this exact format:

<<<<<<< SEARCH
[paste exact lines from source that need to change]
=======
[the replacement lines]
>>>>>>> REPLACE

CRITICAL RULES:
1. ONLY output SEARCH/REPLACE blocks - no other format
2. Copy SEARCH text character-for-character from the source
3. Keep fixes minimal
4. Do NOT use diff format (no --- or +++ lines)
"""

SEARCH_REPLACE_PROMPT = """## Source Code

{source_code}

## Bug Report

{problem_statement}

{hints_section}

## Instructions

Find the buggy code and fix it. Use ONLY this format:

<<<<<<< SEARCH
paste the exact buggy lines here
=======
the fixed lines here
>>>>>>> REPLACE

Do NOT use unified diff format. ONLY use SEARCH/REPLACE blocks.
"""


@dataclass
class SearchReplaceEdit:
    """A single search/replace edit."""
    search: str
    replace: str
    file_path: Optional[str] = None


class SearchReplacePipeline(BaselinePipeline):
    """
    Pipeline using search-and-replace format instead of unified diffs.
    """
    
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        repo_manager: Optional[RepoManager] = None,
        retriever: Optional[HybridRetriever] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ):
        super().__init__(llm_provider, temperature, max_tokens)
        self.repo_manager = repo_manager or RepoManager()
        
        # Optional doc retriever
        if retriever is None:
            corpus = create_sample_corpus()
            self.retriever = HybridRetriever(corpus)
        else:
            self.retriever = retriever
    
    def run(self, instance: SWEBenchInstance) -> PipelineResult:
        """Run search-and-replace pipeline."""
        try:
            # Get source code
            source_code, files = get_source_context(
                repo=instance.repo,
                commit=instance.base_commit,
                problem_statement=instance.problem_statement,
                patch=instance.patch,
                max_chars=8000,
                repo_manager=self.repo_manager,
            )
            
            if not source_code:
                source_code = "(Source files could not be retrieved)"
            
            # Build prompt
            hints_section = ""
            if instance.hints_text:
                hints_section = f"## Hints\n\n{instance.hints_text}"
            
            prompt = SEARCH_REPLACE_PROMPT.format(
                source_code=source_code,
                problem_statement=instance.problem_statement,
                hints_section=hints_section,
            )
            
            # Generate
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=SEARCH_REPLACE_SYSTEM,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Parse search/replace blocks
            edits = self._parse_search_replace(response)
            
            # Convert to unified diff for compatibility
            generated_patch = self._edits_to_diff(edits, files)
            
            return PipelineResult(
                instance_id=instance.instance_id,
                generated_patch=generated_patch,
                ground_truth_patch=instance.patch,
                raw_response=response,
                success=True,
            )
            
        except Exception as e:
            return PipelineResult(
                instance_id=instance.instance_id,
                generated_patch="",
                ground_truth_patch=instance.patch,
                raw_response="",
                success=False,
                error=str(e),
            )
    
    def _parse_search_replace(self, response: str) -> list[SearchReplaceEdit]:
        """Parse SEARCH/REPLACE blocks from response."""
        edits = []
        
        # Pattern for search/replace blocks
        pattern = r'<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for search, replace in matches:
            edits.append(SearchReplaceEdit(
                search=search,
                replace=replace,
            ))
        
        return edits
    
    def _edits_to_diff(self, edits: list[SearchReplaceEdit], files: list[str]) -> str:
        """Convert search/replace edits to unified diff format."""
        if not edits:
            return ""
        
        # Use first file as target
        file_path = files[0] if files else "unknown.py"
        
        diff_lines = [
            f"--- a/{file_path}",
            f"+++ b/{file_path}",
        ]
        
        for edit in edits:
            search_lines = edit.search.split('\n')
            replace_lines = edit.replace.split('\n')
            
            # Create hunk header (approximate)
            diff_lines.append(f"@@ -1,{len(search_lines)} +1,{len(replace_lines)} @@")
            
            # Add context and changes
            for line in search_lines:
                diff_lines.append(f"-{line}")
            for line in replace_lines:
                diff_lines.append(f"+{line}")
        
        return '\n'.join(diff_lines) + '\n'
    
    def apply_edits(
        self,
        edits: list[SearchReplaceEdit],
        repo_path: str,
        file_path: str
    ) -> tuple[bool, str]:
        """
        Apply search/replace edits directly to a file.
        Returns (success, message).
        """
        from pathlib import Path
        
        full_path = Path(repo_path) / file_path
        if not full_path.exists():
            return False, f"File not found: {file_path}"
        
        content = full_path.read_text()
        original = content
        
        for i, edit in enumerate(edits):
            if edit.search in content:
                content = content.replace(edit.search, edit.replace, 1)
            else:
                return False, f"Edit {i+1}: SEARCH block not found in file"
        
        if content == original:
            return False, "No changes made"
        
        full_path.write_text(content)
        return True, f"Applied {len(edits)} edits"
