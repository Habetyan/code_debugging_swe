"""
RAG Pipeline with Source Code (Variant 2b)

Enhanced RAG pipeline that includes actual source code in prompts.
This enables the LLM to generate applicable patches.
"""

from dataclasses import dataclass
from typing import Optional
from src.data import SWEBenchInstance
from src.llm import LLMProvider
from src.retrieval import DocumentCorpus, HybridRetriever
from src.retrieval.corpus import create_sample_corpus
from src.retrieval.source_code import RepoManager, get_source_context
from .baseline import BaselinePipeline, PipelineResult


RAG_CODE_SYSTEM_PROMPT = """You are an expert software engineer. You will be given:
1. The actual source code file(s) to modify
2. A bug report describing the issue
3. Optional documentation

Generate a unified diff patch that fixes the bug. The patch MUST match the exact source code provided.

CRITICAL REQUIREMENTS:
- Copy context lines EXACTLY from the source code (no truncation!)
- Each line in the patch must be COMPLETE - never truncate lines
- Include at least 3 lines of context before and after changes
- Use the EXACT line content from the source files
"""

RAG_CODE_PROMPT_TEMPLATE = """## Source Code

{source_code}

---

## Bug Report

{problem_statement}

{hints_section}

{docs_section}

## Task

Based on the source code above, generate a unified diff patch to fix this bug.
The patch must match the actual source code exactly.

```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
 context line (must match source exactly)
-line to remove
+line to add
 context line
```
"""


class RAGWithCodePipeline(BaselinePipeline):
    """
    RAG pipeline that includes actual source code in prompts.
    """
    
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        retriever: Optional[HybridRetriever] = None,
        repo_manager: Optional[RepoManager] = None,
        retrieval_k: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        include_docs: bool = True,
    ):
        super().__init__(llm_provider, temperature, max_tokens)
        
        # Initialize retriever for docs
        self.include_docs = include_docs
        if include_docs:
            if retriever is None:
                corpus = create_sample_corpus()
                self.retriever = HybridRetriever(corpus)
            else:
                self.retriever = retriever
        else:
            self.retriever = None
        
        # Initialize repo manager
        self.repo_manager = repo_manager or RepoManager()
        self.retrieval_k = retrieval_k
    
    def run(self, instance: SWEBenchInstance) -> PipelineResult:
        """
        Run the RAG pipeline with source code context.
        
        1. Clone repo and extract relevant source files
        2. Optionally retrieve documentation
        3. Generate patch against real code
        """
        try:
            # Step 1: Get source code context
            source_code, files = get_source_context(
                repo=instance.repo,
                commit=instance.base_commit,
                problem_statement=instance.problem_statement,
                patch=instance.patch,  # Use ground truth to find files
                max_chars=6000,
                repo_manager=self.repo_manager,
            )
            
            if not source_code:
                # Fallback: no source found, use basic RAG
                source_code = "(Source files could not be retrieved)"
            
            # Step 2: Get documentation (optional)
            docs_section = ""
            if self.include_docs and self.retriever:
                results = self.retriever.search(
                    instance.problem_statement,
                    top_k=self.retrieval_k,
                )
                docs_context = self.retriever.format_context(results, max_chars=1500)
                if docs_context:
                    docs_section = f"## Relevant Documentation\n\n{docs_context}"
            
            # Step 3: Build prompt
            hints_section = ""
            if instance.hints_text:
                hints_section = f"## Hints\n\n{instance.hints_text}"
            
            prompt = RAG_CODE_PROMPT_TEMPLATE.format(
                source_code=source_code,
                problem_statement=instance.problem_statement,
                hints_section=hints_section,
                docs_section=docs_section,
            )
            
            # Step 4: Generate response
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=RAG_CODE_SYSTEM_PROMPT,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Extract patch
            generated_patch = self._extract_patch(response)
            
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
