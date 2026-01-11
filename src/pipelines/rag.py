"""
RAG Pipeline (Variant 2)

Retrieval-Augmented Generation pipeline for bug fixing.
Retrieves relevant documentation before generating patches.
"""

from dataclasses import dataclass
from typing import Optional
from src.data import SWEBenchInstance
from src.llm import LLMProvider
from src.retrieval import DocumentCorpus, HybridRetriever
from src.retrieval.corpus import create_sample_corpus
from .baseline import BaselinePipeline, PipelineResult


RAG_SYSTEM_PROMPT = """You are an expert software engineer specializing in debugging and fixing Python code.
You have access to relevant documentation that may help you understand the issue.

When generating a patch:
1. Review the provided documentation carefully
2. Analyze the bug report and any error traces
3. Identify the root cause based on documentation insights
4. Generate a minimal, focused fix

Output format:
```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
 context line
-removed line
+added line
 context line
```
"""

RAG_PROMPT_TEMPLATE = """## Relevant Documentation

{retrieved_context}

---

## Bug Report

{problem_statement}

{hints_section}

## Task

Using the documentation above as reference, analyze this bug and generate a patch to fix it.
Provide your fix as a unified diff.
"""


class RAGPipeline(BaselinePipeline):
    """
    RAG pipeline that retrieves relevant documentation before generation.
    Extends the baseline with retrieval augmentation.
    """
    
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        retriever: Optional[HybridRetriever] = None,
        corpus: Optional[DocumentCorpus] = None,
        retrieval_k: int = 5,
        retrieval_method: str = "hybrid",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        super().__init__(llm_provider, temperature, max_tokens)
        
        # Initialize corpus and retriever
        if corpus is None:
            print("Creating sample corpus...")
            corpus = create_sample_corpus()
        
        if retriever is None:
            self.retriever = HybridRetriever(corpus)
        else:
            self.retriever = retriever
            
        self.retrieval_k = retrieval_k
        self.retrieval_method = retrieval_method
    
    def run(self, instance: SWEBenchInstance) -> PipelineResult:
        """
        Run the RAG pipeline on a single bug instance.
        
        1. Retrieve relevant documents using bug description
        2. Include retrieved context in prompt
        3. Generate patch
        """
        try:
            # Step 1: Retrieve relevant documents
            query = instance.problem_statement
            results = self.retriever.search(
                query,
                top_k=self.retrieval_k,
                method=self.retrieval_method
            )
            
            retrieved_context = self.retriever.format_context(results)
            
            # Step 2: Build the prompt with retrieved context
            hints_section = ""
            if instance.hints_text:
                hints_section = f"## Hints\n\n{instance.hints_text}"
            
            prompt = RAG_PROMPT_TEMPLATE.format(
                retrieved_context=retrieved_context,
                problem_statement=instance.problem_statement,
                hints_section=hints_section,
            )
            
            # Step 3: Generate response
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=RAG_SYSTEM_PROMPT,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Extract patch from response
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
