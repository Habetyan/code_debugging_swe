"""
CoT-RAG Pipeline (Variant 3)

Chain-of-Thought Retrieval-Augmented Generation pipeline.
Uses reasoning to diagnose bugs and formulate targeted retrieval queries.

Flow:
1. Diagnosis Phase: Analyze bug, identify root cause
2. Query Formulation: Generate specific search queries
3. Targeted Retrieval: Retrieve docs using formulated queries
4. Repair Generation: Generate patch with full context
"""

from dataclasses import dataclass
from typing import Optional
from src.data import SWEBenchInstance
from src.llm import LLMProvider
from src.retrieval import DocumentCorpus, HybridRetriever
from src.retrieval.corpus import create_sample_corpus
from .baseline import BaselinePipeline, PipelineResult


# Step 1: Diagnosis
DIAGNOSIS_PROMPT = """You are an expert debugger. Analyze this bug report step-by-step.

## Bug Report
{problem_statement}

## Task
Provide a structured diagnosis:

1. **Error Type**: What category of error is this? (e.g., TypeError, API misuse, logic error, edge case)
2. **Affected Component**: Which module/class/function is problematic?
3. **Root Cause**: What is the underlying reason for this bug?
4. **Required Knowledge**: What documentation or examples would help fix this?

Be concise and specific.
"""

# Step 2: Query Formulation
QUERY_FORMULATION_PROMPT = """Based on this bug diagnosis, generate 2-3 specific search queries to find relevant documentation.

## Diagnosis
{diagnosis}

## Task
Generate search queries that would find documentation/examples to help fix this bug.
Format: One query per line, no numbering.
Focus on: library names, specific functions, error patterns.

Example queries:
pandas DataFrame merge duplicate column handling
Django admin list_display None value error
"""

# Step 3: Repair Generation
COT_REPAIR_PROMPT = """You are an expert software engineer. Use the diagnosis and documentation to fix this bug.

## Original Bug Report
{problem_statement}

## Diagnosis
{diagnosis}

## Relevant Documentation
{retrieved_context}

## Task
Based on the diagnosis and documentation above, generate a patch to fix this bug.
Explain your reasoning briefly, then provide the fix as a unified diff.

```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
 context line
-removed line
+added line
```
"""


@dataclass
class CoTResult:
    """Result from CoT pipeline including intermediate steps."""
    instance_id: str
    diagnosis: str
    queries: list[str]
    retrieved_docs: list[str]
    generated_patch: str
    ground_truth_patch: str
    raw_response: str
    success: bool = False
    error: Optional[str] = None


class CoTRAGPipeline(BaselinePipeline):
    """
    Chain-of-Thought RAG pipeline.
    
    1. Diagnose the bug (LLM reasoning)
    2. Formulate retrieval queries (LLM)
    3. Retrieve relevant documentation
    4. Generate patch with full context
    """
    
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        retriever: Optional[HybridRetriever] = None,
        corpus: Optional[DocumentCorpus] = None,
        retrieval_k: int = 3,
        queries_per_diagnosis: int = 3,
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
        self.queries_per_diagnosis = queries_per_diagnosis
    
    def _diagnose(self, instance: SWEBenchInstance) -> str:
        """Step 1: Diagnose the bug using CoT reasoning."""
        prompt = DIAGNOSIS_PROMPT.format(
            problem_statement=instance.problem_statement
        )
        
        diagnosis = self.llm.generate(
            prompt=prompt,
            temperature=0.3,  # Lower temperature for analysis
            max_tokens=1024,
        )
        
        return diagnosis
    
    def _formulate_queries(self, diagnosis: str) -> list[str]:
        """Step 2: Generate targeted retrieval queries."""
        prompt = QUERY_FORMULATION_PROMPT.format(diagnosis=diagnosis)
        
        response = self.llm.generate(
            prompt=prompt,
            temperature=0.5,
            max_tokens=256,
        )
        
        # Parse queries (one per line)
        queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
        return queries[:self.queries_per_diagnosis]
    
    def _retrieve_with_queries(self, queries: list[str]) -> list[tuple]:
        """Step 3: Retrieve docs using formulated queries."""
        all_results = []
        seen_ids = set()
        
        for query in queries:
            results = self.retriever.search(query, top_k=self.retrieval_k)
            for doc, score in results:
                if doc.doc_id not in seen_ids:
                    all_results.append((doc, score))
                    seen_ids.add(doc.doc_id)
        
        # Sort by score and return top results
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:self.retrieval_k * 2]
    
    def run(self, instance: SWEBenchInstance) -> PipelineResult:
        """Run the full CoT-RAG pipeline."""
        try:
            # Step 1: Diagnose
            diagnosis = self._diagnose(instance)
            
            # Step 2: Formulate queries
            queries = self._formulate_queries(diagnosis)
            
            # Step 3: Retrieve with formulated queries
            results = self._retrieve_with_queries(queries)
            retrieved_context = self.retriever.format_context(results)
            
            # Step 4: Generate repair
            prompt = COT_REPAIR_PROMPT.format(
                problem_statement=instance.problem_statement,
                diagnosis=diagnosis,
                retrieved_context=retrieved_context,
            )
            
            response = self.llm.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
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
    
    def run_with_details(self, instance: SWEBenchInstance) -> CoTResult:
        """Run pipeline and return detailed intermediate results."""
        try:
            # Step 1: Diagnose
            diagnosis = self._diagnose(instance)
            
            # Step 2: Formulate queries
            queries = self._formulate_queries(diagnosis)
            
            # Step 3: Retrieve
            results = self._retrieve_with_queries(queries)
            retrieved_docs = [doc.title for doc, _ in results]
            retrieved_context = self.retriever.format_context(results)
            
            # Step 4: Generate
            prompt = COT_REPAIR_PROMPT.format(
                problem_statement=instance.problem_statement,
                diagnosis=diagnosis,
                retrieved_context=retrieved_context,
            )
            
            response = self.llm.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            generated_patch = self._extract_patch(response)
            
            return CoTResult(
                instance_id=instance.instance_id,
                diagnosis=diagnosis,
                queries=queries,
                retrieved_docs=retrieved_docs,
                generated_patch=generated_patch,
                ground_truth_patch=instance.patch,
                raw_response=response,
                success=True,
            )
            
        except Exception as e:
            return CoTResult(
                instance_id=instance.instance_id,
                diagnosis="",
                queries=[],
                retrieved_docs=[],
                generated_patch="",
                ground_truth_patch=instance.patch,
                raw_response="",
                success=False,
                error=str(e),
            )
