"""
Hybrid Retriever: combines semantic search (FAISS) and lexical search (BM25).
Supports adaptive weighting and cross-encoder reranking.
"""
import numpy as np
import re
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
from .corpus import DocumentCorpus, Document


class HybridRetriever:
    """
    Hybrid retrieval using FAISS (semantic) and BM25 (keyword).
    Combines scores with configurable weights.
    Supports Cross-Encoder reranking if available.
    """
    
    def __init__(
        self,
        corpus: DocumentCorpus,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ):
        self.corpus = corpus
        self.embedding_weight = embedding_weight
        self.bm25_weight = bm25_weight
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}...")
        self.encoder = SentenceTransformer(embedding_model)
        
        # Initialize Reranker (optional)
        self.reranker = None
        try:
            from sentence_transformers import CrossEncoder
            # Lightweight cross-encoder
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("CrossEncoder reranker loaded.")
        except Exception as e:
            print(f"Reranker not enabled: {e}")
        
        # Build indices
        self._build_indices()
    
    def _build_indices(self):
        """Build FAISS and BM25 indices."""
        if len(self.corpus) == 0:
            raise ValueError("Cannot build indices on empty corpus")
        
        # Extract texts
        texts = [doc.content for doc in self.corpus]
        
        # Build FAISS index
        print("Building FAISS index...")
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        self.embeddings = np.array(embeddings).astype('float32')
        
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product (cosine with normalized)
        faiss.normalize_L2(self.embeddings)  # Normalize for cosine similarity
        self.faiss_index.add(self.embeddings)
        
        # Build BM25 index
        print("Building BM25 index...")
        tokenized = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized)
        
        print(f"Indices built: {len(self.corpus)} documents")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        method: str = "hybrid"
    ) -> List[Tuple[Document, float]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            method: "embedding", "bm25", or "hybrid"
            
        Returns:
            List of (Document, score) tuples
        """
        if method == "embedding":
            return self._search_embedding(query, top_k)
        elif method == "bm25":
            return self._search_bm25(query, top_k)
        else:  # hybrid
            return self._search_hybrid(query, top_k)
    
    def _search_embedding(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        """Embedding-based search using FAISS."""
        query_vec = self.encoder.encode([query])
        query_vec = np.array(query_vec).astype('float32')
        faiss.normalize_L2(query_vec)
        
        scores, indices = self.faiss_index.search(query_vec, top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(self.corpus.documents):  # Valid index with bounds check
                doc = self.corpus.documents[idx]
                results.append((doc, float(score)))
        
        return results
    
    def _search_bm25(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        """Keyword-based search using BM25."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.corpus.documents):  # Bounds check
                doc = self.corpus.documents[idx]
                results.append((doc, float(scores[idx])))
        
        return results
    
    def _search_hybrid(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        """Hybrid search combining embedding and BM25 with adaptive weighting."""

        has_code_identifiers = re.search(r'(_[a-z]+|([a-z]+[A-Z][a-z]+)|`[a-z_]+`)', query)
        
        bm25_w = self.bm25_weight
        emb_w = self.embedding_weight
        
        if has_code_identifiers:
            bm25_w *= 1.5  # Boost exact keyword matches for code
            emb_w *= 0.8   # Reduce semantic drift

        n_candidates = top_k * 5 if self.reranker else min(len(self.corpus), top_k * 3)
        
        # Embedding search
        query_vec = self.encoder.encode([query])
        query_vec = np.array(query_vec).astype('float32')
        faiss.normalize_L2(query_vec)
        emb_scores, emb_indices = self.faiss_index.search(query_vec, min(n_candidates, len(self.corpus)))
        
        # BM25 search
        tokenized_query = query.lower().split()
        bm25_raw_scores = self.bm25.get_scores(tokenized_query)
        max_bm25 = max(bm25_raw_scores) if len(bm25_raw_scores) > 0 and max(bm25_raw_scores) > 0 else 1.0
        bm25_norm_scores = bm25_raw_scores / max_bm25
        
        # Combine scores
        combined_scores = {}
        
        # Add embedding scores (normalized by FAISS usually to [0,1] for cosine)
        for idx, score in zip(emb_indices[0], emb_scores[0]):
            if idx >= 0:
                combined_scores[idx] = emb_w * score

        top_bm25_indices = np.argsort(bm25_raw_scores)[::-1][:n_candidates]
        
        for idx in top_bm25_indices:
            score = bm25_norm_scores[idx]
            if idx in combined_scores:
                combined_scores[idx] += bm25_w * score
            else:
                combined_scores[idx] = bm25_w * score
        
        # Sort by initial hybrid score
        sorted_candidates = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:n_candidates]
        
        # 3. Reranking (if enabled)
        final_results = []
        if self.reranker:
            candidate_docs = [self.corpus.documents[idx] for idx, _ in sorted_candidates if idx < len(self.corpus.documents)]
            pairs = [[query, doc.content] for doc in candidate_docs]
            rerank_scores = self.reranker.predict(pairs)
            
            # Combine hybrid score with reranker score? Or just use reranker?
            # Usually reranker score is authoritative.
            reranked = sorted(zip(candidate_docs, rerank_scores), key=lambda x: x[1], reverse=True)
            
            for doc, score in reranked[:top_k]:
                final_results.append((doc, float(score)))
        else:
            for idx, score in sorted_candidates[:top_k]:
                doc = self.corpus.documents[idx]
                final_results.append((doc, score))
        
        return final_results
    
    def format_context(self, results: List[Tuple[Document, float]], max_chars: int = 2000) -> str:
        """Format retrieved documents into context string for LLM."""
        context_parts = []
        total_chars = 0
        
        for doc, score in results:
            header = f"[{doc.library}] {doc.title} (Score: {score:.2f})"
            entry = f"{header}\n{doc.content}\n"
            
            if total_chars + len(entry) > max_chars:
                break
                
            context_parts.append(entry)
            total_chars += len(entry)
        
        return "\n---\n".join(context_parts)
