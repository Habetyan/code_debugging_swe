import numpy as np
from typing import Optional
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
from .corpus import DocumentCorpus, Document


class HybridRetriever:
    """
    Hybrid retrieval using FAISS (semantic) and BM25 (keyword).
    Combines scores with configurable weights.
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
    ) -> list[tuple[Document, float]]:
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
    
    def _search_embedding(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        """Embedding-based search using FAISS."""
        query_vec = self.encoder.encode([query])
        query_vec = np.array(query_vec).astype('float32')
        faiss.normalize_L2(query_vec)
        
        scores, indices = self.faiss_index.search(query_vec, top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:  # Valid index
                doc = self.corpus.documents[idx]
                results.append((doc, float(score)))
        
        return results
    
    def _search_bm25(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        """Keyword-based search using BM25."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc = self.corpus.documents[idx]
            results.append((doc, float(scores[idx])))
        
        return results
    
    def _search_hybrid(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        """Hybrid search combining embedding and BM25."""
        # Get embedding scores for all docs
        query_vec = self.encoder.encode([query])
        query_vec = np.array(query_vec).astype('float32')
        faiss.normalize_L2(query_vec)
        
        # Search more than needed to ensure good overlap
        n_search = min(len(self.corpus), top_k * 3)
        emb_scores, emb_indices = self.faiss_index.search(query_vec, n_search)
        
        # Get BM25 scores for all docs
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize BM25 scores to [0, 1]
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_scores_norm = bm25_scores / max_bm25
        
        # Combine scores
        combined_scores = {}
        
        # Add embedding scores
        for idx, score in zip(emb_indices[0], emb_scores[0]):
            if idx >= 0:
                combined_scores[idx] = self.embedding_weight * score
        
        # Add BM25 scores
        for idx, score in enumerate(bm25_scores_norm):
            if idx in combined_scores:
                combined_scores[idx] += self.bm25_weight * score
            else:
                combined_scores[idx] = self.bm25_weight * score
        
        # Sort by combined score
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for idx, score in sorted_indices:
            doc = self.corpus.documents[idx]
            results.append((doc, score))
        
        return results
    
    def format_context(self, results: list[tuple[Document, float]], max_chars: int = 2000) -> str:
        """Format retrieved documents into context string for LLM."""
        context_parts = []
        total_chars = 0
        
        for doc, score in results:
            header = f"[{doc.library}] {doc.title}"
            entry = f"{header}\n{doc.content}\n"
            
            if total_chars + len(entry) > max_chars:
                break
                
            context_parts.append(entry)
            total_chars += len(entry)
        
        return "\n---\n".join(context_parts)
