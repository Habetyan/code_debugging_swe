"""
RepoFileRetriever: Embeds and searches Python files from a cloned repository.
Uses hybrid search (BM25 + embedding) with file-level aggregation.
Supports caching to avoid re-indexing the same repo.
"""
import os
import pickle
import hashlib
import numpy as np
import faiss
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


@dataclass
class CodeChunk:
    """A chunk of code from a repository file."""
    file_path: str  
    content: str  
    start_line: int
    end_line: int


class RepoFileRetriever:
    """
    Embeds and searches Python files from a cloned repository.
    Uses chunk-based embedding with file-level score aggregation.

    Usage:
        retriever = RepoFileRetriever()
        retriever.index_repo("/path/to/cloned/repo")
        results = retriever.search("error handling in database connection", top_k=5)
        # results: [(file_path, score), ...]
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        chunk_size: int = 1500,  
        chunk_overlap: int = 0,
        cache_dir: str = "cache/repo_embeddings",
        use_cache: bool = True,
    ):
        """
        Args:
            embedding_model: SentenceTransformer model name
            chunk_size: Characters per chunk (512 tokens ≈ 2000 chars, use 1500 for safety)
            chunk_overlap: Overlap between chunks (0 = no overlap like Agentless)
            cache_dir: Directory to store cached embeddings
            use_cache: Whether to use caching
        """
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache

        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.encoder: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[CodeChunk] = []  # Store chunks for retrieval
        self.file_to_chunk_indices: Dict[str, List[int]] = {}  # Map file -> chunk indices

        # BM25 index for hybrid search
        self.bm25: Optional[BM25Okapi] = None
        self.file_paths: List[str] = []  # File paths for BM25 results
        self.file_contents: Dict[str, str] = {}  # Store file contents for BM25

    def _load_encoder(self):
        """Lazy load the encoder."""
        if self.encoder is None:
            print(f"Loading embedding model: {self.embedding_model_name}...")
            self.encoder = SentenceTransformer(self.embedding_model_name, trust_remote_code=True)

    def _get_cache_key(self, repo_path: str) -> str:
        """Generate cache key from repo path (includes commit hash in dir name)."""
        # repo_cache/astropy__astropy__d16bfe05 -> astropy__astropy__d16bfe05
        repo_dir = Path(repo_path).name
        return hashlib.md5(f"{repo_dir}_{self.embedding_model_name}".encode()).hexdigest()[:16]

    def _get_cache_paths(self, cache_key: str) -> Tuple[Path, Path, Path]:
        """Return paths for cached data."""
        faiss_path = self.cache_dir / f"{cache_key}_faiss.bin"
        data_path = self.cache_dir / f"{cache_key}_data.pkl"
        bm25_path = self.cache_dir / f"{cache_key}_bm25.pkl"
        return faiss_path, data_path, bm25_path

    def _save_to_cache(self, cache_key: str):
        """Save index and metadata to cache."""
        faiss_path, data_path, bm25_path = self._get_cache_paths(cache_key)

        # Save FAISS index
        faiss.write_index(self.index, str(faiss_path))

        # Save chunks and file data
        data = {
            'chunks': self.chunks,
            'file_to_chunk_indices': self.file_to_chunk_indices,
            'file_contents': self.file_contents,
            'file_paths': self.file_paths,
        }
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)

        # Save BM25
        if self.bm25:
            with open(bm25_path, 'wb') as f:
                pickle.dump(self.bm25, f)

        print(f"  Cached to {cache_key}")

    def _load_from_cache(self, cache_key: str) -> bool:
        """Load index and metadata from cache. Returns True if successful."""
        faiss_path, data_path, bm25_path = self._get_cache_paths(cache_key)

        if not faiss_path.exists() or not data_path.exists():
            return False

        try:
            # Load FAISS index
            self.index = faiss.read_index(str(faiss_path))

            # Load chunks and file data
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            self.chunks = data['chunks']
            self.file_to_chunk_indices = data['file_to_chunk_indices']
            self.file_contents = data['file_contents']
            self.file_paths = data['file_paths']

            # Load BM25 if exists
            if bm25_path.exists():
                with open(bm25_path, 'rb') as f:
                    self.bm25 = pickle.load(f)

            print(f"  Loaded from cache: {len(self.chunks)} chunks")
            return True
        except Exception as e:
            print(f"  Cache load failed: {e}")
            return False

    def _chunk_file(self, content: str, rel_path: str) -> List[CodeChunk]:
        """Split file into chunks by lines, respecting chunk_size."""
        chunks = []
        lines = content.split('\n')

        current_chunk_lines = []
        current_chunk_chars = 0
        start_line = 1

        for i, line in enumerate(lines, 1):
            line_len = len(line) + 1  # +1 for newline

            # If adding this line exceeds chunk_size, save current chunk
            if current_chunk_chars + line_len > self.chunk_size and current_chunk_lines:
                chunk_content = '\n'.join(current_chunk_lines)
                chunks.append(CodeChunk(
                    file_path=rel_path,
                    content=chunk_content,
                    start_line=start_line,
                    end_line=i - 1
                ))

                # Start new chunk
                if self.chunk_overlap > 0:
                    # Keep last N chars worth of lines
                    overlap_chars = 0
                    overlap_lines = []
                    for line in reversed(current_chunk_lines):
                        if overlap_chars + len(line) > self.chunk_overlap:
                            break
                        overlap_lines.insert(0, line)
                        overlap_chars += len(line) + 1
                    current_chunk_lines = overlap_lines
                    current_chunk_chars = overlap_chars
                    start_line = i - len(overlap_lines)
                else:
                    current_chunk_lines = []
                    current_chunk_chars = 0
                    start_line = i

            current_chunk_lines.append(line)
            current_chunk_chars += line_len

        # Last chunk
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            if len(chunk_content.strip()) >= 10:  
                chunks.append(CodeChunk(
                    file_path=rel_path,
                    content=chunk_content,
                    start_line=start_line,
                    end_line=len(lines)
                ))

        return chunks

    def index_repo(self, repo_path: str) -> int:
        """
        Walk the repository, chunk all Python files, and build index.
        Uses caching to avoid re-indexing the same repo.

        Args:
            repo_path: Path to the cloned repository

        Returns:
            Number of chunks indexed
        """
        repo_path = Path(repo_path)

        # Try loading from cache first
        if self.use_cache:
            cache_key = self._get_cache_key(str(repo_path))
            if self._load_from_cache(cache_key):
                return len(self.chunks)

        self._load_encoder()

        self.chunks = []
        self.file_to_chunk_indices = {}
        self.file_contents = {}
        self.file_paths = []
        chunk_texts = []

        # Directories to skip
        skip_dirs = {
            '.git', '__pycache__', 'node_modules', '.venv', 'venv', 'env',
            'build', 'dist', '.eggs', 'site-packages', 'anaconda', 'anaconda3',
            'miniconda', 'miniconda3', '.tox', '.nox', '.pytest_cache',
            'htmlcov', '.mypy_cache', '.ruff_cache', 'eggs', '.hg', '.svn',
            'tests', 'test', 'testing', 'docs', 'doc', 'examples'
        }

        file_count = 0
        for root, dirs, files in os.walk(repo_path):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.endswith('.egg-info')]

            # Also skip if path contains these patterns
            root_lower = root.lower()
            if any(skip in root_lower for skip in ['site-packages', 'anaconda', 'venv', '.egg']):
                continue

            for file in files:
                if file.endswith('.py'):
                    full_path = Path(root) / file
                    rel_path = str(full_path.relative_to(repo_path))

                    try:
                        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()

                        # Skip empty or very small files
                        if len(content.strip()) < 10:
                            continue

                        # Store for BM25
                        self.file_contents[rel_path] = content
                        self.file_paths.append(rel_path)

                        # Chunk the file
                        file_chunks = self._chunk_file(content, rel_path)

                        if file_chunks:
                            # Track which chunk indices belong to this file
                            start_idx = len(self.chunks)
                            self.file_to_chunk_indices[rel_path] = list(range(start_idx, start_idx + len(file_chunks)))

                            for chunk in file_chunks:
                                self.chunks.append(chunk)
                                # Include file path in text for better retrieval
                                text = f"# File: {chunk.file_path}\n{chunk.content}"
                                chunk_texts.append(text)

                            file_count += 1

                        # Limit total files
                        if file_count >= 1000:
                            print(f"Warning: Limiting to {file_count} Python files")
                            break
                    except Exception:
                        pass

            if file_count >= 1000:
                break

        if not self.chunks:
            print("Warning: No Python files found in repo")
            return 0

        print(f"Found {file_count} Python files, {len(self.chunks)} chunks, embedding...")

        # Embed chunks in batches
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i:i + batch_size]
            batch_emb = self.encoder.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            all_embeddings.append(batch_emb)
            if (i // batch_size) % 10 == 0:
                print(f"  Embedded {min(i + batch_size, len(chunk_texts))}/{len(chunk_texts)} chunks")

        embeddings = np.vstack(all_embeddings).astype('float32')

        # Build FAISS index
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        # Build BM25 index
        if self.file_paths:
            corpus = [f"{p} {self.file_contents[p]}".lower().split() for p in self.file_paths]
            self.bm25 = BM25Okapi(corpus)

        print(f"Indexed {len(self.chunks)} chunks from {file_count} files")

        # Save to cache
        if self.use_cache:
            cache_key = self._get_cache_key(str(repo_path))
            self._save_to_cache(cache_key)

        return len(self.chunks)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for files relevant to the query.
        Uses max-score aggregation: each file's score = max score of its chunks.

        Args:
            query: The problem statement or search query
            top_k: Number of files to return

        Returns:
            List of (file_path, score) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        self._load_encoder()

        # Encode query
        query_vec = self.encoder.encode([query], convert_to_numpy=True)
        query_vec = query_vec.astype('float32')
        faiss.normalize_L2(query_vec)

        # Search for more chunks to ensure good file coverage
        n_search = min(top_k * 10, self.index.ntotal)
        scores, indices = self.index.search(query_vec, n_search)

        # Aggregate scores to file level (max score per file)
        file_scores: Dict[str, float] = {}
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.chunks):
                file_path = self.chunks[idx].file_path
                if file_path not in file_scores or score > file_scores[file_path]:
                    file_scores[file_path] = float(score)

        # Sort by score and return top_k files
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_files[:top_k]

    def search_with_chunks(self, query: str, top_k: int = 5) -> List[Tuple[str, float, int, int]]:
        """
        Search and return file paths with the specific chunk locations.

        Returns:
            List of (file_path, score, start_line, end_line) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        self._load_encoder()

        query_vec = self.encoder.encode([query], convert_to_numpy=True)
        query_vec = query_vec.astype('float32')
        faiss.normalize_L2(query_vec)

        n_search = min(top_k * 3, self.index.ntotal)
        scores, indices = self.index.search(query_vec, n_search)

        # Return chunk-level results
        results = []
        seen_files = set()
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.chunks):
                chunk = self.chunks[idx]
                if chunk.file_path not in seen_files:
                    seen_files.add(chunk.file_path)
                    results.append((
                        chunk.file_path,
                        float(score),
                        chunk.start_line,
                        chunk.end_line
                    ))
                    if len(results) >= top_k:
                        break

        return results

    def hybrid_search(self, query: str, top_k: int = 2, bm25_weight: float = 0.1) -> List[Tuple[str, float]]:
        """
        Hybrid search combining BM25 (keyword) and embedding (semantic).
        Uses Reciprocal Rank Fusion to combine rankings.

        Args:
            query: Search query (problem statement)
            top_k: Number of files to return
            bm25_weight: Weight for BM25 (0-1), embedding gets (1-bm25_weight)

        Returns:
            List of (file_path, combined_score) tuples
        """
        # Get BM25 results
        bm25_scores = {}
        if self.bm25 and self.file_paths:
            query_tokens = query.lower().split()
            scores = self.bm25.get_scores(query_tokens)
            for path, score in zip(self.file_paths, scores):
                bm25_scores[path] = score

        # Get embedding results
        embed_scores = {}
        embed_results = self.search(query, top_k=top_k * 5) 
        for path, score in embed_results:
            embed_scores[path] = score

        # Combine using RRF (Reciprocal Rank Fusion)
        k = 60  # RRF constant
        combined_scores = {}

        # Rank BM25 results
        bm25_ranked = sorted(bm25_scores.items(), key=lambda x: -x[1])
        for rank, (path, _) in enumerate(bm25_ranked):
            rrf_score = bm25_weight / (k + rank + 1)
            combined_scores[path] = combined_scores.get(path, 0) + rrf_score

        # Rank embedding results
        embed_ranked = sorted(embed_scores.items(), key=lambda x: -x[1])
        for rank, (path, _) in enumerate(embed_ranked):
            rrf_score = (1 - bm25_weight) / (k + rank + 1)
            combined_scores[path] = combined_scores.get(path, 0) + rrf_score

        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: -x[1])
        return sorted_results[:top_k]
