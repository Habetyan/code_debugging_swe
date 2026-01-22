import pickle
import faiss
from pathlib import Path
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from typing import List, Set, Optional

class ExampleRetriever:
    """
    Retrieves similar solved bugs from a configurable corpus.
    Supports dynamic corpus injection to prevent data leakage.
    """
    
    def __init__(
        self,
        cache_dir: str = "cache/example_retriever_full",
        exclude_ids: Optional[Set[str]] = None,
        use_cache: bool = True
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.cache_dir / "faiss_index.bin"
        self.data_path = self.cache_dir / "examples.pkl"
        
        self.model_name = "BAAI/bge-base-en-v1.5"
        self.encoder = None  
        
        self.index = None
        self.examples = []
        self.exclude_ids = exclude_ids or set()
        self.use_cache = use_cache
        
        self._initialize()

    def _initialize(self):
        """Load from cache or build fresh index."""
        if self.use_cache and self.index_path.exists() and self.data_path.exists():
            print("Loading example index from cache...")
            self.index = faiss.read_index(str(self.index_path))
            with open(self.data_path, 'rb') as f:
                self.examples = pickle.load(f)
            
            # Filter out excluded IDs from cached examples
            if self.exclude_ids:
                self._filter_excluded_ids()
        else:
            print("Building example index (this may take a while)...")
            self._build_index_from_train()

    def _filter_excluded_ids(self):
        """Removing excluded IDs from loaded cache (for leakage prevention)."""
        before = len(self.examples)
        self.examples = [e for e in self.examples if e['instance_id'] not in self.exclude_ids]
        filtered = before - len(self.examples)
        if filtered > 0:
            print(f"Filtered {filtered} examples to prevent leakage.")
            # Rebuild index with filtered examples
            self._rebuild_index()

    def _rebuild_index(self):
        """Rebuilding FAISS index from current examples."""
        if not self.examples:
            return
        
        if not self.encoder:
            self.encoder = SentenceTransformer(self.model_name, trust_remote_code=True)
        
        texts = [f"{e['repo']} {e['problem_statement']}" for e in self.examples]
        embeddings = self.encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

            
    def _build_index_from_train(self):
        """Building index from full SWE-bench training set."""
        print("  Loading SWE-bench (Full) dataset...")
        train_dataset = load_dataset("princeton-nlp/SWE-bench", split="train")
        
        filtered_examples = []
        texts = []
        
        print(f"  Processing {len(train_dataset)} train instances...")
        for item in train_dataset:
            if item['instance_id'] in self.exclude_ids:
                continue
            if item.get('patch'):
                filtered_examples.append({
                    'instance_id': item['instance_id'],
                    'repo': item['repo'],
                    'problem_statement': item['problem_statement'],
                    'patch': item['patch']
                })
                text = f"{item['repo']} {item['problem_statement']}"
                texts.append(text)
        
        print(f"  Retained {len(filtered_examples)} examples.")
        
        # Encoding
        print(f"  Encoding with {self.model_name}...")
        self.encoder = SentenceTransformer(self.model_name, trust_remote_code=True)
        embeddings = self.encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Build FAISS Index
        print("  Building FAISS index...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        # Save
        self.examples = filtered_examples
        
        faiss.write_index(self.index, str(self.index_path))
        with open(self.data_path, 'wb') as f:
            pickle.dump(self.examples, f)
            
        print("  Index built and saved.")

    def set_exclude_ids(self, exclude_ids: Set[str]):
        """Update exclusion set and rebuild index."""
        self.exclude_ids = exclude_ids
        self._filter_excluded_ids()

    def retrieve(self, query: str, repo: str, k: int = 3, prefer_same_repo: bool = True) -> List[dict]:
        """
        Retrieve k most similar examples.

        Args:
            query: The problem statement to search for
            repo: The repository name (e.g., 'django/django')
            k: Number of examples to retrieve
            prefer_same_repo: If True, prioritize examples from the same repo
        """
        if not self.index or self.index.ntotal == 0:
            return []

        if not self.encoder:
            self.encoder = SentenceTransformer(self.model_name, trust_remote_code=True)

        search_text = f"{repo} {query}"

        query_vec = self.encoder.encode([search_text], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)

        # Retrieve more candidates for filtering
        retrieve_k = k * 5 if prefer_same_repo else k
        scores, indices = self.index.search(query_vec, min(retrieve_k, self.index.ntotal))

        all_results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.examples):
                example = self.examples[idx]
                all_results.append({
                    'instance_id': example['instance_id'],
                    'repo': example['repo'],
                    'problem_statement': example['problem_statement'],
                    'patch': example['patch'],
                    'score': float(score)
                })

        if not prefer_same_repo:
            return all_results[:k]

        # Prioritize same-repo examples
        same_repo = [r for r in all_results if r['repo'] == repo]
        other_repo = [r for r in all_results if r['repo'] != repo]

        # Also consider related repos (same org or similar name)
        repo_name = repo.split('/')[-1] if '/' in repo else repo
        related_repo = [r for r in other_repo
                        if repo_name.lower() in r['repo'].lower()]
        unrelated = [r for r in other_repo if r not in related_repo]

        # Combine: same repo first, then related, then others
        results = same_repo + related_repo + unrelated
        return results[:k]

