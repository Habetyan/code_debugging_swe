import os
import pickle
import faiss
import numpy as np
from pathlib import Path
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class ExampleRetriever:
    """
    Retrieves similar solved bugs from the full SWE-bench dataset
    (excluding the Lite test set to prevent leakage).
    """
    
    def __init__(self, cache_dir: str = "cache/example_retriever"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.cache_dir / "faiss_index.bin"
        self.data_path = self.cache_dir / "examples.pkl"
        
        self.model_name = "all-MiniLM-L6-v2"
        self.encoder = None # Lazy load
        
        self.index = None
        self.examples = []
        
        self._initialize()

    def _initialize(self):
        """Load or build the index."""
        if self.index_path.exists() and self.data_path.exists():
            print("Loading example index from cache...")
            self.index = faiss.read_index(str(self.index_path))
            with open(self.data_path, 'rb') as f:
                self.examples = pickle.load(f)
        else:
            print("Building example index (this may take a while)...")
            self._build_index()

    def _build_index(self):
        """Download data, filter, encode, and build index."""
        # 1. Load Datasets
        print("  Loading SWE-bench datasets...")
        full_dataset = load_dataset("princeton-nlp/SWE-bench", split="test")
        lite_dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        
        # 2. Filter out Lite instances
        lite_ids = set(item['instance_id'] for item in lite_dataset)
        
        filtered_examples = []
        texts = []
        
        print(f"  Filtering {len(full_dataset)} instances...")
        for item in full_dataset:
            if item['instance_id'] not in lite_ids:
                # We only want instances that have a patch
                if item.get('patch'):
                    filtered_examples.append({
                        'instance_id': item['instance_id'],
                        'repo': item['repo'],
                        'problem_statement': item['problem_statement'],
                        'patch': item['patch']
                    })
                    # Combine repo name and problem for better semantic search
                    text = f"{item['repo']} {item['problem_statement']}"
                    texts.append(text)
        
        print(f"  Retained {len(filtered_examples)} examples (excluded {len(lite_ids)} Lite instances).")
        
        # 3. Encode
        print(f"  Encoding {len(texts)} examples with {self.model_name}...")
        self.encoder = SentenceTransformer(self.model_name)
        embeddings = self.encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # 4. Build FAISS Index
        print("  Building FAISS index...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        # 5. Save
        self.examples = filtered_examples
        
        faiss.write_index(self.index, str(self.index_path))
        with open(self.data_path, 'wb') as f:
            pickle.dump(self.examples, f)
            
        print("  Index built and saved.")

    def retrieve(self, query: str, repo: str, k: int = 3) -> list[dict]:
        """
        Retrieve k most similar examples.
        """
        if not self.encoder:
            self.encoder = SentenceTransformer(self.model_name)
            
        # Enhance query with repo name
        search_text = f"{repo} {query}"
        
        query_vec = self.encoder.encode([search_text], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        
        scores, indices = self.index.search(query_vec, k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                example = self.examples[idx]
                results.append({
                    'instance_id': example['instance_id'],
                    'repo': example['repo'],
                    'problem_statement': example['problem_statement'],
                    'patch': example['patch'],
                    'score': float(score)
                })
                
        return results
