"""
Expand Corpus Script: Fetches data from StackOverflow and builds a retrieval corpus.
"""
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval.so_fetcher import StackOverflowFetcher
from src.retrieval.corpus import DocumentCorpus

API_KEY = "rl_tYzdHFSnfsBgLyum36RkKnoqV"

def main():
    fetcher = StackOverflowFetcher(API_KEY)
    corpus = DocumentCorpus()
    
    # Common libraries in SWE-bench
    libs = [
        'django', 'flask', 'pandas', 'numpy', 'scikit-learn', 'matplotlib', 
        'requests', 'sympy', 'astropy', 'pylint', 'pytest', 'sphinx', 'xarray'
    ]
    
    total_docs = 0
    for lib in libs:
        print(f"Fetching {lib}...")
        # Fetching 50 pages (1000 items) per lib to get comprehensive data
        docs = fetcher.fetch_questions(lib, pages=50) 
        for doc in docs:
            corpus.add(doc)
        print(f"  Got {len(docs)} docs")
        total_docs += len(docs)
            
    print(f"Collected {len(corpus)} documents.")
    
    os.makedirs("cache", exist_ok=True)
    corpus_path = "cache/expanded_corpus.json"
    corpus.save(corpus_path)
    print(f"Saved to {corpus_path}")

if __name__ == "__main__":
    main()
