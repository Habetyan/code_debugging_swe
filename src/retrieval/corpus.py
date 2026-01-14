"""
Document Corpus: manages storing and loading of retrieval documents.
Supports adding documents from various sources (source code, SO, etc.) and saving to disk.
"""
from dataclasses import dataclass
from typing import Optional
import json
from pathlib import Path


@dataclass
class Document:
    """A document in the retrieval corpus."""
    doc_id: str
    content: str
    title: str = ""
    source: str = ""  # e.g., "python-docs", "stackoverflow", "github"
    library: str = ""  # e.g., "pandas", "django", "sklearn"
    doc_type: str = "documentation"  # documentation, example, qa
    
    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "title": self.title,
            "source": self.source,
            "library": self.library,
            "doc_type": self.doc_type,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        return cls(**data)


class DocumentCorpus:
    """
    Manages the document corpus for retrieval.
    Supports loading from JSON.
    """
    
    def __init__(self, corpus_path: Optional[str] = None):
        self.documents: list[Document] = []
        self.doc_index: dict[str, Document] = {}
        
        if corpus_path and Path(corpus_path).exists():
            self.load(corpus_path)
    
    def add(self, doc: Document):
        """Add a document to the corpus."""
        self.documents.append(doc)
        self.doc_index[doc.doc_id] = doc
    
    def get(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self.doc_index.get(doc_id)
    
    def save(self, path: str):
        """Save corpus to JSON."""
        data = [doc.to_dict() for doc in self.documents]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load corpus from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.documents = [Document.from_dict(d) for d in data]
        self.doc_index = {doc.doc_id: doc for doc in self.documents}
    
    def __len__(self) -> int:
        return len(self.documents)
    
    def __iter__(self):
        return iter(self.documents)
