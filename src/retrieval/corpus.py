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
    Supports loading from JSON and creating sample corpora.
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


def create_sample_corpus() -> DocumentCorpus:
    """
    Create a sample corpus with common Python library documentation.
    This is a minimal corpus for testing; a full implementation would
    scrape actual documentation.
    """
    corpus = DocumentCorpus()
    
    # Sample documents covering common SWE-bench repositories
    sample_docs = [
        # Django
        Document(
            doc_id="django-admin-1",
            title="Django Admin list_display",
            content="The list_display option in Django admin controls which fields are displayed in the change list page. If a field value is None, it will display an empty string by default. To customize None display, use empty_value_display attribute.",
            source="django-docs",
            library="django",
        ),
        Document(
            doc_id="django-models-1",
            title="Django Model Field Options",
            content="Django model fields support null=True to allow NULL values in the database, and blank=True to allow empty form input. When dealing with None values in admin, ensure proper handling in __str__ methods.",
            source="django-docs",
            library="django",
        ),
        # Pandas
        Document(
            doc_id="pandas-merge-1",
            title="Pandas DataFrame Merge",
            content="pd.merge() combines DataFrames based on common columns. Use 'on' parameter for column names, 'how' for join type (left, right, inner, outer). Handle duplicate columns with suffixes parameter. NaN values may appear when no match exists.",
            source="pandas-docs",
            library="pandas",
        ),
        Document(
            doc_id="pandas-nullable-1",
            title="Pandas Nullable Integer Types",
            content="Pandas supports nullable integer types (Int64, Int32) that can hold NA values. Convert with astype('Int64'). These types support proper NA semantics unlike numpy int types which cannot hold NaN.",
            source="pandas-docs",
            library="pandas",
        ),
        # Scikit-learn
        Document(
            doc_id="sklearn-unique-1",
            title="Scikit-learn unique_labels",
            content="unique_labels() extracts unique labels from y. It handles multiple array-like inputs and validates label consistency. For pandas nullable dtypes, convert to numpy first using .to_numpy() or .values.",
            source="sklearn-docs",
            library="scikit-learn",
        ),
        Document(
            doc_id="sklearn-target-1",
            title="Scikit-learn Target Type Checking",
            content="type_of_target() determines if y is binary, multiclass, continuous, etc. Pass check_unknown=False to skip unknown label checking. Pandas nullable dtypes may cause issues; convert with astype() first.",
            source="sklearn-docs",
            library="scikit-learn",
        ),
        # Matplotlib
        Document(
            doc_id="mpl-version-1",
            title="Matplotlib Version Info",
            content="matplotlib.__version__ returns version string. For programmatic version comparison, use packaging.version.parse() or tuple comparison. Version info can be accessed as (major, minor, micro) tuple.",
            source="matplotlib-docs",
            library="matplotlib",
        ),
        # Requests
        Document(
            doc_id="requests-errors-1",
            title="Requests Exception Handling",
            content="requests library wraps low-level exceptions in requests.exceptions. ConnectionError wraps socket errors. Always catch requests.exceptions.RequestException for broad error handling. For socket.error, import socket module.",
            source="requests-docs",
            library="requests",
        ),
        Document(
            doc_id="requests-streaming-1",
            title="Requests Streaming Responses",
            content="Use stream=True for large responses. Iterate with iter_content() or iter_lines(). Handle socket.error and IncompleteRead exceptions within generator. Wrap in ConnectionError for consistency.",
            source="requests-docs",
            library="requests",
        ),
        # Sympy
        Document(
            doc_id="sympy-units-1",
            title="Sympy Unit Dimensions",
            content="Sympy physics.units module handles dimensional analysis. Use get_dimension_system().equivalent_dims() to compare dimensions. Direct == comparison may fail for equivalent but differently represented dimensions.",
            source="sympy-docs",
            library="sympy",
        ),
        Document(
            doc_id="sympy-dimension-1",
            title="Sympy Dimension Equality",
            content="When comparing dimensions in Sympy, use .equals() method or dimension system's equivalent_dims() for proper dimension equivalence checking. Direct != comparison may give false negatives.",
            source="sympy-docs",
            library="sympy",
        ),
        # Pylint
        Document(
            doc_id="pylint-template-1",
            title="Pylint Message Template",
            content="--msg-template option customizes output format. Use {msg_id}, {symbol}, {msg}, {path}, {line}. Nested braces like {{}} are literal. Regex \\w+? matches word characters; .+? may match too much.",
            source="pylint-docs",
            library="pylint",
        ),
        # General Python
        Document(
            doc_id="python-socket-1",
            title="Python Socket Errors",
            content="socket.error is an alias for OSError. Catch socket.error or OSError for network errors. In libraries, wrap in custom exceptions (e.g., ConnectionError) for cleaner API.",
            source="python-docs",
            library="python",
        ),
        Document(
            doc_id="python-version-1",
            title="Python Version Comparison",
            content="sys.version_info is a named tuple (major, minor, micro, releaselevel, serial). Compare with tuples: sys.version_info >= (3, 8). For package versions, use packaging.version module.",
            source="python-docs",
            library="python",
        ),
    ]
    
    for doc in sample_docs:
        corpus.add(doc)
    
    return corpus
