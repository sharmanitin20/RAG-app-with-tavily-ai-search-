"""
Hybrid retriever: BM25 sparse search + FAISS dense search,
fused via Reciprocal Rank Fusion (RRF).

BM25 index is built and persisted alongside the FAISS index during ingest.
At query time both rankers run independently and their result lists are merged.
"""
import pickle
from typing import List, Tuple

from langchain_classic.docstore.document import Document

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

from src.utils.helpers import BM25_INDEX_PATH, load_vector_store

# Standard RRF constant — higher k reduces the impact of top-rank gaps
_RRF_K = 60



def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokeniser (lowercase). Sufficient for BM25."""
    return text.lower().split()



def build_bm25_index(chunks: List[Document]) -> None:
    """
    Build a BM25Okapi index from document chunks and persist it to disk.
    Called from ingest.py after chunking.
    No-op if rank_bm25 is not installed.
    """
    if not _BM25_AVAILABLE:
        return

    corpus = [_tokenize(doc.page_content) for doc in chunks]
    bm25 = BM25Okapi(corpus)

    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as fh:
        pickle.dump({"bm25": bm25, "chunks": chunks}, fh)


def _load_bm25_index() -> Tuple["BM25Okapi | None", List[Document]]:
    """Load persisted BM25 index. Returns (None, []) if unavailable."""
    if not _BM25_AVAILABLE or not BM25_INDEX_PATH.exists():
        return None, []
    with open(BM25_INDEX_PATH, "rb") as fh:
        data = pickle.load(fh)
    return data["bm25"], data["chunks"]



def _rrf_fuse(
    dense_docs: List[Document],
    sparse_docs: List[Document],
    k: int = _RRF_K,
) -> List[Document]:
    """
    Merge two ranked lists with Reciprocal Rank Fusion.
    score(d) = Σ  1 / (k + rank_i(d))   for each ranker i
    Documents are de-duplicated by the first 200 chars of their content.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(dense_docs):
        key = doc.page_content[:200]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(sparse_docs):
        key = doc.page_content[:200]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        doc_map[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked]



def hybrid_search(query: str, k: int = 4) -> List[Document]:
    """
    Retrieve top-k documents using hybrid BM25 + FAISS with RRF fusion.
    Falls back to pure dense search if the BM25 index is unavailable
    (e.g. rank_bm25 not installed, or index not yet built).

    Raises ValueError if no FAISS index exists.
    """
    vector_store = load_vector_store()
    if vector_store is None:
        raise ValueError("No vector index found. Upload files and click Build Index first.")

    # Fetch 2× candidates from each ranker before fusing
    dense_docs = vector_store.similarity_search(query, k=k * 2)

    bm25, chunks = _load_bm25_index()
    if bm25 is None or not chunks:
        # Graceful degradation: BM25 unavailable → pure dense
        return dense_docs[:k]

    token_query = _tokenize(query)
    bm25_scores = bm25.get_scores(token_query)
    top_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[: k * 2]
    sparse_docs = [chunks[i] for i in top_indices]

    return _rrf_fuse(dense_docs, sparse_docs)[:k]
