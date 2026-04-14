import asyncio
import glob
import os
import traceback
from typing import List

from langchain_classic.docstore.document import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.helpers import INDEX_DIR, UPLOAD_DIR, ensure_app_dirs, get_embeddings

DATA_DIR = os.getenv("DATA_DIR", str(UPLOAD_DIR))


def _load_docs(base: str = DATA_DIR) -> List[Document]:
    docs: List[Document] = []

    for path in glob.glob(os.path.join(base, "**", "*"), recursive=True):
        if os.path.isdir(path) or os.path.basename(path).startswith("."):
            continue

        ext = os.path.splitext(path)[1].lower()
        try:
            loaded_docs: List[Document] = []
            if ext == ".md":
                loaded_docs = UnstructuredMarkdownLoader(path).load()
            elif ext == ".pdf":
                loaded_docs = PyMuPDFLoader(path).load()
            elif ext == ".docx":
                loaded_docs = UnstructuredWordDocumentLoader(path).load()
            elif ext == ".txt":
                loaded_docs = TextLoader(path, encoding="utf-8").load()
            else:
                continue

            rel_path = os.path.relpath(path, base)
            for doc in loaded_docs:
                doc.metadata["source"] = rel_path
                doc.metadata["filename"] = os.path.basename(path)
                doc.metadata["extension"] = ext
                docs.append(doc)
        except Exception:
            print(f"INGEST ERROR: failed to load {path}")
            traceback.print_exc()

    return docs


def _chunk(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
    )
    return splitter.split_documents(docs)


def _build_and_save_index(chunks: List[Document]) -> None:
    vector_store = FAISS.from_documents(chunks, get_embeddings())
    INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(INDEX_DIR))


async def run_ingest_async() -> dict:
    ensure_app_dirs()
    docs = await asyncio.to_thread(_load_docs, DATA_DIR)
    if not docs:
        raise ValueError("No supported files found in uploads. Upload PDF, DOCX, MD, or TXT files first.")

    chunks = await asyncio.to_thread(_chunk, docs)
    await asyncio.to_thread(_build_and_save_index, chunks)

    file_count = len({doc.metadata.get("source", "") for doc in docs})
    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "files": file_count,
        "index_path": str(INDEX_DIR),
    }
