import os
import re
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
STORAGE_DIR = PROJECT_ROOT / "storage"
INDEX_DIR = STORAGE_DIR / "faiss_index"
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}

def ensure_app_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    return cleaned or "uploaded_file"


def is_supported_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def get_embeddings() -> HuggingFaceEmbeddings:
    model_name = os.getenv(
        "HF_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vector_store() -> FAISS | None:
    if not INDEX_DIR.exists():
        return None
    return FAISS.load_local(
        str(INDEX_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
