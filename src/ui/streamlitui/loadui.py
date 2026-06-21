import asyncio
import hashlib
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.logic.ingest import run_ingest_async
from src.ui.streamlitui.eval_tab import render_eval_tab
from src.ui.streamlitui.rag_tab import render_rag_tab
from src.ui.streamlitui.uiconfig import Config
from src.ui.streamlitui.web_tab import render_web_tab
from src.utils.helpers import (
    INDEX_DIR,
    UPLOAD_DIR,
    ensure_app_dirs,
    is_supported_file,
    sanitize_filename,
)


class LoadStreamlitUI:
    def __init__(self):
        load_dotenv()
        self.config = Config()
        ensure_app_dirs()

    def _existing_hashes(self) -> dict[str, str]:
        result = {}
        for f in UPLOAD_DIR.iterdir():
            if f.is_file() and not f.name.startswith("."):
                result[hashlib.md5(f.read_bytes()).hexdigest()] = f.name
        return result

    def _save_files(self, uploaded_files) -> list[str]:
        saved = []
        seen_hashes: dict[str, str] = self._existing_hashes()

        for uploaded_file in uploaded_files:
            if not is_supported_file(uploaded_file.name):
                st.error(f"Unsupported file type: `{uploaded_file.name}`")
                continue

            file_bytes = uploaded_file.getbuffer()
            file_hash = hashlib.md5(file_bytes).hexdigest()

            if file_hash in seen_hashes:
                existing_name = seen_hashes[file_hash]
                if existing_name == sanitize_filename(uploaded_file.name):
                    st.warning(f"⏭ Skipped `{uploaded_file.name}` — already uploaded.")
                else:
                    st.warning(
                        f"⏭ Skipped `{uploaded_file.name}` — identical content already "
                        f"exists as `{existing_name}`."
                    )
                continue

            safe_name = sanitize_filename(uploaded_file.name)
            target = UPLOAD_DIR / safe_name
            if target.exists():
                st.warning(
                    f"⏭ Skipped `{uploaded_file.name}` — a file named `{safe_name}` "
                    f"already exists with different content. Delete it first to replace it."
                )
                continue

            target.write_bytes(file_bytes)
            seen_hashes[file_hash] = safe_name
            saved.append(target.name)
        return saved

    def _list_uploads(self) -> list[Path]:
        return sorted(
            [f for f in UPLOAD_DIR.iterdir() if f.is_file() and not f.name.startswith(".")],
            key=lambda f: f.name.lower(),
        )

    def _delete_file(self, file_path: Path) -> None:
        try:
            file_path.unlink(missing_ok=True)
        except Exception as e:
            st.error(f"Could not delete file: {e}")
            return
        if INDEX_DIR.exists():
            shutil.rmtree(INDEX_DIR, ignore_errors=True)
        st.success(f"Deleted **{file_path.name}** and cleared index. Re-build to continue.")

    def _index_status(self) -> tuple[bool, int]:
        index_ready = INDEX_DIR.exists() and any(INDEX_DIR.iterdir())
        file_count = len(self._list_uploads())
        return index_ready, file_count

    def _render_sidebar(self):
        st.title("🧠 DocMind")
        st.caption(self.config.get_page_caption())
        st.divider()

        index_ready, file_count = self._index_status()

        if index_ready and file_count > 0:
            st.success(f"✅ Index ready · {file_count} file{'s' if file_count != 1 else ''}")
        elif file_count > 0:
            st.warning(f"⚠️ {file_count} file{'s' if file_count != 1 else ''} uploaded — build index to query")
        else:
            st.info("Upload files below, then build the index to start querying.")

        st.divider()
        st.subheader("📂 Files")

        uploaded_files = st.file_uploader(
            "PDF, DOCX, MD, TXT",
            accept_multiple_files=True,
        )

        if st.button("Save Files", use_container_width=True):
            if uploaded_files:
                large = [f.name for f in uploaded_files if f.size > 50 * 1024 * 1024]
                if large:
                    st.warning(f"⚠️ Large file(s): {', '.join(large)}")
                saved = self._save_files(uploaded_files)
                if saved:
                    st.success(f"Saved: {', '.join(saved)}")
                    st.rerun()
            else:
                st.warning("No files selected.")

        uploads = self._list_uploads()
        if not uploads:
            st.caption("No files yet.")
        else:
            for f in uploads:
                col_name, col_btn = st.columns([3, 1])
                col_name.markdown(f"📄 `{f.name}`")
                if col_btn.button("🗑", key=f"del_{f.name}", use_container_width=True):
                    self._delete_file(f)
                    st.rerun()

        st.divider()

        if st.button("⚙️ Build Index", use_container_width=True, type="primary"):
            if not self._list_uploads():
                st.warning("Upload at least one file first.")
            else:
                try:
                    with st.spinner("Indexing documents…"):
                        stats = asyncio.run(run_ingest_async())
                    if stats and stats.get("chunks", 0) > 0:
                        st.success(
                            f"✅ {stats['chunks']} chunks from {stats['files']} file(s)"
                        )
                        st.rerun()
                    else:
                        st.error("No content found — check your files are readable.")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    def render(self):
        st.set_page_config(
            page_title=self.config.get_page_title(),
            layout="wide",
            page_icon="🧠",
        )

        with st.sidebar:
            self._render_sidebar()

        tab_rag, tab_web, tab_eval = st.tabs(
            ["📄 Ask My Documents", "🌐 Ask AI (Web)", "📊 Evaluation"]
        )
        with tab_rag:
            render_rag_tab()
        with tab_web:
            render_web_tab()
        with tab_eval:
            render_eval_tab()
