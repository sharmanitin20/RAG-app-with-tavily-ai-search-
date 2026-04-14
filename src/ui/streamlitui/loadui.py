import asyncio
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.logic.ingest import run_ingest_async
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


    def _save_files(self, uploaded_files) -> list[str]:
        saved = []
        for uploaded_file in uploaded_files:
            if not is_supported_file(uploaded_file.name):
                st.error(f"Unsupported file type: {uploaded_file.name}")
                continue
            safe_name = sanitize_filename(uploaded_file.name)
            target = UPLOAD_DIR / safe_name
            counter = 1
            while target.exists():
                target = UPLOAD_DIR / f"{target.stem}_{counter}{target.suffix}"
                counter += 1
            target.write_bytes(uploaded_file.getbuffer())
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

    # ── Render ───────────────────────────────────────────────────────────

    def render(self):
        st.set_page_config(
            page_title=self.config.get_page_title(),
            layout="wide",
            page_icon="📚",
        )
        st.header("🤖 " + self.config.get_page_title())
        st.caption(self.config.get_page_caption())

        with st.sidebar:
            st.subheader("Workflow")
            st.info("1️⃣  Upload  →  2️⃣  Build Index  →  3️⃣  Ask")
            st.divider()
            st.caption("Deleting a file clears the index. Re-build after deleting.")

        left, right = st.columns([1, 1.2], gap="large")
        with left:
            self._render_left_column()
        with right:
            self._render_right_column()


    def _render_left_column(self):
        st.subheader("📂 Upload & Manage Files")

        files = st.file_uploader(
            "Choose files (PDF, DOCX, MD, TXT)",
            accept_multiple_files=True,
        )

        if st.button("Save Files", use_container_width=True):
            if files:
                large = [f.name for f in files if f.size > 50 * 1024 * 1024]
                if large:
                    st.warning(f"⚠️ Large file(s): {', '.join(large)} — batched indexing will handle it.")
                saved = self._save_files(files)
                if saved:
                    st.success(f"Saved: {', '.join(saved)}")
            else:
                st.warning("No files selected.")

        st.divider()

        st.markdown("**Files in upload folder:**")
        uploads = self._list_uploads()

        if not uploads:
            st.info("No files uploaded yet.")
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
                st.warning("Upload at least one file before building the index.")
            else:
                try:
                    with st.spinner("Indexing documents…"):
                        stats = asyncio.run(run_ingest_async())
                    if stats and stats.get("chunks", 0) > 0:
                        st.success(
                            f"✅ Indexed **{stats['chunks']}** chunks "
                            f"from **{stats['files']}** file(s)."
                        )
                    else:
                        st.error("No content found — check that your files are readable.")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")


    def _render_right_column(self):
        tab_rag, tab_web = st.tabs(["📄 Ask My Documents", "🌐 Ask AI (Web)"])
        with tab_rag:
            render_rag_tab()
        with tab_web:
            render_web_tab()