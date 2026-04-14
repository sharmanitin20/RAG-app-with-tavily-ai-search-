
import asyncio

import streamlit as st

from src.logic.rag import answer_with_docs_async
from src.utils.helpers import INDEX_DIR


def render_rag_tab() -> None:
    st.subheader("Ask the RAG Agent")
    query = st.text_area(
        "Your question:",
        placeholder="ASK A QUESTION?",
        height=120,
        key="rag_query",
    )

    if st.button("Get Answer", type="primary", use_container_width=True, key="rag_btn"):
        if not query.strip():
            st.warning("Please enter a question.")
            return

        if not INDEX_DIR.exists():
            st.warning("No index found. Upload files and click Build Index first.")
            return

        with st.spinner("Searching documents…"):
            try:
                answer, sources, contexts = asyncio.run(answer_with_docs_async(query))
            except ValueError as e:
                st.error(str(e))
                return

        st.markdown("### 💬 Answer")
        st.markdown(answer)

        if sources:
            st.markdown("---")
            st.markdown("**📎 Context pulled from:**")
            for src in sources:
                st.markdown(f"- `{src}`")

        if contexts:
            with st.expander("🔍 View retrieved context chunks"):
                for i, chunk in enumerate(contexts, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(chunk[:600] + ("…" if len(chunk) > 600 else ""))