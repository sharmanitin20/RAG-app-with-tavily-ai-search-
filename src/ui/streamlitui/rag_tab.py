import asyncio

import streamlit as st

from src.logic.rag import answer_with_docs_async
from src.utils.helpers import INDEX_DIR


def render_rag_tab() -> None:
    st.subheader("Ask the RAG Agent")
    query = st.text_area(
        "Your question:",
        placeholder="Ask anything about your uploaded documents…",
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

        with st.spinner("Checking query & searching documents…"):
            try:
                answer, sources, contexts, eval_scores = asyncio.run(
                    answer_with_docs_async(query)
                )
            except ValueError as e:
                st.error(str(e))
                return

        if answer.startswith("⚠️ Query blocked:"):
            st.warning(answer)
            return

        st.markdown("### 💬 Answer")
        st.markdown(answer)

        if sources:
            st.markdown("---")
            st.markdown("**📎 Sources:**")
            for src in sources:
                st.markdown(f"- `{src}`")

        if contexts:
            with st.expander("🔍 View retrieved context chunks"):
                for i, chunk in enumerate(contexts, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(chunk[:600] + ("…" if len(chunk) > 600 else ""))

        if eval_scores:
            with st.expander("📊 Quality scores for this answer"):
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Faithfulness", eval_scores["faithfulness"],
                           help="Are all claims supported by the retrieved context?")
                sc2.metric("Context Relevancy", eval_scores["context_relevancy"],
                           help="Were the retrieved chunks relevant to your question?")
                sc3.metric("Answer Relevance", eval_scores["answer_relevance"],
                           help="Does the answer actually address your question?")
                st.caption("Scores 0–1. Full history in the 📊 Evaluation tab.")
