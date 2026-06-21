import asyncio

import streamlit as st

from src.logic.rag import CHECK_PROMPT, PROMPT, _get_llm, _parse_check_response, answer_with_docs_async
from src.logic.web_search import is_tavily_available, tavily_search
from src.utils.helpers import INDEX_DIR


def _build_combined_context(web_results: list[dict], doc_contexts: list[str]) -> str:
    web_context = "\n\n".join(
        f"[Web Source: {r.get('url', '')}]\n{r.get('content', '')}"
        for r in web_results
    )
    if doc_contexts:
        doc_text = "\n\n".join(doc_contexts[:2])
        return web_context + f"\n\n[From your documents]\n{doc_text}"
    return web_context


def _call_llm(query: str, context: str) -> str:
    llm = _get_llm()

    draft_msgs = PROMPT.format_messages(input=query, context=context)
    draft_resp = asyncio.run(asyncio.to_thread(llm.invoke, draft_msgs))
    draft = draft_resp.content if isinstance(draft_resp.content, str) else str(draft_resp.content)

    check_msgs = CHECK_PROMPT.format_messages(question=query, context=context, draft=draft)
    check_resp = asyncio.run(asyncio.to_thread(llm.invoke, check_msgs))
    check_text = check_resp.content if isinstance(check_resp.content, str) else str(check_resp.content)

    supported, final_answer = _parse_check_response(check_text)
    return final_answer if supported else draft


def render_web_tab() -> None:
    st.subheader("Ask AI (Web Search)")
    st.caption(
        "Searches the web via Tavily, scrapes each page for real content, "
        "and optionally combines with your uploaded documents."
    )

    web_query = st.text_area(
        "Your question:",
        placeholder="What are the latest updates on this topic?",
        height=120,
        key="web_query",
    )

    use_doc_context = st.checkbox(
        "Also use my uploaded documents as extra context",
        value=True,
        key="use_doc_ctx",
    )

    if not st.button("Search & Answer", type="primary", use_container_width=True, key="web_btn"):
        return

    if not web_query.strip():
        st.warning("Please enter a question.")
        return

    if not is_tavily_available():
        st.error(
            "Tavily not configured. Add `TAVILY_API_KEY` to your .env file."
        )
        return

    with st.spinner("Searching the web…"):
        web_results = tavily_search(web_query)

    if not web_results:
        st.warning("No web results found. Check your TAVILY_API_KEY.")
        return

    doc_contexts = []
    doc_sources = []
    if use_doc_context and INDEX_DIR.exists():
        try:
            _, doc_sources, doc_contexts, _eval = asyncio.run(answer_with_docs_async(web_query))
        except Exception:
            pass

    combined_context = _build_combined_context(web_results, doc_contexts)

    with st.spinner("Generating answer…"):
        try:
            final_answer = _call_llm(web_query, combined_context)
        except Exception as e:
            st.error(f"LLM call failed: {e}")
            return

    st.markdown("### 💬 Answer")
    st.markdown(final_answer)

    st.markdown("---")
    st.markdown("**🌐 Web sources used:**")
    for r in web_results:
        title = r.get("title") or r.get("url", "Unknown")
        url = r.get("url", "#")
        st.markdown(f"- [{title}]({url})")

    if doc_sources:
        st.markdown("**📄 Also used from your documents:**")
        for src in doc_sources:
            st.markdown(f"- `{src}`")
