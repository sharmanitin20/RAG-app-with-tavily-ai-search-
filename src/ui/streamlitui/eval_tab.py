"""
Evaluation metrics tab.

Shows:
  • Four aggregate metric cards (faithfulness, context relevancy,
    answer relevance, overall) averaged across all logged queries.
  • An expandable history of recent Q&A turns with per-query scores.
  • A button to clear the log.
"""
import streamlit as st

from src.logic.evaluator import clear_eval_log, load_eval_log


def _avg(values: list[float]) -> str:
    if not values:
        return "—"
    return f"{sum(values) / len(values):.2f}"


def render_eval_tab() -> None:
    st.subheader("📊 RAG Evaluation Metrics")
    st.caption(
        "Scores are computed automatically after every RAG answer using the same LLM. "
        "Each dimension is rated 0–1 (higher is better)."
    )

    records = load_eval_log()

    if not records:
        st.info(
            "No evaluations yet. Ask questions in the **Ask My Documents** tab "
            "and scores will appear here automatically."
        )
        return

    # ── Aggregate cards ──────────────────────────────────────────────────────
    faith  = [r["scores"]["faithfulness"]      for r in records]
    ctx    = [r["scores"]["context_relevancy"] for r in records]
    rel    = [r["scores"]["answer_relevance"]  for r in records]
    overall = [(f + c + a) / 3 for f, c, a in zip(faith, ctx, rel)]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Faithfulness",       _avg(faith),   help="Fraction of answer claims supported by retrieved context")
    c2.metric("Context Relevancy",  _avg(ctx),     help="How relevant the retrieved chunks are to the question")
    c3.metric("Answer Relevance",   _avg(rel),     help="How well the answer addresses the question")
    c4.metric("Overall",            _avg(overall), help="Simple average of the three dimensions")
    st.caption(f"Averaged over **{len(records)}** quer{'y' if len(records) == 1 else 'ies'}.")

    st.divider()

    # ── Per-query history (most recent first) ────────────────────────────────
    st.markdown("**Query history** *(last 20)*")

    for rec in reversed(records[-20:]):
        s = rec["scores"]
        label = rec["question"]
        preview = label if len(label) <= 80 else label[:77] + "…"

        with st.expander(f"❓ {preview}"):
            st.markdown(f"**Answer:** {rec['answer'][:400]}{'…' if len(rec['answer']) > 400 else ''}")

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Faithfulness",      s["faithfulness"])
            sc2.metric("Context Relevancy", s["context_relevancy"])
            sc3.metric("Answer Relevance",  s["answer_relevance"])

            ts = rec.get("timestamp", "")
            if ts:
                st.caption(f"🕐 {ts.replace('T', ' ').split('.')[0]} UTC")

    st.divider()

    if st.button("🗑 Clear Evaluation Log", type="secondary", use_container_width=False):
        clear_eval_log()
        st.success("Evaluation log cleared.")
        st.rerun()
