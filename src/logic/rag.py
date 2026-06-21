"""
RAG pipeline — answer_with_docs_async

Flow:
  1. Input guardrail  — block harmful / nonsensical queries early
  2. Hybrid retrieval — BM25 + FAISS fused via RRF
  3. Draft answer     — grounded LLM response from retrieved context
  4. Output guardrail — groundedness check; reject unsupported claims
  5. Evaluation       — async metric scoring logged to eval_log.jsonl
"""
import asyncio
import os
from typing import List, Tuple

from langchain_classic.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from src.logic.evaluator import evaluate_and_log
from src.logic.guardrails import check_input_guardrail
from src.logic.hybrid_retriever import hybrid_search

load_dotenv()

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM = """You are a grounded RAG assistant.
Answer only from the uploaded-file context.
If the context is insufficient, say "I don't know based on the uploaded files."
Do not invent facts.
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        (
            "user",
            "Question:\n{input}\n\nRetrieved Context:\n{context}\n\n"
            "Write a direct answer grounded in the context.",
        ),
    ]
)

CHECK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict groundedness checker. "
            "If any important claim in the draft answer is not clearly supported by the context, "
            "mark it unsupported and replace it with a safe grounded answer.",
        ),
        (
            "user",
            "Question:\n{question}\n\n"
            "Retrieved Context:\n{context}\n\n"
            "Draft Answer:\n{draft}\n\n"
            "Return exactly this format:\n"
            "SUPPORTED: yes|no\n"
            "FINAL_ANSWER: <grounded answer only>",
        ),
    ]
)



def _get_llm() -> ChatGroq:
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY", ""),
    )


def _format_context(docs: List[Document]) -> str:
    parts = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", f"chunk-{idx}")
        parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


def _parse_check_response(text: str) -> tuple[bool, str]:
    supported = False
    final_answer = "I don't know based on the uploaded files."
    for line in text.splitlines():
        if line.startswith("SUPPORTED:"):
            supported = line.split(":", 1)[1].strip().lower() == "yes"
        elif line.startswith("FINAL_ANSWER:"):
            final_answer = line.split(":", 1)[1].strip() or final_answer
    return supported, final_answer


# ── Public API ────────────────────────────────────────────────────────────────

async def answer_with_docs_async(
    question: str,
) -> Tuple[str, List[str], List[str], dict | None]:
    """
    Full RAG pipeline with guardrails, hybrid retrieval, and evaluation.

    Returns:
        answer   (str)           — final grounded answer
        sources  (list[str])     — deduplicated source filenames
        contexts (list[str])     — raw retrieved chunk texts
        eval_scores (dict|None)  — faithfulness / context_relevancy / answer_relevance,
                                   or None if the query was blocked by guardrails
    """
    # ── 1. Input guardrail ────────────────────────────────────────────────────
    is_valid, reason = await check_input_guardrail(question)
    if not is_valid:
        blocked_msg = f"⚠️ Query blocked: {reason}"
        return blocked_msg, [], [], None

    # ── 2. Hybrid retrieval ───────────────────────────────────────────────────
    k = int(os.getenv("RETRIEVAL_K", "4"))
    docs = await asyncio.to_thread(hybrid_search, question, k)

    if not docs:
        return "I don't know based on the uploaded files.", [], [], None

    context = _format_context(docs)
    llm = _get_llm()

    # ── 3. Draft answer ───────────────────────────────────────────────────────
    draft_messages = PROMPT.format_messages(input=question, context=context)
    draft_response = await llm.ainvoke(draft_messages)
    draft_answer = (
        draft_response.content
        if isinstance(draft_response.content, str)
        else str(draft_response.content)
    )

    # ── 4. Output guardrail (groundedness check) ──────────────────────────────
    check_messages = CHECK_PROMPT.format_messages(
        question=question, context=context, draft=draft_answer
    )
    check_response = await llm.ainvoke(check_messages)
    check_text = (
        check_response.content
        if isinstance(check_response.content, str)
        else str(check_response.content)
    )
    supported, final_answer = _parse_check_response(check_text)

    if not supported:
        final_answer = "I don't know based on the uploaded files."

    sources = sorted(
        {doc.metadata.get("source") for doc in docs if doc.metadata.get("source")}
    )
    contexts = [doc.page_content for doc in docs]

    # ── 5. Async evaluation (fire-and-forget style — awaited so scores surface in UI) ──
    eval_scores = await evaluate_and_log(question, final_answer, contexts)

    return final_answer, sources, contexts, eval_scores
