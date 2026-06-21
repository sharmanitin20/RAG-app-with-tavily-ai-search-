"""
LLM-based RAG evaluation metrics.

Scores three dimensions after every answered query:
  • Faithfulness       — are all claims in the answer supported by the context?
  • Context Relevancy  — are the retrieved chunks actually relevant to the question?
  • Answer Relevance   — does the answer address what was asked?

Each score is 0.0–1.0.  Results are appended to a JSON-lines log file so the
Eval tab can show per-query history and running averages.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# ── Storage ───────────────────────────────────────────────────────────────────

EVAL_LOG_PATH = Path(__file__).resolve().parent.parent / "storage" / "eval_log.jsonl"


# ── Prompt ────────────────────────────────────────────────────────────────────

_EVAL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an evaluation system for a RAG (Retrieval-Augmented Generation) pipeline. "
        "Score each dimension from 0 to 10 as an integer only. Be strict and objective.",
    ),
    (
        "user",
        "Question:\n{question}\n\n"
        "Retrieved Context:\n{context}\n\n"
        "Answer:\n{answer}\n\n"
        "Rate the following dimensions:\n"
        "1. FAITHFULNESS (0-10): Are ALL claims in the Answer directly and explicitly "
        "supported by the Context? Deduct points for anything fabricated or extrapolated.\n"
        "2. CONTEXT_RELEVANCY (0-10): How relevant are the retrieved Context chunks "
        "to answering the Question? Deduct points for off-topic or tangential chunks.\n"
        "3. ANSWER_RELEVANCE (0-10): How well does the Answer actually address "
        "the Question? Deduct points for evasive, incomplete, or off-topic answers.\n\n"
        "Reply in exactly this format (integers 0-10, nothing else):\n"
        "FAITHFULNESS: <int>\n"
        "CONTEXT_RELEVANCY: <int>\n"
        "ANSWER_RELEVANCE: <int>",
    ),
])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_llm() -> ChatGroq:
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY", ""),
    )


def _parse_scores(text: str) -> dict[str, float]:
    scores: dict[str, float] = {
        "faithfulness": 0.0,
        "context_relevancy": 0.0,
        "answer_relevance": 0.0,
    }
    mapping = {
        "FAITHFULNESS": "faithfulness",
        "CONTEXT_RELEVANCY": "context_relevancy",
        "ANSWER_RELEVANCE": "answer_relevance",
    }
    for line in text.splitlines():
        for key, field in mapping.items():
            if line.strip().upper().startswith(f"{key}:"):
                try:
                    raw = int(line.split(":", 1)[1].strip())
                    scores[field] = round(min(max(raw, 0), 10) / 10.0, 2)
                except ValueError:
                    pass
    return scores



async def evaluate_and_log(
    question: str,
    answer: str,
    contexts: List[str],
) -> dict[str, float]:
    """
    Compute faithfulness / context_relevancy / answer_relevance for one Q&A turn,
    append the result to the eval log, and return the scores dict.

    Silently returns zero scores on any LLM error so the main flow is never blocked.
    """
    # Cap context to avoid token overflow on large retrievals
    context_text = "\n\n---\n\n".join(contexts[:4])

    try:
        llm = _get_llm()
        messages = _EVAL_PROMPT.format_messages(
            question=question,
            context=context_text,
            answer=answer,
        )
        response = await llm.ainvoke(messages)
        text = response.content if isinstance(response.content, str) else str(response.content)
        scores = _parse_scores(text)
    except Exception:
        scores = {"faithfulness": 0.0, "context_relevancy": 0.0, "answer_relevance": 0.0}

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question[:300],
        "answer": answer[:500],
        "scores": scores,
    }
    EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    return scores


def load_eval_log() -> list[dict]:
    """Return all logged evaluation records, oldest first."""
    if not EVAL_LOG_PATH.exists():
        return []
    records: list[dict] = []
    with open(EVAL_LOG_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def clear_eval_log() -> None:
    """Delete the eval log file."""
    if EVAL_LOG_PATH.exists():
        EVAL_LOG_PATH.unlink()
