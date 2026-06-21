"""
Input guardrails for the RAG pipeline.

Checks that a user question is:
  - Non-empty and within length bounds (fast, no LLM cost)
  - A genuine information-seeking question (LLM check)

Returns (is_valid: bool, reason: str).
is_valid=False means the question should be blocked before hitting retrieval.
"""
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


_GUARD_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a query validator for a document Q&A assistant. "
        "Decide whether the user's question is a valid information-seeking request "
        "that could, in principle, be answered by information stored in documents. "
        "Block questions that are: harmful or abusive, purely social/conversational "
        "(e.g. 'hello', 'thanks'), completely nonsensical, or requests to generate "
        "malicious content. Allow all genuine research or factual questions even if "
        "the documents may not contain the answer.",
    ),
    (
        "user",
        "Question: {question}\n\n"
        "Reply in exactly this format:\n"
        "VALID: yes|no\n"
        "REASON: <one sentence — 'ok' if valid>",
    ),
])



def _get_llm() -> ChatGroq:
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY", ""),
    )


def _parse_response(text: str) -> tuple[bool, str]:
    valid = True
    reason = "ok"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("VALID:"):
            valid = stripped.split(":", 1)[1].strip().lower() == "yes"
        elif stripped.upper().startswith("REASON:"):
            reason = stripped.split(":", 1)[1].strip()
    return valid, reason



async def check_input_guardrail(question: str) -> tuple[bool, str]:
    """
    Validate a user question before it enters the retrieval pipeline.

    Returns:
        (True, "ok")                  — question is fine, proceed normally
        (False, "<reason string>")    — question blocked; reason is user-facing
    """
    # --- Fast heuristic checks (no LLM call) ---
    stripped = question.strip()
    if not stripped:
        return False, "Please enter a question."
    if len(stripped) < 3:
        return False, "Question is too short."
    if len(stripped) > 2000:
        return False, "Question exceeds the 2 000-character limit. Please shorten it."

    # --- LLM semantic check ---
    llm = _get_llm()
    messages = _GUARD_PROMPT.format_messages(question=stripped)
    response = await llm.ainvoke(messages)
    text = response.content if isinstance(response.content, str) else str(response.content)
    return _parse_response(text)
