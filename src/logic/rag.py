import asyncio
import os
from typing import List, Tuple

from langchain_classic.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.utils.helpers import load_vector_store
from dotenv import load_dotenv
load_dotenv()

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
        temperature=0,api_key=os.getenv("GROQ_API_KEY", "")
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


async def answer_with_docs_async(question: str) -> Tuple[str, List[str], List[str]]:
    vector_store = load_vector_store()
    if vector_store is None:
        raise ValueError("No vector index found. Upload files and click Build Index first.")

    docs = await asyncio.to_thread(
        vector_store.similarity_search,
        question,
        int(os.getenv("RETRIEVAL_K", "4")),
    )
    if not docs:
        return "I don't know based on the uploaded files.", [], []

    context = _format_context(docs)
    llm = _get_llm()

    draft_messages = PROMPT.format_messages(input=question, context=context)
    draft_response = await llm.ainvoke(draft_messages)
    draft_answer = draft_response.content if isinstance(draft_response.content, str) else str(draft_response.content)

    check_messages = CHECK_PROMPT.format_messages(question=question, context=context, draft=draft_answer)
    check_response = await llm.ainvoke(check_messages)
    check_text = check_response.content if isinstance(check_response.content, str) else str(check_response.content)
    supported, final_answer = _parse_check_response(check_text)

    if not supported:
        final_answer = "I don't know based on the uploaded files."

    sources = sorted({doc.metadata.get("source") for doc in docs if doc.metadata.get("source")})
    contexts = [doc.page_content for doc in docs]
    return final_answer, sources, contexts
