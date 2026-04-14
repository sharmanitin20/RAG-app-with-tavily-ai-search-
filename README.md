# Uploaded Files RAG

Streamlit RAG app that:

- uploads PDF, DOCX, Markdown, and TXT files
- builds a FAISS index with Hugging Face embeddings
- answers grounded questions from uploaded files
- optionally combines Tavily web search results with document context

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Required secrets

Create a `.env` file locally, or add these as Hugging Face Space secrets:

```env
GROQ_API_KEY=...
TAVILY_API_KEY=...
```

Optional settings:

```env
GROQ_MODEL=llama-3.1-8b-instant
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RETRIEVAL_K=4
CHUNK_SIZE=900
CHUNK_OVERLAP=150
```

## Hugging Face Spaces

This repo is set up as a Streamlit app with `app.py` as the entry point.

After creating a new Streamlit Space, push this repository and add the required secrets in the Space settings:

- `GROQ_API_KEY`
- `TAVILY_API_KEY`

The app stores uploaded files and the FAISS index inside the app filesystem at runtime, so those generated files should not be committed.
