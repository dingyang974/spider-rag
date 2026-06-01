# Streamlit Deployment

## Frontend

Deploy the Streamlit prototype from GitHub:

- Repository: `dingyang974/spider-rag`
- Branch: `main`
- Main file path: `app.py`
- Python runtime: `python-3.11`

## Secrets

Set this in Streamlit Community Cloud app settings:

```toml
DEEPSEEK_API_KEY = "your_deepseek_api_key_here"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# Optional: use this only if a separate FastAPI backend is deployed.
API_BASE_URL = "https://your-fastapi-service.example.com"
```

If `API_BASE_URL` is not set, the Streamlit app falls back to direct in-process RAG: it loads the committed local TF-IDF/FAISS vector store and calls the DeepSeek API from Streamlit.

## Backend API

Streamlit Community Cloud runs the Streamlit app, not the separate FastAPI service. The sample RAG query works online through the direct DeepSeek fallback when `DEEPSEEK_API_KEY` is configured. If you also deploy `api.main:app` to a backend host such as Render, Railway, Fly.io, or another server, paste that public HTTPS URL into `API_BASE_URL`.

Suggested backend start command:

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Required backend environment variables:

```bash
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
DATA_PATH=./data/comments.csv
VECTOR_STORE_PATH=./vector_store
LOG_PATH=./logs
```

The current product prototype does not depend on the API for its main mock-data pages. The API only powers the technical demo in the sample comment library.
