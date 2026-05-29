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
API_BASE_URL = "https://your-fastapi-service.example.com"
```

If `API_BASE_URL` is not set, the app falls back to `http://localhost:8000`, which only works for local development.

## Backend API

Streamlit Community Cloud runs the Streamlit app, not the separate FastAPI service. For the sample RAG query to work online, deploy `api.main:app` to a backend host such as Render, Railway, Fly.io, or another server, then paste that public HTTPS URL into `API_BASE_URL`.

Suggested backend start command:

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Required backend environment variables:

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo
DATA_PATH=./data/comments.csv
VECTOR_STORE_PATH=./vector_store
LOG_PATH=./logs
```

The current product prototype does not depend on the API for its main mock-data pages. The API only powers the technical demo in the sample comment library.
