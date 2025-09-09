# RAG Web API (FastAPI + Supabase)

Retrieval-Augmented Generation (RAG) backend built with FastAPI. It ingests PDFs, chunks and embeds them into a Supabase-backed vector store, and answers questions using OpenAI with both standard and streaming responses. The service also persists basic chat history and ingestion metadata in Postgres (via SQLModel).

## Added Value
- Fast start RAG backend: upload PDFs and query them immediately.
- Managed vector DB via Supabase + pgvector; simple to operate.
- Streaming answers with conversation history support.
- Clean FastAPI interface with OpenAPI docs and typed schemas.
- Async SQLModel access to Postgres for history and ingestion records.

## Technology Stack
- FastAPI + Uvicorn: web framework and ASGI server.
- OpenAI + LangChain: generation and embeddings; `SupabaseVectorStore` for retrieval.
- Supabase (Postgres + pgvector): vector storage for documents.
- SQLModel + asyncpg: async DB access for metadata and history.
- Pydantic Settings: configuration via environment variables.

## Prerequisites
- Python 3.10+ (virtualenv recommended).
- Supabase project (or Postgres with `pgvector` extension enabled) for the `documents` table.
- OpenAI API key.

## Environment Variables
Create a `.env` file in the project root with at least:

```
# Supabase API
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key   # service-role key recommended for inserts
SUPABASE_TABLE=documents         # default used by the app

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-3.5-turbo       # default in config.py
EMBEDDING_MODEL=text-embedding-ada-002

# Postgres (async URL, e.g. Supabase)
# Format: postgresql+asyncpg://USER:PASSWORD@HOST:PORT/DB
POSTGRES_URL=postgresql+asyncpg://... 

# Optional
TOP_K=5
PDF_DIR=pdfs/
```

Notes:
- The app connects with SSL; ensure your Postgres accepts SSL (Supabase does).
- For Supabase vector operations, you may need a service role key unless RLS policies allow inserts.

## Database Schema
You need these tables in Postgres/Supabase. Example SQL (adjust to your environment):

```sql
-- Enable pgvector (on Supabase: `extensions` -> enable `vector`)
create extension if not exists vector;
create extension if not exists pgcrypto; -- for gen_random_uuid()

-- Vector store table that LangChain SupabaseVectorStore expects
create table if not exists public.documents (
  id uuid primary key default gen_random_uuid(),
  content text not null,
  embedding vector(1536),                -- must match EMBEDDING_MODEL dim
  metadata jsonb not null default '{}'::jsonb
);
create index if not exists documents_embedding_idx on public.documents using ivfflat (embedding vector_cosine_ops);

-- Chat history
create table if not exists public.chat_history (
  id bigserial primary key,
  conversation_id uuid not null,
  question text not null,
  answer text not null,
  created_at timestamptz not null default now()
);
create index if not exists chat_history_cid_created_idx on public.chat_history (conversation_id, created_at);

-- PDF ingestion metadata
create table if not exists public.pdf_ingestion (
  id uuid primary key default gen_random_uuid(),
  filename text not null,
  metadata json not null default '{}',
  ingested_at timestamptz not null default now()
);
```

Dimension note: If you switch to a different embedding model (e.g., `text-embedding-3-small` with 1536 dims, or `-large` with 3072), update the `vector(<dims>)` size and reindex.

## How to Run
1) Create and activate a virtual environment:
   `python3 -m venv .venv && source .venv/bin/activate`

2) Install dependencies:
   `pip install -r requirements.txt`

3) Set up `.env` and ensure your Postgres/Supabase tables exist (see schema above).

4) Start the API:
   - `make run`
   - or `uvicorn app:app --reload`

Open the docs at `http://localhost:8000/docs`.

## API Overview
- `POST /v1/upload` — Upload a PDF; chunks and stores embeddings; records ingestion metadata.
- `GET /v1/documents?skip=0&limit=10` — List stored documents (content, embedding, metadata).
- `POST /v1/query` — Non-streaming Q&A over your documents; returns answer and sources.
- `POST /v1/query-stream` — Streaming Q&A; returns token stream; response header `x-conversation-id` is set.
- `GET /v1/history/{conversation_id}` — Returns chat history for a conversation.

### Example Requests
Upload a PDF:
```
curl -F "file=@/path/to/file.pdf" http://localhost:8000/v1/upload
```

Ask a question (non-streaming):
```
curl -X POST http://localhost:8000/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What are the key points?"}'
```

Ask a question (streaming):
```
curl -N -X POST http://localhost:8000/v1/query-stream \
  -H 'Content-Type: application/json' \
  -d '{"question": "Summarize the document", "conversation_id": null}' -i
```
Note: Capture `x-conversation-id` from the response headers and reuse it to maintain history.

## Project Structure
```
.
├── app.py                 # FastAPI app, routes, CORS, lifespan
├── config.py              # Pydantic Settings (env-driven config)
├── schemas.py             # Request/response Pydantic models
├── services/
│   ├── db.py              # Async SQLModel/engine and session management
│   ├── models.py          # SQLModel table mappings (documents, pdf_ingestion)
│   ├── ingest.py          # PDF loading, chunking, embedding insertion
│   ├── documents.py       # Document listing via SQLModel
│   ├── vector_store.py    # Supabase client + LangChain vector store
│   ├── history.py         # Chat history CRUD
│   └── query.py           # Retrieval + OpenAI completion (sync + streaming)
├── pdfs/                  # Local store for uploaded PDFs
├── requirements.txt       # Python dependencies
├── Makefile               # `make run` convenience target
└── README.md
```

## Configuration and Tuning
- CORS origins: update in `app.py` (`origins` list) for your frontend.
- Chunking: adjust `chunk_size` / `chunk_overlap` in `services/ingest.py`.
- Models: configure `OPENAI_MODEL` and `EMBEDDING_MODEL` in `.env`.
- Retrieval `top_k`: set via `TOP_K` (default 5) and reflect in queries if needed.
- Table names: change `SUPABASE_TABLE` if not using `documents`.

## Development Guide
- Add endpoints: extend `router_v1` in `app.py`; define Pydantic schemas in `schemas.py`.
- DB migrations: consider adding Alembic to manage schema evolution (already in requirements).
- Error handling: `services/query.py` uses timeouts; add retry/backoff where needed.
- Testing: factor logic into services and test with async DB sessions and mocked OpenAI.
- Observability: integrate logging/tracing (e.g., OpenTelemetry) if needed.
- Security: avoid shipping service-role keys to untrusted clients; keep this API server-side.

## Troubleshooting
- Connection issues: verify `POSTGRES_URL` uses `postgresql+asyncpg://...` and that SSL is enabled.
- Vector ops failing: ensure `documents` table exists and dimensions match the embedding model.
- Vector search failing: check if the embedding model used for search matches the one used when uploading documents. If `EMBEDDING_MODEL` changed after ingestion, re-embed your documents so query-time vectors and stored vectors share the same dimension and distribution; ensure the `documents.embedding` vector size matches and rebuild the IVFFLAT index if you changed dimensions.
- RLS policies: if using Supabase with RLS enabled, add policies to allow the API to insert/select.
- Streaming stalls: check network proxies; server streams `text/plain` tokens; use `curl -N`.

## Roadmap Ideas
- User auth + per-user namespaces for documents and chat history.
- Better source attribution: return chunk/page references with answers.
- Background ingestion + progress tracking.
- Switchable retrievers (SQL-only vs. SupabaseVectorStore abstraction).
- Caching or hybrid search (BM25 + vector) for improved recall.

---

Happy hacking! Open `http://localhost:8000/docs` to explore the API.
