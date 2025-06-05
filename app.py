import os
from typing import Any
from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import JSONResponse
from sqlalchemy import text
from services.db import get_session, init_db
from services.ingest import ingest_pdf
from services.vector_store import supabase_client
from schemas import UploadResponse, QueryRequest, QueryResponse
from services.documents import list_documents 
from config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Application startup: initialize database
    await init_db()
    yield
    # Application shutdown: (no actions needed currently)

app = FastAPI(
    title="RAG Supabase FastAPI (SQLModel)",
    version="1.0.0",
    description="RAG service using Supabase vector store, OpenAI API, and SQLModel/Postgres",
    lifespan=lifespan
)

router_v1 = APIRouter(prefix="/v1")

@router_v1.get("/test-db")
async def test_db():
    try:
        async for session in get_session():
            result = await session.execute(text("SELECT 1"))
            return {"status": "ok", "result": result.scalar()}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
@router_v1.post(
    "/upload",
    response_model=UploadResponse,
    tags=["Ingestion"],
    summary="Upload a PDF document",
    description="Ingests a PDF, splits into chunks, and stores embeddings in Supabase"
)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    contents = await file.read()
    os.makedirs("pdfs", exist_ok=True)
    path = os.path.join("pdfs", file.filename)
    with open(path, "wb") as f:
        f.write(contents)

    count = await ingest_pdf(path)
    return UploadResponse(message="PDF ingested successfully", inserted_count=count)

@router_v1.get(
    "/documents",
    summary="List all documents from Supabase",
    description="Fetches every row from the Supabase “documents” table and returns them as JSON",
)
async def get_all_documents() -> Any:
    """
    Consume the async generator `list_documents()` and return
    a list of dicts (one per Document).
    """
    try:
        # Build a JSON list by “async for … in” the generator
        documents_list = [
            {
                "id": str(doc.id),
                "content": doc.content,
                "embedding": doc.embedding,    # a List[float] or None
                "metadata": doc.meta,     # a Dict[str,Any]
            }
            async for doc in list_documents()
        ]
        return JSONResponse(documents_list)
    except Exception as e:
        # If something goes wrong in the DB layer, return a 500
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(router_v1)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))