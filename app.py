import os
from typing import Any
from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter, Depends
from fastapi.concurrency import asynccontextmanager
from sqlalchemy import text
from services.db import get_session, init_db
from services.documents import list_documents
from services.ingest import ingest_pdf
from schemas import UploadResponse, QueryRequest, QueryResponse
from typing import Any, List, Dict
from services.db import init_db, get_session
from fastapi.responses import JSONResponse
import logging
from fastapi import Query
from services.query import answer_question

logging.basicConfig(
    level=logging.DEBUG,  # or DEBUG
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

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
            result = await session.exec(text("SELECT 1"))
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

    logger.info(f"Finished ingestion, inserted {count} chunks.")

    return UploadResponse(
        message="PDF ingested successfully",
        inserted_count=count
    )

@router_v1.get(
    "/documents",
    summary="List documents with pagination",
    description="Fetches paginated rows from the Supabase 'documents' table.",
    response_model=List[Dict[str, Any]],
)
async def get_all_documents(skip: int = Query(0, ge=0), limit: int = Query(10, ge=1, le=100)) -> Any:
    """
    Open a single AsyncSession, select all Document rows, and return them.
    """
    try:
        logger.info(f"Fetching documents: skip={skip}, limit={limit}")
        docs = await list_documents(skip=skip, limit=limit)
        return JSONResponse(content=docs)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@router_v1.post(
    "/query",
    response_model=QueryResponse,
    tags=["RAG"],
    summary="Query the knowledge base",
    description="Retrieval-Augmented Generation over ingested documents."
)
async def query_qa(req: QueryRequest):
    answer, sources = await answer_question(req.question)
    return QueryResponse(answer=answer, source_docs=sources)

app.include_router(router_v1)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))