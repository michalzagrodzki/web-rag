import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import asynccontextmanager
from config import settings
from services.db import init_db
from services.ingest import ingest_pdf
#from services.query import answer_question
from schemas import UploadResponse, QueryRequest, QueryResponse

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
)

@app.post(
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))