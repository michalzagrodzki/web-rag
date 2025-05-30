import os
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from config import settings
from services.db import connect_db, disconnect_db
from services.ingest import ingest_pdf
from services.query import answer_question
from schemas import UploadResponse, QueryRequest, QueryResponse

app = FastAPI(
    title="RAG Supabase FastAPI",
    version="1.1.0",
    description="A RAG service using Supabase vector store, OpenAI API, and Prisma"
)

@app.on_event("startup")
async def startup_event():
    await connect_db()

@app.on_event("shutdown")
async def shutdown_event():
    await disconnect_db()

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