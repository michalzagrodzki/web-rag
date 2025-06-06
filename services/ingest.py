import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from services.vector_store import vector_store
from services.models import PdfIngestion
from services.db import get_session
from concurrent.futures import ThreadPoolExecutor
import asyncio
import logging

logger = logging.getLogger(__name__)
PDF_DIR = os.getenv("PDF_DIR", "pdfs/")
executor = ThreadPoolExecutor(max_workers=2)

async def ingest_pdf(file_path: str) -> int:
    logger.info("Starting PDF ingestion.")
    """
    1) Chunk the PDF & push embeddings to Supabase.
    2) Insert a new row into the Supabase Postgres 'Document' table via SQLModel.
    """
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} documents from PDF.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks.")

    # 1) Add embeddings to Supabase vector store
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, vector_store.add_documents, chunks)
    logger.info("Successfully added embeddings to Supabase vector store.")

    # 2) Record ingestion metadata in Postgres
    filename = os.path.basename(file_path)
    metadata = {"chunks": len(chunks), "path": file_path}

    # Use the async session to create a Document row
    async with get_session() as session:
        doc = PdfIngestion(filename=filename, meta=metadata)
        session.add(doc)
        await session.commit()
        logger.info("Inserted ingestion record into database.")

    return len(chunks)