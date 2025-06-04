import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from services.vector_store import vector_store
from services.models import PdfIngestion
from services.db import get_session
from sqlmodel import select

PDF_DIR = os.getenv("PDF_DIR", "pdfs/")

async def ingest_pdf(file_path: str) -> int:
    """
    1) Chunk the PDF & push embeddings to Supabase.
    2) Insert a new row into the Supabase Postgres 'Document' table via SQLModel.
    """
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # 1) Add embeddings to Supabase vector store
    vector_store.add_documents(chunks)

    # 2) Record ingestion metadata in Postgres
    filename = os.path.basename(file_path)
    metadata = {"chunks": len(chunks), "path": file_path}

    # Use the async session to create a Document row
    async for session in get_session():
        doc = PdfIngestion(filename=filename, metadata=metadata)
        session.add(doc)
        await session.commit()

    return len(chunks)