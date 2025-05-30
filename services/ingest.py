from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from services.vector_store import vector_store
from services.db import prisma
import os
from datetime import datetime

PDF_DIR = os.getenv("PDF_DIR", "pdfs/")

async def ingest_pdf(file_path: str) -> int:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    vector_store.add_documents(chunks)

    # Store metadata in Postgres
    filename = os.path.basename(file_path)
    await prisma.document.create(
        data={
            'filename': filename,
            'metadata': {'chunks': len(chunks), 'path': file_path}
        }
    )
    return len(chunks)