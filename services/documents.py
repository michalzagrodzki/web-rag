from typing import AsyncGenerator
from sqlmodel import select
from services.models import Document
from services.db import get_session
import logging

logger = logging.getLogger(__name__)

async def list_documents() -> AsyncGenerator[Document, None]:
    """
    Asynchronously query all rows from the `documents` table
    and yield each Document instance one by one.
    """
    session = None
    try:
        async for session in get_session():
            # 1) Run a simple SELECT * FROM documents
            stmt = select(Document)
            result = await session.execute(stmt)
            documents_db = result.scalars().all()

            # 2) Yield each Document instance
            for document_db in documents_db:
                yield document_db
    except Exception as e:
        logger.error(f"Database error in list_documents: {e}")
        raise