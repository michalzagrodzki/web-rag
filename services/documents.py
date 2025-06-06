from typing import Any, Dict, List
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from services.models import Document
from services.db import get_session
import logging

logger = logging.getLogger(__name__)

async def list_documents() -> List[Dict[str, Any]]:
    """
    Open a single AsyncSession, SELECT * FROM documents, and return
    a list of plain dicts (id, content, embedding, metadata).
    """
    try:
        async with get_session() as session:

            if session is None:
                raise RuntimeError("Could not acquire database session")
        
            # 1) Run a simple SELECT * FROM documents
            stmt = (select(Document)
                        .execution_options(autocommit=True))
            result = await session.execute(stmt)
            docs = result.scalars().all()
            
            await session.commit()

            documents_list: List[Dict[str, Any]] = []
            for doc in docs:
                documents_list.append({
                    "id": str(doc.id),
                    "content": doc.content,
                    "embedding": doc.embedding,
                    "metadata": doc.meta,
                })

            return documents_list
    except Exception as e:
        logger.error(f"Database error in list_documents: {e}")
        raise