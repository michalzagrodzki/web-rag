from typing import List, Dict
from sqlalchemy import text
from services.db import get_session

async def get_history(conversation_id: str) -> List[Dict[str, str]]:
    """
    Fetch all prior turns for this conversation, ordered by timestamp.
    """
    sql = text("""
        SELECT question, answer
        FROM chat_history
        WHERE conversation_id = :cid
        ORDER BY created_at
    """)
    async with get_session() as session:
        result = await session.execute(sql, {"cid": conversation_id})
        rows = result.fetchall()
    # return as list of dicts for easy templating
    return [{"question": r.question, "answer": r.answer} for r in rows]

async def append_history(conversation_id: str, question: str, answer: str) -> None:
    """
    Insert the latest Q&A turn into chat_history.
    """
    sql = text("""
        INSERT INTO chat_history (conversation_id, question, answer)
        VALUES (:cid, :q, :a)
    """)
    async with get_session() as session:
        await session.execute(sql, {
            "cid": conversation_id,
            "q": question,
            "a": answer
        })
        await session.commit()
