from typing import List, Tuple, Dict, Any
from fastapi import HTTPException
from sqlalchemy import text
from services.db import get_session
from langchain_openai import OpenAIEmbeddings
import openai
from config import settings
import logging
import asyncio


logger = logging.getLogger(__name__)

openai.api_key = settings.openai_api_key
embedding_model = OpenAIEmbeddings(
    model=settings.embedding_model,
    openai_api_key=settings.openai_api_key
)

def to_pgvector_literal(vec: list[float]) -> str:
    return f"[{','.join(f'{x:.6f}' for x in vec)}]"

async def answer_question(question: str) -> Tuple[str, List[Dict[str, Any]]]:
    logger.info("✅ Starting to embed query")
    # Step 1: Embed the question
    q_vector = embedding_model.embed_query(question)
    logger.info("✅ Finished embedding query")
    # Step 2: Query top-5 similar documents from Supabase
    sql = text("""
        SELECT id, content, metadata, 1 - (embedding <=> :q) AS similarity
        FROM documents
        ORDER BY embedding <=> :q
        LIMIT 5
    """)

    q_vector_str = to_pgvector_literal(q_vector)

    logger.info("✅ Starting to fetch documents from DB")
    async with get_session() as session:
        try:
            result = await asyncio.wait_for(
                session.execute(sql, {"q": q_vector_str}),
                timeout=10.0
            )
            rows = result.fetchall()
        except asyncio.TimeoutError:
            logger.error("Database query timed out — connection may be stale.")
            raise HTTPException(status_code=504, detail="Database query timed out.")

    logger.info("✅ Fetched documents from DB")

    # Step 3: Construct context string for the LLM
    context_blocks = []
    top_docs = []

    for row in rows:
        context_blocks.append(row.content)
        top_docs.append({
            "id": str(row.id),
            "similarity": float(row.similarity),
            "metadata": row.metadata
        })

    logger.info("✅ Constructed context block")

    context = "\n\n---\n\n".join(context_blocks)
    prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}\nAnswer:"

    # Step 4: Generate answer via OpenAI ChatCompletion (sync call in async context)
    # Wrap blocking call with asyncio to avoid hanging the event loop
    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key)
    logger.info("✅ Prompt ready, calling OpenAI")

    async def run_completion(prompt: str, timeout: float = 20.0):
        return await asyncio.wait_for(
            asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[{"role": "user", "content": prompt}]
                )
            ),
            timeout=timeout
        )

    response = await run_completion(prompt)
    logger.info("✅ Got response from OpenAI")
    answer = response.choices[0].message.content.strip()

    return answer, top_docs
