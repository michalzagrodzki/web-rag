from multiprocessing import pool
import os
import ssl
import uuid
from jiter import from_json
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from config import settings

load_dotenv()

DATABASE_URL = settings.SQLALCHEMY_DATABASE_URI
if not DATABASE_URL:
    raise RuntimeError("`SQLALCHEMY_DATABASE_URI` is empty")
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    echo=True,
    future=True,
    json_deserializer=from_json,
    json_serializer=to_json_str,
    poolclass=pool.NullPool,
    pool_size=16,
    max_overflow=128,
    connect_args={
        "ssl": ssl_context, 
        "prepared_statement_name_func": lambda:  f"__asyncpg_{uuid.uuid4()}__",
        "statement_cache_size": 0,
        "prepared_statement_cache_size": 0,
        },
)

async_session_maker = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

async def init_db() -> None:
    """
    Create any missing tables in Supabase Postgres.
    (For production, you’d probably use Alembic instead of this auto-sync.)
    """
    from services.models import PdfIngestion  # import so SQLModel.metadata knows about it

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

async def get_session() -> AsyncSession: # type: ignore
    """
    Yield an AsyncSession. Use like:

        async for session in get_session():
            ...
    """
    async with async_session_maker() as session:
        yield session