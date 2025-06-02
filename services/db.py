import os
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

engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    connect_args={"ssl": True},
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
    from services.models import Document  # import so SQLModel.metadata knows about it

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