from sqlalchemy.pool import NullPool
import ssl
import uuid
import orjson
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from config import settings
from fastapi.concurrency import asynccontextmanager

load_dotenv()

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Wrap orjson.dumps so that it returns a str (decode bytes → str)
def orjson_serializer(obj: object) -> str:
    # orjson.dumps(obj) returns bytes; decode to utf-8 string
    return orjson.dumps(obj).decode("utf-8")

# We can use orjson.loads directly, since it accepts bytes or str and returns Python objects
def orjson_deserializer(s: str | bytes) -> object:
    return orjson.loads(s)

engine: AsyncEngine = create_async_engine(
    settings.POSTGRES_URL,
    json_serializer=orjson_serializer,
    json_deserializer=orjson_deserializer,
    echo=False,
    pool_recycle=1800,
    future=True,
    pool_pre_ping=True,
    poolclass=NullPool,
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

    #async with engine.begin() as conn:
    #    await conn.run_sync(SQLModel.metadata.create_all)

@asynccontextmanager
async def get_session() -> AsyncSession: # type: ignore
    session: AsyncSession = async_session_maker()
    try:
        yield session
    finally:
        await session.close()