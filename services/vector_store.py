from supabase import create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.embeddings import OpenAIEmbeddings
from config import settings
from services.db import connect_db, disconnect_db
import asyncio

# Connect to Postgres via Prisma on startup
def init_services():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(connect_db())

# Initialize Supabase client & vector store
supabase = create_client(settings.supabase_url, settings.supabase_key)
embeddings = OpenAIEmbeddings(
    model=settings.embedding_model,
    openai_api_key=settings.openai_api_key
)

vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name=settings.supabase_table,
    query_name="match_documents"
)

# Call init
init_services()