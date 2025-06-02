from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # Supabase
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_key: str = Field(..., env="SUPABASE_KEY")
    supabase_table: str = Field("documents", env="SUPABASE_TABLE")

    # OpenAI
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-3.5-turbo", env="OPENAI_MODEL")
    embedding_model: str = Field("text-embedding-ada-002", env="EMBEDDING_MODEL")

    # Prisma/Postgres
    database_url: str = Field(..., env="DATABASE_URL")

    # RAG params
    top_k: int = Field(5, env="TOP_K")

    pdf_dir: str = Field("pdfs/", env="PDF_DIR")

    class Config:
        env_file = ".env"

settings = Settings()