from pydantic_settings import BaseSettings
from pydantic import Field, computed_field, PostgresDsn
from sqlalchemy.engine import URL

class Settings(BaseSettings):
    # Supabase
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_key: str = Field(..., env="SUPABASE_KEY")
    supabase_table: str = Field("documents", env="SUPABASE_TABLE")

    # OpenAI
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-3.5-turbo", env="OPENAI_MODEL")
    embedding_model: str = Field("text-embedding-ada-002", env="EMBEDDING_MODEL")

    # RAG params
    top_k: int = Field(5, env="TOP_K")

    pdf_dir: str = Field("pdfs/", env="PDF_DIR")

    ## PostgreSQL (metadata) credentials, read from .env
    POSTGRES_SERVER: str  = Field(..., env="POSTGRES_SERVER")
    POSTGRES_PORT: int    = Field(5432, env="POSTGRES_PORT")
    POSTGRES_USER: str    = Field(..., env="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field("", env="POSTGRES_PASSWORD")
    POSTGRES_DB: str      = Field("", env="POSTGRES_DB")

    @computed_field
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> PostgresDsn:
        """
        Build a proper `postgresql+psycopg://...` or `postgresql+asyncpg://...`
        connection string with correct quoting. If you later switch to async,
        change `scheme="postgresql+asyncpg"` here.
        """
        url = URL.create(
            drivername="postgresql+asyncpg",
            username=self.POSTGRES_USER,
            password=self.POSTGRES_PASSWORD,
            host=self.POSTGRES_SERVER,
            port=self.POSTGRES_PORT,
            database=self.POSTGRES_DB,
        )
        return str(url)
    class Config:
        env_file = ".env"

settings = Settings()