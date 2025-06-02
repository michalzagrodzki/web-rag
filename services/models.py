from datetime import datetime
from typing import Optional, Any, Dict
from uuid import uuid4

from sqlmodel import SQLModel, Field, Column, JSON

class Document(SQLModel, table=True):
    """
    Represents a stored PDF ingestion record.
    """
    # 1) Let SQLModel create the PK column.
    #    The default_factory ensures we get a uuid4() string at runtime.
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)

    filename: str

    # 2) ingested_at: default to now() in Python.
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    # 3) metadata: use a JSON column in Postgres.
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column("metadata", JSON, nullable=False),
    )