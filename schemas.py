from pydantic import BaseModel
from typing import Any, List, Dict

class UploadResponse(BaseModel):
    message: str
    inserted_count: int

class QueryRequest(BaseModel):
    question: str

class SourceDoc(BaseModel):
    page_content: str | None = None  # optional if not used
    metadata: Dict[str, Any]
    similarity: float | None = None
    id: str

class QueryResponse(BaseModel):
    answer: str
    source_docs: List[SourceDoc]