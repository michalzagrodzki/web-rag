from pydantic import BaseModel
from typing import Any, List, Dict

class UploadResponse(BaseModel):
    message: str
    inserted_count: int

class QueryRequest(BaseModel):
    question: str

class SourceDoc(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    answer: str
    source_docs: List[SourceDoc]