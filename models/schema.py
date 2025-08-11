from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    query: str
    top_k: int = 12

class SourceChunk(BaseModel):
    id: str
    text: str
    title: Optional[str] = ""
    collection: str
    source_file: Optional[str] = ""
    section: Optional[str] = ""
    links: List[str] = Field(default_factory=list)
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class QueryResponse(BaseModel):
    answer: str
    source_chunks: List[SourceChunk]