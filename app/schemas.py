from pydantic import BaseModel
from typing import List, Dict, Any

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    intent: str
    intents: List[str]
    collection_used: str | None
    collections_used: List[str] | None
    context: List[Dict[str, Any]]