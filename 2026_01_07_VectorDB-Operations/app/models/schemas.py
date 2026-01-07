from pydantic import BaseModel

class DocumentCreate(BaseModel):
    title: str
    content: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
