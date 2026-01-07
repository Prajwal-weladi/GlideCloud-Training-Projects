from fastapi import APIRouter, HTTPException
from app.models.schemas import DocumentCreate, SearchRequest
from app.services.embeddings import generate_embedding
from app.core.database import collection

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/")
async def create_document(doc: DocumentCreate):
    embedding = await generate_embedding(doc.content)

    document = {
        "title": doc.title,
        "content": doc.content,
        "embedding": embedding
    }

    result = collection.insert_one(document)
    return {"id": str(result.inserted_id)}


@router.post("/search")
async def search_documents(payload: SearchRequest):
    query_embedding = await generate_embedding(payload.query)

    pipeline = [
        {
            "$vectorSearch": {
                "index": "default",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 100,
                "limit": payload.top_k
            }
        },
        {
            "$project": {
                "title": 1,
                "content": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    results = list(collection.aggregate(pipeline))
    return results
