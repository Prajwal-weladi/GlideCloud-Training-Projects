from app.core.database import collection
from bson import ObjectId

def store_embedding(text, embedding)-> str:
    document = {
        "text": text,
        "embedding": embedding
    }
    result = collection.insert_one(document)
    return str(result.inserted_id)

def get_embedding_by_id(id):
    document = collection.find_one({"_id": ObjectId(id)})
    if document:
        return{
            "id": str(document["_id"]),
            "text": document["text"],
            "embedding": document["embedding"]
        }
    return None