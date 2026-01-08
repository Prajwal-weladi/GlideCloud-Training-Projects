from app.db.mongodb import chunks_collection
from app.core.ollam_client import get_embedding, generate_answer


def query_document(question: str, top_k: int = 5):
    query_embedding = get_embedding(question)

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": top_k
            }
        },
        {
            "$project": {
                "_id": 0,
                "chunk_index": 1,
                "text": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    results = list(chunks_collection.aggregate(pipeline))

    if not results:
        return {
            "answer": "No relevant information found.",
            "chunks_used": []
        }

    # Build context (FULL chunks)
    context = "\n".join(r["text"] for r in results)

    answer = generate_answer(context, question)

    # Reduce response payload
    chunks_used = [
        {
            "chunk_index": r["chunk_index"],
            "score": round(r["score"], 3),
            "preview": r["text"][:400] + "..."
        }
        for r in results
    ]

    return {
        "answer": answer,
        "chunks_used": chunks_used
    }
