import httpx
from app.core.config import settings

async def generate_embedding(text: str) -> list[float]:
    payload = {
        "model": settings.EMBEDDING_MODEL,
        "prompt": text
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/embeddings",
            json=payload,
            timeout=30
        )
    print(response)
    response.raise_for_status()
    return response.json()["embedding"]
