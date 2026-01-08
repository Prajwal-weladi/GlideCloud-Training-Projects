import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_create_document(monkeypatch):
    fake_embedding = [0.1, 0.2, 0.3]

    async def fake_generate_embedding(text):
        return fake_embedding

    monkeypatch.setattr(
        "app.routes.documents.generate_embedding",
        fake_generate_embedding
    )

    mock_collection = MagicMock()
    mock_collection.insert_one.return_value.inserted_id = "fake_id"

    monkeypatch.setattr(
        "app.routes.documents.collection",
        mock_collection
    )

    response = client.post(
        "/documents/",
        json={"title": "Test", "content": "Hello world"}
    )

    assert response.status_code == 200
    assert response.json()["id"] == "fake_id"


def test_search_documents(monkeypatch):
    fake_embedding = [0.5, 0.6]

    async def fake_generate_embedding(text):
        return fake_embedding

    monkeypatch.setattr(
        "app.routes.documents.generate_embedding",
        fake_generate_embedding
    )

    mock_collection = MagicMock()
    mock_collection.aggregate.return_value = [
        {"title": "Doc1", "content": "Text", "score": 0.99}
    ]

    monkeypatch.setattr(
        "app.routes.documents.collection",
        mock_collection
    )

    response = client.post(
        "/documents/search",
        json={"query": "test", "top_k": 3}
    )

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["title"] == "Doc1"
