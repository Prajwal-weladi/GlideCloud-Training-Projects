import pytest


def test_upload_document_success(client, monkeypatch):
    """Test successful document upload"""
    monkeypatch.setattr(
        "app.services.ingestion_service.get_embedding",
        lambda text: [0.1] * 1024
    )

    monkeypatch.setattr(
        "app.services.ingestion_service.chunks_collection.insert_many",
        lambda docs: True
    )

    response = client.post(
        "/documents",
        json={"text": "GlideCloud Solutions is a cloud company."}
    )

    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "doc_id" in data
    assert "chunks" in data
    assert data["message"] == "Document stored and indexed in MongoDB Atlas"


def test_upload_document_empty_text(client, monkeypatch):
    """Test document upload with empty text"""
    monkeypatch.setattr(
        "app.services.ingestion_service.chunks_collection.insert_many",
        lambda docs: True
    )

    response = client.post(
        "/documents",
        json={"text": ""}
    )

    # Should handle empty text gracefully
    assert response.status_code == 200


def test_upload_document_large_text(client, monkeypatch):
    """Test document upload with large text"""
    large_text = "This is a test document. " * 1000  # Large text
    
    monkeypatch.setattr(
        "app.services.ingestion_service.get_embedding",
        lambda text: [0.1] * 1024
    )

    monkeypatch.setattr(
        "app.services.ingestion_service.chunks_collection.insert_many",
        lambda docs: True
    )

    response = client.post(
        "/documents",
        json={"text": large_text}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["chunks"] > 0


def test_upload_document_special_characters(client, monkeypatch):
    """Test document upload with special characters"""
    special_text = "Testing with special chars: @#$%^&*() and 中文 and émojis!"
    
    monkeypatch.setattr(
        "app.services.ingestion_service.get_embedding",
        lambda text: [0.1] * 1024
    )

    monkeypatch.setattr(
        "app.services.ingestion_service.chunks_collection.insert_many",
        lambda docs: True
    )

    response = client.post(
        "/documents",
        json={"text": special_text}
    )

    assert response.status_code == 200


def test_upload_document_response_structure(client, monkeypatch):
    """Test response structure of document upload"""
    monkeypatch.setattr(
        "app.services.ingestion_service.get_embedding",
        lambda text: [0.1] * 1024
    )

    monkeypatch.setattr(
        "app.services.ingestion_service.chunks_collection.insert_many",
        lambda docs: True
    )

    response = client.post(
        "/documents",
        json={"text": "Test document"}
    )

    assert response.status_code == 200
    data = response.json()
    
    # Validate response structure
    assert isinstance(data, dict)
    assert "message" in data
    assert "doc_id" in data
    assert "chunks" in data
    assert isinstance(data["doc_id"], str)
    assert isinstance(data["chunks"], int)
    assert data["chunks"] >= 1
