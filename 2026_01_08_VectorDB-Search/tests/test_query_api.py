import pytest


def test_query_api_success(client, monkeypatch):
    """Test successful query"""
    monkeypatch.setattr(
        "app.services.query_service.get_embedding",
        lambda text: [0.1] * 1024
    )

    monkeypatch.setattr(
        "app.services.query_service.chunks_collection.aggregate",
        lambda pipeline: [
            {
                "text": "GlideCloud Solutions is a cloud and AI-focused company.",
                "score": 0.88,
                "chunk_index": 0
            }
        ]
    )

    monkeypatch.setattr(
        "app.services.query_service.generate_answer",
        lambda context, question: "GlideCloud Solutions is a cloud and AI-focused company."
    )

    response = client.get("/query?q=What is GlideCloud?")

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "GlideCloud" in data["answer"]
    assert "chunks_used" in data


def test_query_api_no_results(client, monkeypatch):
    """Test query with no relevant results"""
    monkeypatch.setattr(
        "app.services.query_service.get_embedding",
        lambda text: [0.1] * 1024
    )

    monkeypatch.setattr(
        "app.services.query_service.chunks_collection.aggregate",
        lambda pipeline: []
    )

    response = client.get("/query?q=What is something unknown?")

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "No relevant information found" in data["answer"]
    assert len(data["chunks_used"]) == 0


def test_query_api_multiple_chunks(client, monkeypatch):
    """Test query with multiple chunks"""
    chunks = [
        {
            "text": "GlideCloud Solutions is a cloud and AI-focused company.",
            "score": 0.88,
            "chunk_index": 0
        },
        {
            "text": "It specializes in vector databases and AI integration.",
            "score": 0.82,
            "chunk_index": 1
        },
        {
            "text": "The company provides cutting-edge solutions.",
            "score": 0.75,
            "chunk_index": 2
        }
    ]

    monkeypatch.setattr(
        "app.services.query_service.get_embedding",
        lambda text: [0.1] * 1024
    )

    monkeypatch.setattr(
        "app.services.query_service.chunks_collection.aggregate",
        lambda pipeline: chunks
    )

    monkeypatch.setattr(
        "app.services.query_service.generate_answer",
        lambda context, question: "GlideCloud is a comprehensive AI solution provider."
    )

    response = client.get("/query?q=Tell me about GlideCloud")

    assert response.status_code == 200
    data = response.json()
    assert len(data["chunks_used"]) == 3
    assert data["chunks_used"][0]["score"] == 0.88
    assert data["chunks_used"][1]["score"] == 0.82
    assert data["chunks_used"][2]["score"] == 0.75


def test_query_api_missing_query_parameter(client):
    """Test query without query parameter"""
    response = client.get("/query")
    
    assert response.status_code == 422  # Unprocessable Entity


def test_query_api_empty_query(client, monkeypatch):
    """Test query with empty string"""
    monkeypatch.setattr(
        "app.services.query_service.get_embedding",
        lambda text: [0.1] * 1024
    )

    monkeypatch.setattr(
        "app.services.query_service.chunks_collection.aggregate",
        lambda pipeline: []
    )

    response = client.get("/query?q=")

    assert response.status_code == 200


def test_query_api_special_characters(client, monkeypatch):
    """Test query with special characters"""
    monkeypatch.setattr(
        "app.services.query_service.get_embedding",
        lambda text: [0.1] * 1024
    )

    monkeypatch.setattr(
        "app.services.query_service.chunks_collection.aggregate",
        lambda pipeline: [
            {
                "text": "Special characters test",
                "score": 0.85,
                "chunk_index": 0
            }
        ]
    )

    monkeypatch.setattr(
        "app.services.query_service.generate_answer",
        lambda context, question: "Answer with special chars: @#$%"
    )

    response = client.get("/query?q=What about @#$%?")

    assert response.status_code == 200


def test_query_api_response_structure(client, monkeypatch):
    """Test response structure of query"""
    monkeypatch.setattr(
        "app.services.query_service.get_embedding",
        lambda text: [0.1] * 1024
    )

    monkeypatch.setattr(
        "app.services.query_service.chunks_collection.aggregate",
        lambda pipeline: [
            {
                "text": "Sample text for chunk",
                "score": 0.85,
                "chunk_index": 0
            }
        ]
    )

    monkeypatch.setattr(
        "app.services.query_service.generate_answer",
        lambda context, question: "Sample answer"
    )

    response = client.get("/query?q=Test query")

    assert response.status_code == 200
    data = response.json()
    
    # Validate response structure
    assert isinstance(data, dict)
    assert "answer" in data
    assert "chunks_used" in data
    assert isinstance(data["answer"], str)
    assert isinstance(data["chunks_used"], list)
    
    # Validate chunks structure
    if data["chunks_used"]:
        chunk = data["chunks_used"][0]
        assert "chunk_index" in chunk
        assert "score" in chunk
        assert "preview" in chunk


def test_query_api_score_rounding(client, monkeypatch):
    """Test that scores are properly rounded"""
    monkeypatch.setattr(
        "app.services.query_service.get_embedding",
        lambda text: [0.1] * 1024
    )

    monkeypatch.setattr(
        "app.services.query_service.chunks_collection.aggregate",
        lambda pipeline: [
            {
                "text": "Test content with precise score",
                "score": 0.8765432109876543,
                "chunk_index": 0
            }
        ]
    )

    monkeypatch.setattr(
        "app.services.query_service.generate_answer",
        lambda context, question: "Answer"
    )

    response = client.get("/query?q=Test")

    assert response.status_code == 200
    data = response.json()
    # Score should be rounded to 3 decimal places
    assert data["chunks_used"][0]["score"] == 0.877
