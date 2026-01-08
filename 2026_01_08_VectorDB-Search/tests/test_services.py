import pytest
from app.services.query_service import query_document


def test_query_document_with_mocked_results(monkeypatch):
    """Test query_document service with mocked results"""
    monkeypatch.setattr(
        "app.services.query_service.get_embedding",
        lambda text: [0.1] * 1024
    )

    monkeypatch.setattr(
        "app.services.query_service.chunks_collection.aggregate",
        lambda pipeline: [
            {
                "text": "GlideCloud Solutions is a cloud and AI-focused company.",
                "score": 0.85,
                "chunk_index": 0
            }
        ]
    )

    monkeypatch.setattr(
        "app.services.query_service.generate_answer",
        lambda context, question: "GlideCloud Solutions is a cloud and AI-focused company."
    )

    response = query_document("What is GlideCloud?")

    assert "answer" in response
    assert "GlideCloud Solutions" in response["answer"]
    assert len(response["chunks_used"]) == 1


def test_query_document_no_results(monkeypatch):
    """Test query_document when no results are found"""
    monkeypatch.setattr(
        "app.services.query_service.get_embedding",
        lambda text: [0.1] * 1024
    )

    monkeypatch.setattr(
        "app.services.query_service.chunks_collection.aggregate",
        lambda pipeline: []
    )

    response = query_document("Unknown topic?")

    assert "answer" in response
    assert response["answer"] == "No relevant information found."
    assert response["chunks_used"] == []


def test_query_document_with_multiple_chunks(monkeypatch):
    """Test query_document with multiple chunks"""
    chunks = [
        {
            "text": "First chunk about GlideCloud.",
            "score": 0.9,
            "chunk_index": 0
        },
        {
            "text": "Second chunk about features.",
            "score": 0.85,
            "chunk_index": 1
        },
        {
            "text": "Third chunk about pricing.",
            "score": 0.8,
            "chunk_index": 2
        },
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
        lambda context, question: "Comprehensive answer"
    )

    response = query_document("Tell me everything")

    assert len(response["chunks_used"]) == 3
    assert response["chunks_used"][0]["score"] == 0.9
    assert response["chunks_used"][1]["score"] == 0.85
    assert response["chunks_used"][2]["score"] == 0.8


def test_query_document_top_k_parameter(monkeypatch):
    """Test query_document with different top_k values"""
    monkeypatch.setattr(
        "app.services.query_service.get_embedding",
        lambda text: [0.1] * 1024
    )

    chunks = [
        {
            "text": f"Chunk {i}",
            "score": 0.9 - (i * 0.05),
            "chunk_index": i
        }
        for i in range(10)
    ]

    monkeypatch.setattr(
        "app.services.query_service.chunks_collection.aggregate",
        lambda pipeline: chunks[:3]  # Simulating top_k=3
    )

    monkeypatch.setattr(
        "app.services.query_service.generate_answer",
        lambda context, question: "Answer"
    )

    response = query_document("Test", top_k=3)

    assert len(response["chunks_used"]) == 3


def test_query_document_chunk_preview_truncation(monkeypatch):
    """Test that chunk previews are truncated to 400 characters"""
    long_text = "A" * 500  # Text longer than 400 chars
    
    monkeypatch.setattr(
        "app.services.query_service.get_embedding",
        lambda text: [0.1] * 1024
    )

    monkeypatch.setattr(
        "app.services.query_service.chunks_collection.aggregate",
        lambda pipeline: [
            {
                "text": long_text,
                "score": 0.85,
                "chunk_index": 0
            }
        ]
    )

    monkeypatch.setattr(
        "app.services.query_service.generate_answer",
        lambda context, question: "Answer"
    )

    response = query_document("Test")

    # Preview should be truncated to 400 chars + "..."
    preview = response["chunks_used"][0]["preview"]
    assert len(preview) <= 403  # 400 + 3 for "..."
    assert preview.endswith("...")


def test_ingest_document_service(monkeypatch):
    """Test document ingestion service"""
    from app.services.ingestion_service import ingest_document
    
    monkeypatch.setattr(
        "app.services.ingestion_service.get_embedding",
        lambda text: [0.1] * 1024
    )

    monkeypatch.setattr(
        "app.services.ingestion_service.chunks_collection.insert_many",
        lambda docs: True
    )

    result = ingest_document("Test document content")

    assert "message" in result
    assert "doc_id" in result
    assert "chunks" in result
    assert result["chunks"] > 0


def test_ingest_document_empty_string(monkeypatch):
    """Test ingestion of empty document"""
    from app.services.ingestion_service import ingest_document
    
    monkeypatch.setattr(
        "app.services.ingestion_service.chunks_collection.insert_many",
        lambda docs: True
    )

    result = ingest_document("")

    assert "message" in result
    assert "doc_id" in result


def test_ingest_pdf_service(monkeypatch):
    """Test PDF ingestion service"""
    from app.services.pdf_ingestion_service import ingest_pdf
    
    monkeypatch.setattr(
        "app.services.pdf_ingestion_service.extract_text_from_pdf",
        lambda path: "Sample PDF content"
    )

    monkeypatch.setattr(
        "app.services.pdf_ingestion_service.get_embedding",
        lambda text: [0.1] * 1024
    )

    monkeypatch.setattr(
        "app.services.pdf_ingestion_service.chunks_collection.insert_many",
        lambda docs: True
    )

    monkeypatch.setattr(
        "app.services.pdf_ingestion_service.os.remove",
        lambda path: True
    )

    result = ingest_pdf("test.pdf")

    assert "message" in result
    assert "doc_id" in result
    assert "chunks" in result


def test_ingest_pdf_empty_content(monkeypatch):
    """Test PDF ingestion with empty content"""
    from app.services.pdf_ingestion_service import ingest_pdf
    
    monkeypatch.setattr(
        "app.services.pdf_ingestion_service.extract_text_from_pdf",
        lambda path: ""
    )

    monkeypatch.setattr(
        "app.services.pdf_ingestion_service.os.remove",
        lambda path: True
    )

    result = ingest_pdf("empty.pdf")

    assert "No readable text found in PDF" in result["message"]
