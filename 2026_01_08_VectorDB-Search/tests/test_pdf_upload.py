from io import BytesIO
import pytest


def test_pdf_upload_success(client, monkeypatch):
    """Test successful PDF upload"""
    monkeypatch.setattr(
        "app.services.pdf_ingestion_service.extract_text_from_pdf",
        lambda path: "This is a test PDF about GlideCloud."
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

    fake_pdf = BytesIO(b"%PDF-1.4 fake content")

    response = client.post(
        "/upload-pdf",
        files={"file": ("test.pdf", fake_pdf, "application/pdf")}
    )

    assert response.status_code == 200
    data = response.json()
    assert "PDF processed" in data["message"]
    assert "doc_id" in data
    assert "chunks" in data


def test_pdf_upload_invalid_format(client):
    """Test PDF upload with invalid file format"""
    fake_file = BytesIO(b"This is not a PDF file")

    response = client.post(
        "/upload-pdf",
        files={"file": ("test.txt", fake_file, "text/plain")}
    )

    assert response.status_code == 200
    assert "error" in response.json()
    assert "Only PDF files are supported" in response.json()["error"]


def test_pdf_upload_no_file(client):
    """Test PDF upload without file"""
    response = client.post("/upload-pdf")
    
    assert response.status_code == 422  # Unprocessable Entity


def test_pdf_upload_empty_pdf(client, monkeypatch):
    """Test PDF upload with empty PDF"""
    monkeypatch.setattr(
        "app.services.pdf_ingestion_service.extract_text_from_pdf",
        lambda path: ""
    )

    monkeypatch.setattr(
        "app.services.pdf_ingestion_service.os.remove",
        lambda path: True
    )

    fake_pdf = BytesIO(b"%PDF-1.4 empty content")

    response = client.post(
        "/upload-pdf",
        files={"file": ("empty.pdf", fake_pdf, "application/pdf")}
    )

    assert response.status_code == 200
    data = response.json()
    assert "No readable text found in PDF" in data["message"]


def test_pdf_upload_response_structure(client, monkeypatch):
    """Test response structure of PDF upload"""
    monkeypatch.setattr(
        "app.services.pdf_ingestion_service.extract_text_from_pdf",
        lambda path: "Test PDF content with multiple words and sentences."
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

    fake_pdf = BytesIO(b"%PDF-1.4 content")

    response = client.post(
        "/upload-pdf",
        files={"file": ("test.pdf", fake_pdf, "application/pdf")}
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
    assert len(data["doc_id"]) == 36  # UUID format


def test_pdf_upload_multiple_files_sequentially(client, monkeypatch):
    """Test multiple PDF uploads"""
    monkeypatch.setattr(
        "app.services.pdf_ingestion_service.extract_text_from_pdf",
        lambda path: "PDF content"
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

    # Upload first PDF
    response1 = client.post(
        "/upload-pdf",
        files={"file": ("test1.pdf", BytesIO(b"%PDF-1.4"), "application/pdf")}
    )

    # Upload second PDF
    response2 = client.post(
        "/upload-pdf",
        files={"file": ("test2.pdf", BytesIO(b"%PDF-1.4"), "application/pdf")}
    )

    assert response1.status_code == 200
    assert response2.status_code == 200
    
    # Both should have different doc_ids
    doc_id_1 = response1.json()["doc_id"]
    doc_id_2 = response2.json()["doc_id"]
    assert doc_id_1 != doc_id_2
