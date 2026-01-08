import pytest
from io import BytesIO
import json


class TestAPIIntegration:
    """Integration tests for the complete API flow"""
    
    def test_document_upload_and_query_flow(self, client, monkeypatch):
        """Test complete flow: upload document -> query"""
        # Mock embedding
        monkeypatch.setattr(
            "app.services.ingestion_service.get_embedding",
            lambda text: [0.1] * 1024
        )
        
        # Mock DB insert for document upload
        uploaded_doc_id = None
        def capture_insert_many(docs):
            nonlocal uploaded_doc_id
            if docs:
                uploaded_doc_id = docs[0].get("doc_id")
            return True
        
        monkeypatch.setattr(
            "app.services.ingestion_service.chunks_collection.insert_many",
            capture_insert_many
        )
        
        # Upload document
        upload_response = client.post(
            "/documents",
            json={"text": "GlideCloud is an AI company specializing in vectors."}
        )
        
        assert upload_response.status_code == 200
        doc_data = upload_response.json()
        assert "doc_id" in doc_data
        
        # Mock query embedding
        monkeypatch.setattr(
            "app.services.query_service.get_embedding",
            lambda text: [0.1] * 1024
        )
        
        # Mock query results
        monkeypatch.setattr(
            "app.services.query_service.chunks_collection.aggregate",
            lambda pipeline: [
                {
                    "text": "GlideCloud is an AI company specializing in vectors.",
                    "score": 0.92,
                    "chunk_index": 0
                }
            ]
        )
        
        # Mock answer generation
        monkeypatch.setattr(
            "app.services.query_service.generate_answer",
            lambda context, question: "GlideCloud is an AI company."
        )
        
        # Query
        query_response = client.get("/query?q=What is GlideCloud?")
        
        assert query_response.status_code == 200
        query_data = query_response.json()
        assert "GlideCloud" in query_data["answer"]
        assert len(query_data["chunks_used"]) > 0
    
    def test_pdf_upload_and_query_flow(self, client, monkeypatch):
        """Test complete flow: upload PDF -> query"""
        pdf_content = "PDF content about GlideCloud"
        
        # Mock PDF extraction
        monkeypatch.setattr(
            "app.services.pdf_ingestion_service.extract_text_from_pdf",
            lambda path: pdf_content
        )
        
        # Mock embedding
        monkeypatch.setattr(
            "app.services.pdf_ingestion_service.get_embedding",
            lambda text: [0.1] * 1024
        )
        
        # Mock DB insert
        monkeypatch.setattr(
            "app.services.pdf_ingestion_service.chunks_collection.insert_many",
            lambda docs: True
        )
        
        # Mock file removal
        monkeypatch.setattr(
            "app.services.pdf_ingestion_service.os.remove",
            lambda path: True
        )
        
        # Upload PDF
        fake_pdf = BytesIO(b"%PDF-1.4")
        upload_response = client.post(
            "/upload-pdf",
            files={"file": ("test.pdf", fake_pdf, "application/pdf")}
        )
        
        assert upload_response.status_code == 200
        upload_data = upload_response.json()
        assert "doc_id" in upload_data
        assert upload_data["chunks"] > 0
    
    def test_multiple_documents_and_queries(self, client, monkeypatch):
        """Test uploading multiple documents and querying"""
        monkeypatch.setattr(
            "app.services.ingestion_service.get_embedding",
            lambda text: [0.1] * 1024
        )
        
        monkeypatch.setattr(
            "app.services.ingestion_service.chunks_collection.insert_many",
            lambda docs: True
        )
        
        # Upload multiple documents
        docs = [
            "GlideCloud specializes in AI and cloud computing.",
            "The company provides vector database solutions.",
            "They offer enterprise-grade AI services."
        ]
        
        responses = []
        for doc in docs:
            response = client.post(
                "/documents",
                json={"text": doc}
            )
            responses.append(response)
            assert response.status_code == 200
        
        # Verify all uploads were successful
        assert len(responses) == 3
        doc_ids = [r.json()["doc_id"] for r in responses]
        assert len(set(doc_ids)) == 3  # All unique


class TestErrorHandling:
    """Test error handling across the API"""
    
    def test_invalid_request_body(self, client):
        """Test API with invalid request body"""
        response = client.post(
            "/documents",
            json={"invalid_field": "test"}
        )
        
        assert response.status_code == 422
    
    def test_missing_required_file(self, client):
        """Test file upload without file"""
        response = client.post("/upload-pdf")
        
        assert response.status_code == 422
    
    def test_query_with_invalid_parameters(self, client, monkeypatch):
        """Test query with invalid parameters"""
        monkeypatch.setattr(
            "app.services.query_service.get_embedding",
            lambda text: [0.1] * 1024
        )
        
        monkeypatch.setattr(
            "app.services.query_service.chunks_collection.aggregate",
            lambda pipeline: []
        )
        
        # Should still work even with moderately long query
        response = client.get("/query?q=" + "a" * 100)
        
        # Should either succeed or return validation error
        assert response.status_code in [200, 422]


class TestResponseValidation:
    """Test response validation and structure"""
    
    def test_document_response_is_json(self, client, monkeypatch):
        """Test that document response is valid JSON"""
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
            json={"text": "test"}
        )
        
        # Should be valid JSON
        data = response.json()
        assert isinstance(data, dict)
    
    def test_query_response_is_json(self, client, monkeypatch):
        """Test that query response is valid JSON"""
        monkeypatch.setattr(
            "app.services.query_service.get_embedding",
            lambda text: [0.1] * 1024
        )
        
        monkeypatch.setattr(
            "app.services.query_service.chunks_collection.aggregate",
            lambda pipeline: []
        )
        
        response = client.get("/query?q=test")
        
        # Should be valid JSON
        data = response.json()
        assert isinstance(data, dict)
    
    def test_response_headers(self, client, monkeypatch):
        """Test response headers"""
        monkeypatch.setattr(
            "app.services.query_service.get_embedding",
            lambda text: [0.1] * 1024
        )
        
        monkeypatch.setattr(
            "app.services.query_service.chunks_collection.aggregate",
            lambda pipeline: []
        )
        
        response = client.get("/query?q=test")
        
        assert response.status_code == 200
        assert "content-type" in response.headers


class TestPerformance:
    """Test performance-related aspects"""
    
    def test_large_text_processing(self, client, monkeypatch):
        """Test processing large text documents"""
        large_text = "word " * 10000  # Large document
        
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
    
    def test_multiple_concurrent_queries(self, client, monkeypatch):
        """Test handling of multiple queries"""
        monkeypatch.setattr(
            "app.services.query_service.get_embedding",
            lambda text: [0.1] * 1024
        )
        
        monkeypatch.setattr(
            "app.services.query_service.chunks_collection.aggregate",
            lambda pipeline: []
        )
        
        # Simulate multiple queries
        for i in range(5):
            response = client.get(f"/query?q=query{i}")
            assert response.status_code == 200
