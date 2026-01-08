import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Provide TestClient for API tests"""
    return TestClient(app)


@pytest.fixture
def mock_embedding():
    """Mock embedding vector"""
    return [0.1] * 1024


@pytest.fixture
def mock_context():
    """Mock document context"""
    return "GlideCloud Solutions is a cloud and AI-focused company specializing in vector databases and AI integration."


@pytest.fixture
def mock_chunks():
    """Mock document chunks"""
    return [
        {
            "text": "GlideCloud Solutions is a cloud and AI-focused company.",
            "score": 0.88,
            "chunk_index": 0
        },
        {
            "text": "It specializes in vector databases and AI integration.",
            "score": 0.82,
            "chunk_index": 1
        }
    ]
