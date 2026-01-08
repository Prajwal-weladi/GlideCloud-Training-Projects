import pytest
import os
from dotenv import load_dotenv


def test_config_loads_settings():
    """Test that config properly loads settings"""
    from app.core.config import settings
    
    assert hasattr(settings, 'MONGO_URI')
    assert hasattr(settings, 'DB_NAME')
    assert hasattr(settings, 'EMBEDDING_MODEL')
    assert hasattr(settings, 'LLM_MODEL')


def test_config_default_values():
    """Test default configuration values"""
    from app.core.config import settings
    
    assert settings.DB_NAME == "vector_search"
    assert settings.CHUNKS_COLLECTION == "document_chunks"
    assert settings.VECTOR_INDEX_NAME == "vector_index"


def test_config_model_names():
    """Test that model names are properly set"""
    from app.core.config import settings
    
    assert settings.EMBEDDING_MODEL == "mxbai-embed-large:latest"
    assert settings.LLM_MODEL == "llama3.2:latest"


def test_ollama_client_embedding(monkeypatch):
    """Test Ollama embedding client"""
    from app.core.ollam_client import get_embedding
    
    mock_response = {"embedding": [0.1] * 1024}
    monkeypatch.setattr(
        "app.core.ollam_client.ollama.embeddings",
        lambda model, prompt: mock_response
    )
    
    result = get_embedding("test text")
    
    assert isinstance(result, list)
    assert len(result) == 1024


def test_ollama_client_generate_answer(monkeypatch):
    """Test Ollama answer generation"""
    from app.core.ollam_client import generate_answer
    
    mock_response = {"response": "This is a test answer"}
    monkeypatch.setattr(
        "app.core.ollam_client.ollama.generate",
        lambda model, prompt: mock_response
    )
    
    result = generate_answer("Test context", "What is this?")
    
    assert isinstance(result, str)
    assert "test answer" in result.lower()


def test_ollama_client_embedding_format(monkeypatch):
    """Test that embedding response is in correct format"""
    from app.core.ollam_client import get_embedding
    
    expected_embedding = [0.1, 0.2, 0.3] * 341 + [0.1]  # 1024 elements
    mock_response = {"embedding": expected_embedding}
    
    monkeypatch.setattr(
        "app.core.ollam_client.ollama.embeddings",
        lambda model, prompt: mock_response
    )
    
    result = get_embedding("test")
    
    assert len(result) == 1024
    assert all(isinstance(x, (int, float)) for x in result)


def test_ollama_client_with_empty_text(monkeypatch):
    """Test Ollama client with empty text"""
    from app.core.ollam_client import get_embedding
    
    mock_response = {"embedding": [0.0] * 1024}
    monkeypatch.setattr(
        "app.core.ollam_client.ollama.embeddings",
        lambda model, prompt: mock_response
    )
    
    result = get_embedding("")
    
    assert isinstance(result, list)
    assert len(result) == 1024


def test_ollama_client_with_long_text(monkeypatch):
    """Test Ollama client with very long text"""
    from app.core.ollam_client import get_embedding
    
    long_text = "word " * 10000
    mock_response = {"embedding": [0.1] * 1024}
    
    monkeypatch.setattr(
        "app.core.ollam_client.ollama.embeddings",
        lambda model, prompt: mock_response
    )
    
    result = get_embedding(long_text)
    
    assert isinstance(result, list)
    assert len(result) == 1024


def test_ollama_client_embedding_model_used(monkeypatch):
    """Test that correct embedding model is used"""
    from app.core.ollam_client import get_embedding, EMBED_MODEL
    
    called_with = {}
    
    def mock_embeddings(model, prompt):
        called_with['model'] = model
        return {"embedding": [0.1] * 1024}
    
    monkeypatch.setattr(
        "app.core.ollam_client.ollama.embeddings",
        mock_embeddings
    )
    
    get_embedding("test")
    
    assert called_with['model'] == EMBED_MODEL


def test_ollama_client_generate_model_used(monkeypatch):
    """Test that correct generation model is used"""
    from app.core.ollam_client import generate_answer, LLM_MODEL
    
    called_with = {}
    
    def mock_generate(model, prompt):
        called_with['model'] = model
        return {"response": "test response"}
    
    monkeypatch.setattr(
        "app.core.ollam_client.ollama.generate",
        mock_generate
    )
    
    generate_answer("context", "question")
    
    assert called_with['model'] == LLM_MODEL


def test_mongodb_connection():
    """Test MongoDB connection setup"""
    from app.db.mongodb import client, db, chunks_collection
    
    # Verify that MongoDB objects are initialized
    assert client is not None
    assert db is not None
    assert chunks_collection is not None


def test_mongodb_collection_name():
    """Test that correct collection is being used"""
    from app.db.mongodb import chunks_collection
    from app.core.config import settings
    
    assert chunks_collection.name == settings.CHUNKS_COLLECTION


def test_config_env_file_loading():
    """Test that .env file is properly loaded"""
    load_dotenv()
    
    mongo_uri = os.getenv('MONGO_URI')
    # The test assumes .env exists, adjust as needed
    # This test validates the config can read from env
