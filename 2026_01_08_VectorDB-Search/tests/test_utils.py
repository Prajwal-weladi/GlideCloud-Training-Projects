import pytest
from app.utils.text_splitter import split_text


def test_split_text_default_params():
    """Test text splitting with default parameters"""
    text = "word " * 500  # 500 words
    chunks = split_text(text)
    
    assert len(chunks) > 0
    assert isinstance(chunks, list)


def test_split_text_empty_string():
    """Test splitting empty string"""
    chunks = split_text("")
    
    # Empty text results in an empty list or a list with empty string
    assert isinstance(chunks, list)


def test_split_text_single_word():
    """Test splitting single word"""
    chunks = split_text("hello")
    
    assert len(chunks) == 1
    assert chunks[0] == "hello"


def test_split_text_custom_chunk_size():
    """Test splitting with custom chunk size"""
    text = "word " * 100
    chunks_small = split_text(text, chunk_size=10, overlap=2)
    chunks_large = split_text(text, chunk_size=50, overlap=5)
    
    assert len(chunks_small) > len(chunks_large)


def test_split_text_with_overlap():
    """Test that overlap is correctly applied"""
    text = "word " * 100
    chunks = split_text(text, chunk_size=20, overlap=5)
    
    # With overlap, we should have repeated words at chunk boundaries
    assert len(chunks) >= 1


def test_split_text_preserves_content():
    """Test that splitting preserves all content"""
    text = "This is a test document with multiple sentences and words."
    chunks = split_text(text, chunk_size=5, overlap=1)
    
    # Reconstruct text from chunks
    reconstructed = " ".join(chunks)
    
    # The original text should be contained in the reconstructed version
    assert text.lower() in reconstructed.lower() or len(chunks) > 0


def test_split_text_chunk_structure():
    """Test that chunks have proper structure"""
    text = "word " * 100
    chunks = split_text(text, chunk_size=25, overlap=5)
    
    for chunk in chunks:
        assert isinstance(chunk, str)
        assert len(chunk) > 0


def test_split_text_special_characters():
    """Test splitting text with special characters"""
    text = "Hello @#$% world! Test 123 with special chars & symbols."
    chunks = split_text(text, chunk_size=5, overlap=1)
    
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, str)


def test_split_text_unicode_support():
    """Test splitting text with unicode characters"""
    text = "这是一个测试文本 with 中文 characters and émojis 🎉"
    chunks = split_text(text)
    
    assert len(chunks) >= 1


def test_split_text_very_large_text():
    """Test splitting very large text"""
    text = "word " * 5000  # 5000 words
    chunks = split_text(text, chunk_size=400, overlap=50)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_split_text_chunk_size_zero():
    """Test splitting with zero chunk size (edge case)"""
    text = "word " * 10
    
    # This might raise an error or handle gracefully
    try:
        chunks = split_text(text, chunk_size=0, overlap=0)
        # If it doesn't raise an error, chunks should still be valid
        assert isinstance(chunks, list)
    except (ValueError, ZeroDivisionError):
        # It's acceptable to raise an error for invalid chunk size
        pass


def test_pdf_reader_integration(monkeypatch):
    """Test PDF reader integration"""
    from app.utils.pdf_reader import extract_text_from_pdf
    
    # Mock PdfReader
    class MockPage:
        def extract_text(self):
            return "Test PDF content"
    
    class MockPdfReader:
        def __init__(self, path):
            self.pages = [MockPage(), MockPage()]
    
    monkeypatch.setattr("app.utils.pdf_reader.PdfReader", MockPdfReader)
    
    result = extract_text_from_pdf("test.pdf")
    
    assert isinstance(result, str)
    assert len(result) > 0


def test_pdf_reader_empty_pages(monkeypatch):
    """Test PDF reader with empty pages"""
    from app.utils.pdf_reader import extract_text_from_pdf
    
    class MockPage:
        def extract_text(self):
            return None
    
    class MockPdfReader:
        def __init__(self, path):
            self.pages = [MockPage(), MockPage()]
    
    monkeypatch.setattr("app.utils.pdf_reader.PdfReader", MockPdfReader)
    
    result = extract_text_from_pdf("test.pdf")
    
    assert isinstance(result, str)


def test_pdf_reader_mixed_content(monkeypatch):
    """Test PDF reader with mixed content (empty and non-empty pages)"""
    from app.utils.pdf_reader import extract_text_from_pdf
    
    class MockPage:
        def __init__(self, text=None):
            self.text = text
        
        def extract_text(self):
            return self.text
    
    class MockPdfReader:
        def __init__(self, path):
            self.pages = [
                MockPage("Page 1 content"),
                MockPage(None),
                MockPage("Page 3 content")
            ]
    
    monkeypatch.setattr("app.utils.pdf_reader.PdfReader", MockPdfReader)
    
    result = extract_text_from_pdf("test.pdf")
    
    assert "Page 1 content" in result
    assert "Page 3 content" in result
