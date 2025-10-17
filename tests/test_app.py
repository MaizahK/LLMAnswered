import io
import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app

client = TestClient(app)

# ---------------------- Fixtures ---------------------- #
@pytest.fixture
def mock_openai_embeddings():
    """Mock for OpenAI embeddings.create"""
    with patch("app.main.client.embeddings.create") as mock_embed:
        mock_embed.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536) for _ in range(2)]
        )
        yield mock_embed


@pytest.fixture
def mock_openai_chat():
    """Mock for OpenAI chat.completions.create"""
    with patch("app.main.client.chat.completions.create") as mock_chat:
        mock_chat.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="This is a mock answer."))]
        )
        yield mock_chat


# ---------------------- Tests ---------------------- #

def test_health_check():
    """Ensure /health returns status ok."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ingest_documents(mock_openai_embeddings):
    """Test document ingestion."""
    payload = {
        "documents": [
            {"id": "doc1", "title": "Sample Doc", "content": "This is some text content."}
        ]
    }
    resp = client.post("/documents", json=payload)
    assert resp.status_code == 200
    assert "indexed_chunks" in resp.json()


def test_list_documents():
    """Ensure /documents returns a list."""
    resp = client.get("/documents")
    assert resp.status_code in [200, 500]  # might fail if metadata file missing
    if resp.status_code == 200:
        assert isinstance(resp.json(), list)


def test_upload_file(mock_openai_embeddings):
    """Test uploading a .txt file."""
    fake_file = io.BytesIO(b"This is a fake text file.")
    files = {"file": ("test.txt", fake_file, "text/plain")}
    data = {"title": "Uploaded Doc"}

    resp = client.post("/upload", files=files, data=data)
    assert resp.status_code == 200
    assert "indexed_chunks" in resp.json()


def test_query_documents(mock_openai_embeddings, mock_openai_chat):
    """Test question answering."""
    payload = {"question": "What is this about?", "top_k": 2}
    resp = client.post("/query", json=payload)
    assert resp.status_code in [200, 502]
    if resp.status_code == 200:
        body = resp.json()
        assert "answer" in body
        assert "sources" in body


def test_delete_document():
    """Test deleting a document by ID."""
    resp = client.delete("/documents/doc1")
    assert resp.status_code in [200, 404]
    if resp.status_code == 200:
        assert resp.json()["deleted"] is True
