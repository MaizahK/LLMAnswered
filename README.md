````markdown
# Document QA API

A minimal Document-based Question Answering API built with FastAPI, FAISS, and OpenAI.

## Features

-   POST `/documents` to ingest documents (JSON or plaintext files)
-   GET `/documents` to list indexed documents
-   DELETE `/documents/{doc_id}` to remove a document and associated vectors
-   POST `/query` to ask a question; retrieves relevant chunks via FAISS and asks an LLM to answer
-   Health check endpoint `/health`

## Requirements

-   Python 3.10+
-   OpenAI API key

## Setup

1. Clone the repo (or copy files)
2. Create a virtual environment and activate it

```bash
python -m venv .venv
source .venv/bin/activate  # windows: .venv\Scripts\activate
pip install -r requirements.txt
```
````

3. Create `.env` from `.env.example` and fill in your `OPENAI_API_KEY`

4. Run:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Usage

### Index documents (JSON)

POST `/documents` with body:

```json
{
	"documents": [
		{ "id": "doc1", "title": "My Doc", "content": "Long content..." }
	]
}
```

### Query

POST `/query` with body:

```json
{ "question": "What is X?", "top_k": 5 }
```

Response includes `answer` and `sources` (list of documents/chunks used).

## Files

-   `app/vector_store.py` manages FAISS index and metadata persistence in `./data/`
-   `app/utils.py` contains chunking and PDF/TXT/MD parsing

```

```
