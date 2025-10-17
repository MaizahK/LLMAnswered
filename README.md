# Document Based QA API

A minimal **Document-based Question Answering API** built with **FastAPI**, **FAISS**, and **OpenAI**.

---

## 🚀 Features

-   **GET** `/docs` → Swagger generated API documentation
-   **GET** `/health` → Health check endpoint
-   **GET** `/documents` → List indexed documents
-   **POST** `/documents` → Ingest documents (JSON or plaintext files)
-   **DELETE** `/documents/{doc_id}` → Remove a document and its associated vectors
-   **POST** `/query` → Ask a question; retrieves relevant chunks via FAISS and generates an answer using OpenAI
-   **POST** `/upload` → Upload and automatically index PDF, TXT, or Markdown files

---

## 📁 Project Structure

```
app/
│
├── main.py                # FastAPI entry point
├── vector_store.py        # Manages FAISS index & metadata in ./data/
├── utils.py               # Chunking and file parsing (PDF, TXT, MD)
└── ...
tests/
│
├── test_app.py            # Unit tests to validate all endpoints
└── ...
requirements.txt
.env
.gitignore
myenv/
README.md
```

---

## 🧩 Requirements

-   Python **3.10+**
-   **OpenAI API key**

---

## ⚙️ Setup

1. **Clone the repository** (or copy the files)

2. **Create and activate a virtual environment**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Create a `.env` file** from `.env.example` and add your `OPENAI_API_KEY`

4. **Run the application**

    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```

5. **Test the application**

    ```bash
    pytest -v
    ```

---

## 📄 Usage

### 📝 API Docs

Visit:

```
GET /docs
```

**Response:**

Rendered HTML page to display all API endpoints

---

### ✅ Health Check

Visit:

```
GET /health
```

**Response:**

```json
{ "status": "ok" }
```

---

### 🗃️ Display Documents (JSON)

**GET** `/documents`

**Response:**

List of all the submitted documents

---

### ➕ Index Documents (JSON)

**POST** `/documents`

**Body example:**

```json
{
	"documents": [
		{ "id": "doc1", "title": "My Doc", "content": "Long content..." }
	]
}
```

---

### ❓ Query

**POST** `/query`

**Body example:**

```json
{
	"question": "What is X?",
	"top_k": 5
}
```

**Response:**

```json
{
	"answer": "Generated answer text...",
	"sources": ["doc1_chunk_1", "doc1_chunk_2"]
}
```

---

### 🗂️ Upload Documents (File)

**POST** `/upload`

Upload a file (PDF, TXT, or Markdown) directly to the API.  
The file will be automatically read, chunked, embedded, and indexed for semantic search and question answering.

**Form Data Example (using cURL):**

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@/path/to/document.pdf" \
  -F "title=My PDF Document"
```

**Response:**

```json
{
	"indexed_chunks": 12
}
```

**Notes:**

-   The file is converted to text using appropriate readers:

    -   `.pdf` → extracted using `read_pdf()`
    -   `.md` / `.markdown` → parsed using `read_markdown()`
    -   `.txt` → read using `read_text()`

-   If the file format is unsupported, a clear error message is returned.
-   Optional fields:

    -   `title`: Custom document title (defaults to filename)
    -   `doc_id`: Custom document ID (auto-generated if not provided)

---
