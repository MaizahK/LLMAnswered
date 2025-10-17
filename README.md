# Document QA API

A minimal **Document-based Question Answering API** built with **FastAPI**, **FAISS**, and **OpenAI**.

---

## 🚀 Features

-   **POST** `/documents` → Ingest documents (JSON or plaintext files)
-   **GET** `/documents` → List indexed documents
-   **DELETE** `/documents/{doc_id}` → Remove a document and its associated vectors
-   **POST** `/query` → Ask a question; retrieves relevant chunks via FAISS and generates an answer using OpenAI
-   **GET** `/health` → Health check endpoint

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

---

## 📄 Usage

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

````

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

## 📁 Project Structure

```
app/
│
├── main.py                # FastAPI entry point
├── vector_store.py        # Manages FAISS index & metadata in ./data/
├── utils.py               # Chunking and file parsing (PDF, TXT, MD)
└── ...
```

---

## ✅ Health Check

Visit:

```
GET /health
```

**Response:**

```json
{ "status": "ok" }
```

```

```

````
