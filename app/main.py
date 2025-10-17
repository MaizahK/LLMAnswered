# app/main.py
import os
import uuid
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import openai
from dotenv import load_dotenv

from .vector_store import VectorStore
from .utils import chunk_text, read_pdf, read_text, read_markdown

# ------------------ Load environment ------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Create a .env with your key or set env var.")

openai.api_key = OPENAI_API_KEY
client = openai.OpenAI()

# ------------------ Initialize app & store ------------------
app = FastAPI(title="Document QA API")

# vector dimension for text-embedding-3-small is 1536
VECTOR_DIM = 1536
store = VectorStore(
    index_path=os.getenv("FAISS_INDEX_PATH", "./data/faiss.index"),
    metadata_path=os.getenv("METADATA_PATH", "./data/metadata.json"),
    dim=VECTOR_DIM,
)

# ------------------ Schemas ------------------
class DocumentIn(BaseModel):
    id: str
    title: str
    content: str

class DocumentsPayload(BaseModel):
    documents: List[DocumentIn]

class QueryIn(BaseModel):
    question: str
    top_k: int = 5

# ------------------ Health ------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

# ------------------ List Documents ------------------
@app.get("/documents")
async def list_documents():
    """
    Returns a list of all documents and their metadata.
    """
    try:
        metadata = store.load_metadata()
        doc_summary = {}
        for meta in metadata.values():
            doc_id = meta.get("doc_id")
            if not doc_id:
                continue
            if doc_id not in doc_summary:
                doc_summary[doc_id] = {"title": meta.get("title"), "chunks": 0}
            doc_summary[doc_id]["chunks"] += 1
        return [{"id": k, **v} for k, v in doc_summary.items()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ Delete Document ------------------
@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    Deletes all chunks and embeddings associated with a specific document ID.
    """
    try:
        deleted = store.delete_by_doc_id(doc_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"deleted": True, "doc_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ Ingest Documents ------------------
@app.post("/documents")
async def ingest_documents(payload: DocumentsPayload):
    docs = payload.documents
    all_embeddings = []
    all_meta = []

    for doc in docs:
        chunks = chunk_text(doc.content)
        texts = chunks

        try:
            # Try to get real embeddings from OpenAI
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            embeddings = [r.embedding for r in resp.data]

        except openai.RateLimitError:
            # User exceeded quota → return
            raise HTTPException(
                status_code=502,
                detail="OpenAI API quota exceeded."
            )

        except openai.AuthenticationError:
            raise HTTPException(
                status_code=401,
                detail="Invalid OpenAI API key. Please check your OPENAI_API_KEY environment variable."
            )

        except openai.APIError as e:
            raise HTTPException(
                status_code=502,
                detail=f"OpenAI API error: {str(e)}"
            )

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error while creating embeddings: {str(e)}"
            )

        # Process embeddings and metadata
        for emb, chunk in zip(embeddings, chunks):
            meta = {"doc_id": doc.id, "title": doc.title, "chunk_text": chunk}
            all_embeddings.append(emb)
            all_meta.append(meta)

    # Store in FAISS
    vector_ids = store.add_vectors(all_embeddings, all_meta)
    for vid, emb in zip(vector_ids, all_embeddings):
        store.persist_embedding_in_metadata(vid, emb)

    return {"indexed_chunks": len(all_embeddings)}

# ------------------ Upload File ------------------
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    title: str = Form(None),
    doc_id: str = Form(None),
):
    """
    Upload a file (PDF, TXT, or Markdown). The file is read, chunked, embedded,
    and indexed via the same ingest_documents flow.
    """
    data = await file.read()
    filename = file.filename or "file"
    lower = filename.lower()

    if lower.endswith(".pdf"):
        content = read_pdf(data)
    elif lower.endswith(".md") or lower.endswith(".markdown"):
        content = read_markdown(data)
    elif lower.endswith(".txt"):
        content = read_text(data)
    else:
        try:
            content = read_text(data)
        except Exception:
            return {"error": "Unsupported file format. Supported: .pdf, .txt, .md/.markdown"}

    if not doc_id:
        doc_id = str(uuid.uuid4())
    if not title:
        title = filename

    payload = DocumentsPayload(documents=[DocumentIn(id=doc_id, title=title, content=content)])
    return await ingest_documents(payload)

# ------------------ Query Endpoint ------------------
@app.post("/query")
async def query_documents(payload: QueryIn):
    """
    Given a user question, this endpoint:
    1. Embeds the question
    2. Finds the most relevant document chunks via FAISS
    3. Uses the LLM to generate a contextual answer
    """
    question = payload.question
    top_k = payload.top_k

    try:
        # Step 1: Embed the user question
        qresp = client.embeddings.create(model=EMBED_MODEL, input=[question])
        query_emb = qresp.data[0].embedding
    except openai.RateLimitError:
        raise HTTPException(
            status_code=502,
            detail="OpenAI API quota exceeded while embedding the query."
        )
    except openai.AuthenticationError:
        raise HTTPException(
            status_code=401,
            detail="Invalid OpenAI API key. Please check your OPENAI_API_KEY environment variable."
        )
    except openai.APIError as e:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI API error while embedding query: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while embedding query: {str(e)}"
        )

    # Step 2: Search FAISS for similar chunks
    try:
        results = store.search(query_emb, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")

    if not results:
        return {"answer": "No relevant documents found.", "sources": []}

    # Step 3: Combine top chunks for context
    context = "\n\n".join([r["chunk_text"] for r in results])

    # Step 4: Ask the LLM for an answer
    prompt = f"Answer the question below based on the provided context.\n\nContext:\n{context}\n\nQuestion: {question}"

    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on documents."},
                {"role": "user", "content": prompt},
            ],
        )
        answer = completion.choices[0].message.content.strip()
    except openai.RateLimitError:
        raise HTTPException(
            status_code=502,
            detail="OpenAI API quota exceeded while generating the answer."
        )
    except openai.APIError as e:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI API error while generating answer: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while generating answer: {str(e)}"
        )

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {"title": r["title"], "score": r["score"], "chunk_text": r["chunk_text"]}
            for r in results
        ],
    }
