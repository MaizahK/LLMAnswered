import os
import json
from typing import List, Dict, Any
import numpy as np
import faiss


class VectorStore:
    """
    Simple FAISS-backed vector store with metadata persisted to JSON.

    Metadata layout (list of items):
    [
        {
            "vector_id": int,
            "doc_id": str,
            "title": str,
            "chunk_text": str,
            "embedding": [float, ...]  # optional for rebuild
        },
        ...
    ]
    """

    def __init__(
        self,
        index_path: str = "./data/faiss.index",
        metadata_path: str = "./data/metadata.json",
        dim: int = 1536,
    ):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dim = dim
        self._load_index()
        self._load_metadata()

    # ------------------ Internal loaders/savers ------------------
    def _load_index(self):
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
            else:
                self.index = faiss.IndexFlatIP(self.dim)
        except Exception:
            self.index = faiss.IndexFlatIP(self.dim)

    def _load_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []

    def _save_metadata(self):
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def _save_index(self):
        faiss.write_index(self.index, self.index_path)

    # ------------------ Public API ------------------
    def add_vectors(self, embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        """
        Add vectors and their metadatas.
        Each metadata dict must include: doc_id, title, chunk_text
        """
        if not embeddings:
            return []

        embs = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embs)

        start_id = len(self.metadata)
        self.index.add(embs)

        for i, md in enumerate(metadatas):
            self.metadata.append({"vector_id": start_id + i, **md})

        self._save_index()
        self._save_metadata()
        return [m["vector_id"] for m in self.metadata[start_id:]]

    def search(self, query_emb: List[float], top_k: int = 5):
        """
        Search for top_k similar embeddings using cosine similarity.
        """
        if self.index.ntotal == 0:
            return []

        vec = np.array([query_emb]).astype("float32")
        faiss.normalize_L2(vec)

        D, I = self.index.search(vec, top_k)
        results = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0 or idx >= len(self.metadata):
                continue
            md = self.metadata[idx]
            results.append({"score": float(score), **md})
        return results

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        Returns all unique document IDs with titles and chunk counts.
        """
        docs = {}
        for md in self.metadata:
            doc_id = md.get("doc_id")
            if not doc_id:
                continue
            if doc_id not in docs:
                docs[doc_id] = {"id": doc_id, "title": md.get("title", ""), "chunks": 0}
            docs[doc_id]["chunks"] += 1
        return list(docs.values())

    def delete_by_doc_id(self, doc_id: str) -> bool:
        """
        Delete all chunks and vectors belonging to a document ID.
        Rebuilds the FAISS index afterward.
        """
        new_meta = [m for m in self.metadata if m["doc_id"] != doc_id]

        if len(new_meta) == len(self.metadata):
            return False  # no match found

        self.metadata = new_meta

        # Rebuild index
        self._rebuild_index()

        self._save_index()
        self._save_metadata()
        return True

    def _rebuild_index(self):
        """
        Rebuilds FAISS index from stored embeddings (if available).
        If embeddings are missing, resets the index.
        """
        if not self.metadata:
            self.index = faiss.IndexFlatIP(self.dim)
            return

        embs = [m.get("embedding") for m in self.metadata if m.get("embedding") is not None]
        if len(embs) == len(self.metadata):
            embs_np = np.array(embs).astype("float32")
            faiss.normalize_L2(embs_np)
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(embs_np)
        else:
            # Cannot rebuild without stored embeddings → reset
            self.index = faiss.IndexFlatIP(self.dim)

        # Reassign vector IDs sequentially
        for i, m in enumerate(self.metadata):
            m["vector_id"] = i

    def persist_embedding_in_metadata(self, vector_id: int, embedding: List[float]):
        """
        Persist embedding into metadata for possible future rebuilds.
        """
        if 0 <= vector_id < len(self.metadata):
            self.metadata[vector_id]["embedding"] = embedding
            self._save_metadata()

    def load_metadata(self) -> Dict[int, Dict[str, Any]]:
        """
        Return metadata as a dict keyed by vector_id for easier lookups.
        """
        return {m["vector_id"]: m for m in self.metadata}
