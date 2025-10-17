import io
from typing import List
from PyPDF2 import PdfReader




def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Simple sliding window chunker."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks




def read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text = []
    for p in reader.pages:
        try:
            text.append(p.extract_text() or "")
        except Exception:
            text.append("")
    return "\n".join(text)




def read_text(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8")




def read_markdown(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8")