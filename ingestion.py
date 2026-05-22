import uuid
import hashlib
from pathlib import Path
from datetime import datetime, timezone

from fastapi import UploadFile

from classification import classify_document
from chunking import chunk_text
from vector_store import store_chunks, find_document_by_hash
from utils import extract_text, extract_author, extract_keywords


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

def calculate_file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

async def process_document(file: UploadFile):
    document_id = str(uuid.uuid4())

    original_filename = file.filename or "documento_sin_nombre"
    file_path = UPLOAD_DIR / f"{document_id}_{original_filename}"

    content = await file.read()
    file_hash = calculate_file_hash(content)
    file_size = len(content)

    existing_document = find_document_by_hash(file_hash)

    if existing_document:
        return {
            "already_ingested": True,
            "message": "documento ya ingestado",
            "documento_id": existing_document.get("documento_id"),
            "nombre_documento": existing_document.get("nombre_documento"),
            "tema": existing_document.get("tema"),
            "fecha_ingesta": existing_document.get("fecha_ingesta"),
            "file_hash": file_hash,
        }
    
    if not content:
        raise ValueError("El archivo está vacío.")

    with open(file_path, "wb") as f:
        f.write(content)

    text = extract_text(str(file_path))

    if not text or not text.strip():
        raise ValueError("No se pudo extraer texto del documento.")

    tema = classify_document(text, original_filename)
    chunks = chunk_text(text)

    if not chunks:
        raise ValueError("No se generaron chunks a partir del documento.")

    author = extract_author(str(file_path))
    keywords = extract_keywords(text)

    fecha_ingesta = datetime.now(timezone.utc).isoformat()

    result = store_chunks(
        chunks=chunks,
        tema=tema,
        filename=original_filename,
        document_id=document_id,
        fecha_ingesta=fecha_ingesta,
        author=author,
        keywords=keywords,
        file_hash=file_hash,
        file_size=file_size,
    )

    return {
        "documento_id": document_id,
        "nombre_documento": original_filename,
        "tema": tema,
        "chunks": len(chunks),
        "indice_elastic": result["index_name"],
        "keywords": keywords,
        "autor": author,
        "fecha_ingesta": fecha_ingesta,
        "already_ingested": False,
        "file_hash": file_hash,
        "file_size": file_size,
    }