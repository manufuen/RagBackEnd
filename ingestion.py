import uuid
from pathlib import Path
from datetime import datetime, timezone

from fastapi import UploadFile

from classification import classify_document
from chunking import chunk_text
from vector_store import store_chunks
from utils import extract_text, extract_author, extract_keywords


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


async def process_document(file: UploadFile):
    document_id = str(uuid.uuid4())

    original_filename = file.filename or "documento_sin_nombre"
    file_path = UPLOAD_DIR / f"{document_id}_{original_filename}"

    content = await file.read()

    if not content:
        raise ValueError("El archivo está vacío.")

    with open(file_path, "wb") as f:
        f.write(content)

    text = extract_text(str(file_path))

    if not text or not text.strip():
        raise ValueError("No se pudo extraer texto del documento.")

    tema = classify_document(text)

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
    }