from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ingestion import process_document
from rag import answer_question
from vector_store import check_elasticsearch_connection, es


app = FastAPI(title="RAG Chat Backend", version="0.2")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"


def get_existing_rag_indices() -> list[str]:
    """
    Devuelve todos los índices RAG existentes en Elasticsearch.
    """
    try:
        indices_response = es.indices.get_alias(index="rag_*")
        return list(indices_response.keys())
    except Exception:
        return []


def delete_uploaded_files_for_document(document_id: str) -> list[str]:
    """
    Elimina archivos físicos de backend/uploads/ cuyo nombre empiece por document_id.
    Ejemplo:
    backend/uploads/<document_id>_Biologia.pdf
    """
    deleted_files = []

    if not UPLOAD_DIR.exists():
        return deleted_files

    for file_path in UPLOAD_DIR.glob(f"{document_id}_*"):
        if file_path.is_file():
            file_path.unlink()
            deleted_files.append(str(file_path.name))

    return deleted_files


def delete_all_uploaded_files() -> list[str]:
    """
    Elimina todos los archivos físicos de backend/uploads/.
    """
    deleted_files = []

    if not UPLOAD_DIR.exists():
        return deleted_files

    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file() and file_path.name != ".gitkeep":
            file_path.unlink()
            deleted_files.append(str(file_path.name))

    return deleted_files

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "RAG Chat Backend funcionando"
    }


@app.get("/health/elasticsearch")
def elasticsearch_health():
    try:
        return check_elasticsearch_connection()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo conectar con Elasticsearch: {str(e)}"
        )


@app.get("/indices")
def list_indices():
    try:
        indices = es.indices.get_alias(index="rag_*")

        return {
            "indices": list(indices.keys())
        }

    except Exception:
        return {
            "indices": []
        }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        result = await process_document(file)

        if result.get("already_ingested"):
            return {
                "status": "duplicated",
                "message": "documento ya ingestado",
                "details": result
            }

        return {
            "status": "ok",
            "details": result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        result = answer_question(request.question)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
@app.delete("/indices/{index_name}")
def delete_index(index_name: str):
    """
    Elimina un índice RAG completo de Elasticsearch.
    Ejemplo:
    DELETE /indices/rag_musica
    """

    if not index_name.startswith("rag_"):
        raise HTTPException(
            status_code=400,
            detail="Solo se pueden eliminar índices RAG que empiecen por 'rag_'."
        )

    try:
        if not es.indices.exists(index=index_name):
            raise HTTPException(
                status_code=404,
                detail=f"El índice '{index_name}' no existe."
            )

        response = es.indices.delete(index=index_name)

        return {
            "status": "ok",
            "message": f"Índice '{index_name}' eliminado correctamente.",
            "details": response,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo eliminar el índice '{index_name}': {str(e)}"
        )
    
@app.delete("/indices")
def delete_all_rag_indices():
    """
    Elimina todos los índices RAG existentes en Elasticsearch.
    Ejemplo:
    DELETE /indices
    """

    try:
        try:
            indices_response = es.indices.get_alias(index="rag_*")
            indices = list(indices_response.keys())
        except Exception:
            indices = []

        if not indices:
            return {
                "status": "ok",
                "message": "No hay índices RAG para eliminar.",
                "deleted_indices": []
            }

        deleted_indices = []

        for index_name in indices:
            if index_name.startswith("rag_"):
                es.indices.delete(index=index_name)
                deleted_indices.append(index_name)

        return {
            "status": "ok",
            "message": "Todos los índices RAG han sido eliminados correctamente.",
            "deleted_indices": deleted_indices
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"No se pudieron eliminar todos los índices RAG: {str(e)}"
        )    
@app.get("/documents")
def list_documents():
    """
    Lista los documentos ingestados agrupando chunks por documento_id.
    """

    indices = get_existing_rag_indices()

    if not indices:
        return {
            "status": "ok",
            "total_documents": 0,
            "documents": []
        }

    try:
        response = es.search(
            index=",".join(indices),
            body={
                "size": 0,
                "aggs": {
                    "documents": {
                        "terms": {
                            "field": "documento_id",
                            "size": 1000
                        },
                        "aggs": {
                            "sample": {
                                "top_hits": {
                                    "size": 1,
                                    "_source": [
                                        "documento_id",
                                        "nombre_documento",
                                        "tema",
                                        "fecha_ingesta",
                                        "file_hash",
                                        "file_size"
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        )

        buckets = (
            response
            .get("aggregations", {})
            .get("documents", {})
            .get("buckets", [])
        )

        documents = []

        for bucket in buckets:
            sample_hits = bucket.get("sample", {}).get("hits", {}).get("hits", [])

            if not sample_hits:
                continue

            source = sample_hits[0].get("_source", {})
            index_name = sample_hits[0].get("_index")

            documents.append({
                "documento_id": source.get("documento_id"),
                "nombre_documento": source.get("nombre_documento"),
                "tema": source.get("tema"),
                "index_name": index_name,
                "chunks": bucket.get("doc_count", 0),
                "fecha_ingesta": source.get("fecha_ingesta"),
                "file_hash": source.get("file_hash"),
                "file_size": source.get("file_size"),
            })

        documents.sort(
            key=lambda item: item.get("fecha_ingesta") or "",
            reverse=True
        )

        return {
            "status": "ok",
            "total_documents": len(documents),
            "documents": documents
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error consultando documentos: {str(e)}"
        )  
@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    """
    Elimina un documento concreto:
    - Borra todos sus chunks en Elasticsearch.
    - Borra el archivo físico asociado en backend/uploads/.
    - Si un índice queda vacío, lo elimina.
    """

    indices = get_existing_rag_indices()

    if not indices:
        raise HTTPException(
            status_code=404,
            detail="No hay índices RAG disponibles."
        )

    total_deleted_chunks = 0
    affected_indices = []

    try:
        for index_name in indices:
            count_response = es.count(
                index=index_name,
                body={
                    "query": {
                        "term": {
                            "documento_id": document_id
                        }
                    }
                }
            )

            count = count_response.get("count", 0)

            if count == 0:
                continue

            delete_response = es.delete_by_query(
                index=index_name,
                body={
                    "query": {
                        "term": {
                            "documento_id": document_id
                        }
                    }
                },
                refresh=True
            )

            deleted = delete_response.get("deleted", 0)
            total_deleted_chunks += deleted
            affected_indices.append(index_name)

            remaining_response = es.count(index=index_name)
            remaining_docs = remaining_response.get("count", 0)

            if remaining_docs == 0:
                es.indices.delete(index=index_name)

        if total_deleted_chunks == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontró ningún documento con id '{document_id}'."
            )

        deleted_files = delete_uploaded_files_for_document(document_id)

        return {
            "status": "ok",
            "message": "Documento eliminado correctamente.",
            "documento_id": document_id,
            "deleted_chunks": total_deleted_chunks,
            "affected_indices": affected_indices,
            "deleted_files": deleted_files,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo eliminar el documento '{document_id}': {str(e)}"
        )   
@app.delete("/documents")
def delete_all_documents():
    """
    Elimina todos los documentos ingestados:
    - Borra todos los índices rag_* de Elasticsearch.
    - Borra los archivos físicos de backend/uploads/.
    """

    indices = get_existing_rag_indices()

    deleted_indices = []

    try:
        for index_name in indices:
            if index_name.startswith("rag_"):
                es.indices.delete(index=index_name)
                deleted_indices.append(index_name)

        deleted_files = delete_all_uploaded_files()

        return {
            "status": "ok",
            "message": "Todos los documentos ingestados han sido eliminados correctamente.",
            "deleted_indices": deleted_indices,
            "deleted_files": deleted_files,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"No se pudieron eliminar todos los documentos: {str(e)}"
        )       