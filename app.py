'''
RAG Chat Backend - app.py
'''
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ingestion import process_document
from rag import answer_question
from vector_store import check_elasticsearch_connection, es


app = FastAPI(title="RAG Chat Backend", version="0.2")

# Configuración de directorios para almacenamiento temporal de archivos subidos
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"

'''
funciones internas del backend
'''
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

def find_uploaded_file_for_document(document_id: str) -> Path | None:
    """
    Busca el archivo físico asociado a un documento.
    Se asume que los archivos se guardan como:
    backend/uploads/<document_id>_<nombre_original>
    """
    if not UPLOAD_DIR.exists():
        return None

    matches = list(UPLOAD_DIR.glob(f"{document_id}_*"))

    if not matches:
        return None

    return matches[0]
''' 
Configuración CORS para permitir solicitudes desde el frontend
En tu proyecto tienes dos servicios separados:
Frontend Streamlit → http://localhost:8501
Backend FastAPI    → http://localhost:8000
Como son puertos distintos, el navegador los considera orígenes diferentes. Sin CORS, algunas peticiones del frontend al backend podrían bloquearse.

'''
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

'''
Modelo de datos que espera recibir el endpoint /chat
'''
class ChatRequest(BaseModel):
    question: str

''' 
Endpoints 
'''
@app.get("/")
# Endpoint raíz para verificar que el backend está funcionando
def root():
    return {
        "status": "ok",
        "message": "RAG Chat Backend funcionando"
    }


@app.get("/health/elasticsearch")
# Endpoint para verificar la conexión con Elasticsearch
def elasticsearch_health():
    try:
        return check_elasticsearch_connection()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo conectar con Elasticsearch: {str(e)}"
        )

@app.get("/documents/{document_id}/file")
def get_document_file(document_id: str):
    """
    Devuelve el archivo físico asociado a un documento ingestados.
    Permite abrirlo desde el frontend.
    """

    file_path = find_uploaded_file_for_document(document_id)

    if not file_path or not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No se encontró el archivo físico para el documento '{document_id}'."
        )

    original_filename = file_path.name

    if "_" in original_filename:
        original_filename = original_filename.split("_", 1)[1]

    return FileResponse(
        path=file_path,
        filename=original_filename,
        media_type="application/octet-stream",
    )

@app.get("/indices")
# Endpoint para listar los índices RAG existentes en Elasticsearch
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
# Endpoint para subir un documento y procesarlo (clasificación, chunking, vectorización, almacenamiento)
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
# Endpoint para recibir una pregunta del frontend, procesarla y devolver una respuesta generada por el sistema RAG
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
# Endpoint para eliminar un índice RAG concreto de Elasticsearch
def delete_index(index_name: str):

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
# Endpoint para eliminar todos los índices RAG existentes de Elasticsearch
def delete_all_rag_indices():

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
# Endpoint para listar los documentos ingestados agrupando chunks por documento_id
def list_documents():

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
# Endpoint para eliminar un documento concreto (todos sus chunks en Elasticsearch + archivo físico asociado en backend/uploads/)
def delete_document(document_id: str):

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
# Endpoint para eliminar todos los documentos ingestados
def delete_all_documents():

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