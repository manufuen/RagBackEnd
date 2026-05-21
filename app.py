from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ingestion import process_document
from rag import answer_question
from vector_store import check_elasticsearch_connection, es


app = FastAPI(title="RAG Chat Backend", version="0.2")

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