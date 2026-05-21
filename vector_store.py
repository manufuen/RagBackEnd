import os
import re
import unicodedata
from typing import Any

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer


load_dotenv()

ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ELASTICSEARCH_USER = os.getenv("ELASTICSEARCH_USER")
ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

EMBEDDING_DIMS = 384

model = SentenceTransformer(EMBEDDING_MODEL)


def get_es_client() -> Elasticsearch:
    if ELASTICSEARCH_USER and ELASTICSEARCH_PASSWORD:
        return Elasticsearch(
            ELASTICSEARCH_URL,
            basic_auth=(ELASTICSEARCH_USER, ELASTICSEARCH_PASSWORD),
            request_timeout=60,
        )

    return Elasticsearch(
        ELASTICSEARCH_URL,
        request_timeout=60,
    )


es = get_es_client()


def check_elasticsearch_connection() -> dict[str, Any]:
    info = es.info()

    return {
        "connected": True,
        "cluster_name": info.get("cluster_name"),
        "version": info.get("version", {}).get("number"),
    }


def normalize_index_part(text: str) -> str:
    text = text.lower().strip()

    text = unicodedata.normalize("NFD", text)
    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Mn"
    )

    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text)

    return text.strip("_")


def topic_to_index_name(tema: str) -> str:
    safe_tema = normalize_index_part(tema)
    return f"rag_{safe_tema}"


def create_index_if_not_exists(index_name: str):
    if es.indices.exists(index=index_name):
        return

    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                "texto": {
                    "type": "text"
                },
                "embedding": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIMS,
                    "index": True,
                    "similarity": "cosine"
                },
                "tema": {
                    "type": "keyword"
                },
                "documento_id": {
                    "type": "keyword"
                },
                "nombre_documento": {
                    "type": "keyword"
                },
                "chunk_id": {
                    "type": "integer"
                },
                "fecha_ingesta": {
                    "type": "date"
                },
                "autor": {
                    "type": "keyword"
                },
                "keywords": {
                    "type": "keyword"
                },
                "resumen_chunk": {
                    "type": "text"
                }
            }
        }
    }

    es.indices.create(index=index_name, body=mapping)


def create_chunk_summary(chunk: str, max_chars: int = 300) -> str:
    chunk = " ".join(chunk.split())

    if len(chunk) <= max_chars:
        return chunk

    return chunk[:max_chars].rsplit(" ", 1)[0] + "..."


def store_chunks(
    chunks: list[str],
    tema: str,
    filename: str,
    document_id: str,
    fecha_ingesta: str,
    author: str | None = None,
    keywords: list[str] | None = None,
) -> dict[str, Any]:
    index_name = topic_to_index_name(tema)
    create_index_if_not_exists(index_name)

    keywords = keywords or []

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    actions = []

    for i, chunk in enumerate(chunks):
        doc = {
            "texto": chunk,
            "embedding": embeddings[i],
            "tema": tema,
            "documento_id": document_id,
            "nombre_documento": filename,
            "chunk_id": i,
            "fecha_ingesta": fecha_ingesta,
            "autor": author,
            "keywords": keywords,
            "resumen_chunk": create_chunk_summary(chunk),
        }

        actions.append({
            "_op_type": "index",
            "_index": index_name,
            "_source": doc,
        })

    helpers.bulk(es, actions)
    es.indices.refresh(index=index_name)

    return {
        "index_name": index_name,
        "chunks_indexed": len(chunks),
    }