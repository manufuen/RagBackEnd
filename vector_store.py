import os # Para manejar variables de entorno
import re # Para procesar texto y extraer JSON
import unicodedata # Para normalizar texto y crear nombres de índices seguros  
from typing import Any 

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers # Para interactuar con Elasticsearch y hacer operaciones bulk
from sentence_transformers import SentenceTransformer # Para generar embeddings de los chunks de texto


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
    # Crea un cliente de Elasticsearch, usando autenticación básica si se proporcionan las credenciales.
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
    # Verifica la conexión a Elasticsearch y devuelve información básica del cluster.
    info = es.info()

    return {
        "connected": True,
        "cluster_name": info.get("cluster_name"),
        "version": info.get("version", {}).get("number"),
    }

def normalize_index_part(text: str) -> str:
    # Normaliza un texto para usarlo como parte de un nombre de índice en Elasticsearch, eliminando acentos, caracteres especiales y espacios.
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
    # Convierte un tema a un nombre de índice seguro para Elasticsearch, asegurándose de que solo contenga caracteres válidos y esté en minúsculas.
    safe_tema = normalize_index_part(tema)
    return f"rag_{safe_tema}"


def create_index_if_not_exists(index_name: str):
    # Crea un índice en Elasticsearch con la configuración y mapeo necesarios si no existe ya.

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
                "file_hash": {
                    "type": "keyword"
                },
                "file_size": {
                    "type": "long"
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
    # Crea un resumen simple de un chunk de texto, truncándolo a un número máximo de caracteres sin cortar palabras por la mitad.
    chunk = " ".join(chunk.split())

    if len(chunk) <= max_chars:
        return chunk

    return chunk[:max_chars].rsplit(" ", 1)[0] + "..."


def find_document_by_hash(file_hash: str) -> dict | None:

    # Busca en Elasticsearch un documento que tenga el mismo hash de archivo, lo que indica que ya ha sido ingestado previamente, y devuelve su información básica si se encuentra.
    try:
        response = es.search(
            index="rag_*",
            body={
                "size": 1,
                "query": {
                    "term": {
                        "file_hash": file_hash
                    }
                },
                "_source": [
                    "documento_id",
                    "nombre_documento",
                    "tema",
                    "fecha_ingesta",
                    "file_hash"
                ]
            },
            ignore_unavailable=True,
        )

        hits = response.get("hits", {}).get("hits", [])

        if not hits:
            return None

        return hits[0].get("_source", {})

    except Exception:
        return None
        
def store_chunks(
    chunks: list[str],
    tema: str,
    filename: str,
    document_id: str,
    fecha_ingesta: str,
    author: str | None = None,
    keywords: list[str] | None = None,
    file_hash: str | None = None,
    file_size: int | None = None,
) -> dict[str, Any]:
    '''
    Almacena los chunks de un documento en Elasticsearch, generando embeddings para cada chunk y guardando metadatos relevantes. Crea el índice correspondiente al tema si no existe.
    '''
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
            "file_hash": file_hash,
            "file_size": file_size,
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