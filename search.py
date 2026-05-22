import os
from typing import Any

from classification import classify_question
from vector_store import es, model, topic_to_index_name


MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.50"))


def normalize_bm25_score(score: float, max_score: float) -> float:
    if not max_score or max_score <= 0:
        return 0.0

    return score / max_score


def normalize_vector_score(score: float) -> float:
    """
    Elasticsearch devuelve:
    cosineSimilarity(...) + 1.0

    Eso deja el score entre 0 y 2 aproximadamente.
    Lo normalizamos a 0-1.
    """
    return max(0.0, min(score / 2.0, 1.0))


def build_result_key(hit: dict[str, Any]) -> str:
    source = hit.get("_source", {})
    documento_id = source.get("documento_id", "")
    chunk_id = source.get("chunk_id", "")

    if documento_id != "" and chunk_id != "":
        return f"{documento_id}_{chunk_id}"

    return hit["_id"]


def get_rag_indices() -> list[str]:
    try:
        indices = es.indices.get_alias(index="rag_*")
        return list(indices.keys())
    except Exception:
        return []


def extract_core_query_terms(question: str) -> str:
    stopwords = {
        "que", "qué", "es", "son", "un", "una", "unos", "unas",
        "el", "la", "los", "las", "de", "del", "en", "sobre",
        "me", "puedes", "puede", "explicar", "explica",
        "define", "definicion", "definición", "concepto"
    }

    words = question.lower().replace("?", "").replace("¿", "").split()

    core_words = [
        word for word in words
        if word not in stopwords and len(word) > 2
    ]

    return " ".join(core_words).strip()


def build_text_query(question: str, document_id: str | None = None) -> dict[str, Any]:
    core_query = extract_core_query_terms(question)

    should_queries = [
        {
            "multi_match": {
                "query": question,
                "fields": [
                    "texto^4",
                    "resumen_chunk^2",
                    "keywords^4",
                    "nombre_documento^2"
                ],
                "type": "best_fields"
            }
        },
        {
            "match": {
                "texto": {
                    "query": question,
                    "operator": "and",
                    "boost": 5
                }
            }
        }
    ]

    if core_query:
        should_queries.extend([
            {
                "match_phrase": {
                    "texto": {
                        "query": core_query,
                        "boost": 10
                    }
                }
            },
            {
                "match": {
                    "texto": {
                        "query": core_query,
                        "operator": "and",
                        "boost": 7
                    }
                }
            },
            {
                "match": {
                    "keywords": {
                        "query": core_query,
                        "boost": 5
                    }
                }
            },
            {
                "match": {
                    "nombre_documento": {
                        "query": core_query,
                        "boost": 3
                    }
                }
            }
        ])

    base_query = {
        "bool": {
            "should": should_queries,
            "minimum_should_match": 1
        }
    }

    if not document_id:
        return base_query

    return {
        "bool": {
            "must": [
                base_query
            ],
            "filter": [
                {
                    "term": {
                        "documento_id": document_id
                    }
                }
            ]
        }
    }


def build_vector_filter_query(document_id: str | None = None) -> dict[str, Any]:
    if not document_id:
        return {
            "match_all": {}
        }

    return {
        "term": {
            "documento_id": document_id
        }
    }


def bm25_search(
    index_name: str,
    question: str,
    top_k: int = 20,
    document_id: str | None = None,
) -> list[dict[str, Any]]:
    body = {
        "size": top_k,
        "query": build_text_query(
            question=question,
            document_id=document_id,
        )
    }

    response = es.search(index=index_name, body=body)
    return response.get("hits", {}).get("hits", [])


def vector_search(
    index_name: str,
    question: str,
    top_k: int = 20,
    document_id: str | None = None,
) -> list[dict[str, Any]]:
    query_vector = model.encode(
        question,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": build_vector_filter_query(document_id=document_id),
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {
                        "query_vector": query_vector
                    }
                }
            }
        }
    }

    response = es.search(index=index_name, body=body)
    return response.get("hits", {}).get("hits", [])


def combine_hits(
    bm25_hits: list[dict[str, Any]],
    vector_hits: list[dict[str, Any]],
    vector_weight: float = 0.65,
    bm25_weight: float = 0.35,
) -> list[dict[str, Any]]:
    max_bm25 = max(
        [hit.get("_score", 0.0) for hit in bm25_hits],
        default=0.0,
    )

    combined: dict[str, dict[str, Any]] = {}

    for hit in bm25_hits:
        key = build_result_key(hit)
        source = hit.get("_source", {})
        raw_score = hit.get("_score", 0.0)
        bm25_score = normalize_bm25_score(raw_score, max_bm25)

        combined[key] = {
            "texto": source.get("texto", ""),
            "documento_id": source.get("documento_id"),
            "nombre_documento": source.get("nombre_documento"),
            "chunk_id": source.get("chunk_id"),
            "tema": source.get("tema"),
            "fecha_ingesta": source.get("fecha_ingesta"),
            "autor": source.get("autor"),
            "keywords": source.get("keywords", []),
            "bm25_score": bm25_score,
            "vector_score": 0.0,
            "score": bm25_weight * bm25_score,
        }

    for hit in vector_hits:
        key = build_result_key(hit)
        source = hit.get("_source", {})
        raw_score = hit.get("_score", 0.0)
        vector_score = normalize_vector_score(raw_score)

        if key not in combined:
            combined[key] = {
                "texto": source.get("texto", ""),
                "documento_id": source.get("documento_id"),
                "nombre_documento": source.get("nombre_documento"),
                "chunk_id": source.get("chunk_id"),
                "tema": source.get("tema"),
                "fecha_ingesta": source.get("fecha_ingesta"),
                "autor": source.get("autor"),
                "keywords": source.get("keywords", []),
                "bm25_score": 0.0,
                "vector_score": vector_score,
                "score": vector_weight * vector_score,
            }
        else:
            combined[key]["vector_score"] = vector_score
            combined[key]["score"] += vector_weight * vector_score

    results = list(combined.values())
    results.sort(key=lambda item: item["score"], reverse=True)

    return results
def is_definition_question(question: str) -> bool:
    question = question.lower().strip()

    definition_patterns = [
        "que es",
        "qué es",
        "que son",
        "qué son",
        "define",
        "definicion",
        "definición",
        "concepto de",
        "significado de",
    ]

    return any(pattern in question for pattern in definition_patterns)


def hit_to_result(hit: dict[str, Any]) -> dict[str, Any]:
    source = hit.get("_source", {})

    return {
        "texto": source.get("texto", ""),
        "documento_id": source.get("documento_id"),
        "nombre_documento": source.get("nombre_documento"),
        "chunk_id": source.get("chunk_id"),
        "tema": source.get("tema"),
        "fecha_ingesta": source.get("fecha_ingesta"),
        "autor": source.get("autor"),
        "keywords": source.get("keywords", []),

        # Estos chunks se añaden manualmente porque suelen contener definiciones.
        # Les damos una puntuación suficiente para no ser rechazados por el umbral.
        "bm25_score": 1.0,
        "vector_score": 0.0,
        "score": 0.95,
    }


def get_intro_chunks(
    index_name: str,
    document_id: str,
    size: int = 3,
) -> list[dict[str, Any]]:
    body = {
        "size": size,
        "query": {
            "term": {
                "documento_id": document_id
            }
        },
        "sort": [
            {
                "chunk_id": {
                    "order": "asc"
                }
            }
        ]
    }

    response = es.search(index=index_name, body=body)
    hits = response.get("hits", {}).get("hits", [])

    return [hit_to_result(hit) for hit in hits]


def merge_results_without_duplicates(
    priority_results: list[dict[str, Any]],
    normal_results: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    merged = []
    seen = set()

    for result in priority_results + normal_results:
        key = f"{result.get('documento_id')}_{result.get('chunk_id')}"

        if key in seen:
            continue

        seen.add(key)
        merged.append(result)

    return merged[:top_k]

def select_best_document(index_name: str, question: str) -> dict[str, Any] | None:
    """
    Primera pasada:
    busca en todo el índice temático y decide qué documento parece más relacionado.
    """

    bm25_hits = bm25_search(
        index_name=index_name,
        question=question,
        top_k=20,
    )

    vector_hits = vector_search(
        index_name=index_name,
        question=question,
        top_k=20,
    )

    combined_hits = combine_hits(
        bm25_hits=bm25_hits,
        vector_hits=vector_hits,
    )

    if not combined_hits:
        return None

    documents: dict[str, dict[str, Any]] = {}

    for hit in combined_hits:
        documento_id = hit.get("documento_id")

        if not documento_id:
            continue

        if documento_id not in documents:
            documents[documento_id] = {
                "documento_id": documento_id,
                "nombre_documento": hit.get("nombre_documento"),
                "tema": hit.get("tema"),
                "score": 0.0,
                "best_chunk_score": 0.0,
                "matched_chunks": 0,
            }

        documents[documento_id]["score"] += hit.get("score", 0.0)
        documents[documento_id]["best_chunk_score"] = max(
            documents[documento_id]["best_chunk_score"],
            hit.get("score", 0.0),
        )
        documents[documento_id]["matched_chunks"] += 1

    if not documents:
        return None

    ranked_documents = list(documents.values())

    ranked_documents.sort(
        key=lambda doc: (
            doc["best_chunk_score"],
            doc["score"],
            doc["matched_chunks"],
        ),
        reverse=True,
    )

    return ranked_documents[0]


def hybrid_search(
    question: str,
    top_k: int = 20,
    vector_weight: float = 0.55,
    bm25_weight: float = 0.45,
) -> dict[str, Any]:
    
    tema = classify_question(question)
    # LO SIGUIENTE ES PARA FORZAR LA TEMATICA A ASTRONOMIA 

    #ndex_name = "rag_astronomia_universo"
    index_name = "rag_biologia"
    #index_name = topic_to_index_name(tema)
  
    # EL QUE ESTA COMENTADO ES EL GENERICO


    if not es.indices.exists(index=index_name):
        return {
            "tema": tema,
            "index_name": index_name,
            "selected_document": None,
            "results": [],
            "found": False,
            "reason": "No existe índice para la temática detectada."
        }

    selected_document = select_best_document(
        index_name=index_name,
        question=question,
    )

    if not selected_document:
        return {
            "tema": tema,
            "index_name": index_name,
            "selected_document": None,
            "results": [],
            "found": False,
            "reason": "No se pudo seleccionar un documento relevante."
        }

    document_id = selected_document["documento_id"]

    bm25_hits = bm25_search(
        index_name=index_name,
        question=question,
        top_k=top_k * 2,
        document_id=document_id,
    )

    vector_hits = vector_search(
        index_name=index_name,
        question=question,
        top_k=top_k * 2,
        document_id=document_id,
    )

    results = combine_hits(
        bm25_hits=bm25_hits,
        vector_hits=vector_hits,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
    )

    results = results[:top_k]

    if is_definition_question(question):
        intro_chunks = get_intro_chunks(
            index_name=index_name,
            document_id=document_id,
            size=3,
        )

        results = merge_results_without_duplicates(
            priority_results=intro_chunks,
            normal_results=results,
            top_k=top_k,
        )

    if not results:
        return {
            "tema": tema,
            "index_name": index_name,
            "selected_document": selected_document,
            "results": [],
            "found": False,
            "reason": "No se encontraron resultados dentro del documento seleccionado."
        }

    top_score = max(result.get("score", 0.0) for result in results)

    if top_score < MIN_RELEVANCE_SCORE:
        return {
            "tema": tema,
            "index_name": index_name,
            "selected_document": selected_document,
            "results": results,
            "found": False,
            "reason": f"Score máximo insuficiente: {top_score:.3f}"
        }

    return {
        "tema": tema,
        "index_name": index_name,
        "selected_document": selected_document,
        "results": results,
        "found": True,
        "reason": "Resultados encontrados dentro del documento seleccionado."
    }