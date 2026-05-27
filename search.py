import os # Para manejar variables de entorno
from typing import Any # Para anotaciones de tipado
from topic_router import route_question_to_existing_topic, UNKNOWN_TOPIC, topic_to_index

from vector_store import es, model, topic_to_index_name


MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.50"))


def normalize_bm25_score(score: float, max_score: float) -> float:
    # Normaliza el score BM25 dividiéndolo por el score máximo obtenido en la búsqueda, para escalarlo entre 0 y 1. Si el score máximo es 0 o negativo, devuelve 0 para evitar divisiones por cero o resultados negativos.
    if not max_score or max_score <= 0:
        return 0.0

    return score / max_score


def normalize_vector_score(score: float) -> float:
    # Normaliza el score de similitud vectorial, que suele estar entre -1 y 1, escalándolo a un rango de 0 a 1. Si el score es menor que -1, devuelve 0; si es mayor que 1, devuelve 1; de lo contrario, lo escala linealmente
    return max(0.0, min(score / 2.0, 1.0))


def build_result_key(hit: dict[str, Any]) -> str:
    # Construye una clave única para un resultado de búsqueda combinando el ID del documento y el ID del chunk. Si por alguna razón no se pueden obtener estos IDs, utiliza el ID del hit de Elasticsearch como fallback. hit = un resultado encontrado por Elasticsearch, como ejemplo, hit["_id"] es el identificador interno que Elasticsearch le da a ese resultado/documento.
    source = hit.get("_source", {})
    documento_id = source.get("documento_id", "")
    chunk_id = source.get("chunk_id", "")

    if documento_id != "" and chunk_id != "":
        return f"{documento_id}_{chunk_id}"

    return hit["_id"]


def get_rag_indices() -> list[str]:
    # Devuelve una lista de los índices RAG que existen actualmente en Elasticsearch, filtrando solo aquellos que comienzan con "rag_".
    try:
        indices = es.indices.get_alias(index="rag_*")
        return list(indices.keys())
    except Exception:
        return []


def extract_core_query_terms(question: str) -> str:
    # Extrae las palabras clave principales de la pregunta del usuario eliminando stopwords comunes y palabras muy cortas, para construir una consulta más enfocada para la búsqueda en Elasticsearch.

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
    # Construye una consulta de texto para Elasticsearch que combina varias estrategias de búsqueda, incluyendo coincidencia de frase, coincidencia de términos y búsqueda en campos específicos, y si se proporciona un document_id, filtra los resultados para ese documento específico.

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
    # Construye una consulta de filtro para la búsqueda vectorial en Elasticsearch, que si se proporciona un document_id, limita los resultados a ese documento específico. Si no se proporciona un document_id, devuelve una consulta que coincide con todos los documentos.
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
    # Realiza una búsqueda BM25 en Elasticsearch usando la consulta de texto construida, y devuelve los hits encontrados. Si se proporciona un document_id, la búsqueda se limita a ese documento específico.
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
    # Realiza una búsqueda vectorial en Elasticsearch calculando el embedding de la pregunta usando el modelo de sentence-transformers, y devuelve los hits encontrados ordenados por similitud. Si se proporciona un document_id, la búsqueda se limita a ese documento específico.
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
    # Combina los hits de las búsquedas BM25 y vectorial, normalizando sus scores y aplicando pesos para obtener un score combinado. Devuelve una lista de resultados ordenados por el score combinado en orden descendente.
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
    # Determina si la pregunta del usuario parece ser una solicitud de definición o explicación de un concepto, buscando patrones comunes en la pregunta. 
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
    # Convierte un hit de Elasticsearch en un formato de resultado más amigable.  
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
    # Obtiene los primeros chunks de un documento específico, que suelen contener la introducción o definición del tema, para darles prioridad en preguntas de definición. 
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
    # Combina dos listas de resultados, dando prioridad a la primera lista (priority_results) y asegurándose de no incluir resultados duplicados basados en el documento_id y chunk_id. Devuelve una lista combinada ordenada por prioridad y limitada a top_k resultados.
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


UNKNOWN_TOPIC = "desconocida"


def get_existing_rag_indices() -> list[str]:
    # Devuelve una lista de los índices RAG que existen actualmente en Elasticsearch, filtrando solo aquellos que comienzan con "rag_".
    try:
        indices = es.indices.get_alias(index="rag_*")
        return list(indices.keys())
    except Exception:
        return []


def resolve_search_index_from_question(question: str) -> tuple[str, str | None, bool, str]:
    # Decide en qué temática buscar usando SOLO los índices ya existentes. Devuelve una tupla con la temática seleccionada, el nombre del índice correspondiente, un booleano indicando si se puede realizar la búsqueda, y una razón explicativa.

    existing_indices = get_existing_rag_indices()

    if not existing_indices:
        return (
            UNKNOWN_TOPIC,
            None,
            False,
            "No hay ningún índice RAG ingestados todavía."
        )

    routing = route_question_to_existing_topic(
        question=question,
        existing_indices=existing_indices,
    )

    topic = routing.get("topic", UNKNOWN_TOPIC)

    if topic == UNKNOWN_TOPIC:
        return (
            UNKNOWN_TOPIC,
            None,
            False,
            routing.get("reason", "No se encontró una temática adecuada.")
        )

    index_name = topic_to_index(topic)

    if index_name not in existing_indices:
        return (
            UNKNOWN_TOPIC,
            None,
            False,
            f"La temática '{topic}' no corresponde a ningún índice ingestados."
        )

    reason = (
        f"Índice seleccionado por {routing.get('method')}: {index_name}. "
        f"Confianza: {routing.get('confidence', 0):.2f}. "
        f"{routing.get('reason', '')}"
    )

    return (
        topic,
        index_name,
        True,
        reason,
    )

def hybrid_search(
    question: str,
    top_k: int = 20,
    vector_weight: float = 0.55,
    bm25_weight: float = 0.45,
) -> dict[str, Any]:
    '''
    Realiza una búsqueda híbrida en los documentos ingestados, combinando BM25 y búsqueda vectorial. Primero resuelve en qué índice temático buscar usando solo los índices existentes, luego selecciona el documento más relevante dentro de ese índice, y finalmente realiza búsquedas BM25 y vectorial dentro de ese documento para obtener los fragmentos más relevantes. Devuelve un diccionario con la temática, el nombre del índice, el documento seleccionado, los resultados encontrados, y una explicación de la razón detrás de la selección del índice y los resultados.
    '''
    
    tema, index_name, can_search, reason = resolve_search_index_from_question(question)

    if not can_search or not index_name:
        return {
            "tema": tema,
            "index_name": index_name,
            "selected_document": None,
            "results": [],
            "found": False,
            "reason": reason,
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