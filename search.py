import os
from typing import Any

from classification import classify_question
from vector_store import es, model, topic_to_index_name


MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.55"))


def normalize_bm25_score(score: float, max_score: float) -> float:
    if not max_score or max_score <= 0:
        return 0.0

    return score / max_score


def normalize_vector_score(score: float) -> float:
    """
    En Elasticsearch usamos:
    cosineSimilarity(...) + 1.0

    Por tanto, el score suele estar entre 0 y 2.
    Lo normalizamos aproximadamente a 0-1.
    """
    return max(0.0, min(score / 2.0, 1.0))


def build_result_key(hit: dict[str, Any]) -> str:
    source = hit.get("_source", {})
    documento_id = source.get("documento_id", "")
    chunk_id = source.get("chunk_id", "")

    if documento_id != "" and chunk_id != "":
        return f"{documento_id}_{chunk_id}"

    return hit["_id"]


def bm25_search(index_name: str, question: str, top_k: int = 8) -> list[dict[str, Any]]:
    query = {
        "size": top_k,
        "query": {
            "multi_match": {
                "query": question,
                "fields": [
                    "texto^3",
                    "resumen_chunk^2",
                    "keywords"
                ],
                "type": "best_fields"
            }
        }
    }

    response = es.search(index=index_name, body=query)
    hits = response.get("hits", {}).get("hits", [])

    return hits


def vector_search(index_name: str, question: str, top_k: int = 8) -> list[dict[str, Any]]:
    query_vector = model.encode(
        question,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    query = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {
                        "query_vector": query_vector
                    }
                }
            }
        }
    }

    response = es.search(index=index_name, body=query)
    hits = response.get("hits", {}).get("hits", [])

    return hits


def hybrid_search(
    question: str,
    top_k: int = 5,
    vector_weight: float = 0.65,
    bm25_weight: float = 0.35,
) -> dict[str, Any]:
    tema = classify_question(question)
    index_name = topic_to_index_name(tema)

    if not es.indices.exists(index=index_name):
        return {
            "tema": tema,
            "index_name": index_name,
            "results": [],
            "found": False,
            "reason": "No existe índice para la temática detectada."
        }

    bm25_hits = bm25_search(index_name, question, top_k=top_k * 2)
    vector_hits = vector_search(index_name, question, top_k=top_k * 2)

    max_bm25 = 0.0

    if bm25_hits:
        max_bm25 = max(hit.get("_score", 0.0) for hit in bm25_hits)

    combined: dict[str, dict[str, Any]] = {}

    for hit in bm25_hits:
        key = build_result_key(hit)
        source = hit.get("_source", {})
        raw_score = hit.get("_score", 0.0)
        normalized_score = normalize_bm25_score(raw_score, max_bm25)

        combined[key] = {
            "texto": source.get("texto", ""),
            "documento_id": source.get("documento_id"),
            "nombre_documento": source.get("nombre_documento"),
            "chunk_id": source.get("chunk_id"),
            "tema": source.get("tema"),
            "fecha_ingesta": source.get("fecha_ingesta"),
            "autor": source.get("autor"),
            "keywords": source.get("keywords", []),
            "bm25_score": normalized_score,
            "vector_score": 0.0,
            "score": bm25_weight * normalized_score,
        }

    for hit in vector_hits:
        key = build_result_key(hit)
        source = hit.get("_source", {})
        raw_score = hit.get("_score", 0.0)
        normalized_score = normalize_vector_score(raw_score)

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
                "vector_score": normalized_score,
                "score": vector_weight * normalized_score,
            }
        else:
            combined[key]["vector_score"] = normalized_score
            combined[key]["score"] += vector_weight * normalized_score

    results = list(combined.values())
    results.sort(key=lambda item: item["score"], reverse=True)

    results = results[:top_k]

    if not results:
        return {
            "tema": tema,
            "index_name": index_name,
            "results": [],
            "found": False,
            "reason": "No se encontraron resultados."
        }

    top_score = results[0]["score"]

    if top_score < MIN_RELEVANCE_SCORE:
        return {
            "tema": tema,
            "index_name": index_name,
            "results": results,
            "found": False,
            "reason": f"Score máximo insuficiente: {top_score:.3f}"
        }

    return {
        "tema": tema,
        "index_name": index_name,
        "results": results,
        "found": True,
        "reason": "Resultados encontrados."
    }