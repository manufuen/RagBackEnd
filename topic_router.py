import json
import os
import re

import classification
from llm_wrapper import CustomLLMWrapper


UNKNOWN_TOPIC = getattr(classification, "UNKNOWN_TOPIC", "desconocida")
TOPIC_KEYWORDS = getattr(classification, "TOPIC_KEYWORDS", {})
classify_question = classification.classify_question

USE_LLM_TOPIC_ROUTER = os.getenv("USE_LLM_TOPIC_ROUTER", "true").lower() == "true"
TOPIC_ROUTER_MIN_CONFIDENCE = float(os.getenv("TOPIC_ROUTER_MIN_CONFIDENCE", "0.55"))


def index_to_topic(index_name: str) -> str:
    if index_name.startswith("rag_"):
        return index_name.replace("rag_", "", 1)

    return index_name


def topic_to_index(topic: str) -> str:
    return f"rag_{topic}"


def extract_json_from_text(text: str) -> dict | None:
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def build_candidate_topics(existing_indices: list[str]) -> tuple[list[str], str]:
    topics = [
        index_to_topic(index_name)
        for index_name in existing_indices
        if index_name.startswith("rag_")
    ]

    lines = []

    for topic in topics:
        keywords = TOPIC_KEYWORDS.get(topic, [])
        keywords_preview = ", ".join(keywords[:20])

        if keywords_preview:
            lines.append(f"- {topic}: {keywords_preview}")
        else:
            lines.append(f"- {topic}")

    return topics, "\n".join(lines)


def route_question_to_existing_topic(
    question: str,
    existing_indices: list[str],
) -> dict:
    """
    Decide en qué temática buscar usando SOLO los índices ya existentes.

    Devuelve:
    {
        "topic": "biologia",
        "confidence": 0.87,
        "method": "llm_router",
        "reason": "..."
    }
    """

    topics, candidate_description = build_candidate_topics(existing_indices)

    if not topics:
        return {
            "topic": UNKNOWN_TOPIC,
            "confidence": 0.0,
            "method": "no_indices",
            "reason": "No hay índices RAG ingestados."
        }

    if not USE_LLM_TOPIC_ROUTER:
        return route_with_keyword_fallback(question, topics)

    llm = CustomLLMWrapper()

    if not llm.is_configured():
        return route_with_keyword_fallback(question, topics)

    system_prompt = """
Eres un clasificador semántico de preguntas para un sistema RAG.

Tu tarea:
- Leer la pregunta del usuario.
- Elegir UNA temática entre las temáticas disponibles.
- Solo puedes elegir una temática que aparezca en la lista.
- Si la pregunta no encaja claramente con ninguna temática disponible, devuelve "desconocida".
- No respondas a la pregunta.
- No expliques nada fuera del JSON.

Debes devolver SOLO JSON válido con este formato:
{
  "topic": "biologia",
  "confidence": 0.87,
  "reason": "La pregunta trata sobre seres vivos y genética."
}
"""

    user_prompt = f"""
Pregunta del usuario:
{question}

Temáticas disponibles, basadas SOLO en índices ya ingestados:
{candidate_description}

Recuerda:
- El valor de "topic" debe ser una de las temáticas disponibles.
- Si ninguna temática encaja claramente, usa "{UNKNOWN_TOPIC}".
- confidence debe ser un número entre 0 y 1.
"""

    try:
        raw_response = llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.0,
        )

        data = extract_json_from_text(raw_response)

        if not data:
            return route_with_keyword_fallback(question, topics)

        topic = str(data.get("topic", UNKNOWN_TOPIC)).strip()
        confidence = float(data.get("confidence", 0.0))
        reason = str(data.get("reason", ""))

        if topic == UNKNOWN_TOPIC:
            return {
                "topic": UNKNOWN_TOPIC,
                "confidence": confidence,
                "method": "llm_router",
                "reason": reason or "El LLM no encontró una temática suficientemente relacionada."
            }

        if topic not in topics:
            return route_with_keyword_fallback(question, topics)

        if confidence < TOPIC_ROUTER_MIN_CONFIDENCE:
            return {
                "topic": UNKNOWN_TOPIC,
                "confidence": confidence,
                "method": "llm_router",
                "reason": f"Confianza insuficiente para la temática '{topic}': {confidence:.2f}"
            }

        return {
            "topic": topic,
            "confidence": confidence,
            "method": "llm_router",
            "reason": reason or f"Pregunta clasificada como {topic}."
        }

    except Exception as e:
        fallback = route_with_keyword_fallback(question, topics)
        fallback["reason"] = f"Fallo en router LLM, usado fallback por keywords. Error: {str(e)}"
        return fallback


def route_with_keyword_fallback(question: str, available_topics: list[str]) -> dict:
    """
    Fallback por keywords si el LLM no está disponible.
    Solo acepta temáticas que tengan índice ya ingestados.
    """

    detected_topic = classify_question(question)

    if detected_topic in available_topics:
        return {
            "topic": detected_topic,
            "confidence": 0.50,
            "method": "keyword_fallback",
            "reason": f"Clasificado por keywords como {detected_topic}."
        }

    return {
        "topic": UNKNOWN_TOPIC,
        "confidence": 0.0,
        "method": "keyword_fallback",
        "reason": f"La temática detectada '{detected_topic}' no tiene índice ingestados."
    }