import json
import os
import re

from llm_wrapper import CustomLLMWrapper


UNKNOWN_TOPIC = "desconocida"

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

    if not topics:
        return [], "No hay temáticas disponibles."

    lines = [f"- {topic}" for topic in topics]

    return topics, "\n".join(lines)


def route_with_fallback(question: str, available_topics: list[str]) -> dict:
    return {
        "topic": UNKNOWN_TOPIC,
        "confidence": 0.0,
        "method": "no_llm_fallback",
        "reason": "No se pudo usar el LLM para enrutar la pregunta."
    }


def route_question_to_existing_topic(
    question: str,
    existing_indices: list[str],
) -> dict:
    topics, candidate_description = build_candidate_topics(existing_indices)

    if not topics:
        return {
            "topic": UNKNOWN_TOPIC,
            "confidence": 0.0,
            "method": "no_indices",
            "reason": "No hay índices RAG disponibles."
        }

    if not USE_LLM_TOPIC_ROUTER:
        return route_with_fallback(question, topics)

    llm = CustomLLMWrapper()

    if not llm.is_configured():
        return route_with_fallback(question, topics)

    system_prompt = """
Eres un router semántico para un sistema RAG.

Tu tarea:
- Leer la pregunta del usuario.
- Elegir UNA temática entre las temáticas disponibles.
- Solo puedes elegir una temática que aparezca en la lista.
- No puedes inventar temáticas nuevas.
- Si ninguna temática encaja claramente, devuelve "desconocida".
- No respondas a la pregunta.
- Devuelve solo JSON válido.

Formato obligatorio:
{
  "topic": "nombre_tematica",
  "confidence": 0.87,
  "reason": "Motivo breve de la elección."
}
"""

    user_prompt = f"""
Pregunta del usuario:
{question}

Temáticas disponibles:
{candidate_description}

Reglas:
- "topic" debe ser exactamente una de las temáticas disponibles.
- Si ninguna encaja, usa "{UNKNOWN_TOPIC}".
- "confidence" debe ser un número entre 0 y 1.
"""

    try:
        raw_response = llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )

        data = extract_json_from_text(raw_response)

        if not data:
            return route_with_fallback(question, topics)

        topic = str(data.get("topic", UNKNOWN_TOPIC)).strip()
        confidence = float(data.get("confidence", 0.0))
        reason = str(data.get("reason", ""))

        if topic == UNKNOWN_TOPIC:
            return {
                "topic": UNKNOWN_TOPIC,
                "confidence": confidence,
                "method": "llm_router",
                "reason": reason or "El LLM no encontró una temática adecuada."
            }

        if topic not in topics:
            return {
                "topic": UNKNOWN_TOPIC,
                "confidence": confidence,
                "method": "llm_router",
                "reason": f"El LLM devolvió una temática no disponible: {topic}."
            }

        if confidence < TOPIC_ROUTER_MIN_CONFIDENCE:
            return {
                "topic": UNKNOWN_TOPIC,
                "confidence": confidence,
                "method": "llm_router",
                "reason": f"Confianza insuficiente para '{topic}': {confidence:.2f}."
            }

        return {
            "topic": topic,
            "confidence": confidence,
            "method": "llm_router",
            "reason": reason or f"Pregunta clasificada como {topic}."
        }

    except Exception as exc:
        return {
            "topic": UNKNOWN_TOPIC,
            "confidence": 0.0,
            "method": "llm_router_error",
            "reason": f"Error usando el router LLM: {str(exc)}"
        }