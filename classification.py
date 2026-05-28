import json
import os
import re
import unicodedata

from llm_wrapper import CustomLLMWrapper


UNKNOWN_TOPIC = "desconocida"

DOCUMENT_CLASSIFIER_MIN_CONFIDENCE = float(
    os.getenv("DOCUMENT_CLASSIFIER_MIN_CONFIDENCE", "0.60")
)


def extract_json_from_text(text: str) -> dict | None:
    # Intenta extraer un JSON válido de un texto, primero intentando parsear el texto completo, y si falla, buscando un bloque JSON dentro del texto. Devuelve None si no se encuentra ningún JSON válido.
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


def slugify_topic(topic: str) -> str:
    """
    Convierte una temática generada por el LLM en un nombre válido para índice.
    """
    topic = topic or ""
    topic = topic.strip().lower()

    topic = unicodedata.normalize("NFD", topic)
    topic = "".join(
        char for char in topic
        if unicodedata.category(char) != "Mn"
    )

    topic = re.sub(r"[^a-z0-9]+", "_", topic)
    topic = re.sub(r"_+", "_", topic)
    topic = topic.strip("_")

    if not topic:
        return UNKNOWN_TOPIC

    return topic[:80]


def build_existing_topics_text(existing_topics: list[str] | None) -> str:
    if not existing_topics:
        return "No hay temáticas existentes todavía."

    return "\n".join(f"- {topic}" for topic in existing_topics)


def classify_document(
    text: str,
    filename: str | None = None,
    existing_topics: list[str] | None = None,
) -> str:
    """
    Clasifica un documento usando el LLM.

    El LLM puede:
    - reutilizar una temática existente
    - crear una nueva temática
    - devolver desconocida si no hay suficiente claridad
    """

    llm = CustomLLMWrapper()

    if not llm.is_configured():
        return UNKNOWN_TOPIC

    filename = filename or "documento_sin_nombre"
    text_sample = text[:10000]

    existing_topics_text = build_existing_topics_text(existing_topics)

    system_prompt = """
Eres un clasificador semántico de documentos para un sistema RAG.

Tu tarea:
- Leer el nombre del archivo y una muestra del contenido.
- Asignar una temática clara y específica al documento.
- Si una temática existente encaja bien, debes reutilizarla exactamente.
- Si ninguna temática existente encaja, puedes crear una nueva.
- La temática debe ser breve, clara y en formato snake_case.
- No uses categorías demasiado genéricas si puedes ser más específico.
- Si no puedes determinar la temática, devuelve "desconocida".
- No resumas el documento.
- No expliques nada fuera del JSON.

Debes responder SOLO con JSON válido:
{
  "topic": "pesca_sector_maritimo",
  "confidence": 0.87,
  "reason": "El documento trata sobre pesca, barcos, caladeros y capturas."
}
"""

    user_prompt = f"""
Nombre del archivo:
{filename}

Temáticas ya existentes en Elasticsearch:
{existing_topics_text}

Contenido del documento:
{text_sample}

Reglas:
- Si el documento encaja con una temática existente, usa exactamente esa temática.
- Si no encaja con ninguna existente, crea una nueva temática en snake_case.
- Usa entre 1 y 4 palabras como máximo.
- Evita nombres genéricos como "documento", "general", "texto" o "otros".
- Si no hay suficiente información, usa "{UNKNOWN_TOPIC}".
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
            return UNKNOWN_TOPIC

        topic = str(data.get("topic", UNKNOWN_TOPIC)).strip()
        confidence = float(data.get("confidence", 0.0))

        if confidence < DOCUMENT_CLASSIFIER_MIN_CONFIDENCE:
            return UNKNOWN_TOPIC

        topic = slugify_topic(topic)

        if topic in ["otros", "general", "documento", "texto"]:
            return UNKNOWN_TOPIC

        return topic

    except Exception:
        return UNKNOWN_TOPIC

