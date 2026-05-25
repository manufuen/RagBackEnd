import json
import os
import re

from llm_wrapper import CustomLLMWrapper
from topics import TOPIC_CATALOG, UNKNOWN_TOPIC


DOCUMENT_CLASSIFIER_MIN_CONFIDENCE = float(
    os.getenv("DOCUMENT_CLASSIFIER_MIN_CONFIDENCE", "0.60")
)


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


def build_topics_description() -> str:
    lines = []

    for topic, description in TOPIC_CATALOG.items():
        lines.append(f"- {topic}: {description}")

    return "\n".join(lines)


def classify_document(text: str, filename: str | None = None) -> str:
    """
    Clasifica un documento usando el LLM.
    Devuelve una temática válida del catálogo o 'desconocida'.
    """

    llm = CustomLLMWrapper()

    if not llm.is_configured():
        return UNKNOWN_TOPIC

    filename = filename or "documento_sin_nombre"

    # No mandamos el documento entero al LLM.
    # Mandamos una muestra representativa para clasificar.
    text_sample = text[:8000]

    system_prompt = """
Eres un clasificador de documentos para un sistema RAG.

Tu tarea:
- Leer el nombre del archivo y una muestra del contenido.
- Elegir UNA temática entre las temáticas disponibles.
- Solo puedes elegir una temática de la lista.
- Si ninguna temática encaja claramente, devuelve "desconocida".
- No resumas el documento.
- No expliques fuera del JSON.

Debes responder SOLO con JSON válido:
{
  "topic": "biologia",
  "confidence": 0.87,
  "reason": "El documento trata sobre seres vivos, genética y bioética."
}
"""

    user_prompt = f"""
Nombre del archivo:
{filename}

Temáticas disponibles:
{build_topics_description()}

Contenido del documento:
{text_sample}

Recuerda:
- "topic" debe ser una de las temáticas disponibles.
- Si no hay una temática clara, usa "{UNKNOWN_TOPIC}".
- "confidence" debe ser un número entre 0 y 1.
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

        if topic == UNKNOWN_TOPIC:
            return UNKNOWN_TOPIC

        if topic not in TOPIC_CATALOG:
            return UNKNOWN_TOPIC

        if confidence < DOCUMENT_CLASSIFIER_MIN_CONFIDENCE:
            return UNKNOWN_TOPIC

        return topic

    except Exception:
        return UNKNOWN_TOPIC


def classify_question(question: str) -> str:
    """
    Ya no es la ruta principal para preguntas.
    Las preguntas se enrutan mejor con topic_router.py usando índices existentes.
    Se deja por compatibilidad.
    """
    return UNKNOWN_TOPIC