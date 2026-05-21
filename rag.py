import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from search import hybrid_search


load_dotenv()


def build_context(results: list[dict[str, Any]]) -> str:
    context_parts = []

    for result in results:
        nombre_documento = result.get("nombre_documento", "documento desconocido")
        chunk_id = result.get("chunk_id", "sin chunk")
        texto = result.get("texto", "")

        context_parts.append(
            f"[Documento: {nombre_documento} | Chunk: {chunk_id}]\n{texto}"
        )

    return "\n\n---\n\n".join(context_parts)


def generate_answer_with_openai(question: str, context: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return ""

    client = OpenAI(api_key=api_key)

    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    system_prompt = """
Eres un asistente RAG.
Debes responder SOLO usando el contexto proporcionado.
Si la respuesta no aparece claramente en el contexto, responde:
"No hay información en los documentos ingestados."

No inventes información.
Cita el documento usado cuando sea posible.
"""

    user_prompt = f"""
Pregunta:
{question}

Contexto:
{context}

Respuesta:
"""

    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return response.choices[0].message.content


def generate_fallback_answer(results: list[dict[str, Any]]) -> str:
    """
    Respuesta sin LLM.
    Sirve para que el proyecto funcione aunque no tengas OPENAI_API_KEY.
    """
    if not results:
        return "No hay información en los documentos ingestados."

    lines = [
        "He encontrado información relacionada en los documentos ingestados.",
        "",
        "Fragmentos más relevantes:"
    ]

    for i, result in enumerate(results[:3], start=1):
        nombre_documento = result.get("nombre_documento", "documento desconocido")
        chunk_id = result.get("chunk_id", "sin chunk")
        texto = result.get("texto", "")

        preview = " ".join(texto.split())[:500]

        lines.append("")
        lines.append(f"{i}. Documento: {nombre_documento} | Chunk: {chunk_id}")
        lines.append(f"{preview}...")

    lines.append("")
    lines.append(
        "Nota: no hay OPENAI_API_KEY configurada, así que todavía no se ha generado una respuesta con LLM."
    )

    return "\n".join(lines)


def answer_question(question: str) -> dict[str, Any]:
    search_response = hybrid_search(question)

    if not search_response["found"]:
        return {
            "answer": "No hay información en los documentos ingestados.",
            "tema": search_response["tema"],
            "index_name": search_response["index_name"],
            "found": False,
            "reason": search_response["reason"],
            "sources": search_response["results"],
        }

    results = search_response["results"]
    context = build_context(results)

    answer = generate_answer_with_openai(question, context)

    if not answer:
        answer = generate_fallback_answer(results)

    return {
        "answer": answer,
        "tema": search_response["tema"],
        "index_name": search_response["index_name"],
        "found": True,
        "reason": search_response["reason"],
        "sources": results,
    }