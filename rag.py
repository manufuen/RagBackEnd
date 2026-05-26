from typing import Any 

from search import hybrid_search 
from llm_wrapper import CustomLLMWrapper


def build_context(results: list[dict[str, Any]]) -> str:
    # Construye un contexto concatenando los fragmentos de texto de los resultados de búsqueda, incluyendo metadatos para que el LLM pueda citar las fuentes.
    context_parts = []

    for result in results:
        nombre_documento = result.get("nombre_documento", "documento desconocido")
        chunk_id = result.get("chunk_id", "sin chunk")
        texto = result.get("texto", "")

        context_parts.append(
            f"[Documento: {nombre_documento} | Chunk: {chunk_id}]\n{texto}"
        )

    return "\n\n---\n\n".join(context_parts)


def generate_answer_with_custom_llm(question: str, context: str) -> str:
    # Genera una respuesta a la pregunta dada usando un LLM personalizado, proporcionando el contexto relevante. Si el LLM no está configurado correctamente, devuelve una cadena vacía.
    llm = CustomLLMWrapper()

    if not llm.is_configured():
        return ""

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

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    return llm.chat(
        messages=messages,
        temperature=0.0,
    )


def generate_fallback_answer(results: list[dict[str, Any]]) -> str:
    # Genera una respuesta de fallback cuando el LLM no está configurado correctamente, mostrando los fragmentos de texto más relevantes.
    if not results:
        return "No hay información en los documentos ingestados."

    lines = [
        "He encontrado información relacionada en los documentos ingestados.",
        "",
        "Fragmentos más relevantes:",
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
        "Nota: no hay LLM configurado correctamente, así que todavía no se ha generado una respuesta final con modelo."
    )

    return "\n".join(lines)


def answer_question(question: str) -> dict[str, Any]:
    # Función principal para responder a una pregunta del usuario. Realiza una búsqueda híbrida en los documentos ingestados, construye un contexto con los resultados, intenta generar una respuesta usando el LLM personalizado, y si no es posible, genera una respuesta de fallback mostrando los fragmentos más relevantes.
    search_response = hybrid_search(question)

    if not search_response["found"]:
        return {
            "answer": "No hay información en los documentos ingestados.",
            "tema": search_response["tema"],
            "index_name": search_response["index_name"],
            "found": False,
            "reason": search_response["reason"],
            "sources": search_response["results"],
            "selected_document": search_response.get("selected_document"),
        }

    results = search_response["results"]
    context = build_context(results)

    answer = generate_answer_with_custom_llm(question, context)

    if not answer:
        answer = generate_fallback_answer(results)

    return {
        "answer": answer,
        "tema": search_response["tema"],
        "index_name": search_response["index_name"],
        "found": True,
        "reason": search_response["reason"],
        "sources": results,
        "selected_document": search_response.get("selected_document"),
    }