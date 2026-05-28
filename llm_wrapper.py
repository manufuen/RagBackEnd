import os # Para manejar variables de entorno
from typing import Any # Para anotaciones de tipado

import httpx 
from dotenv import load_dotenv 

load_dotenv() # Carga las variables de entorno desde un archivo .env

class CustomLLMWrapper:
    ''' 
    Wrapper para llamar a un LLM personalizado compatible con OpenAI API.
    '''
    def __init__(self):
        self.api_key = os.getenv("CUSTOM_LLM_API_KEY")
        self.base_url = os.getenv("CUSTOM_LLM_BASE_URL")
        self.model = os.getenv("CUSTOM_LLM_MODEL", "gpt-5-chat-nextai")

        self.header_provider = os.getenv("CUSTOM_LLM_HEADER_PROVIDER")
        self.header_origin = os.getenv("CUSTOM_LLM_HEADER_ORIGIN")
        self.header_origin_detail = os.getenv(
            "CUSTOM_LLM_HEADER_ORIGIN_DETAIL",
            "openai-compatible",
        )

        # Asegura que la base_url no termine con "/" para evitar problemas al construir la URL de la API.
        if self.base_url: 
            self.base_url = self.base_url.rstrip("/")

    def is_configured(self) -> bool:
        return bool(
            self.api_key
            and self.base_url
            and self.model
            and self.header_provider
            and self.header_origin
        )

    # Construye los headers personalizados para la autenticación y metadatos de la petición al LLM.
    def _build_headers(self) -> dict[str, Any]:
        # Aparece como error porque httpx no acepta headers con valores None, pero nosotros nos aseguramos de que no sean None con is_configured() antes de llamar a esta función.
        return {
            "provider": self.header_provider,
            "origin": self.header_origin,
            "origin-detail": self.header_origin_detail,
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # Función principal para enviar mensajes al LLM y obtener la respuesta.
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> str:
        if not self.is_configured():
            return ""

        url = f"{self.base_url}/chat/completions"

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        headers = self._build_headers()

        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                url,
                headers=headers,
                json=payload,
            )

            response.raise_for_status()
            data = response.json()

        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return ""