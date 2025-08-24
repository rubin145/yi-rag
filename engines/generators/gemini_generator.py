# engines/generators/gemini_generator.py
from __future__ import annotations
from typing import Optional, Dict, Any
from google import genai
from google.genai import types
from engines.contracts import Generator, GeneratorSpec
from core.config import settings


class GeminiGenerator(Generator):
    def __init__(self, spec: GeneratorSpec):
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY no configurada")
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.spec = spec

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        stop: Optional[list[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> dict:
        # En el SDK nuevo, el "system" se pasa en config.system_instruction
        cfg = types.GenerateContentConfig(
            temperature=self.spec.temperature,
            top_p=self.spec.top_p,
            max_output_tokens=self.spec.max_tokens,
            system_instruction=system_prompt,
            stop_sequences=stop or None,
        )
        resp = self.client.models.generate_content(
            model=self.spec.model,  # e.g., "gemini-1.5-flash" o "gemini-1.5-pro"
            contents=[user_prompt],
            config=cfg,
        )
        # Texto
        text = resp.text if hasattr(resp, "text") else ""
        # Gemini no siempre devuelve usage consistente; lo exponemos si está
        usage = getattr(resp, "usage_metadata", None)
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
            "completion_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
            "total_tokens": getattr(usage, "total_token_count", None) if usage else None,
        }
        return {"text": text, "usage": usage_dict, "raw": resp}
