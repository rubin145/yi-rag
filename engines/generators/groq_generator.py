# engines/generators/groq_generator.py
from __future__ import annotations
from typing import Optional, Dict, Any
try:
    from groq import Groq
except Exception:  # pragma: no cover
    Groq = None
from engines.contracts import Generator, GeneratorSpec
from core.config import settings


class GroqGenerator(Generator):
    def __init__(self, spec: GeneratorSpec):
        if Groq is None:
            raise RuntimeError("Instala 'groq' para usar GroqGenerator")
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY no configurada")
        self.client = Groq(api_key=settings.groq_api_key)
        self.spec = spec

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        stop: Optional[list[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> dict:
        resp = self.client.chat.completions.create(
            model=self.spec.model,  # p.ej. "llama-3.1-70b-versatile"
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.spec.max_tokens,
            temperature=self.spec.temperature,
            top_p=self.spec.top_p,
            stop=stop,
        )
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        }
        return {"text": text, "usage": usage_dict, "raw": resp}
