# engines/generators/openai_generator.py
from __future__ import annotations
from typing import Optional, Dict, Any
from openai import OpenAI
from engines.contracts import Generator, GeneratorSpec
from core.config import settings


class OpenAIGenerator(Generator):
    def __init__(self, spec: GeneratorSpec):
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY no configurada")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.spec = spec

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        stop: Optional[list[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> dict:
        resp = self.client.chat.completions.create(
            model=self.spec.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.spec.max_tokens,
            temperature=self.spec.temperature,
            top_p=self.spec.top_p,
            stop=stop,
        )
        choice = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        }
        return {"text": choice, "usage": usage_dict, "raw": resp}
