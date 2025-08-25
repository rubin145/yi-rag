# engines/registry.py
from __future__ import annotations
from typing import Dict, Any
from .contracts import GeneratorSpec, Generator
from .generators.openai_generator import OpenAIGenerator
from .generators.gemini_generator import GeminiGenerator
try:
    from .generators.groq_generator import GroqGenerator
except Exception:  # pragma: no cover
    GroqGenerator = None


def make_generator(spec: GeneratorSpec) -> Generator:
    provider = spec.provider.lower()
    if provider == "openai":
        return OpenAIGenerator(spec)
    if provider == "gemini":
        return GeminiGenerator(spec)
    if provider == "groq" and GroqGenerator is not None:
        return GroqGenerator(spec)
    raise ValueError(f"Proveedor de generator no soportado: {spec.provider}")
