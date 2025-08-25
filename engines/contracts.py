# engines/contracts.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


# -----------------------------
# Especificación del generador
# -----------------------------
@dataclass
class GeneratorSpec:
    """
    Especificación del LLM usado por el pipeline RAG.
    """
    provider: str              # "openai" | "gemini" | "groq" | ...
    model: str
    max_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 1.0
    target_lang: str = "auto"  # "auto" | "es" | "en" | "yi"


# -----------------------------
# Resultado de generación
# -----------------------------
@dataclass
class GenerationUsage:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass
class GenerationResult:
    text: str
    usage: Optional[GenerationUsage] = None
    raw: Any = None  # respuesta cruda del SDK (opcional)


def as_result(obj: Union[GenerationResult, Dict[str, Any]]) -> GenerationResult:
    """
    Helper para compatibilidad: si un generador devuelve dicts,
    los envolvemos en GenerationResult.
    """
    if isinstance(obj, GenerationResult):
        return obj
    text = obj.get("text", "")
    u = obj.get("usage") or {}
    usage = GenerationUsage(
        prompt_tokens=u.get("prompt_tokens"),
        completion_tokens=u.get("completion_tokens"),
        total_tokens=u.get("total_tokens"),
    ) if u else None
    return GenerationResult(text=text, usage=usage, raw=obj.get("raw"))


# -----------------------------
# Contrato de generador (ABC)
# -----------------------------
class Generator(ABC):
    """
    Clase base abstracta: los generadores concretos deben implementar `generate`.
    """

    def __init__(self, spec: GeneratorSpec):
        self.spec = spec

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        stop: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        """
        Debe devolver un GenerationResult con:
          - text (str)
          - usage (opcional)
          - raw (opcional, respuesta cruda del SDK)
        """
        raise NotImplementedError
