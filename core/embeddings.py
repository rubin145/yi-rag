# core/embeddings.py
from __future__ import annotations

import math
import logging
import time
from typing import Any, Dict, Iterable, List, Optional

from .config import settings

log = logging.getLogger("embeddings")


# ---------------------------
# Utils
# ---------------------------
def _batched(xs: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def _l2_normalize(vec: List[float]) -> List[float]:
    s = math.sqrt(sum((x * x) for x in vec))
    if s == 0.0:
        return vec
    return [x / s for x in vec]


def _get_encoder(name: str = "cl100k_base"):
    try:
        import tiktoken

        return tiktoken.get_encoding(name)
    except Exception:
        return None


def _truncate_by_tokens(text: str, max_tokens: Optional[int], tokenizer_name: str = "cl100k_base") -> str:
    if not max_tokens or max_tokens <= 0 or not text:
        return text
    enc = _get_encoder(tokenizer_name)
    if enc is None:
        # Fallback “aprox”: ~3 chars por token
        max_chars = max_tokens * 3
        return text[:max_chars]
    toks = enc.encode(text)
    if len(toks) <= max_tokens:
        return text
    return enc.decode(toks[:max_tokens])


def _coerce_spec(spec: Any) -> Dict[str, Any]:
    """
    Acepta dict o objeto con atributos. Devuelve dict con claves esperadas.
    Claves mínimas: name, provider, model, dim, source_field
    Opcionales: max_input_tokens, normalize (bool), tokenizer, batch_size
    """
    if isinstance(spec, dict):
        d = dict(spec)
    else:
        d = {k: getattr(spec, k) for k in dir(spec) if not k.startswith("_") and not callable(getattr(spec, k))}
    # defaults
    d.setdefault("normalize", True)
    d.setdefault("tokenizer", "cl100k_base")
    d.setdefault("batch_size", None)  # elegimos por provider si es None
    return d


# ---------------------------
# Providers
# ---------------------------
def _embed_openai(texts: List[str], model: str, dim: int, normalize: bool, max_input_tokens: Optional[int],
                  tokenizer: str, batch_size: Optional[int]) -> List[Optional[List[float]]]:
    from openai import OpenAI

    if not settings.openai_api_key:
        log.error("OPENAI_API_KEY no configurada")
        return [None] * len(texts)

    client = OpenAI(api_key=settings.openai_api_key)
    bs = batch_size or 256

    out: List[Optional[List[float]]] = []
    for batch in _batched(texts, bs):
        t0 = time.time()
        # truncado por tokens si corresponde
        prep = [_truncate_by_tokens(t, max_input_tokens, tokenizer) if t else "" for t in batch]
        try:
            resp = client.embeddings.create(model=model, input=prep)
            vecs = [d.embedding for d in resp.data]
            # sanity: OpenAI mantiene orden
            for v in vecs:
                if normalize:
                    v = _l2_normalize(v)
                if dim and len(v) != dim:
                    log.warning("OpenAI dim=%d != spec.dim=%d (modelo=%s)", len(v), dim, model)
                out.append(v)
            log.debug("OpenAI embed batch=%d ms=%.0f", len(batch), (time.time() - t0) * 1000)
        except Exception as e:
            log.exception("OpenAI embeddings fallo: %s", e)
            out.extend([None] * len(batch))
    return out


def _embed_cohere(texts: List[str], model: str, dim: int, normalize: bool, max_input_tokens: Optional[int],
                  tokenizer: str, batch_size: Optional[int], input_type: str = "search_document") -> List[Optional[List[float]]]:
    import cohere

    if not settings.cohere_api_key:
        log.error("COHERE_API_KEY no configurada")
        return [None] * len(texts)

    client = cohere.Client(api_key=settings.cohere_api_key)
    bs = batch_size or 96  # tamaños típicos cohere

    out: List[Optional[List[float]]] = []
    for batch in _batched(texts, bs):
        t0 = time.time()
        prep = [_truncate_by_tokens(t, max_input_tokens, tokenizer) if t else "" for t in batch]
        try:
            # SDKs recientes: client.embed(texts=[...], model="embed-...-v3.0", input_type="search_document")
            try:
                resp = client.embed(texts=prep, model=model, input_type=input_type)
                vecs = resp.embeddings
            except Exception:
                # fallback por si la firma difiere
                resp = client.embed(model=model, input_type=input_type, texts=prep)
                vecs = resp.embeddings

            for v in vecs:
                if normalize:
                    v = _l2_normalize(v)
                if dim and len(v) != dim:
                    log.warning("Cohere dim=%d != spec.dim=%d (modelo=%s)", len(v), dim, model)
                out.append(v)
            log.debug("Cohere embed batch=%d ms=%.0f", len(batch), (time.time() - t0) * 1000)
        except Exception as e:
            log.exception("Cohere embeddings fallo: %s", e)
            out.extend([None] * len(batch))
    return out


def _embed_gemini(texts: List[str], model: str, dim: int, normalize: bool,
                  max_input_tokens: Optional[int], tokenizer: str, batch_size: Optional[int],
                  *, task_type: Optional[str] = None, output_dimensionality: Optional[int] = None
                  ) -> List[Optional[List[float]]]:
    """
    Implementación actualizada para Gemini:
      - SDK preferido: google-genai (client.models.embed_content)
      - Fallback: google.generativeai.embed_content
      - Soporta batch (SDK nuevo) y config: task_type, output_dimensionality
    """
    if not settings.gemini_api_key:
        log.error("GEMINI_API_KEY no configurada")
        return [None] * len(texts)

    # Sanitizar textos + truncado defensivo
    prep_all = [_truncate_by_tokens(t, max_input_tokens, tokenizer) if t else "" for t in texts]

    # -------- Ruta 1: SDK nuevo (google-genai) --------
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore

        client = genai.Client(api_key=settings.gemini_api_key)

        # El SDK nuevo permite lista de strings directamente (batch)
        cfg = None
        if task_type or output_dimensionality:
            cfg = types.EmbedContentConfig(
                task_type=task_type if task_type else None,
                output_dimensionality=output_dimensionality if output_dimensionality else None,
            )

        t0 = time.time()
        resp = client.models.embed_content(
            model=model,                       # p.ej. "gemini-embedding-001"
            contents=prep_all,                 # batch
            config=cfg
        )
        # resp.embeddings es una lista de objetos con .values
        out: List[Optional[List[float]]] = []
        for e in getattr(resp, "embeddings", []):
            v = list(getattr(e, "values", []) or [])
            if not v:
                out.append(None)
                continue
            if normalize:
                v = _l2_normalize(v)
            if dim and len(v) != dim:
                log.warning("Gemini dim=%d != spec.dim=%d (modelo=%s)", len(v), dim, model)
            out.append(v)
        # Alinear por si la API devolvió menos items (errores silenciosos)
        while len(out) < len(prep_all):
            out.append(None)
        log.debug("Gemini embed (genai) batch=%d ms=%.0f", len(prep_all), (time.time() - t0) * 1000)
        return out

    except Exception as e_old:
        log.exception("Gemini embeddings: no se pudo usar ni google-genai ni legacy: %s", e_old)
        return [None] * len(texts)



def _embed_labse(texts: List[str], model: str, dim: int, normalize: bool, max_input_tokens: Optional[int],
                 tokenizer: str, batch_size: Optional[int]) -> List[Optional[List[float]]]:
    """
    Soporte opcional para embebidos locales (LaBSE, etc.).
    model: e.g. "sentence-transformers/LaBSE"
    """
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(model)
    # truncado aproximado por tokens si pediste max_input_tokens (no siempre aplica a ST)
    prep = [_truncate_by_tokens(t, max_input_tokens, tokenizer) if t else "" for t in texts]
    try:
        vecs = m.encode(prep, convert_to_numpy=False, normalize_embeddings=False)
        out: List[Optional[List[float]]] = []
        for v in vecs:
            vlist = list(map(float, v))
            if normalize:
                vlist = _l2_normalize(vlist)
            if dim and len(vlist) != dim:
                log.warning("LaBSE dim=%d != spec.dim=%d (modelo=%s)", len(vlist), dim, model)
            out.append(vlist)
        return out
    except Exception as e:
        log.exception("LaBSE embeddings fallo: %s", e)
        return [None] * len(texts)


# ---------------------------
# Public API
# ---------------------------
def embed_texts(texts: List[Optional[str]], spec_like: Any) -> List[Optional[List[float]]]:
    """
    Calcula embeddings para 'texts' según 'spec'.
    Retorna lista alineada con textos (puede contener None si algún texto falla o es vacío).

    spec_like: dict u objeto con:
      - name: str               # nombre del named vector (no usado acá, pero lo loggeamos)
      - provider: "openai" | "cohere" | "gemini" | "labse"
      - model: str
      - dim: int
      - source_field: str       # yi_text | tr_en_text (no usado aquí)
      - max_input_tokens?: int
      - normalize?: bool = True
      - tokenizer?: str = "cl100k_base"
      - batch_size?: int
    """
    spec = _coerce_spec(spec_like)
    provider = spec.get("provider")
    model = spec.get("model")
    dim = int(spec.get("dim"))
    normalize = bool(spec.get("normalize", True))
    max_input_tokens = spec.get("max_input_tokens")
    tokenizer = spec.get("tokenizer", "cl100k_base")
    batch_size = spec.get("batch_size")

    # Sanitizar textos (alineación garantizada)
    clean_texts: List[str] = [(t or "").strip() for t in texts]
    if not any(clean_texts):
        return [None] * len(texts)

    log.info("Embeddings start provider=%s model=%s dim=%s normalize=%s", provider, model, dim, normalize)

    if provider == "openai":
        return _embed_openai(clean_texts, model, dim, normalize, max_input_tokens, tokenizer, batch_size)
    elif provider == "cohere":
        return _embed_cohere(clean_texts, model, dim, normalize, max_input_tokens, tokenizer, batch_size,
                             input_type=spec.get("input_type", "search_document"))
    elif provider == "gemini":
        return _embed_gemini(
            clean_texts, model, dim, normalize, max_input_tokens, tokenizer, batch_size,
            task_type=spec.get("task_type"),
            output_dimensionality=spec.get("dim"),
        )
    elif provider == "labse":
        return _embed_labse(clean_texts, model, dim, normalize, max_input_tokens, tokenizer, batch_size)
    else:
        raise ValueError(f"Provider no soportado: {provider}")
