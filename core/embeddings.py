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
                  tokenizer: str, batch_size: Optional[int], input_type: str = "search_document",
                  *, job_mode: bool = False, polling_interval: float = 2.0) -> List[Optional[List[float]]]:
    """Embed via Cohere standard endpoint OR async jobs API (job_mode=True).

    job_mode advantages: send whole corpus, fewer network round trips; downside: latency until job ready.
    We only switch if job_mode flag provided because jobs cost a bit more coordination.
    """
    import cohere, time as _time

    if not settings.cohere_api_key:
        log.error("COHERE_API_KEY no configurada")
        return [None] * len(texts)

    client = cohere.Client(api_key=settings.cohere_api_key)

    # ---- Async Jobs path ----
    if job_mode:
        # Prepare (truncate) once
        prep_all = [_truncate_by_tokens(t, max_input_tokens, tokenizer) if t else "" for t in texts]
        try:
            start = _time.time()
            job = client.embed_jobs.create(model=model, texts=prep_all, input_type=input_type)
            job_id = job.id if hasattr(job, 'id') else job.get('id')
            log.info("Cohere embed job submitted id=%s items=%d", job_id, len(prep_all))
            status = getattr(job, 'status', None) or job.get('status')
            while status in (None, 'processing', 'queued'):
                _time.sleep(polling_interval)
                job = client.embed_jobs.get(id=job_id)
                status = getattr(job, 'status', None) or job.get('status')
                log.debug("Cohere job status=%s", status)
            if status != 'completed':
                log.error("Cohere embed job failed status=%s", status)
                return [None] * len(texts)
            # Retrieve embeddings pages
            out_vecs: List[Optional[List[float]]] = [None] * len(prep_all)
            for page in client.embed_jobs.get_embeddings(id=job_id):  # generator of pages
                # page.embeddings is list of {index, embedding}
                records = getattr(page, 'embeddings', []) or []
                for rec in records:
                    idx = rec.get('index') if isinstance(rec, dict) else getattr(rec, 'index', None)
                    emb = rec.get('embedding') if isinstance(rec, dict) else getattr(rec, 'embedding', None)
                    if idx is None:
                        continue
                    if emb:
                        v = list(emb)
                        if normalize:
                            v = _l2_normalize(v)
                        if dim and len(v) != dim:
                            log.warning("Cohere(dim mismatch job) %d != %d", len(v), dim)
                        out_vecs[idx] = v
            log.info("Cohere embed job completed in %.1fs", _time.time() - start)
            return out_vecs
        except Exception as e:
            log.exception("Cohere embed job fallo: %s -- fallback a modo batch normal", e)
            # fall through to normal path

    # ---- Standard batched synchronous path ----
    bs = batch_size or 96
    out: List[Optional[List[float]]] = []
    for batch in _batched(texts, bs):
        t0 = time.time()
        prep = [_truncate_by_tokens(t, max_input_tokens, tokenizer) if t else "" for t in batch]
        try:
            try:
                resp = client.embed(texts=prep, model=model, input_type=input_type)
                vecs = resp.embeddings
            except Exception:
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
    """Embed texts with Gemini with adaptive batching and legacy fallback.

    Adaptations:
      - Split requests into batches (default batch_size or adaptive) to avoid 4MB payload limit.
      - If a batch triggers a payload size error (400 payload size exceeds), halve the batch size and retry.
      - Falls back to legacy `google.generativeai` SDK per-text calls if new SDK unavailable.
    """
    if not settings.gemini_api_key:
        log.error("GEMINI_API_KEY no configurada")
        return [None] * len(texts)

    # Preprocess + defensive truncation
    prep_all = [_truncate_by_tokens(t, max_input_tokens, tokenizer) if t else "" for t in texts]
    total_n = len(prep_all)
    if total_n == 0:
        return []

    # Helper for post-processing vectors
    def _finish_vec(raw: List[float]) -> List[float]:
        v = list(raw)
        if normalize:
            v = _l2_normalize(v)
        if dim and len(v) != dim:
            log.warning("Gemini dim=%d != spec.dim=%d (modelo=%s)", len(v), dim, model)
        return v

    # Try modern google-genai batched path
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore

        client = genai.Client(api_key=settings.gemini_api_key)
        cfg = None
        if task_type or output_dimensionality:
            cfg = types.EmbedContentConfig(
                task_type=task_type if task_type else None,
                output_dimensionality=output_dimensionality if output_dimensionality else None,
            )
        # Gemini API limita a 100 requests por batch
        bs_default = min(batch_size or 100, 100)
        out: List[Optional[List[float]]] = []
        i = 0
        batch_index = 0
        while i < total_n:
            current_bs = min(bs_default, total_n - i)
            # Adaptive shrinking loop for current starting point
            while True:
                sub = prep_all[i : i + current_bs]
                approx_bytes = sum(len(s.encode('utf-8')) for s in sub)
                if approx_bytes > 3_800_000 and current_bs > 1:
                    current_bs = max(1, current_bs // 2)
                    continue
                t0 = time.time()
                try:
                    resp = client.models.embed_content(
                        model=model,
                        contents=sub,
                        config=cfg,
                    )
                    batch_vecs: List[Optional[List[float]]] = []
                    for e in getattr(resp, "embeddings", []) or []:
                        values = list(getattr(e, "values", []) or [])
                        batch_vecs.append(_finish_vec(values) if values else None)
                    while len(batch_vecs) < len(sub):
                        batch_vecs.append(None)
                    out.extend(batch_vecs)
                    log.debug(
                        "Gemini embed batch_index=%d size=%d bytes=%d ms=%.0f",
                        batch_index, current_bs, approx_bytes, (time.time() - t0) * 1000,
                    )
                    i += current_bs
                    batch_index += 1
                    break
                except Exception as e:
                    msg = str(e).lower()
                    if ("payload size exceeds" in msg or "request payload size" in msg) and current_bs > 1:
                        log.warning(
                            "Gemini payload too large. Reducing batch_size from %d to %d", current_bs, max(1, current_bs // 2)
                        )
                        current_bs = max(1, current_bs // 2)
                        continue
                    log.exception(
                        "Gemini embeddings fallo en batch_index=%d size=%d: %s", batch_index, current_bs, e
                    )
                    out.extend([None] * current_bs)
                    i += current_bs
                    batch_index += 1
                    break
        if len(out) < total_n:
            out.extend([None] * (total_n - len(out)))
        return out
    except Exception as modern_err:
        log.warning("Gemini nuevo SDK fallo (%s). Intentando fallback legacy por texto.", modern_err)

    # Legacy fallback: google.generativeai (no batching endpoint, so per-text)
    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=settings.gemini_api_key)
        out: List[Optional[List[float]]] = []
        for idx, t in enumerate(prep_all):
            try:
                resp = genai.embed_content(
                    model=model,
                    content=t,
                    task_type=task_type if task_type else None,
                    output_dimensionality=output_dimensionality if output_dimensionality else None,
                )
                vec = resp.get("embedding") if isinstance(resp, dict) else getattr(resp, "embedding", None)
                if not vec:
                    out.append(None)
                    continue
                out.append(_finish_vec(list(vec)))
            except Exception as e:
                log.exception("Gemini legacy fallo idx=%d: %s", idx, e)
                out.append(None)
        if len(out) < total_n:
            out.extend([None] * (total_n - len(out)))
        return out
    except Exception as legacy_err:
        log.exception("Gemini embeddings: no se pudo usar ni google-genai ni legacy: %s", legacy_err)
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
        return _embed_cohere(
            clean_texts,
            model,
            dim,
            normalize,
            max_input_tokens,
            tokenizer,
            batch_size,
            input_type=spec.get("input_type", "search_document"),
            job_mode=spec.get("job_mode", False),
            polling_interval=spec.get("polling_interval", 2.0),
        )
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
