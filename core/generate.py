# core/generate.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .retrieve import Retriever, SearchFilters
from engines.contracts import GeneratorSpec
from engines.registry import make_generator


# ---------------------------
# Config loader para Generator
# ---------------------------
def load_generator_spec(path_yaml: str = "config/generator.yaml") -> GeneratorSpec:
    with open(path_yaml, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    g = d.get("generator", {})
    if not g:
        raise FileNotFoundError("config/generator.yaml: falta bloque 'generator'")
    return GeneratorSpec(
        provider=g["provider"],
        model=g["model"],
        max_tokens=int(g.get("max_tokens", 512)),
        temperature=float(g.get("temperature", 0.2)),
        top_p=float(g.get("top_p", 1.0)),
        target_lang=str(g.get("target_lang", "auto")),
    )


# ---------------------------
# Utilidades de contexto
# ---------------------------
def _guess_lang(s: str) -> str:
    # heurística mínima: detecta si hay muchos caracteres hebreos → 'yi'
    heb = sum("\u0590" <= ch <= "\u05FF" for ch in s)
    if heb >= 0.2 * max(1, len(s)):
        return "yi"
    # si tiene acentos comunes ES, asumimos 'es'; sino 'en'
    if any(c in s for c in "áéíóúñÁÉÍÓÚÑ"):
        return "es"
    return "en"


def _build_context_blocks(
    hits: List[Dict[str, Any]],
    max_chars: int = 12000,
    include_yi: bool = True,
) -> Tuple[str, List[Tuple[str, int, int]]]:
    """
    Construye un solo string de contexto con encabezados y devuelve también
    la lista de citas [(book_id, page, chunk_idx)] en el orden de aparición.
    """
    parts: List[str] = []
    citations: List[Tuple[str, int, int]] = []
    used = 0

    for h in hits:
        p = h["payload"] if "payload" in h else h
        book_id = p.get("book_id")
        page = p.get("page")
        chunk_idx = p.get("chunk_idx")
        title_en = p.get("title_en") or ""
        title_yi = p.get("title_yi") or ""
        author = p.get("author") or ""
        year = p.get("year") or ""
        tr_en = p.get("tr_en_text") or ""
        yi_tx = p.get("yi_text") or ""

        header = f"[{book_id} | p.{page} | chunk {chunk_idx}] {title_en or title_yi} — {author} ({year})"
        body = tr_en.strip() if tr_en else yi_tx.strip()
        if include_yi and tr_en and yi_tx:
            body = f"{tr_en.strip()}\n\n— Original (YI) —\n{yi_tx.strip()}"

        block = f"### {header}\n{body}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
        if book_id is not None and page is not None and chunk_idx is not None:
            citations.append((str(book_id), int(page), int(chunk_idx)))

    return "\n".join(parts), citations


def _default_system_prompt(target_lang: str) -> str:
    return (
        "You are a meticulous historical research assistant. "
        f"Answer in {target_lang}. "
        "Use only the provided CONTEXT passages. "
        "Cite sources inline using the format [book_id p.PAGE #CHUNK]. "
        "If there is insufficient evidence, say so and summarize what is known from the context."
    )


def _format_user_prompt(query: str, context: str) -> str:
    return f"QUERY:\n{query}\n\nCONTEXT:\n{context}\n\nINSTRUCTIONS:\n- Base your answer strictly on CONTEXT.\n- Add citations like [book123 p.45 #2] where they support claims."


# ---------------------------
# API pública del pipeline
# ---------------------------
@dataclass
class GenerateParams:
    using: str = "dense_yi_gemini"  # named vector para retrieval
    k: int = 6
    filters: Optional[SearchFilters] = None
    include_yi: bool = True
    max_context_chars: int = 12000
    # override opcionales:
    target_lang: Optional[str] = None         # "es" | "en" | "yi" | "auto"
    system_prompt: Optional[str] = None
    stop: Optional[List[str]] = None


def answer(
    query: str,
    gen_spec: Optional[GeneratorSpec] = None,
    params: Optional[GenerateParams] = None,
    vectors_yaml: str = "config/vectors.yaml",
    generator_yaml: str = "config/generator.yaml",
) -> Dict[str, Any]:
    """
    Pipeline RAG:
      1) Retrieve top-k con filtros
      2) Construcción de contexto con límite de chars
      3) Generación con citas
      4) Devuelve: {answer, citations, hits, usage, prompts}
    """
    params = params or GenerateParams()
    retriever = Retriever(vectors_yaml=vectors_yaml)

    # 1) RETRIEVE
    hits = retriever.search(
        query=query,
        k=params.k,
        using=params.using,
        filters=params.filters,
        with_payload=True,
    )

    # 2) CONTEXT
    context, citations = _build_context_blocks(
        hits, max_chars=params.max_context_chars, include_yi=params.include_yi
    )

    # 3) GENERATION
    spec = gen_spec or load_generator_spec(generator_yaml)

    # idioma objetivo
    lang_query = _guess_lang(query)
    target_lang = (params.target_lang or spec.target_lang or "auto")
    if target_lang == "auto":
        target_lang = "es" if lang_query == "es" else "en"

    system_prompt = params.system_prompt or _default_system_prompt(target_lang)
    user_prompt = _format_user_prompt(query, context)

    generator = make_generator(spec)
    out = generator.generate(system_prompt=system_prompt, user_prompt=user_prompt, stop=params.stop)

    return {
        "answer": out["text"],
        "citations": [{"book_id": b, "page": p, "chunk_idx": c} for b, p, c in citations],
        "hits": hits,
        "usage": out.get("usage"),
        "prompts": {"system": system_prompt, "user": user_prompt},
        "meta": {
            "using": params.using,
            "k": params.k,
            "target_lang": target_lang,
            "lang_query": lang_query,
        },
    }
