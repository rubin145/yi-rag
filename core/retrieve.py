# core/retrieve.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml
from qdrant_client import models as qm

from .config import settings
from .embeddings import embed_texts
from .store_qdrant import get_client


# ---------------------------
# VectorSpec loader (desde YAML)
# ---------------------------
@dataclass
class VectorSpec:
    name: str
    provider: str
    model: str
    dim: int
    source_field: str
    max_input_tokens: Optional[int] = None
    normalize: bool = True
    tokenizer: str = "cl100k_base"
    batch_size: Optional[int] = None
    # extras opcionales
    task_type: Optional[str] = None  # Gemini: e.g. RETRIEVAL_DOCUMENT

def _load_specs(path_yaml: str = "config/vectors.yaml") -> Dict[str, VectorSpec]:
    if not os.path.exists(path_yaml):
        raise FileNotFoundError(f"No existe {path_yaml}")
    with open(path_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    specs = {}
    for d in cfg.get("vectors", []):
        vs = VectorSpec(
            name=d["name"],
            provider=d["provider"],
            model=d["model"],
            dim=int(d["dim"]),
            source_field=d["source_field"],
            max_input_tokens=d.get("max_input_tokens"),
            normalize=bool(d.get("normalize", True)),
            tokenizer=d.get("tokenizer", "cl100k_base"),
            batch_size=d.get("batch_size"),
            task_type=d.get("task_type"),
        )
        specs[vs.name] = vs
    return specs


# ---------------------------
# Filtros de búsqueda
# ---------------------------
@dataclass
class SearchFilters:
    book_ids: Optional[List[str]] = None
    authors: Optional[List[str]] = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None

def _to_qdrant_filter(f: Optional[SearchFilters]) -> Optional[qm.Filter]:
    if not f:
        return None
    must: List[qm.FieldCondition | qm.Condition] = []
    if f.book_ids:
        must.append(qm.FieldCondition(key="book_id", match=qm.MatchAny(any=f.book_ids)))
    if f.authors:
        must.append(qm.FieldCondition(key="author", match=qm.MatchAny(any=f.authors)))
    if f.year_from is not None or f.year_to is not None:
        rng = {}
        if f.year_from is not None: rng["gte"] = int(f.year_from)
        if f.year_to   is not None: rng["lte"] = int(f.year_to)
        must.append(qm.FieldCondition(key="year", range=qm.Range(**rng)))
    return qm.Filter(must=must) if must else None


# ---------------------------
# Retriever
# ---------------------------
class Retriever:
    def __init__(self, collection: Optional[str] = None, vectors_yaml: str = "config/vectors.yaml"):
        self.client = get_client()
        self.collection = collection or settings.collection_name
        self.specs = _load_specs(vectors_yaml)

    def list_vectors(self) -> List[str]:
        return sorted(self.specs.keys())

    def _query_embedding(self, query: str, spec: VectorSpec) -> List[float] | None:
        """
        Embebe la query ajustando modo 'query' según provider.
        - Gemini: task_type=RETRIEVAL_QUERY
        - Cohere: input_type='search_query' (ver patch en embeddings.py más abajo)
        - OpenAI: igual (no diferencia query/document)
        """
        s: Dict[str, Any] = dict(
            name=spec.name,
            provider=spec.provider,
            model=spec.model,
            dim=spec.dim,
            source_field="__query__",  # no se usa en embed_texts
            max_input_tokens=spec.max_input_tokens,
            normalize=spec.normalize,
            tokenizer=spec.tokenizer,
            batch_size=spec.batch_size,
        )
        if spec.provider == "gemini":
            s["task_type"] = "RETRIEVAL_QUERY"
        if spec.provider == "cohere":
            s["input_type"] = "search_query"  # requiere patch en embeddings._embed_cohere

        vec = embed_texts([query], s)[0]
        return vec

    def search(
        self,
        query: str,
        k: int = 5,
        using: str = "dense_en_gemini",
        filters: Optional[SearchFilters] = None,
        with_payload: bool = True,
    ) -> List[Dict[str, Any]]:
        if using not in self.specs:
            raise ValueError(f"Named vector desconocido: {using}. Disponibles: {self.list_vectors()}")

        spec = self.specs[using]
        qvec = self._query_embedding(query, spec)
        if qvec is None:
            raise RuntimeError("No se pudo generar embedding de la query")

        qv = qm.NamedVector(name=using, vector=qvec)
        flt = _to_qdrant_filter(filters)

        hits = self.client.search(
            collection_name=self.collection,
            query_vector=qv,
            limit=int(k),
            with_payload=with_payload,
            with_vectors=False,
            query_filter=flt,
        )

        out: List[Dict[str, Any]] = []
        for h in hits:
            out.append({
                "id": h.id,
                "score": float(h.score),
                "payload": h.payload or {},
                # campos prácticos al tope:
                "book_id": h.payload.get("book_id") if h.payload else None,
                "page": h.payload.get("page") if h.payload else None,
                "title_en": h.payload.get("title_en") if h.payload else None,
                "title_yi": h.payload.get("title_yi") if h.payload else None,
                "author": h.payload.get("author") if h.payload else None,
                "year": h.payload.get("year") if h.payload else None,
                "yi_text": h.payload.get("yi_text") if h.payload else None,
                "tr_en_text": h.payload.get("tr_en_text") if h.payload else None,
            })
        return out
