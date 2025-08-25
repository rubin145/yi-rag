# core/retrieve.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml
from qdrant_client import models as qm
from .sparse_vectors import CharTfidfSparseEncoder

# Named sparse vector identifier in Qdrant collection
SPARSE_VECTOR_NAME = "sparse_vector_yi"

from .config import settings
from .embeddings import embed_texts
from .store_qdrant import get_client
from .schemas import SearchFilters


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

def _to_qdrant_filter(f: Optional[SearchFilters]) -> Optional[qm.Filter]:
    if not f:
        return None
    must: List[qm.FieldCondition | qm.Condition] = []
    
    # Single book_id filter
    if f.book_id:
        must.append(qm.FieldCondition(key="book_id", match=qm.MatchValue(value=f.book_id)))
    
    # Author filters (single)
    if f.author:
        must.append(qm.FieldCondition(key="author", match=qm.MatchValue(value=f.author)))
    if f.author_yi:
        must.append(qm.FieldCondition(key="author_yi", match=qm.MatchValue(value=f.author_yi)))
    # Multi-select list OR logic (converted to should inside a nested filter)
    if f.author_list:
        should = [qm.FieldCondition(key="author", match=qm.MatchValue(value=a)) for a in f.author_list if a]
        if should:
            must.append(qm.Filter(should=should, must=[]))
    
    # New metadata filters
    if f.place:
        must.append(qm.FieldCondition(key="place", match=qm.MatchValue(value=f.place)))
    if f.place_list:
        should = [qm.FieldCondition(key="place", match=qm.MatchValue(value=p)) for p in f.place_list if p]
        if should:
            must.append(qm.Filter(should=should, must=[]))
    if f.publisher:
        must.append(qm.FieldCondition(key="publisher", match=qm.MatchValue(value=f.publisher)))
    if f.publisher_list:
        should = [qm.FieldCondition(key="publisher", match=qm.MatchValue(value=p)) for p in f.publisher_list if p]
        if should:
            must.append(qm.Filter(should=should, must=[]))
    def _expand_subject_tokens(subj: str) -> List[str]:
        parts = [p.strip() for p in subj.replace(";", ",").split(",") if p.strip()]
        return parts
    if f.subjects:
        # treat provided string as possibly multi-value
        tokens = _expand_subject_tokens(f.subjects)
        if len(tokens) == 1:
            must.append(qm.FieldCondition(key="subjects", match=qm.MatchText(text=tokens[0])))
        elif tokens:
            should = [qm.FieldCondition(key="subjects", match=qm.MatchText(text=t)) for t in tokens]
            must.append(qm.Filter(should=should, must=[]))
    if f.subjects_list:
        # Prefer token list matching if subjects_tokens exists
        should = [qm.FieldCondition(key="subjects_tokens", match=qm.MatchValue(value=s)) for s in f.subjects_list if s]
        if should:
            must.append(qm.Filter(should=should, must=[]))
    
    # Year range filter
    if f.year_min is not None or f.year_max is not None:
        rng = {}
        if f.year_min is not None: rng["gte"] = int(f.year_min)
        if f.year_max is not None: rng["lte"] = int(f.year_max)
        must.append(qm.FieldCondition(key="year", range=qm.Range(**rng)))
    
    # Page range filter
    if f.page_min is not None or f.page_max is not None:
        rng = {}
        if f.page_min is not None: rng["gte"] = int(f.page_min)
        if f.page_max is not None: rng["lte"] = int(f.page_max)
        must.append(qm.FieldCondition(key="page", range=qm.Range(**rng)))
    
    return qm.Filter(must=must) if must else None


# ---------------------------
# Retriever
# ---------------------------
class Retriever:
    def __init__(self, collection: Optional[str] = None, vectors_yaml: str = "config/vectors.yaml"):
        self.client = get_client()
        self.collection = collection or settings.collection_name
        self.specs = _load_specs(vectors_yaml)
        # Lazy-loaded local TF-IDF sparse encoder (indices/values) for query sparse vectors
        self._sparse_encoder: CharTfidfSparseEncoder | None = None

    def _ensure_sparse_encoder(self):
        if self._sparse_encoder is None:
            try:
                self._sparse_encoder = CharTfidfSparseEncoder.load("data/sparse_model")
            except Exception as e:
                raise RuntimeError(
                    "Sparse encoder artifacts missing in data/sparse_model. Re-run build_sparse step (scripts/2_build_sparse_vectors.py)."
                ) from e

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
        score_threshold: Optional[float] = None,
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

        # Dense cosine similarities already ~0-1; treat them as normalized for UI.
        out: List[Dict[str, Any]] = []
        for h in hits:
            raw = float(h.score)
            norm_score = raw  # already in suitable range
            if score_threshold is not None and norm_score < score_threshold:
                continue
            p = h.payload or {}
            content_field = p.get("tr_en_text") or p.get("en_text") or p.get("yi_text") or p.get("text") or p.get("content") or p.get("chunk")
            out.append({
                "id": h.id,
                "score": norm_score,            # normalized/display score
                "raw_dense_score": raw,          # raw (same here)
                "mode": "dense",
                "content": content_field,
                "payload": p,
                "book_id": p.get("book_id"),
                "page": p.get("page"),
                "chunk_idx": p.get("chunk_idx"),
                "title_en": p.get("title_en"),
                "title_yi": p.get("title_yi"),
                "title_yi_latin": p.get("title_yi_latin"),
                "author": p.get("author"),
                "year": p.get("year"),
                "yi_text": p.get("yi_text"),
                "tr_en_text": p.get("tr_en_text"),
                "ocr_source_url": p.get("ocr_source_url"),
                "source_url": p.get("source_url"),
                "metadata_url": p.get("metadata_url"),
            })
        return out

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        dense: str = "dense_en_gemini",
        filters: Optional[SearchFilters] = None,
        alpha: float = 0.5,
        with_payload: bool = True,
        score_threshold: Optional[float] = None,
        oversample: int = 3,
        normalize: bool = True,
    return_debug: bool = False,
    ) -> List[Dict[str, Any]]:
        """Late-fusion hybrid search (dense + sparse) with optional normalization.

        Implementation performs two independent searches (dense + sparse) to avoid client batch quirks,
        then fuses results. alpha=1 => pure dense, alpha=0 => pure sparse.
        """
        if dense not in self.specs:
            raise ValueError(f"Named vector desconocido: {dense}. Disponibles: {self.list_vectors()}")
        spec = self.specs[dense]
        qvec = self._query_embedding(query, spec)
        if qvec is None:
            raise RuntimeError("No se pudo generar embedding de la query")
        self._ensure_sparse_encoder()
        assert self._sparse_encoder is not None
        s_idx, s_val = self._sparse_encoder.transform_query(query)
        flt = _to_qdrant_filter(filters)
        k_side = int(k * oversample)

        # Dense search
        dense_hits = self.client.search(
            collection_name=self.collection,
            query_vector=qm.NamedVector(name=dense, vector=qvec),
            limit=k_side,
            with_payload=with_payload,
            with_vectors=False,
            query_filter=flt,
        )
        # Sparse search
        sparse_hits = self.client.search(
            collection_name=self.collection,
            query_vector=qm.NamedSparseVector(name=SPARSE_VECTOR_NAME, vector=qm.SparseVector(indices=s_idx, values=s_val)),
            limit=k_side,
            with_payload=with_payload,
            with_vectors=False,
            query_filter=flt,
        )

        # If alpha extremes, short-circuit
        if alpha <= 0.0:
            out_sparse: List[Dict[str, Any]] = []
            for h in sparse_hits[:k]:
                if score_threshold is not None and float(h.score) < score_threshold:
                    continue
                p = h.payload or {}
                content_field = p.get("tr_en_text") or p.get("en_text") or p.get("yi_text") or p.get("text") or p.get("content") or p.get("chunk")
                out_sparse.append({
                    "id": h.id,
                    "score": float(h.score),
                    "dense_score": 0.0,
                    "sparse_score": float(h.score),
                    "content": content_field,
                    "payload": p,
                    "book_id": p.get("book_id"),
                    "page": p.get("page"),
                    "chunk_idx": p.get("chunk_idx"),
                    "title_en": p.get("title_en"),
                    "title_yi": p.get("title_yi"),
                    "title_yi_latin": p.get("title_yi_latin"),
                    "author": p.get("author"),
                    "year": p.get("year"),
                    "yi_text": p.get("yi_text"),
                    "tr_en_text": p.get("tr_en_text"),
                    "ocr_source_url": p.get("ocr_source_url"),
                    "source_url": p.get("source_url"),
                    "metadata_url": p.get("metadata_url"),
                })
            return out_sparse if return_debug else out_sparse[:k]
        if alpha >= 1.0:
            out_dense: List[Dict[str, Any]] = []
            for h in dense_hits[:k]:
                if score_threshold is not None and float(h.score) < score_threshold:
                    continue
                p = h.payload or {}
                content_field = p.get("tr_en_text") or p.get("en_text") or p.get("yi_text") or p.get("text") or p.get("content") or p.get("chunk")
                out_dense.append({
                    "id": h.id,
                    "score": float(h.score),
                    "dense_score": float(h.score),
                    "sparse_score": 0.0,
                    "content": content_field,
                    "payload": p,
                    "book_id": p.get("book_id"),
                    "page": p.get("page"),
                    "chunk_idx": p.get("chunk_idx"),
                    "title_en": p.get("title_en"),
                    "title_yi": p.get("title_yi"),
                    "title_yi_latin": p.get("title_yi_latin"),
                    "author": p.get("author"),
                    "year": p.get("year"),
                    "yi_text": p.get("yi_text"),
                    "tr_en_text": p.get("tr_en_text"),
                    "ocr_source_url": p.get("ocr_source_url"),
                    "source_url": p.get("source_url"),
                    "metadata_url": p.get("metadata_url"),
                })
            return out_dense if return_debug else out_dense[:k]

        # Build combined map
        combined: Dict[Any, Dict[str, Any]] = {}
        for h in dense_hits:
            combined[h.id] = {"payload": h.payload or {}, "dense_score": float(h.score), "sparse_score": 0.0}
        for h in sparse_hits:
            c = combined.get(h.id)
            if c:
                c["sparse_score"] = float(h.score)
            else:
                combined[h.id] = {"payload": h.payload or {}, "dense_score": 0.0, "sparse_score": float(h.score)}

    # Normalization per modality (dense & sparse each 0..1 for fusion)
        if normalize and combined:
            max_dense = max(v["dense_score"] for v in combined.values()) or 1.0
            max_sparse = max(v["sparse_score"] for v in combined.values()) or 1.0
            for v in combined.values():
                v["dense_score_norm"] = v["dense_score"] / max_dense if max_dense else 0.0
                v["sparse_score_norm"] = v["sparse_score"] / max_sparse if max_sparse else 0.0
        else:
            for v in combined.values():
                v["dense_score_norm"] = v["dense_score"]
                v["sparse_score_norm"] = v["sparse_score"]

        out: List[Dict[str, Any]] = []
        for pid, data in combined.items():
            blended = alpha * data["dense_score_norm"] + (1 - alpha) * data["sparse_score_norm"]
            if score_threshold is not None and blended < score_threshold:
                continue
            p = data["payload"]
            content_field = None
            if p:
                content_field = p.get("tr_en_text") or p.get("en_text") or p.get("yi_text") or p.get("text") or p.get("content") or p.get("chunk")
            out.append({
                "id": pid,
                "score": blended,  # blended normalized score for UI / threshold
                "raw_dense_score": data.get("dense_score"),
                "raw_sparse_score": data.get("sparse_score"),
                "dense_score_norm": data.get("dense_score_norm"),
                "sparse_score_norm": data.get("sparse_score_norm"),
                "content": content_field,
                "payload": p,
                "book_id": p.get("book_id") if p else None,
                "page": p.get("page") if p else None,
                "chunk_idx": p.get("chunk_idx") if p else None,
                "title_en": p.get("title_en") if p else None,
                "title_yi": p.get("title_yi") if p else None,
                "title_yi_latin": p.get("title_yi_latin") if p else None,
                "author": p.get("author") if p else None,
                "year": p.get("year") if p else None,
                "yi_text": p.get("yi_text") if p else None,
                "tr_en_text": p.get("tr_en_text") if p else None,
                "ocr_source_url": p.get("ocr_source_url") if p else None,
                "source_url": p.get("source_url") if p else None,
                "metadata_url": p.get("metadata_url") if p else None,
            })
        out.sort(key=lambda x: x["score"], reverse=True)
        return out if return_debug else out[:k]

    def sparse_search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[SearchFilters] = None,
        with_payload: bool = True,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Sparse-only search using locally encoded TF-IDF indices/values.

        Returns normalized 'score' (0..1 per query) plus 'raw_sparse_score'. Threshold applies to normalized score.
        """
        flt = _to_qdrant_filter(filters)
        self._ensure_sparse_encoder()
        assert self._sparse_encoder is not None
        s_idx, s_val = self._sparse_encoder.transform_query(query)
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=qm.NamedSparseVector(name=SPARSE_VECTOR_NAME, vector=qm.SparseVector(indices=s_idx, values=s_val)),
            limit=int(k),
            with_payload=with_payload,
            with_vectors=False,
            query_filter=flt,
        )
        raw_scores = [float(h.score) for h in hits]
        max_raw = max(raw_scores) if raw_scores else 0.0
        out: List[Dict[str, Any]] = []
        for h, raw in zip(hits, raw_scores):
            norm = (raw / max_raw) if max_raw > 0 else 0.0
            if score_threshold is not None and norm < score_threshold:
                continue
            p = h.payload or {}
            content_field = p.get("tr_en_text") or p.get("en_text") or p.get("yi_text") or p.get("text") or p.get("content") or p.get("chunk")
            out.append({
                "id": h.id,
                "score": norm,               # normalized per query
                "raw_sparse_score": raw,      # original TF-IDF magnitude
                "mode": "sparse",
                "content": content_field,
                "payload": p,
                "book_id": p.get("book_id"),
                "page": p.get("page"),
                "chunk_idx": p.get("chunk_idx"),
                "title_en": p.get("title_en"),
                "title_yi": p.get("title_yi"),
                "title_yi_latin": p.get("title_yi_latin"),
                "author": p.get("author"),
                "year": p.get("year"),
                "yi_text": p.get("yi_text"),
                "tr_en_text": p.get("tr_en_text"),
                "ocr_source_url": p.get("ocr_source_url"),
                "source_url": p.get("source_url"),
                "metadata_url": p.get("metadata_url"),
            })
        return out

    def get_distinct_field_values(self, field_name: str, limit: int = 1000) -> List[str]:
        """
        Get distinct values for a specific field from the vector database.
        Used for autocomplete functionality in filters.
        """
        try:
            # Use scroll to get all points with just the specified payload field
            scroll_result = self.client.scroll(
                collection_name=self.collection,
                with_payload=[field_name],
                with_vectors=False,
                limit=limit
            )
            
            values = set()
            for point in scroll_result[0]:  # scroll_result is tuple (points, next_page_offset)
                if point.payload and field_name in point.payload:
                    value = point.payload[field_name]
                    if value and str(value).strip():
                        # Clean and add the value
                        clean_value = str(value).strip()
                        if clean_value and clean_value != "None":
                            values.add(clean_value)
            
            # Return sorted list
            return sorted(list(values))
            
        except Exception as e:
            print(f"Error getting distinct values for {field_name}: {e}")
            return []

    def debug_compare(self, query: str, k: int = 5, using: Optional[str] = None) -> Dict[str, Any]:
        """Return side-by-side dense vs sparse ids and scores for quick debugging."""
        using = using or next(iter(self.specs.keys()))
        dense_hits = self.search(query=query, k=k, using=using, with_payload=False)
        sparse_hits = self.sparse_search(query=query, k=k, with_payload=False)
        return {
            "query": query,
            "dense": [(h["id"], round(h["score"], 4)) for h in dense_hits],
            "sparse": [(h["id"], round(h["score"], 4)) for h in sparse_hits],
        }
