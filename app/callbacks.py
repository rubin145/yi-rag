import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from helpers import _make_session_id, _render_hits_html, _format_citations
import gradio as gr
from core.retrieve import SearchFilters, Retriever
from core.generate import GenerateParams, answer
from observability import LangfuseLogger, PhoenixExporter, QueryEvent, FeedbackEvent

_BACKEND_ERR = "Qdrant backend not configured"


phoenix_exporter = PhoenixExporter()
langfuse_logger = LangfuseLogger()

# Initialize retriever with error handling (configs may not exist in deployment)
try:
    retriever = Retriever(vectors_yaml="config/vectors.yaml")
except Exception:
    retriever = None


def log_query_observability(event):
    langfuse_logger.log_query_event(event)
    # Phoenix disabled to prevent duplicate traces
    # phoenix_exporter.log_query_event(event)

def log_feedback_observability(event):
    langfuse_logger.log_feedback_event(event)
    # Phoenix disabled to prevent duplicate traces  
    # phoenix_exporter.log_feedback_event(event)

    
# ---------------------------
# Callbacks
# ---------------------------



def on_search(query: str, search_type: str, author: List[str], subjects: List[str], place: List[str], publisher: List[str],
              year_from: Optional[float], year_to: Optional[float],
              using: str, k: int, score_threshold: float, alpha: float, include_yi: bool,
              session_id: str, debug_hybrid: bool = False) -> Tuple[str, str, List[Dict[str, Any]], Dict[str, Any], str]:
    t0 = time.time()
    if not retriever:
        return gr.update(value=f"<p><b>Backend unavailable:</b> {_BACKEND_ERR}. Configure QDRANT_URL / QDRANT_API_KEY.</p>"), "", [], {}, session_id or _make_session_id()
    if not query or not query.strip():
        return gr.update(value="<p><em>Enter a query.</em></p>"), "", [], {}, session_id or _make_session_id()
    yf = int(year_from) if year_from else None
    yt = int(year_to) if year_to else None
    author_list = [a for a in (author or []) if a]
    place_list = [p for p in (place or []) if p]
    publisher_list = [p for p in (publisher or []) if p]
    subjects_list: List[str] = []
    for s in (subjects or []):
        if not s: continue
        for token in [t.strip() for t in s.replace(";", ",").split(",") if t.strip()]:
            subjects_list.append(token)
    filters = SearchFilters(
        author_list=author_list or None,
        place_list=place_list or None,
        publisher_list=publisher_list or None,
        subjects_list=subjects_list or None,
        year_min=yf, year_max=yt
    )
    st = (search_type or "semantic").lower()
    if st == "semantic":
        hits = retriever.search(query=query.strip(), k=int(k), using=using, filters=filters, with_payload=True, score_threshold=score_threshold)
    elif st == "hybrid":
        hits = retriever.hybrid_search(query=query.strip(), k=int(k), dense=using, filters=filters, alpha=alpha, with_payload=True, score_threshold=score_threshold, return_debug=debug_hybrid)
    elif st == "lexical":
        hits = retriever.sparse_search(query=query.strip(), k=int(k), filters=filters, with_payload=True, score_threshold=score_threshold)
    else:
        hits = retriever.search(query=query.strip(), k=int(k), using=using, filters=filters, with_payload=True, score_threshold=score_threshold)
    timing_ms = (time.time() - t0) * 1000

    html_out = _render_hits_html(hits, include_yi=include_yi)
    query_id = str(uuid.uuid4())
    session_id = session_id or _make_session_id()
    # Build canonical QueryEvent object
    query_event = QueryEvent(
        query_id=query_id,
        session_id=session_id,
        input={
            "query": query.strip(),
            "search_type": st,
            "using": using,
            "k": k,
            "score_threshold": score_threshold,
            "filters": {
                "author_list": author_list,
                "subjects_list": subjects_list,
                "place_list": place_list,
                "publisher_list": publisher_list,
                "year_min": yf,
                "year_max": yt
            },
            "include_yi": include_yi
        },
        retrieval={
            "hits": [
                {
                    "id": h.get("id"),
                    "score": h.get("score"),
                    "rank": idx,
                    "dense_score": h.get("dense_score") or h.get("raw_dense_score"),
                    "sparse_score": h.get("sparse_score") or h.get("raw_sparse_score"),
                    "content": h.get("content"),
                    "tr_en_text": h.get("tr_en_text") or (h.get("payload") or {}).get("tr_en_text"),
                    "yi_text": h.get("yi_text") or (h.get("payload") or {}).get("yi_text"),
                    "book_id": h.get("book_id"),
                    "page": h.get("page"),
                    "year": h.get("year"),
                    "author": h.get("author"),
                    "title_en": h.get("title_en")
                } for idx, h in enumerate(hits)
            ],
            "timing_ms": timing_ms,
            "vector_name": using,
            "k": k,
            "filters": {
                "author_list": author_list,
                "subjects_list": subjects_list,
                "place_list": place_list,
                "publisher_list": publisher_list,
                "year_min": yf,
                "year_max": yt
            },
            "search_type": st,
            "alpha": alpha if st == "hybrid" else None,
            "score_threshold": score_threshold
        }
    )
    # Log to both observability platforms
    log_query_observability(query_event)
    params_dict = {
        "search_type": st, "using": using, "k": k, "score_threshold": score_threshold, "alpha": alpha,
        "filters": {"author_list": author_list, "subjects_list": subjects_list, "place_list": place_list,
                    "publisher_list": publisher_list, "year_min": yf, "year_max": yt},
        "include_yi": include_yi, "query": query.strip(), "query_id": query_id
    }
    return html_out, "", hits, params_dict, session_id

def on_generate(query: str, search_type: str, author: List[str], subjects: List[str], place: List[str], publisher: List[str],
                year_from: Optional[float], year_to: Optional[float], using: str, k: int, score_threshold: float, alpha: float,
                include_yi: bool, session_id: str) -> Tuple[str, str, List[Dict[str, Any]], Dict[str, Any], str]:
    t0 = time.time()
    if not retriever:
        return "", "<p><b>Backend unavailable:</b> configure QDRANT_URL / QDRANT_API_KEY.</p>", [], {}, session_id or _make_session_id()
    if not query or not query.strip():
        return "", "<em>Enter a query.</em>", [], {}, session_id or _make_session_id()
    yf = int(year_from) if year_from else None
    yt = int(year_to) if year_to else None
    author_list = [a for a in (author or []) if a]
    place_list = [p for p in (place or []) if p]
    publisher_list = [p for p in (publisher or []) if p]
    subjects_list: List[str] = []
    for s in (subjects or []):
        if not s: continue
        for token in [t.strip() for t in s.replace(";", ",").split(",") if t.strip()]:
            subjects_list.append(token)
    params = GenerateParams(
        using=using, k=int(k), include_yi=include_yi, score_threshold=score_threshold,
        search_type=(search_type or "semantic").lower(), alpha=alpha,
        filters=SearchFilters(author_list=author_list or None, subjects_list=subjects_list or None,
                              place_list=place_list or None, publisher_list=publisher_list or None,
                              year_min=yf, year_max=yt)
    )
    res = answer(query=query.strip(), params=params)
    total_timing_ms = (time.time() - t0) * 1000
    answer_md = res.get("answer", "")
    hits = res.get("hits", [])
    cits = _format_citations(res.get("citations", []), hits)
    if cits:
        answer_md += f"\n\n**Citations:** {cits}"
    html_out = _render_hits_html(hits, include_yi=include_yi)
    query_id = str(uuid.uuid4())
    session_id = session_id or _make_session_id()
    meta = res.get("meta", {})
    usage = res.get("usage", {})
    prompts = res.get("prompts", {})
    
    query_event = QueryEvent(
        query_id=query_id,
        session_id=session_id,
        input={
            "query": query.strip(),
            "search_type": search_type,
            "using": using,
            "k": k,
            "score_threshold": score_threshold,
            "filters": {
                "author_list": author_list,
                "subjects_list": subjects_list,
                "place_list": place_list,
                "publisher_list": publisher_list,
                "year_min": yf,
                "year_max": yt
            },
            "include_yi": include_yi
        },
        retrieval={
            "hits": [
                {
                    "id": h.get("id"),
                    "score": h.get("score"),
                    "rank": idx,
                    "dense_score": h.get("dense_score") or h.get("raw_dense_score"),
                    "sparse_score": h.get("sparse_score") or h.get("raw_sparse_score"),
                    "content": h.get("content"),
                    "tr_en_text": h.get("tr_en_text") or (h.get("payload") or {}).get("tr_en_text"),
                    "yi_text": h.get("yi_text") or (h.get("payload") or {}).get("yi_text"),
                    "book_id": h.get("book_id"),
                    "page": h.get("page"),
                    "year": h.get("year"),
                    "author": h.get("author"),
                    "title_en": h.get("title_en")
                } for idx, h in enumerate(hits)
            ],
            "timing_ms": total_timing_ms * 0.3,
            "vector_name": using,
            "k": k,
            "filters": {
                "author_list": author_list,
                "subjects_list": subjects_list,
                "place_list": place_list,
                "publisher_list": publisher_list,
                "year_min": yf,
                "year_max": yt
            },
            "search_type": search_type,
            "alpha": alpha if (search_type or '').lower() == "hybrid" else None,
            "score_threshold": score_threshold
        },
        generation={
            "model": meta.get("model", "unknown"),
            "answer": answer_md,
            "citations": res.get("citations", []),
            "timing_ms": total_timing_ms * 0.7,
            "usage": usage,
            "prompts": prompts
        }
    )

# Then log to both platforms:
    log_query_observability(query_event)
    params_dict = {
        "search_type": search_type, "using": using, "k": k, "score_threshold": score_threshold, "alpha": alpha,
        "filters": {"author_list": author_list, "subjects_list": subjects_list, "place_list": place_list,
                    "publisher_list": publisher_list, "year_min": yf, "year_max": yt},
        "include_yi": include_yi, "query": query.strip(), "query_id": query_id
    }
    return html_out, answer_md, hits, params_dict, session_id

# ---------------------------
# Feedback Handlers (Overall + Doc-level, no JS)
# ---------------------------

def on_feedback_overall(kind: str, annotation: str, session_id: str, last_query_params: Dict[str, Any], last_hits: List[Dict[str, Any]]):
    if not last_query_params:
        return gr.update(value="No query to give feedback on.")
    query_id = last_query_params.get("query_id")
    if not query_id:
        return gr.update(value="No query ID found for feedback.")
    feedback_id = str(uuid.uuid4())
    overall_feedback_event = FeedbackEvent(
        feedback_id=feedback_id,
        query_id=query_id,
        scope="overall",
        sentiment=kind,
        annotation=annotation,
        target=None
    )
    log_feedback_observability(overall_feedback_event)
    ann_text = f" (Note: {annotation})" if (annotation or "").strip() else ""
    return gr.update(value=f"Thanks for the overall feedback ({kind}){ann_text}.")

def on_feedback_overall_annotation(annotation: str, session_id: str, last_query_params: Dict[str, Any], last_hits: List[Dict[str, Any]]):
    if not last_query_params:
        return gr.update(value="No query to give feedback on."), gr.update(value="")
    if not (annotation or "").strip():
        return gr.update(value="Please enter an annotation before sending."), gr.update(value=annotation)
    query_id = last_query_params.get("query_id")
    if not query_id:
        return gr.update(value="No query ID found for feedback."), gr.update(value="")
    
    feedback_id = str(uuid.uuid4())
    overall_feedback_event = FeedbackEvent(
        feedback_id=feedback_id,
        query_id=query_id,
        scope="overall",
        sentiment="annotation_only",
        annotation=(annotation or "").strip()
    )
    log_feedback_observability(overall_feedback_event)

    return gr.update(value=f"✓ Overall annotation: {annotation[:50]}{'...' if len(annotation) > 50 else ''}"), gr.update(value="")

def on_doc_feedback(aspect: str, sentiment: str, doc_idx: int, annotation: str,
                    session_id: str, last_query_params: Dict[str, Any], last_hits: List[Dict[str, Any]]):
    """Doc-level feedback via plain buttons; logs with scope = doc_retrieval|doc_translation and tiny JSON-safe target."""
    if not last_query_params:
        return gr.update(value="No query to give feedback on.")
    query_id = last_query_params.get("query_id")
    if query_id is None:
        return gr.update(value="No query ID found for feedback.")
    if not last_hits:
        return gr.update(value="No hits to give feedback on.")
    if doc_idx is None or doc_idx < 0 or doc_idx >= len(last_hits):
        return gr.update(value=f"Invalid document index: {doc_idx}")

    feedback_id = str(uuid.uuid4())
    scope = "doc_retrieval" if aspect == "retrieval" else "doc_translation"
    target = {"doc_idx": int(doc_idx)}  # keep tiny & JSON-safe
    doc_feedback_event = FeedbackEvent(
        feedback_id=feedback_id,
        query_id=query_id,
        scope=scope,
        sentiment=sentiment,
        annotation=(annotation or "").strip(),
        target=target
    )
    log_feedback_observability(doc_feedback_event)
    
    pretty = "Retrieval" if aspect == "retrieval" else "Translation"
    if sentiment == "annotation_only":
        msg = f"✓ {pretty} annotation for doc {doc_idx}: {(annotation or '')[:50]}{'...' if annotation and len(annotation)>50 else ''}"
    else:
        note = f" (Note: {annotation.strip()})" if annotation and annotation.strip() else ""
        msg = f"✓ {pretty} feedback for doc {doc_idx}: {sentiment}{note}"
    return gr.update(value=msg)

def _doc_choices_from_hits(hits: List[Dict[str, Any]]):
    n = len(hits or [])
    return gr.update(choices=list(range(n)), value=(0 if n > 0 else None), visible=(n > 0))