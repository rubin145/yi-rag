# Clean rebuilt app copy with corrected feedback UI and handlers (NO JS; doc buttons = smaller)
from __future__ import annotations
import os, sys, time, uuid, html, random
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import gradio as gr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.retrieve import Retriever
from core.store_qdrant import QdrantStore
from core.schemas import SearchFilters
from core.generate import answer, GenerateParams
from observability.logger import get_logger

# ---------------------------
# Helpers
# ---------------------------

def _make_session_id() -> str:
    return str(uuid.uuid4())

def _render_hits_html(hits: List[Dict[str, Any]], include_yi: bool = True) -> str:
    """Render search hits as accordions; feedback panels are NOT inline (buttons live in the Feedback section)."""
    if not hits:
        return "<p><em>No results.</em></p>"

    out: List[str] = []
    for i, h in enumerate(hits):
        doc_idx = i
        score = h.get("score")
        score_fmt = f"{score:.3f}" if isinstance(score, (int, float)) else "0.000"
        title_en = h.get("title_en") or ""
        title_yi = h.get("title_yi") or ""
        title_lat = h.get("title_yi_latin") or ""
        author = h.get("author") or ""
        year = h.get("year") or ""
        page = h.get("page")
        book_id = h.get("book_id") or ""
        yi_text = h.get("yi_text") or ""
        tr_en = h.get("tr_en_text") or ""
        ocr_source_url = h.get("ocr_source_url")

        header_title = title_lat or title_yi or title_en or f"Doc {i+1}"
        # Build OCR URL if available
        ocr_link = ""
        if ocr_source_url and page:
            page_ocr_url = ocr_source_url[:-1] + str(page)
            ocr_link = f" | <a href='{page_ocr_url}' target='_blank'>OCR</a>"
        elif book_id and page:
            page_ocr_url = f"https://ocr.yiddishbookcenter.org/text/{book_id}/page/{page}"
            ocr_link = f" | <a href='{page_ocr_url}' target='_blank'>OCR</a>"
        
        header = (
            f"<span style='font-size:18px; font-weight:bold; color:#2563eb;'>{doc_idx}</span> | "
            f"<b>{html.escape(header_title)}</b> {html.escape(author)} {f'({year})' if year else ''} | "
            f"[{html.escape(book_id)}] | p.{page} | score {score_fmt}{ocr_link}"
            if page is not None and book_id
            else f"<span style='font-size:18px; font-weight:bold; color:#2563eb;'>{doc_idx}</span> | {html.escape(header_title)}"
        )

        # Title section with Hebrew and English titles
        title_section = ""
        if title_yi or title_en:
            title_section = f"""
<div style='background:#f8f9fa; padding:8px; margin:4px 0; border-radius:4px; border:1px solid #e9ecef;'>
{f"<div style='font-size:18px; text-align:left; margin-bottom:4px;'>{html.escape(title_yi)}</div>" if title_yi else ""}
{f"<div style='font-size:18px; color:#666; text-align:left;'>{html.escape(title_en)}</div>" if title_en else ""}
</div>
"""
        
        body_parts: List[str] = []
        if title_section:
            body_parts.append(title_section)
        if tr_en:
            body_parts.append(f"<div style='margin-top:4px;'><div style='font-size:11px; color:#e67e22; margin-bottom:2px;'>⚠️ AI generated translation. Need to check with expert.</div><strong>EN</strong><br><pre>{html.escape(tr_en)}</pre></div>")
        if include_yi and yi_text:
            body_parts.append(f"<div style='margin-top:6px;'><strong>YI</strong><br><pre>{html.escape(yi_text)}</pre></div>")
        body = "".join(body_parts) if body_parts else "<em>(empty)</em>"

        out.append(
            f"""
<details>
  <summary>{header}</summary>
  <div style='padding:6px 0 4px 4px; border-left:3px solid #ddd; margin:4px 0 0 0;'>
    {body}
    <div style='margin-top:6px; font-size:11px; color:#666;'>Doc index: <code>{doc_idx}</code></div>
  </div>
</details>
"""
        )
    return "\n".join(out)

def _format_citations(cits: List[Dict[str, Any]], hits: List[Dict[str, Any]] = None) -> str:
    if not cits:
        return ""
    hit_lookup = {}
    if hits:
        for h in hits:
            key = f"{h.get('book_id')}_{h.get('page')}_{h.get('chunk_idx')}"
            hit_lookup[key] = h
    chips: List[str] = []
    for c in cits:
        page = c.get('page', '')
        chunk_idx = c.get('chunk_idx', '')
        book_id = c.get('book_id', '')
        hit_key = f"{book_id}_{page}_{chunk_idx}"
        hit_data = hit_lookup.get(hit_key, {})
        ocr_source_url = c.get('ocr_source_url') or hit_data.get('ocr_source_url')
        title_en = c.get('title_en') or hit_data.get('title_en', '')
        title_yi = c.get('title_yi') or hit_data.get('title_yi', '')
        author = c.get('author') or hit_data.get('author', '')
        year = c.get('year') or hit_data.get('year', '')
        if not ocr_source_url and book_id:
            ocr_source_url = f"https://ocr.yiddishbookcenter.org/text/{book_id}/page/1"
        title_display = title_en or title_yi or book_id
        author_year = f"{author} ({year})" if author and year else (author or str(year) if year else "")
        citation_text = f"[{book_id}] {title_display} — {author_year} p.{page} #{chunk_idx}".strip(" — ")
        if ocr_source_url and page:
            page_ocr_url = ocr_source_url[:-1] + str(page)
            chips.append(f"[<a href='{page_ocr_url}' target='_blank'>{html.escape(citation_text)}</a>]")
        else:
            chips.append(f"[{citation_text}]")
    return " ".join(chips)

# ---------------------------
# Global singletons
# ---------------------------
try:
    retriever = Retriever(vectors_yaml="config/vectors.yaml")
    try:
        QdrantStore(retriever.collection).ensure_payload_indexes()
    except Exception:
        pass
    available_vectors = retriever.list_vectors()
    DEFAULT_VECTOR = "dense_yi_cohere_1536" if "dense_yi_cohere_1536" in available_vectors else (available_vectors[0] if available_vectors else "")
    BACKEND_OK = True
except Exception as _e:  # Qdrant / env misconfig → degrade UI instead of crash
    retriever = None
    available_vectors = []
    DEFAULT_VECTOR = ""
    BACKEND_OK = False
    _BACKEND_ERR = str(_e)

def get_random_defaults():
    raw_alpha = random.uniform(0.4, 0.6)
    alpha_q = round(raw_alpha / 0.05) * 0.05
    return {"k": random.randint(8, 12), "score_threshold": round(random.uniform(0.1, 0.4), 2), "alpha": max(0.0, min(1.0, alpha_q))}

def get_filter_options(field_name: str) -> List[str]:
    if not retriever:
        return []
    try:
        return retriever.get_distinct_field_values(field_name)
    except Exception:
        return []

def randomize_settings():
    d = get_random_defaults()
    return gr.update(value=d["k"]), gr.update(value=d["score_threshold"]), gr.update(value=d["alpha"])

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
    logger = get_logger()
    logger.log_query_event(
        query_id=query_id,
        session_id=session_id,
        input_data={"query": query.strip(), "search_type": st, "using": using, "k": k, "score_threshold": score_threshold,
                    "filters": {"author_list": author_list, "subjects_list": subjects_list, "place_list": place_list,
                                "publisher_list": publisher_list, "year_min": yf, "year_max": yt},
                    "include_yi": include_yi},
        retrieval_data={
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
            "filters": {"author_list": author_list, "subjects_list": subjects_list, "place_list": place_list,
                        "publisher_list": publisher_list, "year_min": yf, "year_max": yt},
            "search_type": st,
            "alpha": alpha if st == "hybrid" else None,
            "score_threshold": score_threshold,
        }
    )
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
    logger = get_logger()
    meta = res.get("meta", {})
    usage = res.get("usage", {})
    prompts = res.get("prompts", {})
    logger.log_query_event(
        query_id=query_id,
        session_id=session_id,
        input_data={"query": query.strip(), "search_type": search_type, "using": using, "k": k, "score_threshold": score_threshold,
                    "filters": {"author_list": author_list, "subjects_list": subjects_list, "place_list": place_list,
                                "publisher_list": publisher_list, "year_min": yf, "year_max": yt},
                    "include_yi": include_yi},
        retrieval_data={
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
            "filters": {"author_list": author_list, "subjects_list": subjects_list, "place_list": place_list,
                        "publisher_list": publisher_list, "year_min": yf, "year_max": yt},
            "search_type": search_type,
            "alpha": alpha if (search_type or '').lower() == "hybrid" else None,
            "score_threshold": score_threshold,
        },
        generation_data={
            "model": meta.get("model", "unknown"),
            "answer": answer_md,
            "citations": res.get("citations", []),
            "timing_ms": total_timing_ms * 0.7,
            "usage": usage,
            "prompts": prompts,
        }
    )
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
    logger = get_logger()
    feedback_id = str(uuid.uuid4())
    logger.log_feedback_event(
        feedback_id=feedback_id,
        query_id=query_id,
        scope="overall",
        sentiment=kind,
        annotation=annotation
    )
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
    logger = get_logger()
    feedback_id = str(uuid.uuid4())
    logger.log_feedback_event(
        feedback_id=feedback_id,
        query_id=query_id,
        scope="overall",
        sentiment="annotation_only",
        annotation=annotation
    )
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

    logger = get_logger()
    feedback_id = str(uuid.uuid4())
    scope = "doc_retrieval" if aspect == "retrieval" else "doc_translation"
    target = {"doc_idx": int(doc_idx)}  # keep tiny & JSON-safe

    logger.log_feedback_event(
        feedback_id=feedback_id,
        query_id=query_id,
        scope=scope,
        sentiment=sentiment,             # "up" | "down" | "annotation_only"
        annotation=(annotation or "").strip(),
        target=target
    )

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

# ---------------------------
# UI
# ---------------------------

css = """
:root { --radius: 12px; }
.details > summary { cursor: pointer; }
pre { white-space: pre-wrap; word-wrap: break-word; }
.markdown-answer { font-size:16px; line-height:1.5;}
.results-container { min-height:10px;}
.answer-container { min-height:10px;}
/* smaller buttons for doc feedback */227383737  
.doc-btn button { font-size: 12px !important; padding: 3px 8px !important; }
"""

SPARSE_AVAILABLE = os.path.isdir(os.path.join(ROOT, "data", "sparse_model"))
search_type_choices = [("Semantic","semantic")]
if SPARSE_AVAILABLE:
    search_type_choices += [("Lexical","lexical"),("Hybrid","hybrid")]

with gr.Blocks(css=css, title="Yiddish RAG") as demo:
    if not BACKEND_OK:
        gr.Markdown(f"⚠️ **Backend not initialized**: {_BACKEND_ERR}. The UI is in read-only mode. Set required environment variables in your Hugging Face Space (QDRANT_URL, QDRANT_API_KEY, etc.) and restart.")
    gr.Markdown("# Yiddish Semantic Search\nAuxiliary retrieval tool for cross-lingual research over Yiddish historical texts.")

    # Query controls
    with gr.Row():
        query = gr.Textbox(label="Query", placeholder="e.g., Conflictos comunitarios en Varsovia", lines=3, scale=4)
    search_type = gr.Dropdown(label="Search Type",
                  choices=search_type_choices,
                  value="semantic", interactive=True, scale=1)
    with gr.Row():
        author = gr.Dropdown(label="Author(s)", choices=get_filter_options("author"),
                             multiselect=True, allow_custom_value=True, value=[], scale=1)
        subjects = gr.Dropdown(label="Subject(s)", choices=get_filter_options("subjects"),
                               multiselect=True, allow_custom_value=True, value=[], scale=1)
        place = gr.Dropdown(label="Place(s)", choices=get_filter_options("place"),
                            multiselect=True, allow_custom_value=True, value=[], scale=1)
        publisher = gr.Dropdown(label="Publisher(s)", choices=get_filter_options("publisher"),
                                multiselect=True, allow_custom_value=True, value=[], scale=1)
    with gr.Row():
        year_from = gr.Number(label="Year from", precision=0, scale=1)
        year_to = gr.Number(label="Year to", precision=0, scale=1)
        include_yi = gr.Checkbox(value=True, label="Show original Yiddish", scale=1)
    with gr.Accordion("Advanced Settings", open=False):
        with gr.Row():
            using = gr.Dropdown(label="Vector Model", choices=retriever.list_vectors(),
                                value=DEFAULT_VECTOR, multiselect=False)
            k = gr.Slider(5, 20, value=10, step=1, label="Results Count (Top-k)")
            score_threshold = gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="Score Threshold")
            alpha = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Hybrid Alpha (dense weight)")
    with gr.Row():
        btn_search = gr.Button("Search", variant="secondary")
        btn_generate = gr.Button("Generate Answer", variant="primary")

    # Answer + Results
    with gr.Row():
        answer_md = gr.Markdown(label="Answer", elem_classes=["markdown-answer","answer-container"], height=150)
    results_html = gr.HTML(label="Results", value="<em>Run a search to see results.</em>", elem_classes=["results-container"])

    # -------- Feedback (Overall + Doc) --------
    gr.Markdown("## Feedback")

    # Overall (outside accordion, as requested)
    with gr.Row():
        fb_overall_text = gr.Markdown(visible=True)
    with gr.Row():
        fb_overall_annotation = gr.Textbox(label="Overall feedback annotation (optional)", placeholder="Additional comments...", scale=3)
        fb_overall_annotation_submit = gr.Button("Send Overall Note", variant="secondary", scale=1)
        fb_overall_up = gr.Button("👍 Overall relevant", variant="primary", scale=0)
        fb_overall_down = gr.Button("👎 Overall not relevant", variant="secondary", scale=0)

    # Doc-level (smaller buttons, pure Gradio)
    with gr.Accordion("Document Feedback", open=True):
        gr.Markdown("Pick a document index and send retrieval/translation feedback. Same sink as overall.")
        with gr.Row():
            fb_doc_idx = gr.Dropdown(label="Doc index", choices=[], value=None, interactive=True, scale=1)
        gr.Markdown("**Retrieval**")
        with gr.Row():
            fb_doc_retrieval_note = gr.Textbox(label="Retrieval note (optional)", placeholder="Short comment…", scale=3)
            fb_doc_retrieval_up = gr.Button("👍 Retrieval relevant", elem_classes=["doc-btn"], scale=0)
            fb_doc_retrieval_down = gr.Button("👎 Retrieval not relevant", elem_classes=["doc-btn"], scale=0)
            fb_doc_retrieval_send_note = gr.Button("Send Retrieval Note", variant="secondary", elem_classes=["doc-btn"], scale=0)
        gr.Markdown("**Translation**")
        with gr.Row():
            fb_doc_translation_note = gr.Textbox(label="Translation note (optional)", placeholder="Short comment…", scale=3)
            fb_doc_translation_up = gr.Button("👍 Translation good", elem_classes=["doc-btn"], scale=0)
            fb_doc_translation_down = gr.Button("👎 Translation poor", elem_classes=["doc-btn"], scale=0)
            fb_doc_translation_send_note = gr.Button("Send Translation Note", variant="secondary", elem_classes=["doc-btn"], scale=0)
        fb_doc_status = gr.Markdown(visible=True, value="")

    # State
    session_id = gr.State(value=_make_session_id())
    last_hits = gr.State(value=[])
    last_params = gr.State(value={})

    # Wiring
    btn_search.click(
        on_search,
        inputs=[query, search_type, author, subjects, place, publisher, year_from, year_to, using, k, score_threshold, alpha, include_yi, session_id],
        outputs=[results_html, answer_md, last_hits, last_params, session_id]
    ).then(
        _doc_choices_from_hits, inputs=[last_hits], outputs=[fb_doc_idx]
    )

    btn_generate.click(
        on_generate,
        inputs=[query, search_type, author, subjects, place, publisher, year_from, year_to, using, k, score_threshold, alpha, include_yi, session_id],
        outputs=[results_html, answer_md, last_hits, last_params, session_id]
    ).then(
        _doc_choices_from_hits, inputs=[last_hits], outputs=[fb_doc_idx]
    )

    # Overall feedback wiring
    fb_overall_up.click(on_feedback_overall, inputs=[gr.State("up"), fb_overall_annotation, session_id, last_params, last_hits], outputs=[fb_overall_text])
    fb_overall_down.click(on_feedback_overall, inputs=[gr.State("down"), fb_overall_annotation, session_id, last_params, last_hits], outputs=[fb_overall_text])
    fb_overall_annotation_submit.click(on_feedback_overall_annotation, inputs=[fb_overall_annotation, session_id, last_params, last_hits], outputs=[fb_overall_text, fb_overall_annotation])

    # Doc feedback wiring
    fb_doc_retrieval_up.click(on_doc_feedback, inputs=[gr.State("retrieval"), gr.State("up"), fb_doc_idx, fb_doc_retrieval_note, session_id, last_params, last_hits], outputs=[fb_doc_status])
    fb_doc_retrieval_down.click(on_doc_feedback, inputs=[gr.State("retrieval"), gr.State("down"), fb_doc_idx, fb_doc_retrieval_note, session_id, last_params, last_hits], outputs=[fb_doc_status])
    fb_doc_retrieval_send_note.click(on_doc_feedback, inputs=[gr.State("retrieval"), gr.State("annotation_only"), fb_doc_idx, fb_doc_retrieval_note, session_id, last_params, last_hits], outputs=[fb_doc_status])

    fb_doc_translation_up.click(on_doc_feedback, inputs=[gr.State("translation"), gr.State("up"), fb_doc_idx, fb_doc_translation_note, session_id, last_params, last_hits], outputs=[fb_doc_status])
    fb_doc_translation_down.click(on_doc_feedback, inputs=[gr.State("translation"), gr.State("down"), fb_doc_idx, fb_doc_translation_note, session_id, last_params, last_hits], outputs=[fb_doc_status])
    fb_doc_translation_send_note.click(on_doc_feedback, inputs=[gr.State("translation"), gr.State("annotation_only"), fb_doc_idx, fb_doc_translation_note, session_id, last_params, last_hits], outputs=[fb_doc_status])

    # Randomize some defaults at load
    demo.load(randomize_settings, outputs=[k, score_threshold, alpha])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
