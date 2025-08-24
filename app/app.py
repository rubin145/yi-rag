# app/app.py
from __future__ import annotations

import os
import sys

# ---- load .env file ----
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ .env file loaded")
except Exception as e:
    print(f"⚠️  Could not load .env file: {e}")

import uuid
import time
import html
from typing import Any, Dict, List, Optional, Tuple

# ---- ensure project root on path (HF Spaces runs from repo root) ----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gradio as gr

from core.retrieve import Retriever, SearchFilters
from core.generate import answer, GenerateParams

# Observability logging
from observability.logger import get_logger


# ---------------------------
# Helpers
# ---------------------------
def _make_session_id() -> str:
    return str(uuid.uuid4())


def _render_hits_html(hits: List[Dict[str, Any]], include_yi: bool = True) -> str:
    """
    Render results as semantic HTML accordions using <details><summary>.
    Works in gr.HTML and is lightweight vs. dynamic component creation.
    """
    if not hits:
        return "<p><em>No results.</em></p>"

    blocks: List[str] = []
    for i, h in enumerate(hits, start=1):
        score = f"{h.get('score', 0.0):.3f}"
        title_en = h.get("title_en") or ""
        title_yi = h.get("title_yi") or ""
        author = h.get("author") or ""
        year = h.get("year") or ""
        page = h.get("page")
        book_id = h.get("book_id")
        yi_text = h.get("yi_text") or ""
        tr_en = h.get("tr_en_text") or ""
        doc_idx = i - 1

        header = (
            f"<b>{html.escape(title_en or title_yi)}</b> — "
            f"{html.escape(str(author))} {f'({year})' if year else ''} "
            f"| p.{html.escape(str(page))} | <code>{html.escape(str(book_id))}</code> "
            f"| score {score}"
        )

        body_parts = []
        if tr_en:
            body_parts.append(f"<div><strong>EN</strong><br><pre>{html.escape(tr_en)}</pre></div>")
        if include_yi and yi_text:
            body_parts.append(f"<div style='margin-top:8px;'><strong>YI</strong><br><pre>{html.escape(yi_text)}</pre></div>")
        body = "\n".join(body_parts) if body_parts else "<em>(empty)</em>"

        # Embedded feedback UI with JavaScript handlers
        feedback_html = f"""
        <div style="margin-top:12px; padding:8px; background:#f8f9fa; border-radius:4px; border:1px solid #e9ecef;">
          <div style="font-weight:bold; margin-bottom:8px; font-size:14px;">📝 Feedback (Doc #{doc_idx})</div>
          
          <div style="margin-bottom:12px; padding:6px; background:white; border-radius:3px;">
            <div style="font-weight:500; margin-bottom:4px;">🔍 Retrieval Quality:</div>
            <div style="margin-bottom:6px;">
              <button onclick="submitFeedback({doc_idx}, 'retrieval', 'up')" style="margin-right:8px; background:#28a745; color:white; border:none; padding:4px 12px; border-radius:4px; cursor:pointer; font-size:12px;">👍 Relevant</button>
              <button onclick="submitFeedback({doc_idx}, 'retrieval', 'down')" style="background:#dc3545; color:white; border:none; padding:4px 12px; border-radius:4px; cursor:pointer; font-size:12px;">👎 Not relevant</button>
            </div>
            <div style="display:flex; gap:6px;">
              <textarea id="retrieval_annotation_{doc_idx}" placeholder="Why was this result relevant/irrelevant?" style="flex:1; height:50px; padding:4px; border:1px solid #ddd; border-radius:3px; font-size:11px; resize:vertical;"></textarea>
              <button onclick="submitAnnotation({doc_idx}, 'retrieval')" style="background:#007bff; color:white; border:none; padding:4px 8px; border-radius:3px; cursor:pointer; font-size:11px; white-space:nowrap;">Send Note</button>
            </div>
          </div>
          
          <div style="padding:6px; background:white; border-radius:3px;">
            <div style="font-weight:500; margin-bottom:4px;">🌍 Translation Quality:</div>
            <div style="margin-bottom:6px;">
              <button onclick="submitFeedback({doc_idx}, 'translation', 'up')" style="margin-right:8px; background:#28a745; color:white; border:none; padding:4px 12px; border-radius:4px; cursor:pointer; font-size:12px;">👍 Good</button>
              <button onclick="submitFeedback({doc_idx}, 'translation', 'down')" style="background:#dc3545; color:white; border:none; padding:4px 12px; border-radius:4px; cursor:pointer; font-size:12px;">👎 Poor</button>
            </div>
            <div style="display:flex; gap:6px;">
              <textarea id="translation_annotation_{doc_idx}" placeholder="What's wrong/good about this translation?" style="flex:1; height:50px; padding:4px; border:1px solid #ddd; border-radius:3px; font-size:11px; resize:vertical;"></textarea>
              <button onclick="submitAnnotation({doc_idx}, 'translation')" style="background:#007bff; color:white; border:none; padding:4px 8px; border-radius:3px; cursor:pointer; font-size:11px; white-space:nowrap;">Send Note</button>
            </div>
          </div>
        </div>
        """

        blocks.append(
            f"""
            <details>
              <summary>{header}</summary>
              <div style="padding:10px 0 6px 4px; border-left:3px solid #ddd; margin:8px 0 0 0;">
                {body}
                {feedback_html}
                <div style="margin-top:6px; font-size:12px; color:#777;">idx: {doc_idx}</div>
              </div>
            </details>
            """
        )

    return "\n".join(blocks)


def _format_citations(cits: List[Dict[str, Any]]) -> str:
    if not cits:
        return ""
    chips = [f"[{c['book_id']} p.{c['page']} #{c['chunk_idx']}]" for c in cits]
    return " ".join(chips)


# ---------------------------
# Global singletons (cheap) & defaults
# ---------------------------
retriever = Retriever(vectors_yaml="config/vectors.yaml")
DEFAULT_VECTOR = next((v for v in retriever.list_vectors() if "yi" in v), retriever.list_vectors()[0])


# ---------------------------
# Gradio Callbacks
# ---------------------------
# Shared state keys: session_id, last_query_id, last_hits, last_params

def on_search(query: str, using: str, k: int, author: str, year_from: Optional[float], year_to: Optional[float], include_yi: bool,
              session_id: str) -> Tuple[str, str, List[Dict[str, Any]], Dict[str, Any], str]:
    t0 = time.time()
    if not query or not query.strip():
        return gr.update(value="<p><em>Enter a query.</em></p>"), "", [], {}, session_id or _make_session_id()

    # filters
    yf = int(year_from) if year_from else None
    yt = int(year_to) if year_to else None
    filters = SearchFilters(
        authors=[author] if author else None,
        year_from=yf,
        year_to=yt,
    )

    hits = retriever.search(query=query.strip(), k=int(k), using=using, filters=filters, with_payload=True)
    timing_ms = (time.time() - t0) * 1000
    html_out = _render_hits_html(hits, include_yi=include_yi)

    query_id = str(uuid.uuid4())
    session_id = session_id or _make_session_id()

    # Log query event (retrieval only)
    logger = get_logger()
    logger.log_query_event(
        query_id=query_id,
        session_id=session_id,
        input_data={
            "query": query.strip(),
            "using": using,
            "k": k,
            "filters": {"author": author, "year_from": yf, "year_to": yt},
            "include_yi": include_yi
        },
        retrieval_data={
            "hits": [{"id": h.get("id"), "score": h.get("score"), "rank": i} for i, h in enumerate(hits)],
            "timing_ms": timing_ms,
            "vector_name": using,
            "k": k,
            "filters": {"author": author, "year_from": yf, "year_to": yt}
        }
    )

    return html_out, "", hits, {"using": using, "k": k, "filters": {"author": author, "year_from": yf, "year_to": yt}, "include_yi": include_yi, "query": query.strip(), "query_id": query_id}, session_id


def on_generate(query: str, using: str, k: int, author: str, year_from: Optional[float], year_to: Optional[float], include_yi: bool,
                session_id: str) -> Tuple[str, str, List[Dict[str, Any]], Dict[str, Any], str]:
    t0 = time.time()
    if not query or not query.strip():
        return "", "<em>Enter a query.</em>", [], {}, session_id or _make_session_id()

    yf = int(year_from) if year_from else None
    yt = int(year_to) if year_to else None
    params = GenerateParams(using=using, k=int(k), include_yi=include_yi,
                            filters=SearchFilters(authors=[author] if author else None, year_from=yf, year_to=yt))
    
    res = answer(query=query.strip(), params=params)
    total_timing_ms = (time.time() - t0) * 1000

    # answer markdown + citations chips
    answer_md = res.get("answer", "")
    cits = _format_citations(res.get("citations", []))
    if cits:
        answer_md += f"\n\n**Citations:** {cits}"

    hits = res.get("hits", [])
    html_out = _render_hits_html(hits, include_yi=include_yi)

    query_id = str(uuid.uuid4())
    session_id = session_id or _make_session_id()

    # Log query event (retrieval + generation)
    logger = get_logger()
    meta = res.get("meta", {})
    usage = res.get("usage", {})
    prompts = res.get("prompts", {})
    
    logger.log_query_event(
        query_id=query_id,
        session_id=session_id,
        input_data={
            "query": query.strip(),
            "using": using,
            "k": k,
            "filters": {"author": author, "year_from": yf, "year_to": yt},
            "include_yi": include_yi
        },
        retrieval_data={
            "hits": [{"id": h.get("id"), "score": h.get("score"), "rank": i} for i, h in enumerate(hits)],
            "timing_ms": total_timing_ms * 0.3,  # aproximado: 30% retrieval
            "vector_name": using,
            "k": k,
            "filters": {"author": author, "year_from": yf, "year_to": yt}
        },
        generation_data={
            "model": meta.get("model", "unknown"),
            "answer": answer_md,
            "citations": res.get("citations", []),
            "timing_ms": total_timing_ms * 0.7,  # aproximado: 70% generation
            "usage": usage,
            "prompts": prompts
        }
    )

    return html_out, answer_md, hits, {"using": using, "k": k, "filters": {"author": author, "year_from": yf, "year_to": yt}, "include_yi": include_yi, "query": query.strip(), "query_id": query_id}, session_id


def on_feedback_overall(kind: str, annotation: str, session_id: str, last_query_params: Dict[str, Any], last_hits: List[Dict[str, Any]]):
    # kind: "up" | "down"
    if not last_query_params:
        return gr.update(value="No query to give feedback on.")
    
    query_id = last_query_params.get("query_id")
    if not query_id:
        return gr.update(value="No query ID found for feedback.")
    
    # Log feedback event
    logger = get_logger()
    feedback_id = str(uuid.uuid4())
    
    logger.log_feedback_event(
        feedback_id=feedback_id,
        query_id=query_id,
        scope="overall",
        sentiment=kind,
        annotation=annotation
    )
    
    ann_text = f" (Note: {annotation})" if annotation.strip() else ""
    msg = f"Thanks for the overall feedback ({kind}){ann_text}."
    return gr.update(value=msg)


def on_feedback_overall_annotation(annotation: str, session_id: str, last_query_params: Dict[str, Any], last_hits: List[Dict[str, Any]]):
    # Handle annotation-only overall feedback
    if not last_query_params:
        return gr.update(value="No query to give feedback on."), gr.update(value="")
    if not annotation.strip():
        return gr.update(value="Please enter an annotation before sending."), gr.update(value=annotation)
    
    query_id = last_query_params.get("query_id")
    if not query_id:
        return gr.update(value="No query ID found for feedback."), gr.update(value="")
    
    # Log annotation-only feedback event
    logger = get_logger()
    feedback_id = str(uuid.uuid4())
    
    logger.log_feedback_event(
        feedback_id=feedback_id,
        query_id=query_id,
        scope="overall",
        sentiment="annotation_only",
        annotation=annotation
    )
    
    msg = f"✓ Overall annotation: {annotation[:50]}{'...' if len(annotation) > 50 else ''}"
    return gr.update(value=msg), gr.update(value="")


def on_feedback_doc_retrieval(kind: str, doc_idx: float, annotation: str, session_id: str, last_query_params: Dict[str, Any], last_hits: List[Dict[str, Any]]):
    if not last_hits:
        return gr.update(value="No results to give feedback on.")
    try:
        i = int(doc_idx)
    except Exception:
        return gr.update(value="Provide a valid doc index (see 'idx' in each accordion).")
    if i < 0 or i >= len(last_hits):
        return gr.update(value="Index out of range.")
    h = last_hits[i]
    # point-level id if present
    pid = h.get("id")
    # TODO: log_feedback_event({... 'scope':'doc_retrieval', 'target_id': pid, 'kind': kind, 'annotation': annotation})
    # This goes to observability (Langfuse/Phoenix/HF Datasets)
    ann_text = f" (Note: {annotation})" if annotation.strip() else ""
    return gr.update(value=f"Thanks for retrieval feedback on doc {i} ({kind}){ann_text}.")


def on_feedback_doc_translation(kind: str, doc_idx: float, annotation: str, session_id: str, last_query_params: Dict[str, Any], last_hits: List[Dict[str, Any]]):
    if not last_hits:
        return gr.update(value="No results to give feedback on.")
    try:
        i = int(doc_idx)
    except Exception:
        return gr.update(value="Provide a valid doc index (see 'idx' in each accordion).")
    if i < 0 or i >= len(last_hits):
        return gr.update(value="Index out of range.")
    h = last_hits[i]
    # point-level id if present
    pid = h.get("id")
    # TODO: Send to vector DB for translation quality tracking
    # TODO: update_translation_feedback_in_vectordb(pid, kind, annotation)
    ann_text = f" (Note: {annotation})" if annotation.strip() else ""
    return gr.update(value=f"Thanks for translation feedback on doc {i} ({kind}){ann_text}. Saved to vector DB.")


def on_embedded_feedback(feedback_json: str, session_id: str, last_query_params: Dict[str, Any], last_hits: List[Dict[str, Any]]):
    """Handle feedback from embedded JavaScript buttons"""
    if not feedback_json.strip():
        return gr.update(value="")
    
    try:
        import json
        data = json.loads(feedback_json)
        doc_idx = int(data["doc_idx"])
        feedback_type = data["feedback_type"]  # "retrieval" or "translation" 
        sentiment = data["sentiment"]  # "up" or "down" or "annotation_only"
        annotation = data["annotation"]
        
        if not last_hits or doc_idx < 0 or doc_idx >= len(last_hits):
            return gr.update(value="Invalid document index.")
        
        query_id = last_query_params.get("query_id")
        if not query_id:
            return gr.update(value="No query ID found for feedback.")
            
        h = last_hits[doc_idx]
        doc_id = h.get("id")
        
        # Log feedback event
        logger = get_logger()
        feedback_id = str(uuid.uuid4())
        
        scope = "doc_retrieval" if feedback_type == "retrieval" else "doc_translation"
        target = {"doc_idx": doc_idx, "doc_id": doc_id}
        
        logger.log_feedback_event(
            feedback_id=feedback_id,
            query_id=query_id,
            scope=scope,
            sentiment=sentiment,
            annotation=annotation,
            target=target
        )
        
        # Prepare response message
        if sentiment == "annotation_only":
            if feedback_type == "retrieval":
                message = f"✓ Retrieval annotation for doc {doc_idx}: {annotation[:50]}{'...' if len(annotation) > 50 else ''}"
            else:  # translation
                message = f"✓ Translation annotation for doc {doc_idx}: {annotation[:50]}{'...' if len(annotation) > 50 else ''} → Vector DB"
        elif feedback_type == "retrieval":
            ann_text = f" (Note: {annotation})" if annotation.strip() else ""
            message = f"✓ Retrieval feedback for doc {doc_idx}: {sentiment}{ann_text}"
        else:  # translation
            ann_text = f" (Note: {annotation})" if annotation.strip() else ""
            message = f"✓ Translation feedback for doc {doc_idx}: {sentiment}{ann_text} → Vector DB"
            
        return gr.update(value=message)
        
    except Exception as e:
        return gr.update(value=f"Error processing feedback: {str(e)}")


# ---------------------------
# UI Layout
# ---------------------------
css = """
:root { --radius: 12px; }
.details > summary { cursor: pointer; }
pre { white-space: pre-wrap; word-wrap: break-word; }
/* Slightly larger font for generated answers */
.markdown-answer { font-size: 16px; line-height: 1.5; }
"""

js_code = """
function submitFeedback(docIdx, feedbackType, sentiment) {
    // Get annotation text from the appropriate textarea
    const textareaId = feedbackType + '_annotation_' + docIdx;
    const annotation = document.getElementById(textareaId)?.value || '';
    
    // Create feedback data object
    const feedbackData = {
        doc_idx: docIdx,
        feedback_type: feedbackType,
        sentiment: sentiment,
        annotation: annotation
    };
    
    // Send to hidden Gradio component
    const hiddenTextbox = document.querySelector('#feedback-trigger textarea');
    if (hiddenTextbox) {
        hiddenTextbox.value = JSON.stringify(feedbackData);
        hiddenTextbox.dispatchEvent(new Event('input', { bubbles: true }));
    }
    
    // Visual feedback on button
    const button = event.target;
    const originalText = button.textContent;
    const originalBg = button.style.background;
    
    button.textContent = '✓ Sent';
    button.style.background = '#6c757d';
    button.disabled = true;
    
    setTimeout(() => {
        button.textContent = originalText;
        button.style.background = originalBg;
        button.disabled = false;
    }, 1500);
}

function submitAnnotation(docIdx, feedbackType) {
    // Get annotation text from the appropriate textarea
    const textareaId = feedbackType + '_annotation_' + docIdx;
    const textarea = document.getElementById(textareaId);
    const annotation = textarea?.value || '';
    
    if (!annotation.trim()) {
        alert('Please enter an annotation before sending.');
        return;
    }
    
    // Create annotation-only feedback data
    const feedbackData = {
        doc_idx: docIdx,
        feedback_type: feedbackType,
        sentiment: 'annotation_only',
        annotation: annotation
    };
    
    // Send to hidden Gradio component
    const hiddenTextbox = document.querySelector('#feedback-trigger textarea');
    if (hiddenTextbox) {
        hiddenTextbox.value = JSON.stringify(feedbackData);
        hiddenTextbox.dispatchEvent(new Event('input', { bubbles: true }));
    }
    
    // Visual feedback and clear textarea
    const button = event.target;
    const originalText = button.textContent;
    const originalBg = button.style.background;
    
    button.textContent = '✓ Sent';
    button.style.background = '#6c757d';
    button.disabled = true;
    
    // Clear textarea
    textarea.value = '';
    
    setTimeout(() => {
        button.textContent = originalText;
        button.style.background = originalBg;
        button.disabled = false;
    }, 1500);
}
"""

with gr.Blocks(css=css, title="Yiddish RAG — Research UI", head=f"<script>{js_code}</script>") as demo:
    gr.Markdown("""
    # Yiddish RAG — Research UI
    _Facilitates cross-lingual retrieval over Yiddish historical texts._
    """)
    
    # Hidden component for JavaScript-to-Python communication
    feedback_trigger = gr.Textbox(visible=False, elem_id="feedback-trigger")

    with gr.Row():
        query = gr.Textbox(label="Query", placeholder="e.g., Conflictos comunitarios en Varsovia", lines=3)
    with gr.Row():
        using = gr.Dropdown(label="Vector (using)", choices=retriever.list_vectors(), value=DEFAULT_VECTOR, multiselect=False)
        k = gr.Slider(1, 20, value=6, step=1, label="Top-k")
        include_yi = gr.Checkbox(value=False, label="Show original Yiddish")
    with gr.Row():
        author = gr.Textbox(label="Filter: Author (exact)", placeholder="e.g., Sholem Aleichem", scale=2)
        year_from = gr.Number(label="Year from", precision=0)
        year_to = gr.Number(label="Year to", precision=0)

    with gr.Row():
        btn_search = gr.Button("Search", variant="secondary")
        btn_generate = gr.Button("Generate Answer", variant="primary")

    with gr.Row():
        answer_md = gr.Markdown(label="Answer", elem_classes=["markdown-answer"])

    results_html = gr.HTML(label="Results", value="<em>Run a search to see results.</em>")

    with gr.Accordion("Overall Feedback", open=False):
        with gr.Row():
            fb_overall_text = gr.Markdown(visible=True)
        with gr.Row():
            fb_overall_annotation = gr.Textbox(label="Overall feedback annotation (optional)", placeholder="Additional comments...")
            fb_overall_annotation_submit = gr.Button("Send Note", variant="secondary", scale=0)
        with gr.Row():
            fb_overall_up = gr.Button("👍 Overall relevant")
            fb_overall_down = gr.Button("👎 Overall not relevant")
        
        gr.Markdown("### Document Feedback Status")
        with gr.Row():
            fb_embedded_status = gr.Markdown(visible=True, value="Use the feedback buttons within each document accordion above.")

    # hidden state
    session_id = gr.State(value=_make_session_id())
    last_query_id = gr.State(value="")
    last_hits = gr.State(value=[])
    last_params = gr.State(value={})

    # wiring
    btn_search.click(
        on_search,
        inputs=[query, using, k, author, year_from, year_to, include_yi, session_id],
        outputs=[results_html, answer_md, last_hits, last_params, session_id],
    )

    btn_generate.click(
        on_generate,
        inputs=[query, using, k, author, year_from, year_to, include_yi, session_id],
        outputs=[results_html, answer_md, last_hits, last_params, session_id],
    )

    # Overall feedback handlers
    fb_overall_up.click(on_feedback_overall, inputs=[gr.State("up"), fb_overall_annotation, session_id, last_params, last_hits], outputs=[fb_overall_text])
    fb_overall_down.click(on_feedback_overall, inputs=[gr.State("down"), fb_overall_annotation, session_id, last_params, last_hits], outputs=[fb_overall_text])
    fb_overall_annotation_submit.click(on_feedback_overall_annotation, inputs=[fb_overall_annotation, session_id, last_params, last_hits], outputs=[fb_overall_text, fb_overall_annotation])

    # Embedded document feedback handler (triggered by JavaScript)
    feedback_trigger.change(on_embedded_feedback, inputs=[feedback_trigger, session_id, last_params, last_hits], outputs=[fb_embedded_status])


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
