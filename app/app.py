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

# Add parent directory to path for imports when running from app/ directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.retrieve import Retriever
from core.store_qdrant import QdrantStore
from core.schemas import SearchFilters
from core.generate import answer, GenerateParams
from observability import PhoenixExporter, LangfuseLogger, QueryEvent
from helpers import _make_session_id, _render_hits_html, _format_citations, get_random_defaults
from callbacks import on_search, on_generate, on_feedback_overall, on_feedback_overall_annotation, on_doc_feedback, _doc_choices_from_hits
from model_downloader import ensure_sparse_model



# ---------------------------
# Global singletons
# ---------------------------
try:
    # Download sparse model if needed (for HF Spaces deployment)
    ensure_sparse_model()
    
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
