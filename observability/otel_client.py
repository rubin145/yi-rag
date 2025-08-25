# observability/otel_client.py
from __future__ import annotations

import os
from typing import Optional

from .schemas import QueryEvent, FeedbackEvent
from .multi_exporter import get_multi_exporter
from .langfuse_scores import get_langfuse_scorer

# Load environment variables early
try:  # pragma: no cover
    from dotenv import load_dotenv
    from pathlib import Path
    # Explicitly find the .env file in the project root, which is 2 levels up
    # from this file's location (observability/otel_client.py)
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.is_file():
        load_dotenv(dotenv_path=env_path)
    else:
        # Fallback for cases where the .env is in the CWD
        if Path('.env').is_file():
            load_dotenv()
        else:
            print("ℹ️ .env file not found in project root or CWD.")
except Exception as e:
    print(f"⚠️  Error loading .env file: {e}")

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode, set_span_in_context
    from openinference.semconv.trace import SpanAttributes as OpenInferenceSpanAttributes, OpenInferenceSpanKindValues
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None


class OpenTelemetryClient:
    """
    Simplified OpenTelemetry client using the new MultiExporterConfig.
    
    This client now delegates all configuration to MultiExporterConfig
    and focuses only on logging events with proper OpenInference conventions.
    """
    
    def __init__(self, service_name: str = "yiddish-rag"):
        self.enabled = False
        self.tracer = None
        # Map query_id -> span & trace context for later feedback attachment
        # Structure: { query_id: {"trace_id": int, "root_span_id": int, "retrieval_span_id": int} }
        self._query_trace_map = {}
        self._scorer = get_langfuse_scorer()

        if not OTEL_AVAILABLE:
            print("⚠️  OpenTelemetry not installed. Skipping OTEL logging.")
            return

        try:
            multi_exporter = get_multi_exporter()
            if multi_exporter.enabled:
                self.tracer = multi_exporter.get_tracer()
                self.enabled = True
                print("✅ OpenTelemetry client initialized with multi-exporter")
            else:
                print("❌ Multi-exporter not enabled, OpenTelemetry client disabled")
        except Exception as e:
            print(f"❌ Failed to initialize OpenTelemetry client: {e}")
    
    def log_query_event(self, event: QueryEvent) -> None:
        """Log a RAG query event using OpenInference semantic conventions."""
        if not self.enabled or not self.tracer:
            return
        
        try:
            # Create root (query) span as current so children share its context/trace
            root_span = self.tracer.start_span(
                name="rag.query",
                kind=trace.SpanKind.SERVER,
                attributes={
                    OpenInferenceSpanAttributes.OPENINFERENCE_SPAN_KIND: "CHAIN",
                    OpenInferenceSpanAttributes.INPUT_VALUE: event.input.get("query", ""),
                    OpenInferenceSpanAttributes.OUTPUT_VALUE: event.generation.answer if event.generation else "No generation",
                    "session.id": event.session_id,
                    "query.id": event.query_id,
                    "correlation.query_id": event.query_id,
                    "rag.query": event.input.get("query", ""),
                    "rag.k": event.input.get("k", 0),
                    "rag.using": event.input.get("using", "unknown"),
                },
            )
            root_ctx = set_span_in_context(root_span)
            # Child retrieval span under query span
            retrieval_span = self.tracer.start_span(
                name="rag.retrieval",
                kind=trace.SpanKind.INTERNAL,
                context=root_ctx,
                attributes={
                    OpenInferenceSpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.RETRIEVER.value,
                    OpenInferenceSpanAttributes.INPUT_VALUE: event.input.get("query", ""),
                    "query.id": event.query_id,
                    "correlation.query_id": event.query_id,
                    "retrieval.vector_name": event.retrieval.vector_name,
                    "retrieval.k": event.retrieval.k,
                    "retrieval.timing_ms": event.retrieval.timing_ms,
                    "retrieval.hits_count": len(event.retrieval.hits),
                    "retrieval.search_type": event.retrieval.search_type or event.input.get("search_type"),
                    "retrieval.alpha": event.retrieval.alpha if event.retrieval.alpha is not None else -1.0,
                    "retrieval.score_threshold": event.retrieval.score_threshold if event.retrieval.score_threshold is not None else -1.0,
                    "retrieval.filters.active": int(bool(event.retrieval.filters)),
                },
            )

            # Populate retrieval span document attributes
            filters = event.retrieval.filters or {}
            for fk, fv in (filters or {}).items():
                try:
                    if fv is None:
                        continue
                    if isinstance(fv, (list, tuple, set)):
                        retrieval_span.set_attribute(f"retrieval.filter.{fk}.count", len(fv))
                        preview_vals = list(fv)[:10]
                        retrieval_span.set_attribute(f"retrieval.filter.{fk}.values", ",".join([str(x) for x in preview_vals]))
                    else:
                        retrieval_span.set_attribute(f"retrieval.filter.{fk}.value", str(fv))
                except Exception:
                    pass
            try:
                snippet_chars = int(os.getenv("RETRIEVAL_DOC_SNIPPET_CHARS", "220"))
            except ValueError:
                snippet_chars = 220
            for i, hit in enumerate(event.retrieval.hits):
                content = (
                    hit.get("tr_en_text") or hit.get("en_text") or hit.get("yi_text") or hit.get("content") or hit.get("text") or hit.get("chunk") or ""
                )
                full_len = len(content)
                if content and full_len > snippet_chars:
                    content = content[:snippet_chars] + "…"
                retrieval_span.set_attribute(f"document.{i}.id", str(hit.get("id", f"doc_{i}")))
                retrieval_span.set_attribute(f"document.{i}.content", content)
                retrieval_span.set_attribute(f"document.{i}.content_length", full_len)
                try:
                    base_score = hit.get("score", 0.0) or 0.0
                    retrieval_span.set_attribute(f"document.{i}.score", float(base_score))
                except Exception:
                    retrieval_span.set_attribute(f"document.{i}.score", 0.0)
                retrieval_span.set_attribute(f"document.{i}.rank", i)
                # Metadata extraction (configurable + safe defaults)
                try:
                    default_keys = (
                        "title,title_en,title_yi,source,year,file_name,book_id,doc_id,chunk_start,chunk_end,language,"
                        "author,author_yi,page,publisher,place,subjects,subject_list,author_list"
                    )
                    allowed_keys_env = os.getenv("RETRIEVAL_META_KEYS", default_keys)
                    allowed_meta_keys = [k.strip() for k in allowed_keys_env.split(",") if k.strip()]
                    meta_count = 0
                    # Auto include unified title alias if present
                    title_val = (
                        hit.get("title")
                        or hit.get("title_en")
                        or hit.get("title_yi")
                    )
                    if title_val:
                        sval = str(title_val)
                        if len(sval) > 300:
                            sval = sval[:300] + "…"
                        retrieval_span.set_attribute(f"document.{i}.metadata.title_unified", sval)
                    for mk in allowed_meta_keys:
                        if mk in hit and hit[mk] is not None:
                            val = hit[mk]
                            # Handle list-like subjects/author_list gracefully
                            if isinstance(val, (list, tuple, set)):
                                seq = list(val)
                                retrieval_span.set_attribute(f"document.{i}.metadata.{mk}.count", len(seq))
                                preview = seq[:5]
                                sval = ",".join([str(x) for x in preview])
                            else:
                                sval = str(val)
                            if len(sval) > 300:
                                sval = sval[:300] + "…"
                            retrieval_span.set_attribute(f"document.{i}.metadata.{mk}", sval)
                            meta_count += 1
                    retrieval_span.set_attribute(f"document.{i}.metadata.keys_count", meta_count)
                except Exception:
                    pass
            retrieval_span.set_attribute(
                OpenInferenceSpanAttributes.OUTPUT_VALUE,
                "Retrieved {} documents".format(len(event.retrieval.hits)),
            )
            retrieval_span.set_status(Status(StatusCode.OK))
            root_span.set_status(Status(StatusCode.OK))

            # Store span contexts for future feedback linkage
            try:
                root_ctx = root_span.get_span_context()
                ret_ctx = retrieval_span.get_span_context()
                self._query_trace_map[event.query_id] = {
                    "trace_id": root_ctx.trace_id,
                    "root_span_id": root_ctx.span_id,
                    "retrieval_span_id": ret_ctx.span_id,
                }
                # Pre-create placeholder Langfuse scores
                if getattr(self._scorer, "enabled", False):
                    trace_hex = f"{root_ctx.trace_id:032x}"
                    self._scorer.create_placeholder_scores(event.query_id, trace_hex, event.retrieval.hits)
            except Exception:
                pass

            # Generation span (child of root span) - ends immediately
            if event.generation:
                gen_attributes = {
                    OpenInferenceSpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                    OpenInferenceSpanAttributes.LLM_MODEL_NAME: event.generation.model,
                    OpenInferenceSpanAttributes.INPUT_VALUE: event.input.get("query", ""),
                    OpenInferenceSpanAttributes.OUTPUT_VALUE: event.generation.answer,
                    "generation.timing_ms": event.generation.timing_ms,
                    "generation.citations_count": len(event.generation.citations),
                }
                gen_span = self.tracer.start_span("child",
                    name="rag.generation", context=root_ctx, attributes=gen_attributes
                )
                try:
                    gen_span.set_attribute("query.id", event.query_id)
                    gen_span.set_attribute("correlation.query_id", event.query_id)
                except Exception:
                    pass
                gen_span.end()
            # Close retrieval then root (Option A: no open spans retained)
            retrieval_span.end()
            root_span.end()
                
        except Exception as e:
            print(f"❌ Error in OpenTelemetry query event logging: {e}")
            import traceback
            traceback.print_exc()
    
def log_feedback_event(self, event: FeedbackEvent) -> None:
    """Log feedback as an independent span, parenting to original root or retrieval span via stored SpanContext."""
    if not self.enabled or not self.tracer:
        return

    try:
        parent_ctx = None
        mapping = self._query_trace_map.get(event.query_id)
        chosen_parent_span_id = None
        root_span_id = None
        retrieval_span_id = None
        trace_id_val = None
        if mapping:
            trace_id_val = mapping.get("trace_id")
            root_span_id = mapping.get("root_span_id")
            retrieval_span_id = mapping.get("retrieval_span_id")
            chosen_parent_span_id = root_span_id
            if chosen_parent_span_id and trace_id_val:
                try:
                    from opentelemetry.trace import SpanContext, TraceFlags, TraceState, NonRecordingSpan, set_span_in_context
                    sc = SpanContext(
                        trace_id=trace_id_val,
                        span_id=chosen_parent_span_id,
                        is_remote=False,
                        trace_flags=TraceFlags(TraceFlags.SAMPLED),
                        trace_state=TraceState()
                    )
                    parent_ctx = set_span_in_context(NonRecordingSpan(sc))
                except Exception:
                    parent_ctx = None

        # --- normalize scope for scoring & add aspect
        if event.scope in ("doc_retrieval", "doc_translation"):
            scope_mapped = "doc"
            aspect = "retrieval" if event.scope == "doc_retrieval" else "translation"
        else:
            scope_mapped = "overall"
            aspect = None

        # --- sanitize/flatten target
        safe_target = None
        doc_idx_val = None
        doc_id_val = None
        if isinstance(event.target, dict):
            raw_idx = event.target.get("doc_idx")
            if raw_idx is not None:
                try:
                    doc_idx_val = int(raw_idx)
                except Exception:
                    # last resort: keep as string
                    doc_idx_val = int(str(raw_idx)) if str(raw_idx).isdigit() else str(raw_idx)
            raw_id = event.target.get("doc_id") or event.target.get("id")
            if raw_id is not None:
                doc_id_val = str(raw_id)
            safe_target = {}
            if doc_idx_val is not None:
                safe_target["doc_idx"] = doc_idx_val
            if doc_id_val is not None:
                safe_target["doc_id"] = doc_id_val

        # Span attributes (unchanged + a couple of extras for visibility)
        attributes = {
            "feedback.id": event.feedback_id,
            "feedback.query_id": event.query_id,
            "query.id": event.query_id,
            "correlation.query_id": event.query_id,
            "feedback.scope": event.scope,                # original scope
            "feedback.scope_mapped": scope_mapped,        # "overall" | "doc"
            "feedback.aspect": aspect or "",
            "feedback.sentiment": event.sentiment,
            "feedback.annotation": event.annotation or "",
            "feedback.trace_id": f"{trace_id_val}" if trace_id_val is not None else "",
            "feedback.parent_span_id": f"{chosen_parent_span_id}" if chosen_parent_span_id is not None else "",
            "feedback.root_span_id": f"{root_span_id}" if root_span_id is not None else "",
            "feedback.retrieval_span_id": f"{retrieval_span_id}" if retrieval_span_id is not None else "",
        }
        if doc_idx_val is not None:
            attributes["feedback.target.doc_idx"] = doc_idx_val
        if doc_id_val is not None:
            attributes["feedback.target.doc_id"] = doc_id_val

        from opentelemetry import trace
        with self.tracer.start_as_current_span(
            "rag.feedback", context=parent_ctx, kind=trace.SpanKind.INTERNAL, attributes=attributes
        ) as fb_span:
            try:
                from opentelemetry.trace import Status, StatusCode  # type: ignore
                fb_span.set_status(Status(StatusCode.OK))
            except Exception:
                pass

        # --- Update Langfuse custom score (SAFE: mapped scope + sanitized target)
        try:
            if getattr(self._scorer, "enabled", False):
                mapping = self._query_trace_map.get(event.query_id)
                trace_hex = f"{mapping.get('trace_id'):032x}" if mapping and mapping.get('trace_id') is not None else None
                self._scorer.update_score(
                    query_id=event.query_id,
                    trace_hex=trace_hex,
                    scope=scope_mapped,              # "overall" | "doc"  ✅
                    sentiment=event.sentiment,
                    annotation=event.annotation,
                    target=safe_target,               # sanitized ✅ (or drop if your scorer doesn't need it)
                )
        except Exception as e:
            print(f"⚠️ Langfuse score update failed: {e}")

    except Exception as e:
        print(f"❌ Error in OpenTelemetry feedback logging: {e}")


    def flush(self):
        """Force sending of pending spans."""
        try:
            multi_exporter = get_multi_exporter()
            multi_exporter.flush()
        except Exception as e:
            print(f"⚠️  Error flushing OpenTelemetry spans: {e}")


# Global singleton instance
_otel_client: Optional[OpenTelemetryClient] = None

def get_otel_client() -> OpenTelemetryClient:
    """Get or create the global OpenTelemetry client instance."""
    global _otel_client
    if _otel_client is None:
        _otel_client = OpenTelemetryClient()
    return _otel_client