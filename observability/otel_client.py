# observability/otel_client.py
from __future__ import annotations

import os
from typing import Optional

from .schemas import QueryEvent, FeedbackEvent
from .multi_exporter import get_multi_exporter

# Load environment variables early
try:  # pragma: no cover
    from dotenv import load_dotenv
    from pathlib import Path
    # Explicitly find the .env file in the project root, which is 2 levels up
    # from this file's location (observability/otel_client.py)
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.is_file():
        load_dotenv(dotenv_path=env_path)
        print(f"✅ Loaded .env file from: {env_path}")
    else:
        # Fallback for cases where the .env is in the CWD
        if Path('.env').is_file():
            load_dotenv()
            print("✅ Loaded .env file from current working directory.")
        else:
            print("ℹ️ .env file not found in project root or CWD.")
except Exception as e:
    print(f"⚠️  Error loading .env file: {e}")

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
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
        
        if not OTEL_AVAILABLE:
            print("⚠️  OpenTelemetry not installed. Skipping OTEL logging.")
            return
        
        try:
            # Use the new multi-exporter configuration
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
            # Create a root span for the entire RAG query
            with self.tracer.start_as_current_span(
                name="rag.query",
                kind=trace.SpanKind.SERVER,
                attributes={
                    OpenInferenceSpanAttributes.OPENINFERENCE_SPAN_KIND: "CHAIN",
                    OpenInferenceSpanAttributes.INPUT_VALUE: event.input.get("query", ""),
                    OpenInferenceSpanAttributes.OUTPUT_VALUE: event.generation.answer if event.generation else "No generation",
                    "session.id": event.session_id,
                    "query.id": event.query_id,
                    "rag.query": event.input.get("query", ""),
                    "rag.k": event.input.get("k", 0),
                    "rag.using": event.input.get("using", "unknown"),
                    # Arize specific
                    "external_model_id": event.generation.model if event.generation else "retrieval-only",
                    "model_id": event.generation.model if event.generation else "retrieval-only",
                }
            ) as root_span:
                
                # Create a nested span for the retrieval step
                with self.tracer.start_as_current_span(
                    name="rag.retrieval",
                    kind=trace.SpanKind.INTERNAL,
                    attributes={
                        OpenInferenceSpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.RETRIEVER.value,
                        OpenInferenceSpanAttributes.INPUT_VALUE: event.input.get("query", ""),
                        "retrieval.vector_name": event.retrieval.vector_name,
                        "retrieval.k": event.retrieval.k,
                        "retrieval.timing_ms": event.retrieval.timing_ms,
                        "retrieval.hits_count": len(event.retrieval.hits),
                        # Arize specific
                        "external_model_id": event.retrieval.vector_name,
                        "model_id": event.retrieval.vector_name,
                    }
                ) as retrieval_span:
                    # Add documents using OpenInference standard format
                    for i, hit in enumerate(event.retrieval.hits):
                        # Each document gets its own set of attributes with index
                        # Try multiple content fields in order of preference
                        content = hit.get("tr_en_text") or hit.get("yi_text") or hit.get("content", "")
                        
                        retrieval_span.set_attribute(f"document.{i}.id", str(hit.get("id", f"doc_{i}")))
                        retrieval_span.set_attribute(f"document.{i}.content", content)
                        retrieval_span.set_attribute(f"document.{i}.score", float(hit.get("score", 0.0)))
                        retrieval_span.set_attribute(f"document.{i}.rank", i)
                        
                        # Optional metadata from payload or direct fields
                        if hit.get("book_id"):
                            retrieval_span.set_attribute(f"document.{i}.metadata.book_id", str(hit["book_id"]))
                        if hit.get("page"):
                            retrieval_span.set_attribute(f"document.{i}.metadata.page", str(hit["page"]))
                        if hit.get("author"):
                            retrieval_span.set_attribute(f"document.{i}.metadata.author", str(hit["author"]))
                        if hit.get("title_en"):
                            retrieval_span.set_attribute(f"document.{i}.metadata.title_en", str(hit["title_en"]))
                        if hit.get("year"):
                            retrieval_span.set_attribute(f"document.{i}.metadata.year", str(hit["year"]))
                    
                    # Set output to documents as well
                    documents_summary = f"Retrieved {len(event.retrieval.hits)} documents with scores: " + \
                                      ", ".join([f"{hit.get('score', 0):.3f}" for hit in event.retrieval.hits[:3]])
                    retrieval_span.set_attribute(OpenInferenceSpanAttributes.OUTPUT_VALUE, documents_summary)
                    retrieval_span.set_attribute("retrieval.query", event.input.get("query", ""))
                    retrieval_span.set_status(Status(StatusCode.OK))

                # Create a nested span for the generation step (if it exists)
                if event.generation:
                    gen_attributes = {
                        OpenInferenceSpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                        OpenInferenceSpanAttributes.LLM_MODEL_NAME: event.generation.model,
                        OpenInferenceSpanAttributes.INPUT_VALUE: event.input.get("query", ""),
                        OpenInferenceSpanAttributes.OUTPUT_VALUE: event.generation.answer,
                        "generation.timing_ms": event.generation.timing_ms,
                        "generation.citations_count": len(event.generation.citations),
                        # Arize specific: external_model_id required
                        "external_model_id": event.generation.model,
                        "model_id": event.generation.model,
                    }
                    
                    # Add usage information if available
                    usage = event.generation.usage or {}
                    prompt_tokens = usage.get("input_tokens")
                    completion_tokens = usage.get("output_tokens")
                    
                    if prompt_tokens is not None:
                        gen_attributes[OpenInferenceSpanAttributes.LLM_TOKEN_COUNT_PROMPT] = prompt_tokens
                    if completion_tokens is not None:
                        gen_attributes[OpenInferenceSpanAttributes.LLM_TOKEN_COUNT_COMPLETION] = completion_tokens
                    
                    # Add prompts if available
                    prompts = event.generation.prompts or {}
                    if prompts.get("user"):
                        gen_attributes["generation.prompt.user"] = prompts["user"]
                    if prompts.get("system"):
                        gen_attributes["generation.prompt.system"] = prompts["system"]

                    with self.tracer.start_as_current_span(
                        name="rag.generation",
                        kind=trace.SpanKind.INTERNAL,
                        attributes=gen_attributes
                    ) as generation_span:
                        # Add citations as structured data
                        if event.generation.citations:
                            import json
                            generation_span.set_attribute("generation.citations", json.dumps(event.generation.citations))
                        generation_span.set_status(Status(StatusCode.OK))
                
                root_span.set_status(Status(StatusCode.OK))
                
        except Exception as e:
            print(f"❌ Error in OpenTelemetry query event logging: {e}")
            import traceback
            traceback.print_exc()
    
    def log_feedback_event(self, event: FeedbackEvent) -> None:
        """Log a feedback event as a separate span linked to the original query trace."""
        if not self.enabled or not self.tracer:
            return
        
        try:
            with self.tracer.start_as_current_span(
                "rag.feedback",
                attributes={
                    "feedback.id": event.feedback_id,
                    "feedback.query_id": event.query_id, # Used for linking in UIs
                    "feedback.scope": event.scope,
                    "feedback.sentiment": event.sentiment,
                    "feedback.annotation": event.annotation or "",
                }
            ) as feedback_span:
                if event.target:
                    import json
                    feedback_span.set_attribute("feedback.target", json.dumps(event.target))
                feedback_span.set_status(Status(StatusCode.OK))
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