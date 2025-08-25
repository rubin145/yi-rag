# observability/logger.py
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

from .schemas import QueryEvent, FeedbackEvent, RetrievalSpan, GenerationSpan
from .otel_client import get_otel_client


class MultiLogger:
    """
    Logger principal que usa OpenTelemetry como estándar
    Compatible con Langfuse, Phoenix, Jaeger, y cualquier backend OTLP
    """
    
    def __init__(self):
        # Cliente OpenTelemetry único
        self.otel_client = get_otel_client()

        # TODO: Añadir clientes específicos para casos no-OTLP
        # self.hf_datasets = HFDatasetsLogger()  # Dataset propio
        # self.qdrant_feedback = QdrantFeedbackLogger()  # Translation feedback

        self.enabled = True
    
    def log_query_event(
        self,
        query_id: str,
        session_id: str,
        input_data: Dict[str, Any],
        retrieval_data: Dict[str, Any],
        generation_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log un evento de query usando OpenTelemetry
        
        Args:
            query_id: ID único del query
            session_id: ID de la sesión del usuario
            input_data: datos de entrada del usuario
            retrieval_data: resultados del retrieval
            generation_data: resultados de la generación (opcional)
        """
        if not self.enabled:
            return
        
        try:
            # Construir spans
            retrieval_span = RetrievalSpan(
                hits=retrieval_data.get("hits", []),
                timing_ms=retrieval_data.get("timing_ms", 0),
                vector_name=retrieval_data.get("vector_name", ""),
                k=retrieval_data.get("k", 0),
                filters=retrieval_data.get("filters"),
                search_type=retrieval_data.get("search_type") or input_data.get("search_type"),
                alpha=retrieval_data.get("alpha"),
                score_threshold=retrieval_data.get("score_threshold") or input_data.get("score_threshold")
            )
            
            generation_span = None
            if generation_data:
                generation_span = GenerationSpan(
                    model=generation_data.get("model", ""),
                    answer=generation_data.get("answer", ""),
                    citations=generation_data.get("citations", []),
                    timing_ms=generation_data.get("timing_ms", 0),
                    usage=generation_data.get("usage"),
                    prompts=generation_data.get("prompts")
                )
            
            # Crear evento
            event = QueryEvent(
                query_id=query_id,
                session_id=session_id,
                input=input_data,
                retrieval=retrieval_span,
                generation=generation_span
            )
            
            # Enviar a OpenTelemetry (automáticamente va a todos los backends configurados)
            self.otel_client.log_query_event(event)
            
            # TODO: Enviar a clientes específicos no-OTLP
            # self.hf_datasets.log_query_event(event)
            
        except Exception as e:
            print(f"❌ Error in log_query_event: {e}")
            import traceback
            traceback.print_exc()
    
    def log_feedback_event(
        self,
        feedback_id: str,
        query_id: str,
        scope: str,
        sentiment: str,
        annotation: Optional[str] = None,
        target: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log un evento de feedback usando OpenTelemetry
        
        Args:
            feedback_id: ID único del feedback
            query_id: ID del query original
            scope: "overall" | "doc_retrieval" | "doc_translation"
            sentiment: "up" | "down" | "annotation_only"
            annotation: comentario opcional
            target: información del documento (para feedback específico)
        """
        if not self.enabled:
            return
        
        try:
            # Crear evento
            event = FeedbackEvent(
                feedback_id=feedback_id,
                query_id=query_id,
                scope=scope,
                sentiment=sentiment,
                annotation=annotation,
                target=target
            )
            
            # Siempre enviamos a OpenTelemetry (Langfuse, Phoenix, etc.) para mantener consistencia
            otel = getattr(self, "otel_client", None)

            # Newer client: has log_feedback_event(event)
            if hasattr(otel, "log_feedback_event"):
                otel.log_feedback_event(event)
            else:
                # Fallback for older clients: emit span parented to the query's root span
                tracer = getattr(otel, "tracer", None)
                try:
                    # --- Build minimal, JSON-safe attributes
                    attrs = {
                        "feedback.id": event.feedback_id,
                        "feedback.query_id": event.query_id,
                        "feedback.scope": event.scope,
                        "feedback.sentiment": event.sentiment,
                        "feedback.annotation": event.annotation or "",
                    }
                    if isinstance(event.target, dict):
                        if "doc_idx" in event.target and event.target["doc_idx"] is not None:
                            try:
                                attrs["feedback.target.doc_idx"] = int(event.target["doc_idx"])
                            except Exception:
                                attrs["feedback.target.doc_idx"] = str(event.target["doc_idx"])
                        if "doc_id" in event.target and event.target["doc_id"] is not None:
                            attrs["feedback.target.doc_id"] = str(event.target["doc_id"])

                    if tracer is None:
                        print("⚠️ OTEL client has no tracer; feedback span not emitted.")
                        return

                    # --- Parent to the query’s root span if we have it
                    parent_ctx = None
                    try:
                        mapping_all = getattr(otel, "_query_trace_map", {}) or {}
                        mapping = mapping_all.get(event.query_id)
                        if mapping and mapping.get("trace_id") is not None and mapping.get("root_span_id") is not None:
                            from opentelemetry.trace import SpanContext, TraceFlags, TraceState, NonRecordingSpan, set_span_in_context
                            sc = SpanContext(
                                trace_id=mapping["trace_id"],
                                span_id=mapping["root_span_id"],
                                is_remote=False,
                                trace_flags=TraceFlags(TraceFlags.SAMPLED),
                                trace_state=TraceState(),
                            )
                            parent_ctx = set_span_in_context(NonRecordingSpan(sc))
                    except Exception as _e:
                        parent_ctx = None
                        print(f"⚠️ Could not set parent context for feedback: {_e}")

                    from opentelemetry import trace as _trace  # type: ignore
                    with tracer.start_as_current_span("rag.feedback", context=parent_ctx, kind=_trace.SpanKind.INTERNAL, attributes=attrs):
                        pass

                    # Optional: update your scorer
                    if getattr(self, "_scorer", None) and getattr(self._scorer, "enabled", False):
                        try:
                            trace_hex = None
                            if mapping:
                                trace_id_val = mapping.get("trace_id")
                                if trace_id_val is not None:
                                    trace_hex = f"{trace_id_val:032x}"
                            self._scorer.update_score(
                                query_id=event.query_id,
                                trace_hex=trace_hex,
                                scope=("doc" if event.scope in ("doc_retrieval", "doc_translation") else "overall"),
                                sentiment=event.sentiment,
                                annotation=event.annotation,
                                target={"doc_idx": attrs.get("feedback.target.doc_idx")} if "feedback.target.doc_idx" in attrs else None,
                            )
                        except Exception as _e:
                            print(f"⚠️ Langfuse score update failed (shim): {_e}")

                except Exception as _e:
                    print(f"⚠️ OTEL fallback failed: {_e}")

        except Exception as _e:
            print(f"⚠️ OTEL failed somewhere: {_e}")   

    
    def flush(self):
        """Fuerza el envío de eventos pendientes en todos los clientes"""
        self.otel_client.flush()
        
        # TODO: flush otros clientes no-OTLP
        # self.hf_datasets.flush()
        # self.qdrant_feedback.flush()


# Instancia global singleton
_logger_instance: Optional[MultiLogger] = None

def get_logger() -> MultiLogger:
    """Get or create global logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = MultiLogger()
    return _logger_instance