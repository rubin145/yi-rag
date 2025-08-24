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
                filters=retrieval_data.get("filters")
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
            
            if scope == "doc_translation":
                # Feedback de traducción va solo a Qdrant (específico, no OTLP)
                # TODO: implementar qdrant feedback
                # self.qdrant_feedback.update_translation_quality(event)
                print(f"🔄 Translation feedback {feedback_id} → Qdrant (TODO)")
            else:
                # Feedback de retrieval va a OpenTelemetry (Langfuse, Phoenix, etc.)
                self.otel_client.log_feedback_event(event)
                
                # TODO: También enviar a dataset propio si está configurado
                # self.hf_datasets.log_feedback_event(event)
                
        except Exception as e:
            print(f"❌ Error in log_feedback_event: {e}")
    
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