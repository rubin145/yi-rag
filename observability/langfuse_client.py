# """Langfuse client used directly and as a bridge from OpenTelemetry."""
# from __future__ import annotations

# import os
# from typing import Optional
# from datetime import datetime

# from .schemas import QueryEvent, FeedbackEvent

# try:
#     # Langfuse Python SDK v3
#     from langfuse import Langfuse
#     LANGFUSE_AVAILABLE = True
# except ImportError:
#     Langfuse = None
#     LANGFUSE_AVAILABLE = False


# class LangfuseLogger:
#     """Cliente para logging a Langfuse via SDK (independiente de OTel)."""

#     def __init__(self):
#         self.enabled = False
#         self.client: Optional[Langfuse] = None

#         if not LANGFUSE_AVAILABLE:
#             print("⚠️  Langfuse not installed. Skipping Langfuse logging.")
#             return

#         public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
#         secret_key = os.getenv("LANGFUSE_SECRET_KEY")
#         host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

#         if not public_key or not secret_key:
#             print("⚠️  Langfuse credentials not found. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY.")
#             return

#         try:
#             self.client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
#             self.enabled = True
#             print("✅ Langfuse logging enabled")
#         except Exception as e:
#             print(f"❌ Failed to initialize Langfuse: {e}")

#     async def log_query_event(self, event: QueryEvent) -> Optional[str]:
#         """Log query event to Langfuse como trace con spans/generation."""
#         if not self.enabled or not self.client:
#             return None

#         try:
#             # Trace principal
#             trace = self.client.trace(
#                 id=event.query_id,
#                 name="yiddish-rag-query",
#                 input=event.input,
#                 timestamp=event.timestamp if isinstance(event.timestamp, datetime) else None,
#                 metadata={
#                     "pipeline": "RAG",
#                     "vector_name": event.retrieval.vector_name if event.retrieval else None,
#                 },
#             )

#             # Span de retrieval
#             if event.retrieval:
#                 self.client.span(
#                     trace_id=event.query_id,
#                     name="retrieval",
#                     input={
#                         "query": event.input.get("query"),
#                         "vector_name": event.retrieval.vector_name,
#                         "k": event.retrieval.k,
#                         "filters": event.retrieval.filters,
#                     },
#                     output={
#                         "hits": event.retrieval.hits,
#                         "num_results": len(event.retrieval.hits),
#                     },
#                     metadata={
#                         "timing_ms": event.retrieval.timing_ms,
#                         "vector_name": event.retrieval.vector_name,
#                     },
#                     start_time=event.timestamp,
#                     end_time=event.timestamp,
#                 )

#             # Span de generación
#             if getattr(event, "generation", None):
#                 self.client.generation(
#                     trace_id=event.query_id,
#                     name="generation",
#                     model=event.generation.model,
#                     input={
#                         "query": event.input.get("query"),
#                         "context_docs": len(event.retrieval.hits) if event.retrieval else 0,
#                     },
#                     output={
#                         "answer": event.generation.answer,
#                         "citations": event.generation.citations,
#                     },
#                     usage=event.generation.usage,
#                     metadata={
#                         "timing_ms": event.generation.timing_ms,
#                         "model": event.generation.model,
#                     },
#                     start_time=event.timestamp,
#                     end_time=event.timestamp,
#                 )

#                 trace.update(
#                     output={
#                         "answer": event.generation.answer,
#                         "citations": event.generation.citations,
#                         "num_retrieved": len(event.retrieval.hits) if event.retrieval else 0,
#                     }
#                 )
#             else:
#                 trace.update(
#                     output={
#                         "hits": event.retrieval.hits if event.retrieval else [],
#                         "num_retrieved": len(event.retrieval.hits) if event.retrieval else 0,
#                     }
#                 )

#             return event.query_id
#         except Exception as e:
#             print(f"❌ Error logging to Langfuse: {e}")
#             return None

#     async def log_feedback_event(self, event: FeedbackEvent) -> bool:
#         """Log feedback event to Langfuse (score on the trace)."""
#         if not self.enabled or not self.client:
#             return False

#         try:
#             score_map = {"up": 1, "down": 0, "annotation_only": None}
#             score = score_map.get(event.sentiment)

#             self.client.score(
#                 trace_id=event.query_id,
#                 name=f"user_feedback_{event.scope}",
#                 value=score,
#                 comment=event.annotation,
#                 metadata={
#                     "feedback_id": event.feedback_id,
#                     "scope": event.scope,
#                     "sentiment": event.sentiment,
#                     "target": event.target,
#                     "timestamp": event.timestamp.isoformat(),
#                 },
#             )
#             return True
#         except Exception as e:
#             print(f"❌ Error logging feedback to Langfuse: {e}")
#             return False

#     def flush(self):
#         """Fuerza el envío de eventos pendientes."""
#         if self.enabled and self.client:
#             self.client.flush()