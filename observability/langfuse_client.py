"""Langfuse client used directly alongside OpenTelemetry for richer feedback linking."""
from __future__ import annotations

import os
from typing import Optional
from datetime import datetime

from .schemas import QueryEvent, FeedbackEvent

try:  # pragma: no cover
	from langfuse import Langfuse
	LANGFUSE_AVAILABLE = True
except ImportError:  # pragma: no cover
	Langfuse = None
	LANGFUSE_AVAILABLE = False


class LangfuseLogger:
	"""Synchronous wrapper around Langfuse SDK (keeps IDs aligned with query_id)."""

	def __init__(self):
		self.enabled = False
		self.client: Optional[Langfuse] = None
		self._trace_cache = {}  # Store trace IDs for feedback
		if not LANGFUSE_AVAILABLE:
			return
		public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
		secret_key = os.getenv("LANGFUSE_SECRET_KEY")
		host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
		if not public_key or not secret_key:
			return
		try:
			self.client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
			self.enabled = True
		except Exception as e:  # pragma: no cover
			print(f"❌ Failed to init Langfuse: {e}")

	def log_query_event(self, event: QueryEvent) -> Optional[str]:
		if not self.enabled or not self.client:
			return None
		try:
			# Create a single root trace (not span) - manually create trace
			trace_id = self.client.create_trace_id()
			
			# Create root span for the query
			root_span = self.client.start_span(
				name="rag.query",
				input={"query": event.input.get("query"), **{k: v for k, v in event.input.items() if k != "query"}},
				metadata={
					"vector_name": event.retrieval.vector_name if event.retrieval else None,
					"search_type": event.retrieval.search_type if event.retrieval else None,
					"alpha": event.retrieval.alpha if event.retrieval else None,
					"score_threshold": event.retrieval.score_threshold if event.retrieval else None,
				}
			)
			
			if event.retrieval:
				# Create retrieval span as child of root span
				retrieval_span = root_span.start_span(
					name="retrieval",
					input={"query": event.input.get("query"), "k": event.retrieval.k},
					output={"hits": event.retrieval.hits[:5], "hits_count": len(event.retrieval.hits)},
					metadata={"timing_ms": event.retrieval.timing_ms}
				)
				retrieval_span.end()
			
			if getattr(event, "generation", None):
				# Create generation span as child of root span
				generation_span = root_span.start_generation(
					name="generation",
					model=event.generation.model,
					input={"query": event.input.get("query"), "context_docs": len(event.retrieval.hits) if event.retrieval else 0},
					output={"answer": event.generation.answer, "citations": event.generation.citations},
					usage=event.generation.usage,
					metadata={"timing_ms": event.generation.timing_ms}
				)
				generation_span.end()
				root_span.update(output={"answer": event.generation.answer})
			else:
				root_span.update(output={"hits_count": len(event.retrieval.hits) if event.retrieval else 0})
			
			root_span.end()
			
			# Store trace ID for feedback linking - use the trace ID, not span ID
			actual_trace_id = root_span.trace_id if hasattr(root_span, 'trace_id') else root_span.id
			self._trace_cache[event.query_id] = actual_trace_id
			print(f"🔗 Stored trace ID {actual_trace_id} for query {event.query_id}")
			
			return event.query_id
		except Exception as e:  # pragma: no cover
			print(f"❌ Langfuse query log error: {e}")
			return None

	def log_feedback_event(self, event: FeedbackEvent) -> bool:
		if not self.enabled or not self.client:
			return False
		try:
			score_map = {"up": 1, "down": 0, "annotation_only": None}
			score = score_map.get(event.sentiment)
			name_parts = ["user_feedback", event.scope]
			if event.target and isinstance(event.target, dict) and "doc_idx" in event.target:
				name_parts.append(str(event.target["doc_idx"]))
			score_name = ".".join(name_parts)
			
			# Get trace ID from cache
			trace_id = self._trace_cache.get(event.query_id)
			print(f"🔍 Looking for trace ID for query {event.query_id}, found: {trace_id}")
			
			# Only create score if we have a numeric value and trace ID
			if score is not None and trace_id:
				self.client.create_score(
					name=score_name,
					value=score,
					trace_id=trace_id,
					comment=event.annotation,
					metadata={
						"feedback_id": event.feedback_id,
						"scope": event.scope,
						"sentiment": event.sentiment,
						"target": event.target,
					}
				)
				print(f"✅ Created score {score_name}={score} for trace {trace_id}")
			else:
				print(f"⚠️ Skipped score creation: score={score}, trace_id={trace_id}")
			# For annotation_only, we can skip or just log the annotation
			# Langfuse doesn't require scores for every piece of feedback
			return True
		except Exception as e:  # pragma: no cover
			print(f"❌ Langfuse feedback log error: {e}")
			return False

	def flush(self):  # pragma: no cover
		if self.enabled and self.client:
			try:
				self.client.flush()
			except Exception:
				pass