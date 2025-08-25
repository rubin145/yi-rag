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
			ts = event.timestamp if isinstance(event.timestamp, datetime) else None
			trace = self.client.trace(
				id=event.query_id,
				name="rag.query",
				input={"query": event.input.get("query"), **{k: v for k, v in event.input.items() if k != "query"}},
				timestamp=ts,
				metadata={
					"vector_name": event.retrieval.vector_name if event.retrieval else None,
					"search_type": event.retrieval.search_type if event.retrieval else None,
					"alpha": event.retrieval.alpha if event.retrieval else None,
					"score_threshold": event.retrieval.score_threshold if event.retrieval else None,
				},
			)
			if event.retrieval:
				self.client.span(
					trace_id=event.query_id,
					name="retrieval",
					input={"query": event.input.get("query"), "k": event.retrieval.k},
					output={"hits": event.retrieval.hits[:5], "hits_count": len(event.retrieval.hits)},
					metadata={"timing_ms": event.retrieval.timing_ms},
					start_time=ts,
					end_time=ts,
				)
			if getattr(event, "generation", None):
				self.client.generation(
					trace_id=event.query_id,
					name="generation",
					model=event.generation.model,
					input={"query": event.input.get("query"), "context_docs": len(event.retrieval.hits) if event.retrieval else 0},
					output={"answer": event.generation.answer, "citations": event.generation.citations},
					usage=event.generation.usage,
					metadata={"timing_ms": event.generation.timing_ms},
					start_time=ts,
					end_time=ts,
				)
				trace.update(output={"answer": event.generation.answer})
			else:
				trace.update(output={"hits_count": len(event.retrieval.hits) if event.retrieval else 0})
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
			self.client.score(
				trace_id=event.query_id,
				name=score_name,
				value=score,
				comment=event.annotation,
				metadata={
					"feedback_id": event.feedback_id,
					"scope": event.scope,
					"sentiment": event.sentiment,
					"target": event.target,
				},
			)
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