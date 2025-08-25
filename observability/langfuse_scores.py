"""Minimal Langfuse scoring helper.

This bypasses OTEL for custom scores (not standard OTLP),
using Langfuse Python SDK if available and enabled via env ENABLE_LANGFUSE_SCORES=1.

Usage:
  scorer = get_langfuse_scorer()
  scorer.create_placeholder_scores(query_id, trace_hex, hits)
  scorer.update_score(query_id=query_id, trace_hex=trace_hex, scope=..., sentiment=..., annotation=..., target=...)
"""
from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

try:  # pragma: no cover
    from langfuse import Langfuse  # type: ignore
    _LANGFUSE_LIB = True
except Exception:  # pragma: no cover
    Langfuse = None  # type: ignore
    _LANGFUSE_LIB = False

PLACEHOLDER_VALUE = -1.0  # sentinel distinct from real (0/1) values

class _NoOpScorer:
    enabled = False
    def create_placeholder_scores(self, *_, **__):
        return
    def update_score(self, *_, **__):
        return

class LangfuseScorer:
    def __init__(self):
        self.enabled = bool(os.getenv("ENABLE_LANGFUSE_SCORES", "0") == "1" and _LANGFUSE_LIB)
        if not self.enabled:
            self.client = None
            if os.getenv("ENABLE_LANGFUSE_SCORES", "0") == "1" and not _LANGFUSE_LIB:
                print("⚠️ Langfuse scorer enabled but langfuse package not installed.")
            return
        try:
            self.client = Langfuse()
            print("📝 Langfuse scorer active (custom scores)")
        except Exception as e:  # pragma: no cover
            print(f"❌ Failed to init Langfuse scorer: {e}")
            self.enabled = False
            self.client = None

    def _score_id(self, *, query_id: str, scope: str, doc_idx: Optional[int] = None) -> str:
        if scope == "overall":
            return f"score.overall.{query_id}"
        if doc_idx is not None:
            return f"score.{scope}.doc{doc_idx}.{query_id}"
        return f"score.{scope}.{query_id}"

    def create_placeholder_scores(self, query_id: str, trace_hex: str, hits: List[Dict[str, Any]]):
        if not self.enabled or not self.client:
            return
        try:
            self.client.score(
                id=self._score_id(query_id=query_id, scope="overall"),
                trace_id=trace_hex,
                name="overall_relevance",
                value=PLACEHOLDER_VALUE,
                comment="placeholder",
            )
            for idx, _hit in enumerate(hits):
                self.client.score(
                    id=self._score_id(query_id=query_id, scope="doc_retrieval", doc_idx=idx),
                    trace_id=trace_hex,
                    name="doc_retrieval_relevance",
                    value=PLACEHOLDER_VALUE,
                    comment="placeholder",
                )
                self.client.score(
                    id=self._score_id(query_id=query_id, scope="doc_translation", doc_idx=idx),
                    trace_id=trace_hex,
                    name="doc_translation_quality",
                    value=PLACEHOLDER_VALUE,
                    comment="placeholder",
                )
        except Exception as e:  # pragma: no cover
            print(f"⚠️ Langfuse placeholder score error: {e}")

    def update_score(
        self,
        *,
        query_id: str,
        trace_hex: Optional[str],
        scope: str,
        sentiment: str,
        annotation: Optional[str],
        target: Optional[Dict[str, Any]],
    ):
        if not self.enabled or not self.client or not trace_hex:
            return
        try:
            doc_idx = None
            if target and isinstance(target, dict):
                doc_idx = target.get("doc_idx")
            if sentiment == "up":
                val = 1.0
            elif sentiment == "down":
                val = 0.0
            else:
                val = PLACEHOLDER_VALUE
            score_id = self._score_id(query_id=query_id, scope=scope, doc_idx=doc_idx)
            name_map = {
                "overall": "overall_relevance",
                "doc_retrieval": "doc_retrieval_relevance",
                "doc_translation": "doc_translation_quality",
            }
            self.client.score(
                id=score_id,
                trace_id=trace_hex,
                name=name_map.get(scope, scope),
                value=val,
                comment=annotation or None,
            )
        except Exception as e:  # pragma: no cover
            print(f"⚠️ Langfuse score update error: {e}")

_scorer_instance: Optional[LangfuseScorer] = None

def get_langfuse_scorer() -> LangfuseScorer | _NoOpScorer:
    global _scorer_instance
    if _scorer_instance is None:
        scorer = LangfuseScorer()
        if scorer.enabled:
            _scorer_instance = scorer
        else:
            _scorer_instance = _NoOpScorer()  # type: ignore
    return _scorer_instance
