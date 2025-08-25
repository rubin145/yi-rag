# observability/schemas.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RetrievalSpan(BaseModel):
    """Información sobre la fase de retrieval"""
    hits: List[Dict[str, Any]]
    timing_ms: float
    vector_name: str
    k: int
    filters: Optional[Dict[str, Any]] = None
    # Nuevos campos para diagnóstico de configuración de búsqueda
    search_type: Optional[str] = None  # semantic | lexical | hybrid
    alpha: Optional[float] = None      # peso dense en híbrido
    score_threshold: Optional[float] = None  # umbral aplicado sobre score normalizado


class GenerationSpan(BaseModel):
    """Información sobre la fase de generación"""
    model: str
    answer: str
    citations: List[Dict[str, Any]]
    timing_ms: float
    usage: Optional[Dict[str, Any]] = None
    prompts: Optional[Dict[str, str]] = None


class QueryEvent(BaseModel):
    """Evento principal de query (retrieval + generation)"""
    query_id: str
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Input del usuario
    input: Dict[str, Any]  # query, using, k, filters, etc.
    
    # Resultados
    retrieval: RetrievalSpan
    generation: Optional[GenerationSpan] = None  # None si solo retrieval


class FeedbackEvent(BaseModel):
    """Evento de feedback del usuario"""
    feedback_id: str
    query_id: str  # referencia al query original
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    scope: str  # "overall" | "doc_retrieval" | "doc_translation"
    sentiment: str  # "up" | "down" | "annotation_only"
    annotation: Optional[str] = None
    
    # Para feedback de documento específico
    target: Optional[Dict[str, Any]] = None  # {"doc_idx": 2, "doc_id": "abc123"}