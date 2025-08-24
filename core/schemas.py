from pydantic import BaseModel, Field
from typing import Optional, Dict, List

class Chunk(BaseModel):
    id: str                    # book_id:page:chunk_idx
    book_id: str
    title_yi: Optional[str] = None   # título en yiddish
    title_en: Optional[str] = None   # título en inglés (opcional)
    author: Optional[str] = None
    year: Optional[int] = None
    page: int
    chunk_idx: int
    yi_text: str
    tr_en_text: Optional[str] = None
    source_file: Optional[str] = None
    extra: Dict = Field(default_factory=dict)

class SearchFilters(BaseModel):
    book_id: Optional[str] = None
    author: Optional[str] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    page_min: Optional[int] = None
    page_max: Optional[int] = None

# (opcional) forma en que representarás un “hit” del retriever
class Hit(BaseModel):
    id: str
    score: float
    book_id: str
    title: Optional[str]
    author: Optional[str]
    year: Optional[int]
    page: int
    chunk_idx: int
    yi_text: str
    tr_en_text: Optional[str]

# Feedback mejorado para el sistema actual
class Feedback(BaseModel):
    scope: str                    # "overall" | "doc_retrieval" | "doc_translation"
    sentiment: str                # "up" | "down" | "annotation_only"
    annotation: Optional[str] = None
    target_doc_idx: Optional[int] = None
    target_doc_id: Optional[str] = None
    query_id: Optional[str] = None
    session_id: Optional[str] = None