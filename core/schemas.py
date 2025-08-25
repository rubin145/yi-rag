from pydantic import BaseModel, Field
from typing import Optional, Dict, List

class Chunk(BaseModel):
    id: str                    # book_id:page:chunk_idx
    book_id: str
    title_yi_latin: Optional[str] = None # Yiddish title in Latin characters
    title_yi: Optional[str] = None   # Yiddish title in Hebrew characters
    title_en: Optional[str] = None   # English title
    author: Optional[str] = None     # English author name
    author_yi: Optional[str] = None  # Yiddish author name
    year: Optional[int] = None
    place: Optional[str] = None          # publication place
    publisher: Optional[str] = None      # publisher
    storage_locations: Optional[str] = None  # where book is stored
    subjects: Optional[str] = None       # subject categories
    total_pages: Optional[int] = None    # total pages in book
    source_url: Optional[str] = None     # original URL
    ocr_source_url: Optional[str] = None # OCR text URL
    metadata_url: Optional[str] = None   # metadata URL
    page: int
    chunk_idx: int
    yi_text: str
    tr_en_text: Optional[str] = None
    tr_en_metadata: Dict = Field(default_factory=dict)
    sparse_vector_yi: Optional[Dict] = None
    vectors: Optional[Dict[str, List[float]]] = Field(default_factory=dict)
    source_file: Optional[str] = None
    extra: Dict = Field(default_factory=dict)

class SearchFilters(BaseModel):
    book_id: Optional[str] = None
    author: Optional[str] = None
    author_yi: Optional[str] = None
    place: Optional[str] = None
    publisher: Optional[str] = None
    subjects: Optional[str] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    page_min: Optional[int] = None
    page_max: Optional[int] = None
    # Multi-select lists (OR semantics)
    author_list: Optional[List[str]] = None
    place_list: Optional[List[str]] = None
    publisher_list: Optional[List[str]] = None
    subjects_list: Optional[List[str]] = None

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