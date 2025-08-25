import os
from typing import Literal, Optional
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    # --- Preproceso ---
    # default_index_lang: Literal["en","yi"] = os.getenv("DEFAULT_INDEX_LANG", "yi")
    
    chunk_strategy: Literal["page","tokens","chars"] = os.getenv("CHUNK_STRATEGY","chars")
    chunk_tokens: int = int(os.getenv("CHUNK_TOKENS","450"))
    chunk_tokens_overlap: int = int(os.getenv("CHUNK_TOKENS_OVERLAP","50"))
    chunk_chars: int = int(os.getenv("CHUNK_CHARS","2500"))
    chunk_chars_overlap: int = int(os.getenv("CHUNK_CHARS_OVERLAP","200"))

    # --- Traducción (desacoplada del index) ---
    tr_provider: Literal["openai","cohere","gemini","none"] = os.getenv("TR_PROVIDER","none")
    translate_chunks_en: bool = os.getenv("TRANSLATE_CHUNKS_EN","false").lower() == "true"
    translate_titles_en: bool = os.getenv("TRANSLATE_TITLES_EN","true").lower() == "true"
    log_every_n_translations: int = int(os.getenv("LOG_EVERY_N_TRANSL", "50"))


    # Modelos / keys (como ya tenías)
    tr_model_openai: str = os.getenv("OPENAI_TRANSLATION_MODEL","gpt-4o-mini")
    tr_model_cohere: str = os.getenv("COHERE_TRANSLATION_MODEL","command-r")
    tr_model_gemini: str = os.getenv("GEMINI_TRANSLATION_MODEL","gemini-1.5-flash")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    cohere_api_key: Optional[str] = os.getenv("COHERE_API_KEY")
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")

    # --- Qdrant (para fases posteriores) ---
    qdrant_url: Optional[str]      = os.getenv("QDRANT_URL")
    qdrant_api_key: Optional[str]  = os.getenv("QDRANT_API_KEY")
    collection_name: str           = os.getenv("QDRANT_COLLECTION", "yi_rag")

    @field_validator("chunk_tokens", "chunk_tokens_overlap", "chunk_chars", "chunk_chars_overlap")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Debe ser > 0")
        return v

settings = Settings()