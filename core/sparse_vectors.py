# core/sparse_vectors.py
from __future__ import annotations

import json
import os
from typing import Iterable, List, Tuple, Optional, Dict

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


class CharTfidfSparseEncoder:
    """
    TF-IDF de n-gramas de caracteres (3–5) para Yidis (alfabeto hebreo).
    - Robusto a OCR/variantes.
    - Devuelve top-K índices/valores por documento.
    - Persistencia: vectorizer + config para reproducir queries.
    """

    def __init__(
        self,
        analyzer: str = "char_wb",          # "char_wb" o "char"
        ngram_range: Tuple[int, int] = (3, 5),
        min_df: int | float = 2,            # >=2 docs
        max_df: int | float = 0.9,          # descarta n-grams muy frecuentes
        max_features: Optional[int] = 200_000,
        sublinear_tf: bool = True,          # log-tf
        topk_per_doc: int = 400,
        min_value: float = 1e-6,
        lowercase: bool = False,
        dtype=np.float32,
    ):
        self.cfg = dict(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            topk_per_doc=topk_per_doc,
            min_value=min_value,
            lowercase=lowercase,
            dtype=str(dtype.__name__),
        )
        self.dtype = dtype
        self.vectorizer: Optional[TfidfVectorizer] = None

    # ---------- core ----------
    def fit(self, texts: Iterable[str]) -> "CharTfidfSparseEncoder":
        vec = TfidfVectorizer(
            analyzer=self.cfg["analyzer"],
            ngram_range=self.cfg["ngram_range"],
            min_df=self.cfg["min_df"],
            max_df=self.cfg["max_df"],
            max_features=self.cfg["max_features"],
            lowercase=self.cfg["lowercase"],
            sublinear_tf=self.cfg["sublinear_tf"],
            norm=None,                # sin normalizar; Qdrant fusiona con RRF si luego sumás dense
            dtype=self.dtype,
        )
        self.matrix_ = vec.fit_transform(texts)   # solo para stats si querés
        self.vectorizer = vec
        return self

    def transform_matrix(self, texts: Iterable[str]) -> sp.csr_matrix:
        assert self.vectorizer is not None, "fit primero o load(...)"
        return self.vectorizer.transform(texts)

    def _topk_row(self, row_csr: sp.csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
        # Retorna indices/values ordenados por peso desc, aplicando top-K y umbral
        k = self.cfg["topk_per_doc"]
        min_val = self.cfg["min_value"]
        coo = row_csr.tocoo()
        if coo.nnz == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=self.dtype)
        order = np.argsort(-coo.data)
        data = coo.data[order]
        cols = coo.col[order]
        mask = data > min_val
        data = data[mask][:k].astype(self.dtype, copy=False)
        cols = cols[mask][:k].astype(np.int32, copy=False)
        return cols, data

    def transform_to_indices_values(self, texts: Iterable[str]) -> List[Tuple[np.ndarray, np.ndarray]]:
        X = self.transform_matrix(texts)
        out: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(X.shape[0]):
            idxs, vals = self._topk_row(X[i])
            out.append((idxs, vals))
        return out

    # ---------- persistencia ----------
    def save(self, artifacts_dir: str) -> None:
        os.makedirs(artifacts_dir, exist_ok=True)
        assert self.vectorizer is not None, "Nada para guardar; fit/load primero"
        joblib.dump(self.vectorizer, os.path.join(artifacts_dir, "tfidf_vectorizer.joblib"))
        with open(os.path.join(artifacts_dir, "sparse_cfg.json"), "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, ensure_ascii=False, indent=2)

        # opcional: vocab_size, idf stats
        meta = {
            "vocab_size": len(self.vectorizer.vocabulary_),
            "stop_words": None,
        }
        with open(os.path.join(artifacts_dir, "sparse_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, artifacts_dir: str) -> "CharTfidfSparseEncoder":
        with open(os.path.join(artifacts_dir, "sparse_cfg.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        enc = cls(**{**cfg, "dtype": getattr(np, cfg.get("dtype", "float32"))})
        enc.vectorizer = joblib.load(os.path.join(artifacts_dir, "tfidf_vectorizer.joblib"))
        return enc

    # ---------- helpers ----------
    @property
    def vocab_size(self) -> int:
        assert self.vectorizer is not None
        return len(self.vectorizer.vocabulary_)

    def transform_query(self, text: str) -> Tuple[List[int], List[float]]:
        """Para query-time: indices/values ya listos para Qdrant."""
        (idxs, vals) = self.transform_to_indices_values([text])[0]
        return idxs.tolist(), vals.tolist()
