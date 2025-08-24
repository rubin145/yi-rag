# scripts/index_corpus.py
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from core.config import settings
from core.embeddings import embed_texts
from core.store_qdrant import (
    ensure_collection_named_vectors,
    upsert_batch_with_named_vectors,
    upsert_or_update_vectors
)

log = logging.getLogger("index_corpus")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------
# VectorSpec loader
# ---------------------------
@dataclass
class VectorSpec:
    name: str
    provider: str
    model: str
    dim: int
    source_field: str            # "yi_text" | "tr_en_text"
    max_input_tokens: Optional[int] = None
    normalize: bool = True
    tokenizer: str = "cl100k_base"
    batch_size: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VectorSpec":
        return cls(
            name=d["name"],
            provider=d["provider"],
            model=d["model"],
            dim=int(d["dim"]),
            source_field=d["source_field"],
            max_input_tokens=d.get("max_input_tokens"),
            normalize=bool(d.get("normalize", True)),
            tokenizer=d.get("tokenizer", "cl100k_base"),
            batch_size=d.get("batch_size"),
        )


def load_vector_specs(path_yaml: Optional[str], env_json: Optional[str]) -> List[VectorSpec]:
    if path_yaml and os.path.exists(path_yaml):
        with open(path_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        vecs = [VectorSpec.from_dict(v) for v in cfg.get("vectors", [])]
        return vecs
    if env_json:
        d = json.loads(env_json)
        vecs = [VectorSpec.from_dict(v) for v in d.get("vectors", [])]
        return vecs
    raise FileNotFoundError("No se encontraron vector specs (config/vectors.yaml o VECTORS_JSON)")


# ---------------------------
# Data loader
# ---------------------------
def iter_rows_from_parquet(single_path: Optional[str], parts_dir: Optional[str]) -> List[Dict[str, Any]]:
    """
    Devuelve lista de dicts (rows) desde un parquet único o lista de parts/*.parquet
    """
    rows: List[Dict[str, Any]] = []
    if parts_dir:
        files = sorted(glob.glob(os.path.join(parts_dir, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"No hay parts/*.parquet en {parts_dir}")
        for p in files:
            df = pd.read_parquet(p)
            rows.extend(df.to_dict(orient="records"))
    elif single_path:
        df = pd.read_parquet(single_path)
        rows.extend(df.to_dict(orient="records"))
    else:
        raise ValueError("Debes pasar --parquet o --parts")
    return rows


# ---------------------------
# Indexado
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Indexar corpus a Qdrant con named vectors")
    parser.add_argument("--parquet", default="data/processed/chunks.parquet", help="Ruta a parquet único")
    parser.add_argument("--parts", default=None, help="Directorio con parts/*.parquet (si usás modo por libro)")
    parser.add_argument("--vectors-yaml", default="config/vectors.yaml", help="Ruta al YAML de vectores")
    parser.add_argument("--vectors-json", default=None, help="Alternativa a YAML: JSON en env/archivo")
    parser.add_argument("--only", default=None, help="Nombre de vector a indexar (si querés solo uno)")
    parser.add_argument("--recreate", action="store_true", help="Recrear colección (borra datos)")
    parser.add_argument("--batch", type=int, default=256, help="Tamaño de lote para Qdrant")
    args = parser.parse_args()

    # cargar specs
    vec_specs = load_vector_specs(args.vectors_yaml, args.vectors_json)
    if args.only:
        vec_specs = [s for s in vec_specs if s.name == args.only]
        if not vec_specs:
            raise ValueError(f"--only={args.only} no coincide con ningún spec")

    # asegurar colección con *todos* los named vectors (aunque hoy subamos 1)
    ensure_collection_named_vectors(settings.collection_name, [s.__dict__ for s in vec_specs], recreate=args.recreate)

    # cargar datos
    rows = iter_rows_from_parquet(args.parquet if not args.parts else None, args.parts)
    n = len(rows)
    if n == 0:
        log.warning("No hay filas en el dataset.")
        return
    log.info("Rows a indexar: %d", n)

    # Por cada vector spec, indexamos
    for spec in vec_specs:
        log.info("==== Vector %s (provider=%s model=%s dim=%d source=%s) ====",
                 spec.name, spec.provider, spec.model, spec.dim, spec.source_field)

        # extraer textos fuente
        texts: List[Optional[str]] = [r.get(spec.source_field) for r in rows]
        # calcular embeddings
        vecs: List[Optional[List[float]]] = embed_texts(texts, spec.__dict__)

        # sanity y métricas
        present = sum(1 for v in vecs if v is not None)
        log.info("Embeddings listos: %d/%d", present, len(vecs))
        if present == 0:
            log.warning("No se generaron vectores para %s; salteando upsert.", spec.name)
            continue

        # upsert en lotes para no exceder tamaños
        for idx_range in _batched_idx(n, args.batch):
            batch_rows = [rows[i] for i in idx_range]
            batch_vecs = {spec.name: [vecs[i] for i in idx_range]}
            upsert_or_update_vectors(settings.collection_name, batch_rows, batch_vecs)

        log.info("OK: subido vector %s", spec.name)


def _batched_idx(n: int, batch: int):
    for i in range(0, n, batch):
        yield range(i, min(i + batch, n))


if __name__ == "__main__":
    main()
