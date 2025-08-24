# core/store_qdrant.py
from __future__ import annotations

import logging
from uuid import uuid5, NAMESPACE_URL
from typing import Any, Dict, Iterable, List, Optional

from qdrant_client import QdrantClient, models as qm

from .config import settings

log = logging.getLogger("store_qdrant")


# ---------------------------
# Client
# ---------------------------
def get_client() -> QdrantClient:
    if not settings.qdrant_url:
        raise ValueError("QDRANT_URL no configurado")
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


# ---------------------------
# Collection / schema
# ---------------------------
def ensure_collection_named_vectors(
    collection: str,
    vector_specs: List[Dict[str, Any]],
    distance: qm.Distance = qm.Distance.COSINE,
    recreate: bool = False,
) -> None:
    """
    Crea (o recrea) la colección con *todos* los named vectors declarados en vector_specs.
    Cada spec debe tener: {name, dim}
    """
    client = get_client()

    vec_cfg: Dict[str, qm.VectorParams] = {}
    for spec in vector_specs:
        name = spec["name"]
        dim = int(spec["dim"])
        vec_cfg[name] = qm.VectorParams(size=dim, distance=distance)

    if recreate:
        log.warning("Recreando colección %s", collection)
        client.recreate_collection(collection_name=collection, vectors_config=vec_cfg)
    else:
        # create si no existe; si existe, no la toca
        try:
            client.get_collection(collection)
            # Podrías validar aquí que las dims/nombres coincidan
            log.info("Colección %s ya existe", collection)
        except Exception:
            log.info("Creando colección %s", collection)
            client.create_collection(collection_name=collection, vectors_config=vec_cfg)

    # índices de payload útiles
    try:
        client.create_payload_index(collection, field_name="book_id", field_schema=qm.PayloadSchemaType.KEYWORD)
    except Exception:
        pass
    try:
        client.create_payload_index(collection, field_name="year", field_schema=qm.PayloadSchemaType.INTEGER)
    except Exception:
        pass
    for txt_field in ("title_yi", "title_en"):
        try:
            client.create_payload_index(collection, field_name=txt_field, field_schema=qm.PayloadSchemaType.TEXT)
        except Exception:
            pass


# ---------------------------
# Helpers
# ---------------------------
def point_id_from_chunk_id(chunk_id: str) -> int:
    # UUID5 determinístico → int64 Qdrant
    return uuid5(NAMESPACE_URL, chunk_id).int % (1 << 63)


def payload_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    # row viene de pandas: dict con las columnas del Chunk
    return {
        "book_id": row.get("book_id"),
        "title_yi": row.get("title_yi"),
        "title_en": row.get("title_en"),
        "author": row.get("author"),
        "year": int(row["year"]) if row.get("year") not in (None, "", "None") else None,
        "page": int(row["page"]) if row.get("page") not in (None, "", "None") else None,
        "chunk_idx": int(row["chunk_idx"]) if row.get("chunk_idx") not in (None, "", "None") else None,
        "yi_text": row.get("yi_text"),
        "tr_en_text": row.get("tr_en_text"),
        "source_file": row.get("source_file"),
        "extra_json": row.get("extra_json"),  # string JSON (seguro para Parquet)
    }


def _batched_idx(n: int, batch: int) -> Iterable[range]:
    for i in range(0, n, batch):
        yield range(i, min(i + batch, n))


# ---------------------------
# Upsert por lotes
# ---------------------------
def upsert_batch_with_named_vectors(
    collection: str,
    rows: List[Dict[str, Any]],
    vectors_by_name: Dict[str, List[Optional[List[float]]]],
) -> None:
    """
    Sube un lote de points.
    - rows: lista de dicts (cada uno es una fila del parquet)
    - vectors_by_name: { vector_name: [vec_or_None_para_cada_row] }
      (alineado con rows)
    Solo incluirá en cada point los named vectors disponibles (no-None).
    """
    client = get_client()
    points: List[qm.PointStruct] = []

    for i, row in enumerate(rows):
        vectors: Dict[str, List[float]] = {}
        for vname, vecs in vectors_by_name.items():
            vec = vecs[i]
            if vec is not None:
                vectors[vname] = vec

        if not vectors:
            # si este row no tiene ningún vector en este paso, lo salteamos
            continue

        pid = point_id_from_chunk_id(row["id"])
        payload = payload_from_row(row)

        points.append(qm.PointStruct(id=pid, vector=vectors, payload=payload))

    if not points:
        return

    client.upsert(collection_name=collection, points=points)

from qdrant_client import models as qm

def upsert_or_update_vectors(
    collection: str,
    rows: list[dict],
    vectors_by_name: dict[str, list[Optional[list[float]]]],
) -> None:
    client = get_client()
    ids = [point_id_from_chunk_id(r["id"]) for r in rows]

    # ¿cuáles ya existen?
    existing = set()
    if ids:
        try:
            got = client.retrieve(collection_name=collection, ids=ids, with_payload=False, with_vectors=False)
            for p in got:
                existing.add(p.id)
        except Exception:
            pass

    idx_update = [i for i, pid in enumerate(ids) if pid in existing]
    idx_insert = [i for i, pid in enumerate(ids) if pid not in existing]

    # A) añadir/actualizar vectores en points existentes (NO toca payload ni otros vectores)
    if idx_update:
        for vname, vecs in vectors_by_name.items():
            batch = [
                qm.PointVectors(id=ids[i], vector={vname: vecs[i]})
                for i in idx_update
                if vecs[i] is not None
            ]
            if batch:
                client.update_vectors(collection_name=collection, points=batch)

    # B) crear points que aún no existen (con los vectores disponibles y payload)
    if idx_insert:
        points = []
        for i in idx_insert:
            vectors = {vn: vectors_by_name[vn][i] for vn in vectors_by_name if vectors_by_name[vn][i] is not None}
            if not vectors:
                continue
            points.append(qm.PointStruct(id=ids[i], vector=vectors, payload=payload_from_row(rows[i])))
        if points:
            client.upsert(collection_name=collection, points=points)
