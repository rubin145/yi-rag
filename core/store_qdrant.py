# core/store_qdrant.py
from __future__ import annotations

import logging
from uuid import uuid5, NAMESPACE_URL
from typing import Any, Dict, Iterable, List, Optional

from qdrant_client import QdrantClient, models as qm

from .config import settings

log = logging.getLogger("store_qdrant")

# Increment if payload structure changes in a way that clients should detect
SCHEMA_VERSION = 1


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
    # Text fields for full-text search
    for txt_field in ("title_yi", "title_en", "storage_locations", "yi_text"):
        try:
            client.create_payload_index(collection, field_name=txt_field, field_schema=qm.PayloadSchemaType.TEXT)
        except Exception:
            pass
    
    # Keyword fields for exact matching in filters
    for keyword_field in ("author", "author_yi", "place", "publisher"):
        try:
            client.create_payload_index(collection, field_name=keyword_field, field_schema=qm.PayloadSchemaType.KEYWORD)
        except Exception:
            pass
    
    # Subjects as text for partial matching
    try:
        client.create_payload_index(collection, field_name="subjects", field_schema=qm.PayloadSchemaType.TEXT)
    except Exception:
        pass
    try:
        client.create_payload_index(collection, field_name="total_pages", field_schema=qm.PayloadSchemaType.INTEGER)
    except Exception:
        pass
    # New indexes for faster lookups / filtering
    for fname, ftype in [
        ("id", qm.PayloadSchemaType.KEYWORD),           # original chunk string id
        ("page", qm.PayloadSchemaType.INTEGER),
        ("chunk_idx", qm.PayloadSchemaType.INTEGER),
        ("source_file", qm.PayloadSchemaType.KEYWORD),
        ("has_translation", qm.PayloadSchemaType.INTEGER),  # 0/1 flag
    ]:
        try:
            client.create_payload_index(collection, field_name=fname, field_schema=ftype)
        except Exception:
            pass


# ---------------------------
# Helpers
# ---------------------------
def point_id_from_chunk_id(chunk_id: str) -> int:
    # UUID5 determinístico → int64 Qdrant
    return uuid5(NAMESPACE_URL, chunk_id).int % (1 << 63)

import yaml
from qdrant_client import models
from qdrant_client.http.models import UpdateStatus
from tqdm.auto import tqdm
from .schemas import Chunk, SearchFilters


class QdrantStore:
    def __init__(self, collection_name: str, config_path: str = "config/vectors.yaml"):
        self.collection_name = collection_name
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=60,
        )
        with open(config_path, "r") as f:
            self.vector_config = yaml.safe_load(f)
        # Opportunistically ensure critical payload indexes exist (idempotent)
        self.ensure_payload_indexes()

    def ensure_payload_indexes(self):
        """Create missing payload indexes needed for filtering. Safe to call multiple times."""
        idx_specs = [
            ("author", qm.PayloadSchemaType.KEYWORD),
            ("author_yi", qm.PayloadSchemaType.KEYWORD),
            ("place", qm.PayloadSchemaType.KEYWORD),
            ("publisher", qm.PayloadSchemaType.KEYWORD),
            ("subjects", qm.PayloadSchemaType.TEXT),
            ("subjects_tokens", qm.PayloadSchemaType.KEYWORD),
            ("year", qm.PayloadSchemaType.INTEGER),
            ("book_id", qm.PayloadSchemaType.KEYWORD),
        ]
        for field, schema in idx_specs:
            try:
                self.client.create_payload_index(self.collection_name, field_name=field, field_schema=schema)
            except Exception:
                pass

    def recreate_collection(self):
        log.info(f"Recreating Qdrant collection: {self.collection_name}")
        
        # Dynamically build named vectors from config
        vectors_config = {
            cfg["name"]: models.VectorParams(
                size=cfg["dim"],
                distance=models.Distance.COSINE,
            )
            for cfg in self.vector_config.get("vectors", [])
        }

        # Add sparse vector config
        sparse_vectors_config = {
            "sparse_vector_yi": models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=False,
                )
            )
        }

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
        log.info("Collection recreated successfully.")

    def index_chunks(self, chunks: List[Chunk], batch_size: int = 100):
        log.info(f"Indexing {len(chunks)} chunks into '{self.collection_name}'...")
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Indexing batches"):
            batch_chunks = chunks[i : i + batch_size]
            points_to_upsert = []

            for chunk in batch_chunks:
                # The payload is everything in the chunk EXCEPT the vector fields.
                # This ensures the payload is always up-to-date with the schema.
                payload = chunk.model_dump(
                    exclude={"vectors", "sparse_vector_yi"},
                    exclude_none=True,
                )
                # Expand subjects -> subjects_tokens list (split on comma/semicolon)
                subj_raw = payload.get("subjects")
                if subj_raw and isinstance(subj_raw, str):
                    tokens = [t.strip() for t in subj_raw.replace(";", ",").split(",") if t.strip()]
                    if tokens:
                        payload["subjects_tokens"] = tokens
                # Augment payload with schema & convenience flags
                payload["schema_version"] = SCHEMA_VERSION
                payload["has_translation"] = 1 if payload.get("tr_en_text") else 0
                # Mark has_vectors if we have any dense or sparse vector
                payload["has_vectors"] = 1 if (chunk.vectors or chunk.sparse_vector_yi) else 0

                # The vectors are passed separately.
                point_vectors = {}
                if chunk.vectors:
                    point_vectors.update(chunk.vectors)
                if chunk.sparse_vector_yi:
                    point_vectors["sparse_vector_yi"] = models.SparseVector(
                        indices=chunk.sparse_vector_yi["indices"],
                        values=chunk.sparse_vector_yi["values"],
                    )

                points_to_upsert.append(
                    models.PointStruct(
                        id=point_id_from_chunk_id(chunk.id),  # deterministic int64
                        payload=payload,
                        vector=point_vectors,
                    )
                )

            if points_to_upsert:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points_to_upsert,
                    wait=True,
                )
        log.info("Indexing complete.")

    def update_chunk_translation(self, chunk_id: str, tr_en_text: str, title_en: str) -> bool:
        """Updates a chunk with its English translation and title."""
        payload = {
            "tr_en_text": tr_en_text,
            "title_en": title_en,
        }
        res = self.client.set_payload(
            collection_name=self.collection_name,
            payload=payload,
            points=[point_id_from_chunk_id(chunk_id)],
            wait=True,
        )
        if res.status != UpdateStatus.COMPLETED:
            log.warning(f"Failed to update chunk {chunk_id} with translation.")
            return False
        return True

    def get_all_chunk_ids(self) -> List[str]:
        """Fetches all chunk IDs from the collection."""
        log.info("Fetching all chunk IDs...")
        all_points, _ = self.client.scroll(
            collection_name=self.collection_name,
            with_payload=False,
            with_vectors=False,
            limit=10_000,  # Adjust if you have more chunks
        )
        return [p.id for p in all_points]

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieves a single chunk by its ID."""
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id_from_chunk_id(chunk_id)],
            with_payload=True,
        )
        if not points:
            return None
        
        payload = points[0].payload
        payload['id'] = points[0].id # Add the id back for Pydantic model
        return Chunk(**payload)



def _batched_idx(n: int, batch: int) -> Iterable[range]:
    for i in range(0, n, batch):
        yield range(i, min(i + batch, n))


def payload_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Construye el payload para Qdrant desde la fila original.

    Excluye campos de vectores y valores None. Mantiene metadatos útiles.
    """
    exclude_keys = {"vectors", "sparse_vector_yi"}
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if k in exclude_keys or v is None:
            continue
        out[k] = v
    # Ensure schema flags present
    out.setdefault("schema_version", SCHEMA_VERSION)
    out.setdefault("has_translation", 1 if out.get("tr_en_text") else 0)
    dense_has = False
    if isinstance(row.get("vectors"), dict):
        vec_dict = row.get("vectors", {})
        dense_has = any(isinstance(v, list) and v for v in vec_dict.values())
    sparse_has = bool(row.get("sparse_vector_yi")) and isinstance(row.get("sparse_vector_yi"), dict)
    out.setdefault("has_vectors", 1 if (dense_has or sparse_has) else 0)
    return out


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
