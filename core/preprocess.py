# core/preprocess.py
from __future__ import annotations

import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Literal

import json
import pandas as pd

# opcionales (solo si usás incremental="append")
import pyarrow as pa
import pyarrow.parquet as pq

from tqdm.auto import tqdm

from .config import settings
from .schemas import Chunk
from .translate import Translator
from .utils import extract_title, parse_filename_parts, split_pages


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("preprocess")



# -----------------------------------------------------------------------------
# Splitters
# -----------------------------------------------------------------------------
def split_by_tokens(text: str, max_tokens: int, overlap: int) -> List[str]:
    """
    Divide 'text' en ventanas de 'max_tokens' tokens con solapamiento 'overlap'.
    Usa cl100k_base (general-purpose). Si tiktoken no está disponible, cae a chars.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        toks = enc.encode(text or "")
        if not toks:
            return []
        step = max(1, max_tokens - overlap)
        out: List[str] = []
        for start in range(0, len(toks), step):
            end = min(start + max_tokens, len(toks))
            piece = enc.decode(toks[start:end]).strip()
            if piece:
                out.append(piece)
            if end == len(toks):
                break
        return out
    except Exception:
        # Fallback aproximado: 1 token ~ 3 chars (heurístico)
        return _split_by_chars(text, max_tokens * 3, overlap * 3)


def _split_by_chars(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    s = (text or "").strip()
    if not s:
        return []
    step = max(1, max_chars - overlap_chars)
    out: List[str] = []
    for i in range(0, len(s), step):
        seg = s[i : i + max_chars].strip()
        if seg:
            out.append(seg)
        if i + max_chars >= len(s):
            break
    return out


# -----------------------------------------------------------------------------
# Metadata CSV (opcional)
# -----------------------------------------------------------------------------
def load_book_metadata(path_csv: str = "data/metadata.csv") -> Dict[str, Dict]:
    """
    CSV opcional con columnas: book_id, title_yi, title_en, author, year
    Devuelve dict[book_id] = {title_yi?, title_en?, author?, year?}
    """
    if not os.path.exists(path_csv):
        log.info("No hay metadata.csv; se continúa sin metadata externa.")
        return {}
    df = pd.read_csv(path_csv).fillna("")
    meta: Dict[str, Dict] = {}
    for _, r in df.iterrows():
        bid = str(r.get("book_id", "")).strip()
        if not bid:
            continue
        meta[bid] = {
            "title_yi": (str(r.get("title_yi", "")).strip() or None),
            "title_en": (str(r.get("title_en", "")).strip() or None),
            "author": (str(r.get("author", "")).strip() or None),
            "year": (int(r["year"]) if str(r.get("year", "")).strip().isdigit() else None),
        }
    log.info(f"Metadata externa cargada: {len(meta)} registros.")
    return meta


# -----------------------------------------------------------------------------
# Construcción de chunks (con contadores)
# -----------------------------------------------------------------------------
def make_chunks_from_pages(
    pages: List[Tuple[int, str]],
    book_id: str,
    title_yi: Optional[str],
    title_en: Optional[str],
    author: Optional[str],
    year: Optional[int],
    translator: Translator,
) -> Tuple[List[Chunk], Dict[str, int]]:
    chunks: List[Chunk] = []

    # contadores locales
    stat_pages = 0
    stat_subchunks = 0
    tr_attempt = 0
    tr_ok = 0
    tr_fail = 0

    for page_num, page_text in pages:
        page_text = (page_text or "").strip()
        if not page_text:
            continue
        stat_pages += 1

        if settings.chunk_strategy == "page":
            subchunks = [page_text]
        elif settings.chunk_strategy == "tokens":
            subchunks = split_by_tokens(page_text, settings.chunk_tokens, settings.chunk_overlap)
        else:  # "chars"
            subchunks = _split_by_chars(page_text, settings.chunk_chars, settings.char_overlap)

        # Normalizar y filtrar vacíos
        subchunks = [s.strip() for s in subchunks if s and s.strip()]
        if not subchunks:
            continue

        for idx, sub in enumerate(subchunks):
            tr_en = None
            if settings.translate_chunks_en and settings.tr_provider != "none":
                tr_attempt += 1
                tr_en = translator.translate_to_en(sub)
                if tr_en:
                    tr_ok += 1
                else:
                    tr_fail += 1

            chunks.append(
                Chunk(
                    id=f"{book_id}:{page_num}:{idx}",
                    book_id=book_id,
                    title_yi=title_yi,
                    title_en=title_en,
                    author=author,
                    year=year,
                    page=page_num,
                    chunk_idx=idx,
                    yi_text=sub,
                    tr_en_text=tr_en,
                    source_file=f"{book_id}.txt",
                )
            )
            stat_subchunks += 1

    stats = {
        "pages": stat_pages,
        "subchunks": stat_subchunks,
        "chunks": stat_subchunks,  # 1:1 con subchunks
        "tr_attempt": tr_attempt,
        "tr_ok": tr_ok,
        "tr_fail": tr_fail,
    }
    return chunks, stats


def process_txt_file(
    path_txt: str,
    meta_lookup: Dict[str, Dict],
    translator: Translator,
) -> Tuple[List[Chunk], Dict[str, int], str]:
    # Leer crudo
    with open(path_txt, "r", encoding="utf-8") as f:
        raw = f.read()

    # book_id (sufijo) y posible título desde el filename (prefijo)
    filename_no_ext = os.path.splitext(os.path.basename(path_txt))[0]
    book_id_from_name, title_guess_from_name = parse_filename_parts(filename_no_ext)

    # Separar páginas y detectar Title: dentro del documento (suele ser YI)
    pages = split_pages(raw)
    title_in_doc_yi = extract_title(raw)

    # Metadata externa (si existe)
    ext = meta_lookup.get(book_id_from_name, {}) if meta_lookup else {}
    # Prioridad de títulos: metadata > Title: del doc > prefijo del filename
    title_yi = ext.get("title_yi") or title_in_doc_yi or title_guess_from_name
    title_en = ext.get("title_en")

    # Si falta title_en y queremos traducir títulos, hacelo ahora (barato y útil)
    if (
        settings.translate_titles_en
        and not title_en
        and title_yi
        and settings.tr_provider != "none"
    ):
        title_en = translator.translate_to_en(title_yi)

    author = ext.get("author")
    year = ext.get("year")

    chunks, stats = make_chunks_from_pages(
        pages=pages,
        book_id=book_id_from_name,
        title_yi=title_yi,
        title_en=title_en,
        author=author,
        year=year,
        translator=translator,
    )

    return chunks, stats, book_id_from_name


# -----------------------------------------------------------------------------
# Utilidad: reordenar columnas
# -----------------------------------------------------------------------------
def _sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Convierte 'extra' (dict) en 'extra_json' (string JSON) para evitar structs vacíos.
    - Asegura esquema estable entre libros, incluso si 'extra' está vacío.
    """
    if "extra" in df.columns:
        df["extra_json"] = df["extra"].apply(lambda x: json.dumps(x or {}))
        df = df.drop(columns=["extra"])
    else:
        # si por algún motivo no está, garantizamos la columna para esquema estable
        df["extra_json"] = None
    return df

def _reorder_cols(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "id","book_id","title_yi","title_en","author","year",
        "page","chunk_idx","yi_text","tr_en_text","source_file","extra_json",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]



# -----------------------------------------------------------------------------
# Runner principal (con modos de guardado)
# -----------------------------------------------------------------------------
def run_preprocess(
    input_dir: str = "data/raw",
    output_path: str = "data/processed/chunks.parquet",
    metadata_csv: str = "data/metadata.csv",
    incremental: Literal["none", "book", "append"] = "none",
) -> None:
    """
    Lee .txt de input_dir, genera chunks (page/tokens/chars), traduce opcionalmente
    títulos y chunks a EN (según config), y guarda Parquet.
    - incremental="none": un solo parquet al final (output_path).
    - incremental="book": un parquet por libro en data/processed/parts/{book_id}.parquet
    - incremental="append": apendea a output_path usando ParquetWriter (pyarrow).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    meta_lookup = load_book_metadata(metadata_csv)
    translator = Translator()

    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".txt")])
    if not files:
        log.warning("No se encontraron .txt en %s", input_dir)
        return

    # contadores globales
    G_pages = 0
    G_chunks = 0
    G_tr_attempt = 0
    G_tr_ok = 0
    G_tr_fail = 0

    writer: Optional[pq.ParquetWriter] = None
    t_start = time.time()

    try:
        if incremental == "book":
            os.makedirs("data/processed/parts", exist_ok=True)

        all_chunks: List[Chunk] = []  # solo en modo "none"

        for fname in tqdm(files, desc="Libros", unit="libro"):
            t0 = time.time()
            path_txt = os.path.join(input_dir, fname)

            try:
                file_chunks, stats, book_id = process_txt_file(path_txt, meta_lookup, translator)
            except Exception as e:
                log.exception("Fallo procesando %s: %s", fname, e)
                continue

            # acumular contadores globales
            G_pages += stats["pages"]
            G_chunks += stats["chunks"]
            G_tr_attempt += stats["tr_attempt"]
            G_tr_ok += stats["tr_ok"]
            G_tr_fail += stats["tr_fail"]

            # logging por archivo
            log.info(
                "%s → chunks=%d, páginas=%d, subchunks=%d, tr: ok=%d fail=%d (%.1fs)",
                fname,
                stats["chunks"],
                stats["pages"],
                stats["subchunks"],
                stats["tr_ok"],
                stats["tr_fail"],
                time.time() - t0,
            )

            # salida según modo
            if incremental == "book":
                df_book = pd.DataFrame([c.model_dump() for c in file_chunks])
                df_book = _sanitize_for_parquet(df_book)
                df_book = _reorder_cols(df_book)
                out_path = os.path.join("data/processed/parts", f"{book_id}.parquet")
                df_book.to_parquet(out_path, index=False)
            elif incremental == "append":
                df_book = pd.DataFrame([c.model_dump() for c in file_chunks])
                df_book = _sanitize_for_parquet(df_book)
                df_book = _reorder_cols(df_book)
                table = pa.Table.from_pandas(df_book)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)
                writer.write_table(table)
            else:
                all_chunks.extend(file_chunks)

        # escritura final
        if incremental == "none":
            if not all_chunks:
                log.warning("No se generaron chunks.")
                return
            df = pd.DataFrame([c.model_dump() for c in all_chunks])
            df = _sanitize_for_parquet(df_book)
            df = _reorder_cols(df)
            df.to_parquet(output_path, index=False)
            log.info("[DONE] %d chunks → %s (%.1fs)", len(df), output_path, time.time() - t_start)
        else:
            log.info(
                "[DONE] modo=%s total: páginas=%d, chunks=%d, tr_ok=%d tr_fail=%d (%.1fs)",
                incremental, G_pages, G_chunks, G_tr_ok, G_tr_fail, time.time() - t_start
            )

    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    # Permitimos configurar el modo por env var, sin tocar código
    mode = os.getenv("PREPROCESS_INCREMENTAL", "none").lower()
    if mode not in {"none", "book", "append"}:
        mode = "none"
    run_preprocess(incremental=mode)