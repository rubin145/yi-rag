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
    
    i = 0
    while i < len(s):
        # Ideal end position
        ideal_end = min(i + max_chars, len(s))
        if ideal_end >= len(s):
            seg = s[i:].strip()
            if seg:
                out.append(seg)
            break

        # Final snapping: allow bigger window (more tolerant)
        search_window = min(300, max_chars // 3)  # hasta 300 chars o 1/3
        actual_end = _find_natural_break(s, ideal_end, search_window, prefer="end")

        # Si el chunk resultante es demasiado chico (<50% del target),
        # forzamos corte duro en ideal_end
        if actual_end - i < max_chars * 0.5:
            actual_end = ideal_end

        seg = s[i:actual_end].strip()
        if seg:
            out.append(seg)

        # Next start = actual_end - overlap
        raw_next_start = max(0, actual_end - overlap_chars)
        if raw_next_start > 0 and raw_next_start < len(s):
            # Start snapping: smaller window (menos tolerancia)
            search_window_start = min(80, overlap_chars)
            next_start = _find_natural_break(s, raw_next_start, search_window_start, prefer="start")
        else:
            next_start = raw_next_start

        # Evitar loops infinitos
        i = max(i + 1, next_start)
    
    return out


def _find_natural_break(text: str, ideal_pos: int, search_window: int, prefer: str) -> int:
    """
    Find natural break near ideal_pos.
    prefer = "end" → se usa para el final de chunk
    prefer = "start" → se usa para el inicio del próximo chunk
    Priority: newline > sentence punctuation > space
    """
    start_search = max(0, ideal_pos - search_window)
    end_search = min(len(text), ideal_pos + search_window)

    # Orden de direcciones: para finales, priorizar atrás; para inicios, priorizar adelante
    directions = [-1, 1] if prefer == "end" else [1, -1]

    # 1. Buscar newline o sentence punctuation
    for distance in range(search_window + 1):
        for direction in directions:
            pos = ideal_pos + direction * distance
            if pos < start_search or pos >= end_search:
                continue
            if pos >= len(text) or pos < 0:
                continue

            char = text[pos]

            # Newline (párrafo)
            if char == '\n':
                # saltar espacios posteriores
                next_pos = pos + 1
                while next_pos < len(text) and text[next_pos] in ' \n\t':
                    next_pos += 1
                return min(next_pos, len(text))

            # Puntuación fuerte
            if char in '.!?;:״׳':
                next_pos = pos + 1
                while next_pos < len(text) and text[next_pos] in ' \t':
                    next_pos += 1
                return min(next_pos, len(text))

    # 2. Buscar espacios
    for distance in range(search_window + 1):
        for direction in directions:
            pos = ideal_pos + direction * distance
            if pos < start_search or pos >= end_search or pos < 0 or pos >= len(text):
                continue
            if text[pos] == ' ':
                return min(pos + 1, len(text))

    # 3. Fallback: hard break
    return min(ideal_pos, len(text))


# -----------------------------------------------------------------------------
# Limpieza de páginas (nuevo)
# -----------------------------------------------------------------------------
def _clean_page_text(page_text: str) -> str:
    import re
    if not page_text:
        return ""

    text = page_text

    # 1. Quitar líneas que sean SOLO números (número de página en OCR)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # 1bis. Número de página al FINAL de la línea (p.ej. "... װעמען עס 183\n")
    # OJO: restringido a 1–3 dígitos y SOLO si están solos al final de línea
    #text = re.sub(r'(?m)(?<=\S)\s(?<!\d)(\d{1,3})\s*$', '', text)

    # 2. Quitar banners estilo “--- Page 123 ---”
    text = re.sub(r'-{2,}\s*Page\s*\d+\s*-{2,}', '', text, flags=re.IGNORECASE)

    # 2bis. Banner estilo “— Page 188 —” además de "---"
    text = re.sub(r'[—-]{2,}\s*Page\s*\d+\s*[—-]{2,}\s*', '', text, flags=re.IGNORECASE)

    # 3. Unir guiones de fin de línea (palabras partidas en OCR)
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    # 4. Normalizar saltos múltiples → como máximo dos \n
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# -----------------------------------------------------------------------------
# Metadata CSV (opcional)
# -----------------------------------------------------------------------------
def load_book_metadata(path_csv: str = "data/corpus_metadata.csv") -> Dict[str, Dict]:
    """Carga metadata de libros.

    IMPORTANTE (corrección): la columna CSV 'title' contiene el título Yiddish romanizado
    (latin) – NO una traducción al inglés. Antes se asignaba erróneamente a 'title_en'.
    Ahora:
      - 'title'  -> title_yi_latin
      - 'title_yi' -> título en alfabeto hebreo
      - title_en queda None (se podrá poblar luego por traducción).

    También se incorporan ocr_source_url y metadata_url que antes se ignoraban.
    """
    if not os.path.exists(path_csv):
        log.info("No hay corpus_metadata.csv; se continúa sin metadata externa.")
        return {}
    df = pd.read_csv(path_csv).fillna("")
    meta: Dict[str, Dict] = {}
    for _, r in df.iterrows():
        bid = str(r.get("book_id", "")).strip()
        if not bid:
            continue
        title_latin = (str(r.get("title", "")).strip() or None)
        meta[bid] = {
            "title_yi_latin": title_latin,
            "title_yi": (str(r.get("title_yi", "")).strip() or None),
            # No asignamos más title_en aquí; se traducirá posteriormente.
            "author": (str(r.get("author", "")).strip() or None),
            "author_yi": (str(r.get("author_yi", "")).strip() or None),
            "year": (int(r["publication_year"]) if str(r.get("publication_year", "")).strip().isdigit() else None),
            "place": (str(r.get("place", "")).strip() or None),
            "publisher": (str(r.get("publisher", "")).strip() or None),
            "storage_locations": (str(r.get("storage_locations", "")).strip() or None),
            "subjects": (str(r.get("subjects", "")).strip() or None),
            "pages": (int(r["pages"]) if str(r.get("pages", "")).strip().isdigit() else None),
            "source_url": (str(r.get("source_url", "")).strip() or None),
            "ocr_source_url": (str(r.get("ocr_source_url", "")).strip() or None),
            "metadata_url": (str(r.get("metadata_url", "")).strip() or None),
        }
    log.info(f"Metadata externa cargada: {len(meta)} registros (con title_yi_latin y URLs).")
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
    author_yi: Optional[str],
    year: Optional[int],
    place: Optional[str],
    publisher: Optional[str],
    storage_locations: Optional[str],
    subjects: Optional[str],
    total_pages: Optional[int],
    source_url: Optional[str],
    ocr_source_url: Optional[str],
    metadata_url: Optional[str],
    translator: Translator,
) -> Tuple[List[Chunk], Dict[str, int]]:
    chunks: List[Chunk] = []

    # contadores locales
    stat_pages = 0
    stat_subchunks = 0
    tr_attempt = 0
    tr_ok = 0
    tr_fail = 0

    # Contar total de páginas con contenido para progreso
    pages_with_content = [p for p in pages if (p[1] or "").strip()]
    total_pages_to_process = len(pages_with_content)
    
    log.info(f"📖 Procesando libro {book_id}: {total_pages_to_process} páginas con contenido")

    for page_idx, (page_num, page_text) in enumerate(pages, 1):
        page_text = (page_text or "").strip()
        if not page_text:
            continue
        stat_pages += 1

        # 🔹 limpieza OCR pero preservando metadata de página
        page_text = _clean_page_text(page_text)
        if not page_text:
            continue

        if settings.chunk_strategy == "page":
            subchunks = [page_text]
        elif settings.chunk_strategy == "tokens":
            subchunks = split_by_tokens(page_text, settings.chunk_tokens, settings.chunk_tokens_overlap)
        else:  # "chars"
            subchunks = _split_by_chars(page_text, settings.chunk_chars, settings.chunk_chars_overlap)

        # Normalizar y filtrar vacíos
        subchunks = [s.strip() for s in subchunks if s and s.strip()]
        if not subchunks:
            continue

        # Log progreso de páginas cada 10 páginas o si es la última
        if stat_pages % 10 == 0 or stat_pages == total_pages_to_process:
            log.info(f"   📄 Página {stat_pages}/{total_pages_to_process} (#{page_num}) → {len(subchunks)} chunks")

        for idx, sub in enumerate(subchunks):
            tr_en = None
            tr_en_metadata = {}
            if settings.translate_chunks_en and settings.tr_provider != "none":
                tr_attempt += 1
                tr_en = translator.translate_to_en(sub)
                if tr_en:
                    tr_ok += 1
                    tr_en_metadata = {
                        "provider": settings.tr_provider,
                        "model": translator.model,
                        "system_prompt": translator.get_system_prompt()
                    }
                else:
                    tr_fail += 1

            chunks.append(
                Chunk(
                    id=f"{book_id}:{page_num}:{idx}",
                    book_id=book_id,
                    title_yi=title_yi,
                    title_en=title_en,
                    author=author,
                    author_yi=author_yi,
                    year=year,
                    place=place,
                    publisher=publisher,
                    storage_locations=storage_locations,
                    subjects=subjects,
                    total_pages=total_pages,
                    source_url=source_url,
                    ocr_source_url=ocr_source_url,
                    metadata_url=metadata_url,
                    page=page_num,
                    chunk_idx=idx,
                    yi_text=sub,
                    tr_en_text=tr_en,
                    tr_en_metadata=tr_en_metadata,
                    source_file=f"{book_id}.txt",
                )
            )
            stat_subchunks += 1

            # Log progreso de chunks cada 50 chunks
            if stat_subchunks % 50 == 0:
                log.info(f"   🔗 {stat_subchunks} chunks procesados, traducciones: {tr_ok}/{tr_attempt}")

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

    # Separar páginas - no need to extract title since metadata is robust
    pages = split_pages(raw)

    # Use metadata directly (no extraction or translation needed)
    ext = meta_lookup.get(book_id_from_name, {}) if meta_lookup else {}
    title_yi_latin = ext.get("title_yi_latin") or title_guess_from_name
    title_yi = ext.get("title_yi") or title_guess_from_name  # fallback if missing
    title_en = ext.get("title_en")  # likely None now; translation script will fill
    author = ext.get("author")
    author_yi = ext.get("author_yi")
    year = ext.get("year")
    place = ext.get("place")
    publisher = ext.get("publisher")
    storage_locations = ext.get("storage_locations")
    subjects = ext.get("subjects")
    total_pages = ext.get("pages")  # total pages in book
    source_url = ext.get("source_url")
    ocr_source_url = ext.get("ocr_source_url")
    metadata_url = ext.get("metadata_url")

    chunks, stats = make_chunks_from_pages(
        pages=pages,
        book_id=book_id_from_name,
    title_yi_latin=title_yi_latin,
    title_yi=title_yi,
        title_en=title_en,
        author=author,
        author_yi=author_yi,
        year=year,
        place=place,
        publisher=publisher,
        storage_locations=storage_locations,
        subjects=subjects,
        total_pages=total_pages,
        source_url=source_url,
        ocr_source_url=ocr_source_url,
        metadata_url=metadata_url,
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
    - Garantiza tipos consistentes para todas las columnas.
    """
    if "extra" in df.columns:
        df["extra_json"] = df["extra"].apply(lambda x: json.dumps(x or {}))
        df = df.drop(columns=["extra"])
    else:
        # si por algún motivo no está, garantizamos la columna para esquema estable
        df["extra_json"] = pd.Series([json.dumps({})] * len(df), dtype=str)

    if "tr_en_metadata" in df.columns:
        df["tr_en_metadata_json"] = df["tr_en_metadata"].apply(lambda x: json.dumps(x or {}))
        df = df.drop(columns=["tr_en_metadata"])
    else:
        df["tr_en_metadata_json"] = pd.Series([json.dumps({})] * len(df), dtype=str)

    # Garantizar tipos consistentes para evitar schema mismatch en append mode
    string_cols = [
        "id", "book_id", "title_yi_latin", "title_yi", "title_en", "author", "author_yi", 
        "place", "publisher", "storage_locations", "subjects", "source_url", 
        "ocr_source_url", "metadata_url", "yi_text", "tr_en_text", "source_file", 
        "extra_json", "tr_en_metadata_json"
    ]
    
    for col in string_cols:
        if col in df.columns:
            # Convertir None/NaN a string vacío para tener schema consistente
            df[col] = df[col].astype(str)
            df[col] = df[col].replace("nan", "").replace("None", "")
    
    # Garantizar tipos numéricos
    numeric_cols = ["year", "total_pages", "page", "chunk_idx"]
    for col in numeric_cols:
        if col in df.columns:
            # Convertir a nullable int para manejar NaN correctamente
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    
    return df

def _reorder_cols(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "id","book_id","title_yi_latin","title_yi","title_en","author","author_yi","year",
        "place","publisher","storage_locations","subjects","total_pages",
        "source_url","ocr_source_url","metadata_url",
        "page","chunk_idx","yi_text","tr_en_text","tr_en_metadata_json","source_file","extra_json",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]


# -----------------------------------------------------------------------------
# Runner principal (con modos de guardado)
# -----------------------------------------------------------------------------
def run_preprocess(
    input_dir: str = "data/raw",
    output_path: str = "data/processed/chunks.parquet",
    metadata_csv: str = "data/corpus_metadata.csv",
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
                "📚 %s → ✅ %d chunks (%d páginas, %d subchunks) | 🌐 tr: ✅%d ❌%d | ⏱️%.1fs",
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
                
                try:
                    writer.write_table(table)
                except ValueError as e:
                    if "schema does not match" in str(e):
                        log.error("Schema mismatch detectado. El archivo existente tiene un esquema incompatible.")
                        log.error("Solución: elimina el archivo %s y vuelve a ejecutar, o usa --mode none", output_path)
                        raise ValueError(f"Schema mismatch en append mode. Elimina {output_path} y vuelve a ejecutar.") from e
                    else:
                        raise
            else:
                all_chunks.extend(file_chunks)

        # escritura final
        if incremental == "none":
            if not all_chunks:
                log.warning("No se generaron chunks.")
                return
            df = pd.DataFrame([c.model_dump() for c in all_chunks])
            df = _sanitize_for_parquet(df)
            df = _reorder_cols(df)
            df.to_parquet(output_path, index=False)
            log.info("✅ [DONE] %d chunks → %s (%.1fs)", len(df), output_path, time.time() - t_start)
        else:
            log.info(
                "✅ [DONE] modo=%s | 📄 páginas=%d | 🔗 chunks=%d | 🌐 tr: ✅%d ❌%d | ⏱️%.1fs",
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
