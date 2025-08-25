import re
from typing import List, Tuple, Optional

# Reconoce cabeceras tipo: --- Page 3 ---
PAGE_RE = re.compile(r"^--- Page (\d+) ---\s*$", re.MULTILINE)
# Reconoce Title: <lo que sea> al inicio del archivo
TITLE_RE = re.compile(r"^Title:\s*(.+)$", re.MULTILINE)

def split_pages(text: str) -> List[Tuple[int, str]]:
    """
    Divide por separadores '--- Page N ---'.
    Devuelve lista de (page_number, page_text).
    Si no encuentra separadores, devuelve [(1, texto entero)].
    """
    if not text:
        return []
    parts = PAGE_RE.split(text)
    if len(parts) <= 1:
        return [(1, text.strip())]
    pages: List[Tuple[int, str]] = []
    # parts = [header, num1, page1, num2, page2, ...]
    for i in range(1, len(parts), 2):
        try:
            page_num = int(parts[i])
        except ValueError:
            continue
        page_text = (parts[i + 1] or "").strip()
        pages.append((page_num, page_text))
    return pages

def extract_title(text: str) -> str | None:
    """
    Busca 'Title: ...' en el documento y devuelve el título si lo encuentra.
    """
    if not text:
        return None
    m = TITLE_RE.search(text)
    return m.group(1).strip() if m else None

def parse_filename_parts(filename_no_ext: str) -> Tuple[str, Optional[str]]:
    """
    Devuelve (book_id, title_yi_from_name)
    Regla: book_id = lo que va DESPUÉS del último '_'
           title_yi_from_name = lo que va ANTES (si existe)
    Ej: 'אבן נגף_nybc212428' -> ('nybc212428', 'אבן נגף')
        'nybc212428'         -> ('nybc212428', None)
    """
    if "_" not in filename_no_ext:
        return filename_no_ext, None
    left, right = filename_no_ext.rsplit("_", 1)
    book_id = right.strip() or filename_no_ext
    title_guess = (left.strip() or None)
    return book_id, title_guess