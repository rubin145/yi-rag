import uuid
import html
import random
from typing import Any, Dict, List, Optional

def _make_session_id() -> str:
    return str(uuid.uuid4())

def _render_hits_html(hits: List[Dict[str, Any]], include_yi: bool = True) -> str:
    if not hits:
        return "<p><em>No results.</em></p>"
    out: List[str] = []
    for i, h in enumerate(hits):
        doc_idx = i
        score = h.get("score")
        score_fmt = f"{score:.3f}" if isinstance(score, (int, float)) else "0.000"
        title_en = h.get("title_en") or ""
        title_yi = h.get("title_yi") or ""
        title_lat = h.get("title_yi_latin") or ""
        author = h.get("author") or ""
        year = h.get("year") or ""
        page = h.get("page")
        book_id = h.get("book_id") or ""
        yi_text = h.get("yi_text") or ""
        tr_en = h.get("tr_en_text") or ""
        ocr_source_url = h.get("ocr_source_url")
        header_title = title_lat or title_yi or title_en or f"Doc {i+1}"
        ocr_link = ""
        if ocr_source_url and page:
            page_ocr_url = ocr_source_url[:-1] + str(page)
            ocr_link = f" | <a href='{page_ocr_url}' target='_blank'>OCR</a>"
        elif book_id and page:
            page_ocr_url = f"https://ocr.yiddishbookcenter.org/text/{book_id}/page/{page}"
            ocr_link = f" | <a href='{page_ocr_url}' target='_blank'>OCR</a>"
        header = (
            f"<span style='font-size:18px; font-weight:bold; color:#2563eb;'>{doc_idx}</span> | "
            f"<b>{html.escape(header_title)}</b> {html.escape(author)} {f'({year})' if year else ''} | "
            f"[{html.escape(book_id)}] | p.{page} | score {score_fmt}{ocr_link}"
            if page is not None and book_id
            else f"<span style='font-size:18px; font-weight:bold; color:#2563eb;'>{doc_idx}</span> | {html.escape(header_title)}"
        )
        title_section = ""
        if title_yi or title_en:
            title_section = f"""
<div style='background:#f8f9fa; padding:8px; margin:4px 0; border-radius:4px; border:1px solid #e9ecef;'>
{f"<div style='font-size:18px; text-align:left; margin-bottom:4px;'>{html.escape(title_yi)}</div>" if title_yi else ""}
{f"<div style='font-size:18px; color:#666; text-align:left;'>{html.escape(title_en)}</div>" if title_en else ""}
</div>
"""
        body_parts: List[str] = []
        if title_section:
            body_parts.append(title_section)
        if tr_en:
            body_parts.append(f"<div style='margin-top:4px;'><div style='font-size:11px; color:#e67e22; margin-bottom:2px;'>⚠️ AI generated translation. Need to check with expert.</div><strong>EN</strong><br><pre>{html.escape(tr_en)}</pre></div>")
        if include_yi and yi_text:
            body_parts.append(f"<div style='margin-top:6px;'><strong>YI</strong><br><pre>{html.escape(yi_text)}</pre></div>")
        body = "".join(body_parts) if body_parts else "<em>(empty)</em>"
        out.append(
            f"""
<details>
  <summary>{header}</summary>
  <div style='padding:6px 0 4px 4px; border-left:3px solid #ddd; margin:4px 0 0 0;'>
    {body}
    <div style='margin-top:6px; font-size:11px; color:#666;'>Doc index: <code>{doc_idx}</code></div>
  </div>
</details>
"""
        )
    return "\n".join(out)

def _format_citations(cits: List[Dict[str, Any]], hits: List[Dict[str, Any]] = None) -> str:
    if not cits:
        return ""
    hit_lookup = {}
    if hits:
        for h in hits:
            key = f"{h.get('book_id')}_{h.get('page')}_{h.get('chunk_idx')}"
            hit_lookup[key] = h
    chips: List[str] = []
    for c in cits:
        page = c.get('page', '')
        chunk_idx = c.get('chunk_idx', '')
        book_id = c.get('book_id', '')
        hit_key = f"{book_id}_{page}_{chunk_idx}"
        hit_data = hit_lookup.get(hit_key, {})
        ocr_source_url = c.get('ocr_source_url') or hit_data.get('ocr_source_url')
        title_en = c.get('title_en') or hit_data.get('title_en', '')
        title_yi = c.get('title_yi') or hit_data.get('title_yi', '')
        author = c.get('author') or hit_data.get('author', '')
        year = c.get('year') or hit_data.get('year', '')
        if not ocr_source_url and book_id:
            ocr_source_url = f"https://ocr.yiddishbookcenter.org/text/{book_id}/page/1"
        title_display = title_en or title_yi or book_id
        author_year = f"{author} ({year})" if author and year else (author or str(year) if year else "")
        citation_text = f"[{book_id}] {title_display} — {author_year} p.{page} #{chunk_idx}".strip(" — ")
        if ocr_source_url and page:
            page_ocr_url = ocr_source_url[:-1] + str(page)
            chips.append(f"[<a href='{page_ocr_url}' target='_blank'>{html.escape(citation_text)}</a>]")
        else:
            chips.append(f"[{citation_text}]")
    return " ".join(chips)

def get_random_defaults():
    raw_alpha = random.uniform(0.4, 0.6)
    alpha_q = round(raw_alpha / 0.05) * 0.05
    return {"k": random.randint(8, 12), "score_threshold": round(random.uniform(0.1, 0.4), 2), "alpha": max(0.0, min(1.0, alpha_q))}


