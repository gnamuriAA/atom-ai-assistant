import re
import html
from typing import List, TYPE_CHECKING, Any, Dict
from bs4 import BeautifulSoup

if TYPE_CHECKING:
    # Only import for type checking, not at runtime
    import importlib
    markdown_to_structural_block = importlib.import_module('markdown-to-structural-block')
    StructuralBlock = markdown_to_structural_block.StructuralBlock


EMPTY_RE = re.compile(r"^\s*$")

def _clean(s: str) -> str:
    return " ".join(s.replace("\xa0", " ").split()).strip()

def _is_meaningful_header_row(cells): 
    return any(c.name.lower() == "th" and _clean(c.get_text(" ", strip=True)) for c in cells)

def _extract_title_if_any(first_tr):
    ths = first_tr.find_all("th", recursive=False)
    if len(ths) == 1 and ths[0].get("colspan"):
        title =_clean(ths[0].get_text(" ", strip=True))
        return title if title else None
    return None

def html_table_to_grid(table_tag):
    """
    Convert <table> to a rectangular grid of cell texts, expanding rowspan/colspan.
    Returns: grid (list of rows), max_cols
    """
    rows = table_tag.find_all("tr", recursive=True)
    grid = []
    span_map = {}  # (row_idx, col_idx) -> remaining span text

    max_cols = 0
    for r_idx, tr in enumerate(rows):
        cells = tr.find_all(["th", "td"], recursive=False)
        row = []
        c_idx = 0

        def fill_spans():
            nonlocal c_idx
            while (r_idx, c_idx) in span_map:
                row.append(span_map[(r_idx, c_idx)])
                c_idx += 1

        fill_spans()

        for cell in cells:
            fill_spans()

            text = _clean(cell.get_text(" ", strip=True))
            rowspan = int(cell.get("rowspan", 1) or 1)
            colspan = int(cell.get("colspan", 1) or 1)

            # place this cell text in the current row for colspan times
            for k in range(colspan):
                row.append(text)
                # for rowspan, place same text into future rows at same columns
                if rowspan > 1:
                    for rr in range(1, rowspan):
                        span_map[(r_idx + rr, c_idx + k)] = text

            c_idx += colspan

        # fill any trailing spans
        fill_spans()

        max_cols = max(max_cols, len(row))
        grid.append(row)

    # normalize row lengths
    for r in grid:
        if len(r) < max_cols:
            r.extend([""] * (max_cols - len(r)))

    return grid, max_cols

def table_to_kv_string(
    html_table_escaped: str,
    *,
    fallback_headers=None,
    include_row_index=True,
    separator=" | "
) -> str:
    
    """
    Convert an escaped HTML table string (&lt;table&gt;...) into
    multi-line text where each row is "Header: Value".
    """
    fallback_headers = [h for h in (fallback_headers or []) if _clean(h)]

    html_table = html.unescape(html_table_escaped)
    soup = BeautifulSoup(html_table, "lxml")
    table = soup.find("table")
    if table is None:
        return ""

    # title detection (optional)
    title = None
    first_tr = table.find("tr")
    if first_tr:
        title = _extract_title_if_any(first_tr)

    # build grid
    grid, ncols = html_table_to_grid(table)

    # choose header row from <th> if meaningful, otherwise fallback
    headers = None
    for tr in table.find_all("tr", recursive=True):
        cells = tr.find_all(["th", "td"], recursive=False)
        if _is_meaningful_header_row(cells):
            # use the texts of that row as headers
            header_texts = [_clean(c.get_text(" ", strip=True)) for c in cells]
            # expand to ncols if needed (colspan not reflected here, but grid is normalized)
            # We'll just pad/truncate.
            if len(header_texts) < ncols:
                header_texts += [""] * (ncols - len(header_texts))
            headers = header_texts[:ncols]
            break

    if headers is None or all(not _clean(h) for h in headers):
        # fallback to paragraph-derived headers if available
        if fallback_headers:
            headers = (fallback_headers + [""] * ncols)[:ncols]
        else:
            headers = [f"COL_{i+1}" for i in range(ncols)]

    # row-wise kv output
    lines = []
    if title:
        lines.append(f"TABLE: {title}")

    # Heuristic: skip header-only row if it matches headers
    # We'll compare first grid row to headers for exact match; if close, skip.
    start_row = 0
    if grid:
        first = [_clean(x) for x in grid[0]]
        hdr = [_clean(x) for x in headers]
        if first == hdr:
            start_row = 1

    row_counter = 1
    for r in grid[start_row:]:
        kvs = []
        for h, v in zip(headers, r):
            h = _clean(h)
            v = _clean(v)
            if not h and not v:
                continue
            if not h:
                h = "VALUE"
            if v:
                kvs.append(f"{h}: {v}")
        if not kvs:
            continue

        prefix = f"ROW {row_counter}: " if include_row_index else ""
        lines.append(prefix + separator.join(kvs))
        row_counter += 1

    return "\n".join(lines).strip()

def extract_column_headers_from_context(paragraph_text: str, max_headers=12):
    """
    Given a paragraph block text (escaped markdown), try to extract column headers.
    Works well for your pattern where each header is on its own line.
    """
    lines = [ln.strip() for ln in paragraph_text.splitlines() if ln.strip()]
    # Keep lines that look like labels: mostly uppercase words, not too long.
    candidates = []
    for ln in lines:
        # "mostly uppercase" heuristic
        alpha = [c for c in ln if c.isalpha()]
        if not alpha:
            continue
        upper_ratio = sum(1 for c in alpha if c.isupper()) / max(1, len(alpha))
        if upper_ratio >= 0.75 and len(ln) <= 60:
            candidates.append(ln)

    # If we found a reasonable list, return it
    if 2 <= len(candidates) <= max_headers:
        return candidates
    return []

def structural_blocks_to_docs_with_table_strings(structural_blocks: List[Any], base_metadata: Dict[str, Any]) -> List[Any]:
    """
    structural_blocks: list of StructuralBlock objects
    base_metadata: headers metadata from MarkdownHeaderTextSplitter
    Returns: List of StructuralBlock objects with enhanced table processing
    
    Note: Uses duck typing to avoid circular imports
    """
    # Import StructuralBlock at runtime to avoid circular import
    import importlib
    markdown_to_structural_block = importlib.import_module('markdown-to-structural-block')
    StructuralBlock = markdown_to_structural_block.StructuralBlock
    
    out = []
    last_paragraph_text = None

    for idx, b in enumerate(structural_blocks):
        # Update meta with base_metadata
        b.meta.update({
            **base_metadata,
            "block_index": idx
        })
        
        if b.kind == "paragraph":
            last_paragraph_text = b.raw_escaped
            out.append(b)

        elif b.kind == "html_table":
            # derive fallback headers from the paragraph above
            fallback_headers = extract_column_headers_from_context(last_paragraph_text or "")
            kv_text = table_to_kv_string(
                b.raw_escaped,
                fallback_headers=fallback_headers
            )
            
            # Create new StructuralBlock with kv_text as raw_unescaped for better embeddings
            enhanced_block = StructuralBlock(
                kind=b.kind,
                start_line=b.start_line,
                end_line=b.end_line,
                raw_escaped=b.raw_escaped,
                raw_unescaped=kv_text if kv_text else b.raw_unescaped,
                meta={
                    **b.meta,
                    "table_raw_escaped": b.raw_escaped,
                    "table_headers_used": fallback_headers,
                    "table_kv_text": kv_text,
                }
            )
            out.append(enhanced_block)
        else:
            # other blocks: code, lists, etc.
            out.append(b)

    return out