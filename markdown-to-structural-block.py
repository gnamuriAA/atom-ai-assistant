from dataclasses import dataclass
import html
import re
from typing import Any, Dict, List, Tuple
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode
from html_table_raw_text import structural_blocks_to_docs_with_table_strings

RE_TABLE_OPEN = re.compile(r"<\s*table\b", re.IGNORECASE)
RE_HEADING_TAG = re.compile(r"^h([1-6])$", re.IGNORECASE)
RE_PAGEBREAK = re.compile(r"<!--\s*PageBreak\s*-->", re.IGNORECASE)
RE_PAGENUMBER = re.compile(r'<!--\s*PageNumber\s*=\s*"([^"]+)"\s*-->', re.IGNORECASE)
RE_PAGE_NUMERIC = re.compile(r"\bPage\s*(\d+)\b", re.IGNORECASE)
RE_REV = re.compile(r"^\s*Rev\b", re.IGNORECASE)

def build_heading_path_from_metadata(md: Dict[str, Any], keys=("H1", "H2", "H3", "H4")) -> Tuple[str,...]:
    """ Create a heading path tuple from MarkdownHeaderTextSplitter metadata."""
    out = []
    for k in keys:
        v = md.get(k)
        if v:
            out.append(v)
    return tuple(out)

def slice_lines(lines: List[str], start0: int, end0: int) -> str:
    """ Slice a list of lines[strt0:end0] using 0-based indices and return the joined string."""
    start0 = max(0, start0)
    end0 = min(len(lines), end0)
    return "\n".join(lines[start0:end0]).rstrip("\n")

def node_plaintext(node: SyntaxTreeNode) -> str:
    """ Best-effort plain text extraction from a node. 
        For self-closing tags (html_block, fence), content is available on node.content.
    """
    if node.content:
        return node.content.strip()
    parts: List[str] = []

    def walk(n: SyntaxTreeNode):
        if n.type == "text" and n.content:
            parts.append(n.content)
        for c in n.children:
            walk(c)

    for ch in node.children:
        walk(ch)

    return " ".join(parts).strip()

# ----------------------------
# Structural splitter core
# ----------------------------

@dataclass
class StructuralBlock:
    kind: str 
    start_line: int
    end_line: int
    raw_escaped: str
    raw_unescaped: str
    meta: Dict[str, Any]

@dataclass
class HeadingSection:
    """Groups multiple StructuralBlocks under the same heading for embedding."""
    heading_path: Tuple[str, ...]  # e.g., ('Chapter 1', 'Section 1.1')
    heading_metadata: Dict[str, str]  # e.g., {'H1': 'Chapter 1', 'H2': 'Section 1.1'}
    blocks: List[StructuralBlock]
    combined_text: str  # All block texts combined for embedding
    meta: Dict[str, Any]  # Section-level metadata (doc_name, page_number, etc.)

def structural_split_markdown_chunk(
        md_chunk_raw: str,
        *,
        html_tables_are_escaped: bool = True,
        enable_html_parsing: bool = True,
        include_lists: bool = True,
        include_blockquotes: bool = True,
        full_document_lines: List[str] = None,
        chunk_offset: int = 0
) -> List[StructuralBlock]:
    """
    Split one markdown chunk (already header-scoped) into structural blocks.
    Uses markdown-it-py token stream + SyntaxTreeNode tree.

    Docstring for structural_split_markdown_chunk
    
    :param md_chunk_raw: Description
    :type md_chunk_raw: str
    :param html_tables_are_escaped: Description
    :type html_tables_are_escaped: bool
    :param enable_html_parsing: Description
    :type enable_html_parsing: bool
    :param include_lists: Description
    :type include_lists: bool
    :param include_blockquotes: Description
    :type include_blockquotes: bool
    :return: Description
    :rtype: List[StructuralBlock]
    """

    raw_lines = md_chunk_raw.splitlines()
    md_for_parse = html.unescape(md_chunk_raw) if html_tables_are_escaped else md_chunk_raw

    md = MarkdownIt("commonmark", {"html": enable_html_parsing})
    tokens = md.parse(md_for_parse)
    tree = SyntaxTreeNode(tokens)

    blocks: List[StructuralBlock] = []

    for node in tree.children:
        if not node.block or not node.map:
            continue
        
        start0, end0 = node.map

        raw_escaped = slice_lines(raw_lines, start0, end0)
        raw_unescaped = html.unescape(raw_escaped) if html_tables_are_escaped else raw_escaped

        kind = None
        
        if node.type == "paragraph":
            kind = "paragraph"

        elif node.type == "html_block":
            if RE_TABLE_OPEN.search(raw_unescaped) or (node.content and RE_TABLE_OPEN.search(node.content)):
                kind = "html_table"
            else :
                kind = "html_block"
        elif node.type == "fence":
            kind = "code_fence"
        
        elif include_lists and node.type in ("bullet_list", "ordered_list"):
            kind = "list"
        
        elif include_blockquotes and node.type == "blockquote":
            kind = "blockquote"

        if kind:
            # Use full document lines if available, otherwise use chunk lines
            lines_for_context = full_document_lines if full_document_lines else raw_lines
            position_in_doc = chunk_offset + start0 if full_document_lines else start0
            
            page_ctx = extract_page_context(lines_for_context, position_in_doc)
            blocks.append(StructuralBlock(
                kind=kind,
                start_line=start0 + 1,
                end_line=end0,
                raw_escaped=raw_escaped,
                raw_unescaped=raw_unescaped,
                meta={
                    "page_number": page_ctx["page_number"],
                    "node_type": node.type,
                    "pageLabel": page_ctx["page_label"],
                    "revision": page_ctx["revision"],
                    "tag": node.tag,
                    "map": node.map,
                    "preview_text": node_plaintext(node)[:120]
                }
            ))
        
    return blocks


# ----------------------------
# Grouping blocks by heading
# ----------------------------

def group_blocks_by_heading(blocks: List[StructuralBlock]) -> List[HeadingSection]:
    """
    Groups StructuralBlocks by their heading path.
    Returns a list of HeadingSection objects, each containing all blocks under the same heading.
    """
    from collections import OrderedDict
    
    grouped: OrderedDict[Tuple[str, ...], List[StructuralBlock]] = OrderedDict()
    
    for block in blocks:
        heading_path = block.meta.get('heading_path', ())
        if heading_path not in grouped:
            grouped[heading_path] = []
        grouped[heading_path].append(block)
    
    sections: List[HeadingSection] = []
    
    for heading_path, section_blocks in grouped.items():
        if not section_blocks:
            continue
            
        # Extract heading metadata from first block
        first_block = section_blocks[0]
        heading_metadata = {}
        for key in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']:
            if key in first_block.meta:
                heading_metadata[key] = first_block.meta[key]
        
        # Separate html_blocks from content blocks
        text_parts = []
        html_blocks_raw = []
        for block in section_blocks:
            if block.kind == 'html_block':
                # Collect html_blocks into metadata instead of combined text
                html_blocks_raw.append(block.raw_escaped)
            elif block.raw_unescaped:
                # Use raw_unescaped for embedding (includes table kv_text)
                text_parts.append(block.raw_unescaped)
        
        combined_text = "\n\n".join(text_parts)
        
        # Aggregate metadata (using first block's metadata as base)
        section_meta = {
            'doc_name': first_block.meta.get('doc_name'),
            'source': first_block.meta.get('source'),
            'page_number': first_block.meta.get('page_number'),
            'pageLabel': first_block.meta.get('pageLabel'),
            'revision': first_block.meta.get('revision'),
            'doc_index': first_block.meta.get('doc_index'),
            'block_count': len(section_blocks),
            'start_line': section_blocks[0].start_line,
            'end_line': section_blocks[-1].end_line,
            'html_blocks': html_blocks_raw,
        }
        
        sections.append(HeadingSection(
            heading_path=heading_path,
            heading_metadata=heading_metadata,
            blocks=section_blocks,
            combined_text=combined_text,
            meta=section_meta
        ))
    
    return sections

# ----------------------------
# Pipeline: Header split -> structural blocks -> Documents
# ----------------------------

def header_split_then_structural_blocks(
    markdown_path: Path,
    headers_to_split_on: None,    
) -> List[StructuralBlock]:
    """
    1) Use MarkdownHeaderTextSplitter to split into header-scoped
    2) For each doc, split into structural blocks using markdown-it-py + SyntaxTreeNode
    3) Return a flattened list of StructuralBlocks with enriched metadata
    """

    if headers_to_split_on is None:
        headers_to_split_on = [
            ("#", "H1"),
            ("##", "H2"),
            ("###", "H3"),
            ("####", "H4")
        ]

    markdown_document = markdown_path.read_text(encoding="utf-8")
    full_doc_lines = markdown_document.splitlines()
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=True)
    header_docs = header_splitter.split_text(markdown_document)
    doc_name = markdown_path.name
    source = str(markdown_path)

    all_blocks: List[StructuralBlock] = []
    
    # Track position in the original document
    current_position = 0

    for doc_index, doc in enumerate(header_docs): 
        hp = build_heading_path_from_metadata(doc.metadata)
        
        # Find this chunk's position in the full document
        chunk_start = markdown_document.find(doc.page_content, current_position)
        if chunk_start == -1:
            chunk_start = current_position
        
        # Count lines from start to chunk_start
        chunk_offset = markdown_document[:chunk_start].count('\n')

        blocks = structural_split_markdown_chunk(doc.page_content, 
                                                 html_tables_are_escaped=True, 
                                                 enable_html_parsing=True,
                                                 include_lists=True,
                                                 include_blockquotes=True,
                                                 full_document_lines=full_doc_lines,
                                                 chunk_offset=chunk_offset)
        
        # Base metadata for all blocks in this header section
        base_metadata = {
            **doc.metadata,
            "heading_path": hp,
            "doc_index": doc_index,
            "doc_name": doc_name,
            "source": source,
        }
        
        # Process blocks with table-aware enhancement
        enhanced_blocks = structural_blocks_to_docs_with_table_strings(blocks, base_metadata)
        all_blocks.extend(enhanced_blocks)
        
        # Update position for next chunk
        current_position = chunk_start + len(doc.page_content)
        
    return all_blocks

def extract_page_context(lines_raw: List[str], start0: int, lookback: int = 300) -> Dict[str, Any]:
    """
    Scan backwards from start0 to find nearest PageNumber markers.
    Stops at the nearest PageBreak after it has found PageNumber(s).
    Works with escaped comments by unescaping each scanned line.
    """
    labels = []
    marker_line = None
    seen_any = False

    for k in range(start0, max(-1, start0 - lookback), -1):
        line = html.unescape(lines_raw[k])
        found = RE_PAGENUMBER.findall(line)
        if found:
            if marker_line is None:
                marker_line = k + 1  # 1-based
            labels.extend(found)
            seen_any = True
        if seen_any and RE_PAGEBREAK.search(line):
            break

    page_number = None
    page_label = None
    revision = None

    for lab in labels:
        m = RE_PAGE_NUMERIC.search(lab)
        if m:
            page_number = int(m.group(1))
            page_label = lab.strip()
        elif RE_REV.match(lab):
            revision = lab.strip()

    return {
        "page_number": page_number,
        "page_label": page_label,
        "revision": revision,
        "page_marker_line": marker_line,
        "found_labels": labels,
    }

if __name__ == "__main__":
    markdown_path = Path("./resources/B737.md")
    markdown_document = markdown_path.read_text(encoding="utf-8")

    # Get structural blocks
    blocks = header_split_then_structural_blocks(markdown_path, headers_to_split_on=None)
    print(f"Total structural blocks: {len(blocks)}\n")
    
    # Group blocks by heading
    sections = group_blocks_by_heading(blocks)
    print(f"Total heading sections: {len(sections)}\n")
    
    # Print first 5 heading sections
    for i, section in enumerate(sections):
        print(f"\n{'='*80}")
        print(f"SECTION {i+1}")
        print(f"{'='*80}")
        print(f"Heading Path: {' > '.join(section.heading_path) if section.heading_path else '(root)'}")
        print(f"Heading Metadata: {section.heading_metadata}")
        print(f"Page Number: {section.meta.get('page_number', 'N/A')}")
        print(f"Number of Blocks: {section.meta.get('block_count')}")
        print(f"Block Types: {[block.kind for block in section.blocks]}")
        print(f"\nCombined Text Preview (first 300 chars):")
        print(f"{section.combined_text[:300]}...")
        print(f"\nText Length: {len(section.combined_text)} characters")