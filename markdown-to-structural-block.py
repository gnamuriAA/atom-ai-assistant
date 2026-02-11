from dataclasses import dataclass
import html
import re
from typing import Any, Dict, List, Tuple
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode

RE_TABLE_OPEN = re.compile(r"<\s*table\b", re.IGNORECASE)
RE_HEADING_TAG = re.compile(r"^h([1-6])$", re.IGNORECASE)

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

def structural_split_markdown_chunk(
        md_chunk_raw: str,
        *,
        html_tables_are_escaped: bool = True,
        enable_html_parsing: bool = True,
        include_lists: bool = True,
        include_blockquotes: bool = True
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
            blocks.append(StructuralBlock(
                kind=kind,
                start_line=start0 + 1,
                end_line=end0,
                raw_escaped=raw_escaped,
                raw_unescaped=raw_unescaped,
                meta={
                    "node_type": node.type,
                    "tag": node.tag,
                    "map": node.map,
                    "preview_text": node_plaintext(node)[:120]
                }
            ))
        
    return blocks


# ----------------------------
# Pipeline: Header split -> structural blocks -> Documents
# ----------------------------

def header_split_then_structural_blocks(
    markdonw_document: str,
    headers_to_split_on: None,    
) -> List[Document]:
    """
    1) Use MarkdownHeaderTextSplitter to split into header-scoped
    2) For each doc, split into structural blocks using markdown-it-py + SyntaxTreeNode
    3) Return a flattened list of Documents (one per structural block) with rich metadata
    """

    if headers_to_split_on is None:
        headers_to_split_on = [
            ("#", "H1"),
            ("##", "H2"),
            ("###", "H3"),
            ("####", "H4")
        ]

        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=True)
        header_docs = header_splitter.split_text(markdonw_document)

        out_docs: List[Document] = []

        for doc_index, doc in enumerate(header_docs): 
            hp = build_heading_path_from_metadata(doc.metadata)

            blocks = structural_split_markdown_chunk(doc.page_content, 
                                                     html_tables_are_escaped=True, 
                                                     enable_html_parsing=True,
                                                     include_lists=True,
                                                     include_blockquotes=True)
            
            for block_index, block in enumerate(blocks):
                out_docs.append(Document(
                    page_content=block.raw_unescaped,
                      metadata={
                          **doc.metadata,
                          "heading_path": hp,
                          "doc_index": doc_index,
                          "block_index": block_index,
                          "block_kind": block.kind,
                          "chunk_rel_start_line": block.start_line,
                          "chunk_rel_end_line": block.end_line,
                          "raw_unescaped_preview": block.raw_unescaped[:200]
                      }))
    return out_docs

if __name__ == "__main__":
    markdown_path = Path("./resources/B737.md")
    markdown_document = markdown_path.read_text(encoding="utf-8")

    docs = header_split_then_structural_blocks(markdown_document, headers_to_split_on=None)

    print(docs[:10])