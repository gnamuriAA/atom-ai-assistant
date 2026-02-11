from langchain_text_splitters import MarkdownHeaderTextSplitter
from pathlib import Path


markdown_path = Path("./resources/B737.md")
markdown_document = markdown_path.read_text(encoding="utf-8")

headers_to_split_on = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
    ("####", "H4")
]

if __name__ == "__main__":
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, 
                                                   strip_headers=True)
    md_header_splits = markdown_splitter.split_text(markdown_document)

    print(md_header_splits[:10])