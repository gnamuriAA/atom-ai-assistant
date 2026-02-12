"""
Generate embeddings for HeadingSections using Azure OpenAI.

Creates EmbeddingSection dataclass that extends HeadingSection data
with embedding vectors for use in RAG / vector search.
"""

import os
import logging
import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from copy import deepcopy

from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Runtime import to avoid circular dependency (hyphenated module name)
_structural_module = importlib.import_module("markdown-to-structural-block")
HeadingSection = _structural_module.HeadingSection
header_split_then_structural_blocks = _structural_module.header_split_then_structural_blocks
group_blocks_by_heading = _structural_module.group_blocks_by_heading


# ----------------------------
# EmbeddingSection dataclass
# ----------------------------

@dataclass
class EmbeddingSection:
    """HeadingSection enriched with an embedding vector."""

    heading_path: Tuple[str, ...]
    heading_metadata: Dict[str, str]
    combined_text: str
    meta: Dict[str, Any]
    embedding: List[float] = field(default_factory=list)
    embedding_model: str = ""
    token_count: Optional[int] = None
    sub_section_index: Optional[int] = None  # None = not split, 0/1/2... = sub-section index
    total_sub_sections: Optional[int] = None  # Total sub-sections if split

    @classmethod
    def from_heading_section(
        cls,
        section: "HeadingSection",
        embedding: List[float],
        model: str,
        token_count: Optional[int] = None,
    ) -> "EmbeddingSection":
        """Create an EmbeddingSection from a HeadingSection and its embedding vector.

        Args:
            section: The HeadingSection to convert.
            embedding: The embedding vector.
            model: The model/deployment used to generate the embedding.
            token_count: Optional token count returned by the API.

        Returns:
            An EmbeddingSection instance.
        """
        return cls(
            heading_path=section.heading_path,
            heading_metadata=section.heading_metadata,
            combined_text=section.combined_text,
            meta=section.meta,
            embedding=embedding,
            embedding_model=model,
            token_count=token_count,
        )


# ----------------------------
# Azure OpenAI client helpers
# ----------------------------

def _build_client() -> AzureOpenAI:
    """Build an AzureOpenAI client using AzureKeyCredential.

    Returns:
        An authenticated AzureOpenAI client.
    """
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")

    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT must be set in .env or environment")
    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY must be set in .env or environment")

    logger.info("Authenticating with AzureKeyCredential")
    return AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
    )


# ----------------------------
# Embedding generation
# ----------------------------

MAX_BATCH_SIZE = 16  # Azure OpenAI embedding batch limit
MAX_TOKEN_CHARS = 20_000  # ~6,000 tokens safety limit (8,192 max for text-embedding-3-small)
OVERLAP_BLOCKS = 1  # Number of blocks to repeat at sub-section boundaries


def split_oversized_sections(
    sections: List["HeadingSection"],
    max_chars: int = MAX_TOKEN_CHARS,
    overlap_blocks: int = OVERLAP_BLOCKS,
) -> List["HeadingSection"]:
    """Split HeadingSections that exceed token limits at block boundaries.

    Groups StructuralBlocks together until approaching the character limit,
    then starts a new sub-section. Overlaps the last N blocks from the
    previous sub-section into the next for context continuity.

    Args:
        sections: List of HeadingSections to process.
        max_chars: Maximum character count per sub-section (~4 chars/token).
        overlap_blocks: Number of blocks to repeat at boundaries.

    Returns:
        List of HeadingSections (original if under limit, split otherwise).
    """
    StructuralBlock = _structural_module.StructuralBlock
    result: List[HeadingSection] = []

    for section in sections:
        # Check if section fits within limit
        if len(section.combined_text) <= max_chars:
            result.append(section)
            continue

        # Get content blocks only (exclude html_block — already in meta)
        content_blocks = [b for b in section.blocks if b.kind != "html_block"]

        if not content_blocks:
            result.append(section)
            continue

        # Group blocks into sub-sections
        sub_sections_blocks: List[List] = []
        current_group: List = []
        current_chars = 0

        for block in content_blocks:
            block_text = block.raw_unescaped or ""
            block_len = len(block_text)

            # If adding this block exceeds limit and we have content, start new group
            if current_chars + block_len > max_chars and current_group:
                sub_sections_blocks.append(current_group)
                # Overlap: carry last N blocks into next group
                overlap = current_group[-overlap_blocks:] if overlap_blocks > 0 else []
                current_group = list(overlap)
                current_chars = sum(len(b.raw_unescaped or "") for b in current_group)

            current_group.append(block)
            current_chars += block_len

        # Don't forget the last group
        if current_group:
            sub_sections_blocks.append(current_group)

        # If only 1 group resulted, no split needed
        if len(sub_sections_blocks) == 1:
            result.append(section)
            continue

        # Create sub-sections
        total_subs = len(sub_sections_blocks)
        logger.info(
            "Splitting section '%s' into %d sub-sections (original: %d chars, %d blocks)",
            " > ".join(section.heading_path) if section.heading_path else "(root)",
            total_subs,
            len(section.combined_text),
            len(content_blocks),
        )

        for sub_idx, sub_blocks in enumerate(sub_sections_blocks):
            combined = "\n\n".join(
                b.raw_unescaped for b in sub_blocks if b.raw_unescaped
            )
            sub_meta = deepcopy(section.meta)
            sub_meta["sub_section_index"] = sub_idx
            sub_meta["total_sub_sections"] = total_subs
            sub_meta["block_count"] = len(sub_blocks)
            sub_meta["start_line"] = sub_blocks[0].start_line
            sub_meta["end_line"] = sub_blocks[-1].end_line

            result.append(HeadingSection(
                heading_path=section.heading_path,
                heading_metadata=section.heading_metadata,
                blocks=sub_blocks,
                combined_text=combined,
                meta=sub_meta,
            ))

    return result


def generate_embeddings(
    sections: List["HeadingSection"],
    *,
    deployment: Optional[str] = None,
    batch_size: int = MAX_BATCH_SIZE,
) -> List[EmbeddingSection]:
    """Generate embeddings for a list of HeadingSections.

    Oversized sections are automatically split at block boundaries
    with overlap before embedding.

    Args:
        sections: HeadingSections to embed.
        deployment: Azure OpenAI deployment name (defaults to env var).
        batch_size: Number of texts per API call (max 16 for Azure).

    Returns:
        List of EmbeddingSection objects with embedding vectors.
    """
    deployment = deployment or os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")
    if not deployment:
        raise ValueError(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT must be set in .env or passed as argument"
        )

    # Split oversized sections at block boundaries
    sections = split_oversized_sections(sections)
    logger.info("Sections after splitting oversized: %d", len(sections))

    client = _build_client()

    # Filter out sections with empty text
    valid_sections = [(i, s) for i, s in enumerate(sections) if s.combined_text.strip()]
    skipped = len(sections) - len(valid_sections)
    if skipped:
        logger.warning("Skipping %d sections with empty combined_text", skipped)

    results: List[Optional[EmbeddingSection]] = [None] * len(sections)

    # Process in batches
    for batch_start in range(0, len(valid_sections), batch_size):
        batch = valid_sections[batch_start: batch_start + batch_size]
        texts = [s.combined_text for _, s in batch]

        logger.info(
            "Embedding batch %d-%d of %d sections",
            batch_start + 1,
            min(batch_start + len(batch), len(valid_sections)),
            len(valid_sections),
        )

        response = client.embeddings.create(
            input=texts,
            model=deployment,
        )

        for resp_item, (original_idx, section) in zip(response.data, batch):
            sub_idx = section.meta.get("sub_section_index")
            total_subs = section.meta.get("total_sub_sections")
            results[original_idx] = EmbeddingSection.from_heading_section(
                section=section,
                embedding=resp_item.embedding,
                model=response.model,
                token_count=resp_item.index,
            )
            results[original_idx].sub_section_index = sub_idx
            results[original_idx].total_sub_sections = total_subs

    # For skipped (empty) sections, create entries with empty embedding
    for i, section in enumerate(sections):
        if results[i] is None:
            results[i] = EmbeddingSection.from_heading_section(
                section=section,
                embedding=[],
                model=deployment,
                token_count=0,
            )

    logger.info(
        "Generated embeddings for %d sections (%d had content, %d empty)",
        len(results),
        len(valid_sections),
        skipped,
    )

    return results


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    markdown_path = Path("./resources/B737.md")

    # Step 1: Split markdown into structural blocks
    logger.info("Splitting markdown into structural blocks...")
    blocks = header_split_then_structural_blocks(markdown_path, headers_to_split_on=None)
    logger.info("Total structural blocks: %d", len(blocks))

    # Step 2: Group blocks by heading
    sections = group_blocks_by_heading(blocks)
    logger.info("Total heading sections: %d", len(sections))

    # Step 3: Generate embeddings
    logger.info("Generating embeddings...")
    embedding_sections = generate_embeddings(sections)

    # Step 4: Print summary
    print(f"\n{'='*80}")
    print(f"EMBEDDING RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Total sections processed: {len(embedding_sections)}")
    print(f"Sections with embeddings: {sum(1 for e in embedding_sections if e.embedding)}")
    split_count = sum(1 for e in embedding_sections if e.sub_section_index is not None)
    print(f"Sub-sections from splitting: {split_count}")
    print(f"Embedding dimensions: {len(embedding_sections[0].embedding) if embedding_sections and embedding_sections[0].embedding else 'N/A'}")
    print(f"Model used: {embedding_sections[0].embedding_model if embedding_sections else 'N/A'}")

    # Print first 5 sections
    for i, es in enumerate(embedding_sections[:5]):
        print(f"\n--- Section {i+1} ---")
        print(f"Heading: {' > '.join(es.heading_path) if es.heading_path else '(root)'}")
        print(f"Text length: {len(es.combined_text)} chars")
        print(f"Embedding dims: {len(es.embedding)}")
        if es.sub_section_index is not None:
            print(f"Sub-section: {es.sub_section_index + 1} of {es.total_sub_sections}")
        print(f"Embedding preview: {es.embedding[:5]}...")
