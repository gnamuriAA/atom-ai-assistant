"""
Store EmbeddingSections into MongoDB Atlas with Vector Search support.

Converts EmbeddingSection dataclass instances into MongoDB documents
and upserts them into a collection with a unique index to support
idempotent re-runs.
"""

import os
import logging
import importlib
from datetime import datetime, timezone
from typing import Any, Dict, List

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Runtime import (hyphenated module name)
_embed_module = importlib.import_module("generate-embeddings-from-heading")
EmbeddingSection = _embed_module.EmbeddingSection
generate_embeddings = _embed_module.generate_embeddings

_structural_module = importlib.import_module("markdown-to-structural-block")
header_split_then_structural_blocks = _structural_module.header_split_then_structural_blocks
group_blocks_by_heading = _structural_module.group_blocks_by_heading


# ----------------------------
# MongoDB helpers
# ----------------------------

def _get_collection():
    """Connect to MongoDB and return the target collection."""
    uri = os.environ.get("MONGODB_URI", "")
    db_name = os.environ.get("MONGODB_DATABASE", "atom_ai")
    coll_name = os.environ.get("MONGODB_COLLECTION", "heading_embeddings")

    if not uri:
        raise ValueError("MONGODB_URI must be set in .env or environment")

    client = MongoClient(uri)

    # Verify connection
    client.admin.command("ping")
    logger.info("Connected to MongoDB successfully")

    db = client[db_name]
    collection = db[coll_name]

    # Create unique index for idempotent upserts (skip if user lacks permission)
    try:
        collection.create_index(
            [("doc_name", 1), ("heading_path", 1), ("sub_section_index", 1)],
            unique=True,
            name="unique_section_idx",
        )
        logger.info("Ensured unique index on (doc_name, heading_path, sub_section_index)")
    except Exception as e:
        logger.warning("Could not create index (may need manual creation): %s", e)

    return collection


def embedding_section_to_doc(es: EmbeddingSection) -> Dict[str, Any]:
    """Convert an EmbeddingSection to a MongoDB document dict.

    Flattens meta fields to top-level for easier filtering and indexing.
    Excludes raw StructuralBlock objects to keep documents lean.
    """
    return {
        "heading_path": list(es.heading_path),
        "heading_metadata": es.heading_metadata,
        "combined_text": es.combined_text,
        "embedding": es.embedding,
        "embedding_model": es.embedding_model,
        "token_count": es.token_count,
        "doc_name": es.meta.get("doc_name"),
        "source": es.meta.get("source"),
        "page_number": es.meta.get("page_number"),
        "page_label": es.meta.get("pageLabel"),
        "revision": es.meta.get("revision"),
        "block_count": es.meta.get("block_count"),
        "start_line": es.meta.get("start_line"),
        "end_line": es.meta.get("end_line"),
        "html_blocks": es.meta.get("html_blocks", []),
        "sub_section_index": es.sub_section_index,
        "total_sub_sections": es.total_sub_sections,
        "created_at": datetime.now(timezone.utc),
    }


# ----------------------------
# Upsert to MongoDB
# ----------------------------

BATCH_SIZE = 100


def store_embeddings(
    embedding_sections: List[EmbeddingSection],
    batch_size: int = BATCH_SIZE,
) -> int:
    """Upsert EmbeddingSections into MongoDB.

    Uses bulk_write with UpdateOne(upsert=True) keyed on
    (doc_name, heading_path, sub_section_index) to avoid duplicates.

    Args:
        embedding_sections: List of EmbeddingSection objects to store.
        batch_size: Number of documents per bulk write operation.

    Returns:
        Total number of documents upserted.
    """
    collection = _get_collection()
    total_upserted = 0

    for batch_start in range(0, len(embedding_sections), batch_size):
        batch = embedding_sections[batch_start: batch_start + batch_size]
        operations = []

        for es in batch:
            doc = embedding_section_to_doc(es)

            # Build the unique filter for upsert
            filter_key = {
                "doc_name": doc["doc_name"],
                "heading_path": doc["heading_path"],
                "sub_section_index": doc["sub_section_index"],
            }

            operations.append(
                UpdateOne(filter_key, {"$set": doc}, upsert=True)
            )

        if operations:
            result = collection.bulk_write(operations)
            upserted = result.upserted_count + result.modified_count
            total_upserted += upserted
            logger.info(
                "Batch %d-%d: %d upserted, %d modified",
                batch_start + 1,
                min(batch_start + len(batch), len(embedding_sections)),
                result.upserted_count,
                result.modified_count,
            )

    logger.info("Total documents stored: %d", total_upserted)
    return total_upserted


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    from pathlib import Path

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
    logger.info("Total embedding sections: %d", len(embedding_sections))

    # Step 4: Store in MongoDB
    logger.info("Storing embeddings in MongoDB...")
    count = store_embeddings(embedding_sections)

    print(f"\n{'='*80}")
    print(f"MONGODB STORAGE SUMMARY")
    print(f"{'='*80}")
    print(f"Documents stored: {count}")
    print(f"Database: {os.environ.get('MONGODB_DATABASE', 'atom_ai')}")
    print(f"Collection: {os.environ.get('MONGODB_COLLECTION', 'heading_embeddings')}")