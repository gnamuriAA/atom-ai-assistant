import os
import logging
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

logger = logging.getLogger(__name__)


def _build_embedding_client() -> AzureOpenAI:
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