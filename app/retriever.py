import logging
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from openai import AsyncAzureOpenAI
from app.config import settings

logger = logging.getLogger(__name__)

# Module-level singletons — populated on first use, never at import time.
_qdrant_client: AsyncQdrantClient | None = None
_azure_client: AsyncAzureOpenAI | None = None


def _get_qdrant_client() -> AsyncQdrantClient:
    """Return a cached AsyncQdrantClient, creating it on first call."""
    global _qdrant_client
    if _qdrant_client is None:
        if not settings.QDRANT_URL:
            raise ValueError("QDRANT_URL is not configured")
        logger.info(f"🔌 [RETRIEVER] Connecting to Qdrant at: {settings.QDRANT_URL}")
        _qdrant_client = AsyncQdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=10.0,
            check_compatibility=False,
        )
    return _qdrant_client


def _get_azure_client() -> AsyncAzureOpenAI:
    """Return a cached AsyncAzureOpenAI client, creating it on first call."""
    global _azure_client
    if _azure_client is None:
        if not settings.AZURE_OPENAI_ENDPOINT:
            raise ValueError("AZURE_OPENAI_ENDPOINT is not configured")
        if not settings.AZURE_OPENAI_API_KEY:
            raise ValueError("AZURE_OPENAI_API_KEY is not configured")
        logger.info(
            f"🧠 [RETRIEVER] Configuring Azure OpenAI - Endpoint: {settings.AZURE_OPENAI_ENDPOINT}"
        )
        _azure_client = AsyncAzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version="2024-02-01",
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        )
    return _azure_client


async def search_context(query: str, collection_name: str, limit: int = 3) -> list:
    logger.info(f"🔎 [RETRIEVER] Searching: '{query}' in collection '{collection_name}'")

    try:
        qdrant = _get_qdrant_client()
        azure = _get_azure_client()

        # 1. Generate embedding with Azure OpenAI
        response = await azure.embeddings.create(
            input=[query],
            model=settings.AZURE_OPENAI_DEPLOYMENT,
        )
        query_vector = response.data[0].embedding

        # 2. Semantic search in Qdrant
        response_qdrant = await qdrant.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        search_results = response_qdrant.points
        logger.info(f"🎯 [RETRIEVER] Found {len(search_results)} results.")

        # 3. Format for the LLM
        return [
            {"score": hit.score, "payload": hit.payload}
            for hit in search_results
        ]

    except UnexpectedResponse as e:
        if e.status_code == 404:
            logger.warning(
                f"⚠️ [QDRANT] Collection '{collection_name}' does not exist. Returning empty context."
            )
            return []
        logger.error(f"❌ [QDRANT HTTP ERROR] Retriever failed: {str(e)}")
        return []

    except ValueError as e:
        # Missing configuration — surface clearly rather than silently swallowing
        logger.error(f"❌ [RETRIEVER CONFIG ERROR]: {str(e)}")
        raise

    except Exception as e:
        logger.error(f"❌ [RETRIEVER CRITICAL ERROR]: {type(e).__name__} - {str(e)}")
        return []