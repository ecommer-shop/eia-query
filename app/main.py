import asyncio
import logging

from fastapi import FastAPI, HTTPException
from app.schemas import QueryRequest, QueryResponse
from app.llm_router import classify_intents, IntentClass
from app.retriever import search_context, _get_qdrant_client, _get_azure_client
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Query API - E-commerce",
    description="Microservicio de enrutamiento y recuperación vectorial",
    version="1.0.0",
)


@app.post("/retrieve_context", response_model=QueryResponse)
async def retrieve_context(request: QueryRequest):
    query = request.query
    if not query.strip():
        raise HTTPException(status_code=400, detail="La consulta no puede estar vacía.")

    # 1. Intent routing via Groq Llama-3.1-8b
    intents = await classify_intents(query)

    # 2. Map intents to Qdrant collections
    intent_map = {
        IntentClass.CATALOGO: (settings.COLLECTION_CATALOG, 5),
        IntentClass.POLITICAS: (settings.COLLECTION_POLICIES, 5),
        IntentClass.INFO_GENERAL: (settings.COLLECTION_GENERAL, 5),
    }

    tasks = []
    collections_involved = []
    for intent in intents:
        if intent in intent_map:
            coll, limit = intent_map[intent]
            collections_involved.append(coll)
            tasks.append(search_context(query, coll, limit=limit))

    # 3. Run all retrieval tasks in parallel
    context_results = []
    if tasks:
        try:
            logger.info(f"📡 [API] Running {len(tasks)} parallel retrieval task(s)...")
            search_outputs = await asyncio.gather(*tasks, return_exceptions=True)
            for output in search_outputs:
                if isinstance(output, Exception):
                    logger.error(f"❌ [API] Retrieval task failed: {type(output).__name__} - {output}")
                else:
                    context_results.extend(output)
        except Exception as e:
            logger.error(f"❌ [API CRITICAL] Unexpected error during gather: {type(e).__name__} - {e}")
            raise HTTPException(status_code=500, detail="Error interno al recuperar contexto.")

    # 4. Consolidated response
    return QueryResponse(
        query=query,
        intent=", ".join(intents),
        collection_used=", ".join(collections_involved) if collections_involved else "N/A",
        context=context_results,
    )


@app.get("/health")
async def health_check():
    """
    Validates that all critical dependencies are reachable and configured.
    Returns a per-dependency status so callers can pinpoint what is missing.
    """
    checks: dict[str, str] = {}

    # --- Groq ---
    checks["groq"] = "ok" if settings.GROQ_API_KEY else "missing GROQ_API_KEY"

    # --- Azure OpenAI ---
    azure_missing = [
        name
        for name in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT")
        if not getattr(settings, name, None)
    ]
    checks["azure_openai"] = "ok" if not azure_missing else f"missing {', '.join(azure_missing)}"

    # --- Qdrant (connectivity probe) ---
    if not settings.QDRANT_URL:
        checks["qdrant"] = "missing QDRANT_URL"
    else:
        try:
            client = _get_qdrant_client()
            await client.get_collections()
            checks["qdrant"] = "ok"
        except Exception as e:
            checks["qdrant"] = f"unreachable: {type(e).__name__}"

    overall = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    status_code = 200 if overall == "ok" else 503

    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status_code,
        content={
            "status": overall,
            "service": "RAG Query API",
            "checks": checks,
        },
    )