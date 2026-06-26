from fastapi import FastAPI, HTTPException
from app.schemas import QueryRequest, QueryResponse
from app.llm_router import classify_intent
from app.retriever import search_context
from app.config import settings
import asyncio

app = FastAPI(
    title="RAG Query API - E-commerce",
    description="Microservicio de enrutamiento y recuperación vectorial",
    version="1.0.0"
)

@app.post("/retrieve_context", response_model=QueryResponse)
async def retrieve_context(request: QueryRequest):
    query = request.query
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="La consulta no puede estar vacía.")
    
    # 1. Enrutamiento (Groq)
    print(f"\n🚀 [API] Nueva consulta recibida: '{query}'")
    intents = await classify_intent(query)
    print(f"🔀 [API] Intenciones resueltas: {intents}")
    
    # 2. Selección de Colecciones y Búsqueda Vectorial
    context_data = []
    collections_used = []
    collection_used = None
    
    if all(intent == "CONVERSACIONAL" for intent in intents):
        collection_used = "N/A"
        print("⏭️ [API] Intención CONVERSACIONAL. Saltando búsqueda en Qdrant.")
    else:
        collection_map = {
            "CATALOGO": (settings.COLLECTION_CATALOG, 10),
            "POLITICAS": (settings.COLLECTION_POLICIES, 4),
            "INFO_GENERAL": (settings.COLLECTION_GENERAL, 5),
        }
        tasks = []
        for intent in intents:
            if intent in collection_map:
                collection_name, limit = collection_map[intent]
                if collection_name not in collections_used:
                    collections_used.append(collection_name)
                    print(f"🔎 [API] Buscando en Qdrant -> Colección: {collection_name}")
                    tasks.append(search_context(query, collection_name, limit=limit))
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for collection_name, result in zip(collections_used, results):
                if isinstance(result, Exception):
                    print(f"❌ [API CRITICAL ERROR] Falló el retriever para {collection_name}: {str(result)}")
                    continue
                for item in result:
                    item["collection"] = collection_name
                    context_data.append(item)
        except Exception as e:
            print(f"❌ [API CRITICAL ERROR] Falló el retriever: {str(e)}")
            context_data = []
        
        collection_used = ", ".join(collections_used) if collections_used else "N/A"
    
    # 3. Respuesta empaquetada para tu segunda API (Gateway/Bot)
    return QueryResponse(
        query=query,
        intent=", ".join(intents),
        intents=intents,
        collection_used=collection_used,
        collections_used=collections_used,
        context=context_data
    )

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "RAG Query API", "groq_configured": bool(settings.GROQ_API_KEY)}