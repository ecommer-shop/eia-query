from fastapi import FastAPI, HTTPException
from app.schemas import QueryRequest, QueryResponse
import asyncio

from app.llm_router import classify_intents, IntentClass 
from app.retriever import search_context
from app.config import settings

app = FastAPI(
    title="RAG Query API - E-commerce",
    description="Microservicio de enrutamiento y recuperación vectorial",
    version="1.0.0"
)

@app.post("/retrieve_context", response_model=QueryResponse)
async def retrieve_context(request: QueryRequest):
    query = request.query
    if not query.strip():
        raise HTTPException(status_code=400, detail="La consulta vacía.")

    # 1. Enrutamiento Multi-Intento (Groq Llama-3.1-8b)
    # Ahora devuelve una lista: ["CATALOGO", "POLITICAS"]
    intents = await classify_intents(query)
    
    # 2. Preparación de Tareas Concurrentes
    tasks = []
    collections_involved = []

    # Mapeo de intenciones a colecciones y límites
    intent_map = {
        IntentClass.CATALOGO: (settings.COLLECTION_CATALOG, 5),
        IntentClass.POLITICAS: (settings.COLLECTION_POLICIES, 5),
        IntentClass.INFO_GENERAL: (settings.COLLECTION_GENERAL, 5),
    }

    for intent in intents:
        if intent in intent_map:
            coll, limit = intent_map[intent]
            collections_involved.append(coll)
            tasks.append(search_context(query, coll, limit=limit))

    # 3. Ejecución en Paralelo (Optimización de Latencia)
    # Si Qdrant tarda 100ms por búsqueda, el tiempo total seguirá siendo ~100ms
    context_results = []
    if tasks:
        try:
            print(f"📡 [API] Ejecutando {len(tasks)} búsquedas en paralelo...")
            search_outputs = await asyncio.gather(*tasks, return_exceptions=True)
            
            for output in search_outputs:
                if isinstance(output, Exception):
                    print(f"❌ [API ERROR] Tarea fallida: {output}")
                else:
                    context_results.extend(output)
        except Exception as e:
            print(f"❌ [API CRITICAL] Error en gather: {str(e)}")

    # 4. Respuesta consolidada
    return QueryResponse(
        query=query,
        intent=", ".join(intents), # Informamos todas las intenciones detectadas
        collection_used=", ".join(collections_involved) if collections_involved else "N/A",
        context=context_results
    )
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "RAG Query API", "groq_configured": bool(settings.GROQ_API_KEY)}