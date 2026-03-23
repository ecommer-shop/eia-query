from fastapi import FastAPI, HTTPException
from app.schemas import QueryRequest, QueryResponse
from app.llm_router import classify_intent
from app.retriever import search_context
from app.config import settings

app = FastAPI(
    title="RAG Query API - E-commerce",
    description="Microservicio de enrutamiento y recuperación vectorial",
    version="1.0.0"
)

# Ya no necesitamos el @app.on_event("startup") porque Groq y Qdrant 
# se inicializan globalmente en sus respectivos módulos al importarlos.

@app.post("/retrieve_context", response_model=QueryResponse)
async def retrieve_context(request: QueryRequest):
    query = request.query
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="La consulta no puede estar vacía.")
    
    # 1. Enrutamiento (Groq)
    print(f"\n🚀 [API] Nueva consulta recibida: '{query}'")
    intent = classify_intent(query)
    print(f"🔀 [API] Intención resuelta: {intent}")
    
    # 2. Selección de Colección y Búsqueda Vectorial
    context_data = []
    collection_used = None
    
    if intent == "CATALOGO":
        collection_used = settings.COLLECTION_CATALOG
        print(f"🔎 [API] Buscando en Qdrant -> Colección: {collection_used}")
        context_data = search_context(query, collection_used, limit=10) 
        
    elif intent == "POLITICAS":
        collection_used = settings.COLLECTION_POLICIES
        print(f"🔎 [API] Buscando en Qdrant -> Colección: {collection_used}")
        context_data = search_context(query, collection_used, limit=4)
        
    else:
        print("⏭️ [API] Intención GENERAL. Saltando búsqueda en Qdrant.")
        
    # 3. Respuesta empaquetada para tu segunda API (Gateway/Bot)
    return QueryResponse(
        query=query,
        intent=intent,
        collection_used=collection_used,
        context=context_data
    )

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "RAG Query API", "groq_configured": bool(settings.GROQ_API_KEY)}