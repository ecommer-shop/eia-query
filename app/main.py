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

@app.post("/retrieve_context", response_model=QueryResponse)
async def retrieve_context(request: QueryRequest):
    query = request.query
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="La consulta no puede estar vacía.")
    
    # 1. Enrutamiento (Groq)
    print(f"\n🚀 [API] Nueva consulta recibida: '{query}'")
    intent = await classify_intent(query)
    print(f"🔀 [API] Intención resuelta: {intent}")
    
    # 2. Selección de Colección y Búsqueda Vectorial
    context_data = []
    collection_used = None
    
    try:
        if intent == "CATALOGO":
            collection_used = settings.COLLECTION_CATALOG
            print(f"🔎 [API] Buscando en Qdrant -> Colección: {collection_used}")
            # Importante: Mantener el await aquí
            context_data = await search_context(query, collection_used, limit=10) 
            
        elif intent == "POLITICAS":
            collection_used = settings.COLLECTION_POLICIES
            print(f"🔎 [API] Buscando en Qdrant -> Colección: {collection_used}")
            context_data = await search_context(query, collection_used, limit=4)
            
        elif intent == "INFO_GENERAL":
            # 🔥 AQUÍ ENTRAN TUS PDFs (Wompi, DIAN, Costos, Misión)
            collection_used = settings.COLLECTION_GENERAL
            print(f"🔎 [API] Buscando en Qdrant -> Colección: {collection_used}")
            context_data = await search_context(query, collection_used, limit=5)
            
        else: # Si Groq devuelve "CONVERSACIONAL" u otra cosa
            collection_used = "N/A"
            print("⏭️ [API] Intención CONVERSACIONAL. Saltando búsqueda en Qdrant.")
            context_data = [] 

    except Exception as e:
        # 🔥 BLINDAJE: Si Qdrant falla (ej. colección no existe), no tumbamos la API
        print(f"❌ [API CRITICAL ERROR] Falló el retriever: {str(e)}")
        context_data = []
        
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