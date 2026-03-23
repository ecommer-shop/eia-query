from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from app.config import settings

print(f"🔌 [INIT] Iniciando cliente Qdrant...")
qdrant_client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=10)

print(f"🧠 [INIT] Cargando modelo de embeddings en memoria: {settings.EMBEDDING_MODEL}...")
embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

def search_context(query: str, collection_name: str, limit: int = 5) -> list:
    print(f"\n🔎 [RETRIEVER] Iniciando búsqueda para: '{query}'")
    print(f"📂 [RETRIEVER] Colección objetivo: '{collection_name}'")
    
    try:
        # 1. Convertir pregunta a vector
        query_vector = embedding_model.encode(query).tolist()
        print(f"🔢 [RETRIEVER] Vector generado exitosamente (Dimensión: {len(query_vector)})")
        
        # 2. Buscar en Qdrant usando la NUEVA API (query_points)
        response = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector, # Nota: ahora el parámetro es 'query', ya no 'query_vector'
            limit=limit
        )
        
        # En la nueva API, los resultados vienen dentro del atributo .points
        search_results = response.points 
        
        print(f"🎯 [RETRIEVER] Qdrant encontró {len(search_results)} resultados.")
        
        # 3. Formatear resultados
        context = []
        for hit in search_results:
            print(f"   -> Match Score: {hit.score:.4f} | ID: {hit.payload.get('product_id', 'N/A')}")
            context.append({
                "score": hit.score,
                "payload": hit.payload
            })
            
        return context
        
    except Exception as e:
        print(f"❌ [RETRIEVER CRITICAL ERROR] Fallo en la comunicación con Qdrant: {type(e).__name__} - {str(e)}")
        return []