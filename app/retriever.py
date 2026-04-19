from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from openai import AsyncAzureOpenAI
from app.config import settings

# 1. Iniciamos clientes en su versión ASÍNCRONA
print(f"🔌 [INIT] Conectando a Qdrant (Async) en: {settings.QDRANT_URL}")
qdrant_client = AsyncQdrantClient(
    url=settings.QDRANT_URL, 
    api_key=settings.QDRANT_API_KEY, 
    timeout=10.0
)

print(f"🧠 [INIT] Configurando Azure OpenAI (Async) - Endpoint: {settings.AZURE_OPENAI_ENDPOINT}")
azure_client = AsyncAzureOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    api_version="2024-02-01", # Asegúrate de que esta versión soporte tu deployment
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
)

# 2. Convertimos la función a corrutina (async def)
async def search_context(query: str, collection_name: str, limit: int = 3) -> list:
    print(f"\n🔎 [RETRIEVER] Buscando: '{query}' en colección '{collection_name}'")
    
    try:
        # 1. Generar embedding con Azure (AWAIT OBLIGATORIO)
        response = await azure_client.embeddings.create(
            input=[query],
            model=settings.AZURE_OPENAI_DEPLOYMENT
        )
        
        query_vector = response.data[0].embedding
        
        # 2. Búsqueda semántica asíncrona en Qdrant (AWAIT OBLIGATORIO)
        response_qdrant = await qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True 
        )
        
        search_results = response_qdrant.points 
        print(f"🎯 [RETRIEVER] Encontrados {len(search_results)} resultados.")
        
        # 3. Formatear para el LLM
        return [
            {"score": hit.score, "payload": hit.payload} 
            for hit in search_results
        ]
        
    except UnexpectedResponse as e:
        # 🔥 MANEJO DE ERROR LIMPIO PARA EL 404 🔥
        if e.status_code == 404:
            print(f"⚠️ [QDRANT] La colección '{collection_name}' no existe. Devolviendo contexto vacío.")
            return []
        print(f"❌ [QDRANT HTTP ERROR] Falló el retriever: {str(e)}")
        return []
        
    except Exception as e:
        # Captura de errores de Azure (rate limits, timeouts) o problemas generales
        print(f"❌ [RETRIEVER CRITICAL ERROR]: {type(e).__name__} - {str(e)}")
        return []