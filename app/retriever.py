from qdrant_client import QdrantClient
from openai import AzureOpenAI
from app.config import settings

# 1. Iniciamos clientes
print(f"🔌 [INIT] Conectando a Qdrant en: {settings.QDRANT_URL}")
qdrant_client = QdrantClient(
    url=settings.QDRANT_URL, 
    api_key=settings.QDRANT_API_KEY, 
    timeout=10
)

print(f"🧠 [INIT] Configurando Azure OpenAI (Endpoint: {settings.AZURE_OPENAI_ENDPOINT})")
azure_client = AzureOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    api_version="2024-02-01",
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
)

def search_context(query: str, collection_name: str, limit: int = 5) -> list:
    print(f"\n🔎 [RETRIEVER] Buscando: '{query}'")
    
    try:
        # 1. Generar embedding con Azure
        response = azure_client.embeddings.create(
            input=[query],
            model=settings.AZURE_OPENAI_DEPLOYMENT
        )
        
        query_vector = response.data[0].embedding
        
        # 2. Búsqueda semántica en Qdrant
        # query_points es la forma moderna y eficiente de Qdrant
        response_qdrant = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True # Aseguramos que traiga la info del producto
        )
        
        search_results = response_qdrant.points 
        print(f"🎯 [RETRIEVER] Encontrados {len(search_results)} resultados.")
        
        # 3. Formatear para el LLM
        return [
            {"score": hit.score, "payload": hit.payload} 
            for hit in search_results
        ]
        
    except Exception as e:
        print(f"❌ [ERROR] Falló el retriever: {str(e)}")
        return []