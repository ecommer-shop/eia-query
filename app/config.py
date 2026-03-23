import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Modelo de embeddings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

    # Nombres de las colecciones exactos
    COLLECTION_CATALOG = "ecommerce-rag-collection"
    COLLECTION_POLICIES = "Info_Legal_General"

settings = Settings()

def validate_settings(required: list[str] | None = None) -> None:
    """
    Valida que las variables de entorno críticas estén disponibles.
    Llamar durante el `startup` de la aplicación para fallar con mensaje claro.
    """
    if required is None:
        required = ["QDRANT_URL", "QDRANT_API_KEY", "GROQ_API_KEY"]

    missing = []
    for name in required:
        if getattr(settings, name, None) in (None, ""):
            missing.append(name)

    if missing:
        raise ValueError(f"❌ Faltan variables de entorno críticas: {', '.join(missing)}")