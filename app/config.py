import os
from dotenv import load_dotenv

# Cargamos el archivo .env
load_dotenv()

class Settings:
    # --- Qdrant ---
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    # --- Azure OpenAI (IMPORTANTE: Deben estar aquí declaradas) ---
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    
    # --- Groq (Opcional, si sigues usándolo para chat) ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # --- Colecciones ---
    COLLECTION_CATALOG = "ecommerce-rag-collection_openAI"
    COLLECTION_POLICIES = "Info_Legal_AZURE"
    COLLECTION_GENERAL = "Info_GENERAL_AZURE"

settings = Settings()

def validate_settings(required: list[str] | None = None) -> None:
    """
    Valida que las variables de entorno críticas estén disponibles.
    """
    if required is None:
        required = [
            "QDRANT_URL", 
            "QDRANT_API_KEY", 
            "AZURE_OPENAI_API_KEY", 
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_DEPLOYMENT"
        ]

    missing = []
    for name in required:
        # Buscamos si el atributo existe en el objeto 'settings'
        value = getattr(settings, name, None)
        if value in (None, ""):
            missing.append(name)

    if missing:
        # Esto lanzará un error amigable si te falta algo en el .env
        raise ValueError(f"❌ Faltan variables en tu .env: {', '.join(missing)}")