import logging
import json
from enum import Enum
from typing import List
from groq import AsyncGroq
from app.config import settings

logger = logging.getLogger(__name__)

# Module-level singleton — populated on first use, never at import time.
_groq_client: AsyncGroq | None = None


def _get_groq_client() -> AsyncGroq:
    """Return a cached AsyncGroq client, creating it on first call.

    Raises ValueError if GROQ_API_KEY is not configured so callers can
    surface a meaningful error instead of an opaque authentication failure.
    """
    global _groq_client
    if _groq_client is None:
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not configured")
        logger.info("🤖 [ROUTER] Initializing Groq client")
        _groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    return _groq_client


class IntentClass(str, Enum):
    CATALOGO = "CATALOGO"
    POLITICAS = "POLITICAS"
    INFO_GENERAL = "INFO_GENERAL"
    CONVERSACIONAL = "CONVERSACIONAL"


ROUTER_SYSTEM_PROMPT = """
Eres el enrutador de intenciones de la plataforma Ecommer. 
Analiza la entrada del usuario y clasifícala en una o varias categorías si es necesario.

Categorías disponibles:
1. "CATALOGO": Búsqueda de productos, stock, tallas, colores o recomendaciones.
2. "POLITICAS": Envíos, devoluciones, garantías, métodos de pago.
3. "INFO_GENERAL": Sobre la empresa, misión, contacto, soporte técnico.
4. "CONVERSACIONAL": Saludos, agradecimientos o charla informal.

Responde ESTRICTAMENTE en formato JSON:
{"intents": ["CATEGORIA1", "CATEGORIA2"]}
"""


async def classify_intents(query: str) -> List[str]:
    logger.info(f"🧠 [ROUTER] Classifying intent for: '{query}'")
    try:
        groq_client = _get_groq_client()
        chat_completion = await groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=50,
        )

        content = chat_completion.choices[0].message.content
        data = json.loads(content)
        intents = data.get("intents", [IntentClass.CONVERSACIONAL.value])

        logger.info(f"✅ [ROUTER] Detected intents: {intents}")
        return intents

    except ValueError as e:
        # Missing configuration — log and fall back so the request can still proceed
        logger.error(f"❌ [ROUTER CONFIG ERROR]: {str(e)}")
        return [IntentClass.CONVERSACIONAL.value]

    except Exception as e:
        logger.error(f"❌ [ROUTER ERROR]: {type(e).__name__} - {str(e)}")
        # Safe fallback: treat as conversational so the request is not blocked
        return [IntentClass.CONVERSACIONAL.value]