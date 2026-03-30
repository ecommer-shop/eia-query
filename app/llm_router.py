import logging
from enum import Enum
from groq import AsyncGroq
from app.config import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
aclient = AsyncGroq(api_key=settings.GROQ_API_KEY)

class IntentClass(str, Enum):
    CATALOGO = "CATALOGO"
    POLITICAS = "POLITICAS"
    INFO_GENERAL = "INFO_GENERAL" # <-- NUEVA INTENCIÓN PARA TUS PDFs
    CONVERSACIONAL = "CONVERSACIONAL" # <-- NUEVA INTENCIÓN PARA SALTAR QDRANT

ROUTER_SYSTEM_PROMPT = """
Eres el enrutador de intenciones de la plataforma Ecommer. Clasifica la entrada del usuario en UNA de estas cuatro categorías:

1. "CATALOGO": Busca comprar, pregunta por productos, características o disponibilidad de stock.
2. "POLITICAS": Pregunta por envíos, devoluciones, garantías o reglas específicas de una compra.
3. "INFO_GENERAL": Pregunta sobre qué es Ecommer, cómo funciona, costos de suscripción, pasarelas de pago (Wompi), facturación (DIAN), soporte técnico, misión o visión de la empresa.
4. "CONVERSACIONAL": Saludos ("hola", "buenos días"), agradecimientos o preguntas totalmente fuera de contexto.

Responde ÚNICAMENTE con la palabra exacta de la categoría, en mayúsculas.
"""

async def classify_intent(query: str) -> str:
    logger.info(f"🧠 [ROUTER] Analizando pregunta: '{query}'")
    try:
        chat_completion = await aclient.chat.completions.create(
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=10,
        )
        
        intent_raw = chat_completion.choices[0].message.content.strip().upper()
        logger.debug(f"✅ [ROUTER] Respuesta cruda: '{intent_raw}'")
        
        if IntentClass.CATALOGO.value in intent_raw: return IntentClass.CATALOGO.value
        if IntentClass.POLITICAS.value in intent_raw: return IntentClass.POLITICAS.value
        if IntentClass.INFO_GENERAL.value in intent_raw: return IntentClass.INFO_GENERAL.value
            
        return IntentClass.CONVERSACIONAL.value
        
    except Exception as e:
        logger.error(f"❌ [ROUTER ERROR]: {str(e)}")
        return IntentClass.CONVERSACIONAL.value