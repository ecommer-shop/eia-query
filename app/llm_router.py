import logging
import json
from enum import Enum
from typing import List  # <--- Crítico: Importar List para el tipado
from groq import AsyncGroq
from app.config import settings

logger = logging.getLogger(__name__)
aclient = AsyncGroq(api_key=settings.GROQ_API_KEY)

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
    logger.info(f"🧠 [ROUTER] Analizando complejidad de: '{query}'")
    try:
        chat_completion = await aclient.chat.completions.create(
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"}, # Forzamos salida JSON en Groq
            temperature=0.0,
            max_tokens=50,
        )
        
        # Parseo seguro del JSON
        content = chat_completion.choices[0].message.content
        data = json.loads(content)
        intents = data.get("intents", [IntentClass.CONVERSACIONAL.value])
        
        logger.info(f"✅ [ROUTER] Intenciones detectadas: {intents}")
        return intents

    except Exception as e:
        logger.error(f"❌ [ROUTER ERROR]: {str(e)}")
        # Fallback de seguridad para no romper el flujo principal
        return [IntentClass.CONVERSACIONAL.value]