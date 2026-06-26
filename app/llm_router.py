import logging
import re
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
Eres el enrutador de intenciones de la plataforma Ecommer. Clasifica la entrada del usuario en una o varias de estas categorías:

1. "CATALOGO": Busca comprar, pregunta por productos, características o disponibilidad de stock.
2. "POLITICAS": Pregunta por envíos, devoluciones, garantías o reglas específicas de una compra.
3. "INFO_GENERAL": Pregunta sobre qué es Ecommer, cómo funciona, costos de suscripción, pasarelas de pago (Wompi), facturación (DIAN), soporte técnico, misión o visión de la empresa.
4. "CONVERSACIONAL": Saludos ("hola", "buenos días"), agradecimientos o preguntas totalmente fuera de contexto.

Si la consulta cubre más de una categoría, responde con las categorías separadas por comas.
Si la consulta menciona tanto compra como devolución o políticas, incluye ambas categorías.
Responde ÚNICAMENTE con las palabras exactas de las categorías, en mayúsculas.
"""

def _parse_intent_output(intent_raw: str) -> list[str]:
    normalized = re.sub(r"[\n;]+", ",", intent_raw)
    tokens = [token.strip() for token in normalized.split(",") if token.strip()]
    detected = []
    for token in tokens:
        for intent in IntentClass:
            if intent.value == token or intent.value in token:
                detected.append(intent.value)
    return list(dict.fromkeys(detected))


def _keyword_intent_fallback(query: str) -> list[str]:
    query_lower = query.lower()
    detected = []

    if re.search(r"\b(compra|comprar|producto|productos|stock|precio|caracter[ií]stica|caracter[ií]sticas|link|enlace|cat[aá]logo|catalogo|tiene|tengo|buscar|dime)\b", query_lower):
        detected.append(IntentClass.CATALOGO.value)

    if re.search(r"\b(devoluci[oó]n|devoluciones|garant[ií]a|garantias|env[ií]o|envios|pol[ií]tica|pol[ií]ticas|cambio|reembolso|refund|devoluci[oó]n)\b", query_lower):
        detected.append(IntentClass.POLITICAS.value)

    if re.search(r"\b(ecommer|suscripci[oó]n|costos|coste|wompi|dian|facturaci[oó]n|pago|pasarela|misi[oó]n|visi[oó]n|soporte|empresa|como funciona|informaci[oó]n)\b", query_lower):
        detected.append(IntentClass.INFO_GENERAL.value)

    if not detected and re.search(r"\b(hola|buenos|buenas|gracias|qué tal|como estas|qué tal|buen dia|buenas tardes|saludos)\b", query_lower):
        detected.append(IntentClass.CONVERSACIONAL.value)

    return list(dict.fromkeys(detected))


async def classify_intent(query: str) -> list[str]:
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

        detected = _parse_intent_output(intent_raw)
        fallback = _keyword_intent_fallback(query)

        if not detected:
            detected = fallback

        combined = list(dict.fromkeys(detected + fallback))
        if not combined:
            return [IntentClass.CONVERSACIONAL.value]

        if IntentClass.CONVERSACIONAL.value in combined and len(combined) > 1:
            combined = [i for i in combined if i != IntentClass.CONVERSACIONAL.value]

        return combined

    except Exception as e:
        logger.error(f"❌ [ROUTER ERROR]: {str(e)}")
        return [IntentClass.CONVERSACIONAL.value]