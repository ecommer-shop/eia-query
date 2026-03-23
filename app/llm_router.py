from groq import Groq
from app.config import settings

# Instanciamos el cliente asegurándonos de pasarle la key explícitamente
client = Groq(api_key=settings.GROQ_API_KEY)

def classify_intent(query: str) -> str:
    system_prompt = """
    Eres el enrutador de intenciones de un e-commerce. Clasifica la entrada del usuario en UNA de estas tres categorías:
    
    1. "CATALOGO": El usuario busca comprar, pregunta por productos, características o disponibilidad.
    2. "POLITICAS": El usuario pregunta por envíos, devoluciones, garantías o reglas de la tienda, informacion legal, o una descriipcion general sobre que es la tienda o como registrarse en la plataforma.
    3. "GENERAL": Saludos o preguntas fuera de contexto.
    
    Responde ÚNICAMENTE con la palabra exacta de la categoría, en mayúsculas.
    """
    
    print(f"🧠 [ROUTER] Analizando pregunta: '{query}'")
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            model="llama-3.1-8b-instant", # <--- LA SOLUCIÓN
            temperature=0.0,
            max_tokens=10,
        )
        
        intent = chat_completion.choices[0].message.content.strip().upper()
        print(f"✅ [ROUTER] Respuesta cruda de Groq: '{intent}'")
        
        # Limpieza por si Groq devuelve algún signo de puntuación extra
        if "CATALOGO" in intent: return "CATALOGO"
        if "POLITICAS" in intent: return "POLITICAS"
        
        return "GENERAL"
        
    except Exception as e:
        # 🔥 AQUÍ ESTÁ LA MAGIA DE DEBUGGING 🔥
        print(f"❌ [ROUTER CRITICAL ERROR]: {type(e).__name__} - {str(e)}")
        print("⚠️ Cayendo al Fallback: GENERAL")
        return "GENERAL"