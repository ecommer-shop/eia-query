# Usamos Python 3.13 para match con pyproject.toml (requires-python)
FROM python:3.13-slim

# Evitar escritura de .pyc y forzar logs directos
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Dependencias base del sistema (build tools temporales)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Instalar uv (gestor de entornos y deps)
RUN pip install --no-cache-dir uv

# 3. Instalar dependencias desde uv.lock (mejor caching por capas)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 4. Copiar el código de la aplicación
COPY ./app ./app

# 5. Pre-descargar el modelo de embeddings para evitar descargas en runtime
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# 6. Limpiar dependencias de construcción para adelgazar la imagen final
RUN apt-get purge -y --auto-remove build-essential

EXPOSE 8000

# Lanzar Uvicorn (sin --reload en contenedor)
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]