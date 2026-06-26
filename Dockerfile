# Usamos python 3.11 slim (buen balance entre tamaño y compatibilidad)
FROM python:3.11-slim

# Evitar escritura de .pyc y forzar logs directos
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar uv (gestor de paquetes Python)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 1. Instalar dependencias base del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Copiar archivos de dependencias
COPY pyproject.toml uv.lock ./

# 3. Sincronizar dependencias con uv
RUN uv sync --frozen --no-dev

# 4. Pre-descargar el modelo de embeddings en la imagen para evitar descargas en Runtime
RUN uv run --no-dev python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# 5. Limpiar dependencias de construcción para adelgazar la imagen final
RUN apt-get purge -y --auto-remove build-essential

# 6. Copiar el código de la aplicación
COPY ./app ./app

EXPOSE 8000

# Lanzar Uvicorn
CMD ["uv", "run", "--no-dev", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]