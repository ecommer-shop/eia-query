# Etapa 1: Constructor (Build Stage)
FROM ghcr.io/astral-sh/uv:latest AS uv_bin
FROM python:3.13-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar uv
COPY --from=uv_bin /uv /uv/bin/uv
ENV PATH="/uv/bin:$PATH"

# Dependencias de compilación (solo si alguna de tus libs lo pide)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de configuración
COPY pyproject.toml uv.lock ./

# Instalamos todo lo que esté en tu pyproject.toml (menos torch si lo quitas de ahí)
# uv sync crea el .venv automáticamente
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Etapa 2: Imagen Final (Runtime Stage)
FROM python:3.13-slim

WORKDIR /app

# Configuración de puerto para Railway y entorno virtual
ENV PORT=8000 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copiar solo el entorno virtual y el código
COPY --from=builder /app/.venv /app/.venv
COPY app/ /app/app/

# Exponer el puerto dinámico
EXPOSE ${PORT}

# Comando de ejecución adaptado para Railway
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]