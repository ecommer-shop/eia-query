# Usamos python 3.12 slim
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copiar dependencias
COPY pyproject.toml uv.lock README.md ./

# Sincronizar (solo fastapi, uvicorn, qdrant, groq, openai - SIN sentence-transformers)
RUN uv sync --frozen --no-dev

# Copiar código
COPY ./app ./app

EXPOSE 8000

CMD ["uv", "run", "--no-dev", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]