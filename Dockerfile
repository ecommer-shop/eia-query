# Usamos python 3.11 slim (buen balance entre tamaño y compatibilidad)
FROM python:3.11-slim

# Evitar escritura de .pyc y forzar logs directos
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Instalar dependencias base del sistema y limpiar basura de apt
# Solo instalamos build-essential temporalmente por si alguna librería requiere compilar en C
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. EL TRUCO MAGISTRAL: Instalar PyTorch en versión CPU explícitamente primero.
# Esto evita que sentence-transformers descargue los ~3GB de CUDA.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 3. Copiar e instalar los requirements que me pasaste
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Pre-descargar el modelo de embeddings en la imagen para evitar descargas en Runtime
# Usamos una capa específica para esto. 
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# 5. Limpiar dependencias de construcción (build-essential) para adelgazar la imagen final
RUN apt-get purge -y --auto-remove build-essential

# 6. Copiar el código de la aplicación
COPY ./app ./app

EXPOSE 8000

# Lanzar Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]