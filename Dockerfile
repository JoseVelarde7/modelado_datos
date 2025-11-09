# Usar Python 3.11 (compatible con pandas 2.2.2)
FROM python:3.11.9-slim

# Establecer directorio de trabajo
WORKDIR /app

# Variables de entorno para optimización
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (para aprovechar cache de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el proyecto
COPY . .

# Exponer el puerto (Railway asigna dinámicamente, pero documentamos)
EXPOSE 8000

# Comando para ejecutar la aplicación con Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "120", "app:server"]