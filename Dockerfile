# Imagen de la API de riesgo de credito (FastAPI + Uvicorn)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

WORKDIR /app

# 1. Dependencias (se instalan primero para aprovechar la cache de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Codigo fuente y datos
COPY Base_de_datos.csv .
COPY mlops_pipeline/ ./mlops_pipeline/

# 3. Entrena el modelo dentro de la imagen para que coincida con las versiones instaladas
RUN python mlops_pipeline/model_training_evaluation.py

# 4. El servicio corre con un usuario sin privilegios
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "model_deploy:app", "--app-dir", "mlops_pipeline", "--host", "0.0.0.0", "--port", "8000"]
