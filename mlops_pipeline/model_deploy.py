"""
model_deploy.py
---------------
Despliegue del mejor modelo como servicio (API REST con FastAPI).

- Carga el modelo entrenado (artefactos/mejor_modelo.joblib).
- Recibe los datos crudos de las solicitudes de credito, les aplica la misma limpieza y los
  mismos atributos derivados de ft_engineering.py y retorna la prediccion.
- Expone el endpoint /predict, que acepta JSON, y /predict/csv, que acepta un archivo CSV.
  Ambos permiten prediccion por lotes (varios registros en una sola solicitud).

Uso local:
    uvicorn model_deploy:app --app-dir mlops_pipeline --port 8000
    Documentacion interactiva: http://127.0.0.1:8000/docs
"""

import io
import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_engineering import OBJETIVO, VARIABLES_ELIMINAR, crear_atributos, limpiar_datos  # noqa: E402

CARPETA_ARTEFACTOS = Path(__file__).resolve().parent / "artefactos"
RUTA_MODELO = Path(os.getenv("RUTA_MODELO", CARPETA_ARTEFACTOS / "mejor_modelo.joblib"))

CLASE_NO_PAGO = 0
MAX_REGISTROS = 10_000
MAX_BYTES_CSV = 10 * 1024 * 1024
UMBRAL_RIESGO_MEDIO = 0.30
UMBRAL_RIESGO_ALTO = 0.50
ETIQUETAS = {0: "No paga a tiempo", 1: "Paga a tiempo"}

EJEMPLO_REGISTRO = {
    "tipo_credito": 4,
    "fecha_prestamo": "2025-01-07T14:40:36",
    "capital_prestado": 1852560,
    "plazo_meses": 12,
    "edad_cliente": 32,
    "tipo_laboral": "Empleado",
    "salario_cliente": 3500000,
    "total_otros_prestamos": 1000000,
    "cuota_pactada": 128650,
    "puntaje_datacredito": 795,
    "cant_creditosvigentes": 0,
    "huella_consulta": 2,
    "saldo_mora": 0,
    "saldo_total": 0,
    "creditos_sectorFinanciero": 0,
    "creditos_sectorCooperativo": 0,
    "creditos_sectorReal": 0,
    "promedio_ingresos_datacredito": 916148,
    "tendencia_ingresos": "Creciente",
}


# ---------------------------------------------------------------------------
# Esquemas de entrada y salida
# ---------------------------------------------------------------------------
class SolicitudCredito(BaseModel):
    """Datos de una solicitud de credito, en el mismo formato de Base_de_datos.csv.

    Los campos que pueden faltar (por ejemplo, la informacion de Datacredito) son opcionales:
    el modelo los imputa igual que en el entrenamiento. Las columnas adicionales se ignoran.
    """

    model_config = ConfigDict(extra="ignore", json_schema_extra={"example": EJEMPLO_REGISTRO})

    tipo_credito: int
    fecha_prestamo: Optional[datetime] = Field(None, description="Si no se envia, se usa la fecha actual")
    capital_prestado: float = Field(gt=0)
    plazo_meses: int = Field(gt=0)
    edad_cliente: Optional[float] = None
    tipo_laboral: str
    salario_cliente: Optional[float] = None
    total_otros_prestamos: Optional[float] = None
    cuota_pactada: float = Field(gt=0)
    puntaje_datacredito: Optional[float] = None
    cant_creditosvigentes: int = Field(ge=0)
    huella_consulta: int = Field(ge=0)
    saldo_mora: Optional[float] = None
    saldo_total: Optional[float] = None
    creditos_sectorFinanciero: int = Field(ge=0)
    creditos_sectorCooperativo: int = Field(ge=0)
    creditos_sectorReal: int = Field(ge=0)
    promedio_ingresos_datacredito: Optional[float] = None
    tendencia_ingresos: Optional[str] = None


class SolicitudPrediccion(BaseModel):
    """Lote de solicitudes para predecir en una sola llamada."""

    registros: List[SolicitudCredito] = Field(min_length=1, max_length=MAX_REGISTROS)


class Prediccion(BaseModel):
    indice: int
    prediccion: int = Field(description="1 = paga a tiempo, 0 = no paga a tiempo")
    etiqueta: str
    probabilidad_no_pago: float
    nivel_riesgo: str


class RespuestaPrediccion(BaseModel):
    modelo: str
    total_registros: int
    total_alto_riesgo: int
    predicciones: List[Prediccion]


LISTA_SOLICITUDES = TypeAdapter(List[SolicitudCredito])


# ---------------------------------------------------------------------------
# Logica de prediccion
# ---------------------------------------------------------------------------
def cargar_modelo(ruta=None):
    """Carga el pipeline entrenado y el nombre del modelo."""
    ruta = Path(ruta) if ruta is not None else RUTA_MODELO
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontro el modelo en {ruta}. Ejecute model_training_evaluation.py")
    modelo = joblib.load(ruta)
    info = ruta.with_name("mejor_modelo_info.json")
    nombre = json.loads(info.read_text(encoding="utf-8")).get("modelo", "desconocido") if info.exists() \
        else type(modelo[-1]).__name__
    return modelo, nombre


def preparar_entrada(solicitudes):
    """Convierte las solicitudes en las variables que espera el modelo.

    Aplica la misma limpieza y los mismos atributos derivados del entrenamiento.
    """
    datos = pd.DataFrame([s.model_dump() for s in solicitudes])
    datos["fecha_prestamo"] = pd.to_datetime(datos["fecha_prestamo"]).fillna(pd.Timestamp.now().floor("s"))
    for columna in VARIABLES_ELIMINAR:
        if columna not in datos:
            datos[columna] = np.nan
    datos = crear_atributos(limpiar_datos(datos))
    datos["tendencia_ingresos"] = datos["tendencia_ingresos"].astype(object)
    return datos.drop(columns=[OBJETIVO], errors="ignore")


def nivel_riesgo(probabilidad):
    """Nivel de riesgo segun la probabilidad de no pago."""
    if probabilidad >= UMBRAL_RIESGO_ALTO:
        return "Alto"
    return "Medio" if probabilidad >= UMBRAL_RIESGO_MEDIO else "Bajo"


def predecir(modelo, solicitudes):
    """Retorna una prediccion por solicitud (prediccion por lotes)."""
    X = preparar_entrada(solicitudes)
    X = X[list(modelo.feature_names_in_)]
    indice_no_pago = list(modelo.classes_).index(CLASE_NO_PAGO)
    probabilidades = modelo.predict_proba(X)[:, indice_no_pago]
    clases = modelo.predict(X)
    return [
        Prediccion(
            indice=i,
            prediccion=int(clase),
            etiqueta=ETIQUETAS[int(clase)],
            probabilidad_no_pago=round(float(probabilidad), 4),
            nivel_riesgo=nivel_riesgo(probabilidad),
        )
        for i, (clase, probabilidad) in enumerate(zip(clases, probabilidades))
    ]


def leer_csv(contenido):
    """Lee un CSV separado por coma o por punto y coma."""
    texto = contenido.decode("utf-8-sig")
    separador = ";" if texto.splitlines()[0].count(";") > texto.splitlines()[0].count(",") else ","
    return pd.read_csv(io.StringIO(texto), sep=separador)


# ---------------------------------------------------------------------------
# Aplicacion FastAPI
# ---------------------------------------------------------------------------
@asynccontextmanager
async def ciclo_de_vida(aplicacion: FastAPI):
    """Carga el modelo una sola vez al iniciar el servicio."""
    try:
        aplicacion.state.modelo, aplicacion.state.nombre_modelo = cargar_modelo()
    except FileNotFoundError:
        aplicacion.state.modelo, aplicacion.state.nombre_modelo = None, None
    yield


app = FastAPI(
    title="API de riesgo de crédito",
    description="Predice si un cliente pagará a tiempo su crédito. Soporta predicción por lotes.",
    version="1.0.0",
    lifespan=ciclo_de_vida,
)


def _modelo_disponible(request: Request):
    if request.app.state.modelo is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado. Entrene el modelo primero.")
    return request.app.state.modelo


def _respuesta(request: Request, solicitudes):
    predicciones = predecir(_modelo_disponible(request), solicitudes)
    return RespuestaPrediccion(
        modelo=request.app.state.nombre_modelo,
        total_registros=len(predicciones),
        total_alto_riesgo=sum(p.nivel_riesgo == "Alto" for p in predicciones),
        predicciones=predicciones,
    )


@app.get("/")
def inicio():
    """Informacion general del servicio."""
    return {
        "servicio": "API de riesgo de crédito",
        "endpoints": {"salud": "/health", "prediccion_json": "/predict", "prediccion_csv": "/predict/csv"},
        "documentacion": "/docs",
    }


@app.get("/health")
def salud(request: Request):
    """Estado del servicio y del modelo."""
    cargado = request.app.state.modelo is not None
    return {"estado": "ok" if cargado else "sin_modelo", "modelo_cargado": cargado,
            "modelo": request.app.state.nombre_modelo}


@app.post("/predict", response_model=RespuestaPrediccion)
def predecir_json(request: Request, solicitud: SolicitudPrediccion):
    """Prediccion por lotes a partir de un JSON: {"registros": [ {...}, {...} ]}."""
    return _respuesta(request, solicitud.registros)


@app.post("/predict/csv", response_model=RespuestaPrediccion)
async def predecir_csv(
    request: Request,
    archivo: UploadFile = File(..., description="CSV con las mismas columnas de Base_de_datos.csv"),
    formato: str = Query("json", pattern="^(json|csv)$", description="Formato de la respuesta"),
):
    """Prediccion por lotes a partir de un archivo CSV."""
    contenido = await archivo.read()
    if not contenido:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")
    if len(contenido) > MAX_BYTES_CSV:
        raise HTTPException(status_code=413, detail="El archivo supera el tamaño máximo de 10 MB.")
    try:
        datos = leer_csv(contenido)
    except (UnicodeDecodeError, pd.errors.ParserError) as error:
        raise HTTPException(status_code=400, detail=f"No se pudo leer el CSV: {error}") from error
    if datos.empty or len(datos) > MAX_REGISTROS:
        raise HTTPException(status_code=400, detail=f"El CSV debe tener entre 1 y {MAX_REGISTROS} registros.")

    registros = datos.replace({np.nan: None}).to_dict(orient="records")
    try:
        solicitudes = LISTA_SOLICITUDES.validate_python(registros)
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=error.errors(include_url=False, include_context=False)) \
            from error

    respuesta = _respuesta(request, solicitudes)
    if formato == "json":
        return respuesta

    resultado = datos.copy()
    resultado["prediccion"] = [p.prediccion for p in respuesta.predicciones]
    resultado["probabilidad_no_pago"] = [p.probabilidad_no_pago for p in respuesta.predicciones]
    resultado["nivel_riesgo"] = [p.nivel_riesgo for p in respuesta.predicciones]
    return StreamingResponse(
        io.StringIO(resultado.to_csv(index=False)),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predicciones.csv"},
    )


def main():
    """Levanta el servicio localmente."""
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
