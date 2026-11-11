"""Configuracion comun de las pruebas: rutas del proyecto y un modelo pequeno de prueba."""

import json
import sys
from pathlib import Path

import joblib
import matplotlib
import pytest

matplotlib.use("Agg")

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "mlops_pipeline"))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

import ft_engineering  # noqa: E402


@pytest.fixture(scope="session")
def datos_divididos():
    """Conjuntos de entrenamiento/evaluacion y preprocesador reales del proyecto."""
    return ft_engineering.ejecutar_ft_engineering()


@pytest.fixture(scope="session")
def modelo_prueba(datos_divididos):
    """Pipeline rapido (preprocesador + regresion logistica) para no depender del modelo guardado."""
    X_train, _, y_train, _, preprocesador = datos_divididos
    modelo = Pipeline([
        ("preprocesador", preprocesador),
        ("modelo", LogisticRegression(class_weight="balanced", max_iter=2000)),
    ])
    return modelo.fit(X_train, y_train)


@pytest.fixture(scope="session")
def ruta_modelo_prueba(modelo_prueba, tmp_path_factory):
    """Guarda el modelo de prueba y su ficha en una carpeta temporal."""
    carpeta = tmp_path_factory.mktemp("artefactos")
    ruta = carpeta / "mejor_modelo.joblib"
    joblib.dump(modelo_prueba, ruta)
    (carpeta / "mejor_modelo_info.json").write_text(json.dumps({"modelo": "Modelo de prueba"}), encoding="utf-8")
    return ruta
