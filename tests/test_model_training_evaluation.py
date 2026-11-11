"""Pruebas del entrenamiento y la evaluacion de modelos."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import model_training_evaluation as mte


def test_summarize_classification_prediccion_perfecta():
    y_real = np.array([0, 0, 1, 1, 1])
    metricas = mte.summarize_classification(y_real, y_real, np.array([0.9, 0.8, 0.1, 0.2, 0.3]))
    assert metricas["accuracy"] == 1.0
    assert metricas["recall_no_pago"] == 1.0
    assert metricas["roc_auc"] == 1.0


def test_summarize_classification_mide_la_clase_no_pago():
    y_real = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 1, 1])  # nunca detecta el no pago
    metricas = mte.summarize_classification(y_real, y_pred, np.array([0.4, 0.3, 0.2, 0.1]))
    assert metricas["accuracy"] == 0.5
    assert metricas["recall_no_pago"] == 0.0
    assert metricas["precision_no_pago"] == 0.0


def test_build_model_retorna_pipeline_y_metricas(datos_divididos):
    X_train, X_test, y_train, y_test, preprocesador = datos_divididos
    modelo = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    resultado = mte.build_model("Logistica", modelo, preprocesador, X_train, y_train, X_test, y_test)
    resumen = resultado["resumen"]
    assert resumen["modelo"] == "Logistica"
    assert 0.6 < resumen["roc_auc"] < 0.8
    assert len(resultado["cv_roc_auc"]) == mte.N_FOLDS
    assert len(resultado["prob_test"]) == len(X_test)
    assert hasattr(resultado["pipeline"], "predict_proba")


def test_seleccionar_mejor_modelo_descarta_sobreajuste():
    tabla = pd.DataFrame([
        {"modelo": "Linea base (Dummy)", "diferencia_train_test": 0.0, "cv_roc_auc_desv": 0.0,
         "cv_roc_auc_media": 0.5, "tiempo_entrenamiento_s": 0.01},
        {"modelo": "Sobreajustado", "diferencia_train_test": 0.30, "cv_roc_auc_desv": 0.01,
         "cv_roc_auc_media": 0.75, "tiempo_entrenamiento_s": 1.0},
        {"modelo": "Consistente", "diferencia_train_test": 0.02, "cv_roc_auc_desv": 0.01,
         "cv_roc_auc_media": 0.70, "tiempo_entrenamiento_s": 2.0},
    ])
    assert mte.seleccionar_mejor_modelo(tabla) == "Consistente"


def test_seleccionar_mejor_modelo_desempata_por_tiempo():
    tabla = pd.DataFrame([
        {"modelo": "Lento", "diferencia_train_test": 0.01, "cv_roc_auc_desv": 0.01,
         "cv_roc_auc_media": 0.700, "tiempo_entrenamiento_s": 5.0},
        {"modelo": "Rapido", "diferencia_train_test": 0.01, "cv_roc_auc_desv": 0.01,
         "cv_roc_auc_media": 0.698, "tiempo_entrenamiento_s": 0.1},
    ])
    assert mte.seleccionar_mejor_modelo(tabla) == "Rapido"


def test_obtener_modelos_incluye_linea_base():
    modelos = mte.obtener_modelos()
    assert "Linea base (Dummy)" in modelos
    assert len(modelos) == 5


@pytest.mark.filterwarnings("ignore")
def test_main_guarda_modelo_tabla_y_graficos(tmp_path, monkeypatch):
    monkeypatch.setattr(mte, "CARPETA_ARTEFACTOS", tmp_path)
    monkeypatch.setattr(mte, "obtener_modelos", lambda: {
        "Linea base (Dummy)": mte.DummyClassifier(strategy="most_frequent", random_state=42),
        "Regresion Logistica": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
        "Arbol de Decision": DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=1),
    })
    mte.main()
    for archivo in ["mejor_modelo.joblib", "mejor_modelo_info.json", "tabla_resumen.csv",
                    "comparacion_modelos.png", "curvas_roc_pr.png", "consistencia_cv.png",
                    "matriz_confusion.png", "importancia_variables.png"]:
        assert (tmp_path / archivo).exists(), archivo
    info = json.loads((tmp_path / "mejor_modelo_info.json").read_text(encoding="utf-8"))
    assert info["modelo"] in {"Regresion Logistica", "Arbol de Decision"}
