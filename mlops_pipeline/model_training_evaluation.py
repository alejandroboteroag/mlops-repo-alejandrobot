"""
model_training_evaluation.py
----------------------------
Entrenamiento y evaluacion de modelos supervisados.

- Toma los conjuntos de entrenamiento/evaluacion y el preprocesador de ft_engineering.py.
- Entrena varios modelos con funciones reutilizables (build_model, summarize_classification).
- Compara los modelos en tres ejes:
    * performance: metricas en el conjunto de evaluacion (ROC-AUC, PR-AUC, recall, F1)
    * consistency: validacion cruzada estratificada (promedio y desviacion) y diferencia train/test
    * scalability: tiempo de entrenamiento y de prediccion
- Genera una tabla resumen y graficos comparativos.
- Guarda el mejor modelo (pipeline completo: preprocesador + modelo) en formato .joblib.

La clase de interes es Pago_atiempo = 0 (el cliente NO paga a tiempo).
"""

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_engineering import OBJETIVO, SEMILLA, ejecutar_ft_engineering  # noqa: E402

CARPETA_ARTEFACTOS = Path(__file__).resolve().parent / "artefactos"
CLASE_NO_PAGO = 0
N_FOLDS = 5

# Criterios de consistencia para que un modelo pueda ser elegido
MAX_DIFERENCIA_TRAIN_TEST = 0.10   # diferencia maxima de ROC-AUC entre train y test
MAX_DESVIACION_CV = 0.05           # desviacion maxima del ROC-AUC en validacion cruzada


def obtener_modelos():
    """Modelos candidatos. Todos compensan el desbalance de clases."""
    return {
        "Linea base (Dummy)": DummyClassifier(strategy="most_frequent"),
        "Regresion Logistica": LogisticRegression(class_weight="balanced", max_iter=2000),
        "Arbol de Decision": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=50, class_weight="balanced", random_state=SEMILLA
        ),
        # Los modelos de arboles se regularizan (poca profundidad, hojas grandes)
        # porque con valores por defecto se sobreajustan mucho a esta base
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=100, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEMILLA
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=150, max_depth=2, min_samples_leaf=150,
            l2_regularization=5.0, class_weight="balanced", random_state=SEMILLA
        ),
    }


def probabilidad_no_pago(modelo, X):
    """Probabilidad de que el cliente NO pague a tiempo (clase 0)."""
    indice = list(modelo.classes_).index(CLASE_NO_PAGO)
    return modelo.predict_proba(X)[:, indice]


def summarize_classification(y_real, y_pred, prob_no_pago):
    """Calcula las metricas de clasificacion enfocadas en la clase de no pago."""
    es_no_pago = (np.asarray(y_real) == CLASE_NO_PAGO).astype(int)
    return {
        "accuracy": accuracy_score(y_real, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_pred),
        "precision_no_pago": precision_score(y_real, y_pred, pos_label=CLASE_NO_PAGO, zero_division=0),
        "recall_no_pago": recall_score(y_real, y_pred, pos_label=CLASE_NO_PAGO, zero_division=0),
        "f1_no_pago": f1_score(y_real, y_pred, pos_label=CLASE_NO_PAGO, zero_division=0),
        "roc_auc": roc_auc_score(es_no_pago, prob_no_pago),
        "pr_auc": average_precision_score(es_no_pago, prob_no_pago),
    }


def build_model(nombre, estimador, preprocesador, X_train, y_train, X_test, y_test):
    """Construye el pipeline (preprocesador + modelo), lo valida, entrena y evalua.

    Retorna un diccionario con el pipeline entrenado y todas sus metricas.
    """
    pipeline = Pipeline(steps=[
        ("preprocesador", clone(preprocesador)),
        ("modelo", estimador),
    ])

    # Consistency: validacion cruzada estratificada sobre el conjunto de entrenamiento
    validacion = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA)
    metricas_cv = {
        "roc_auc": "roc_auc",
        "recall_no_pago": make_scorer(recall_score, pos_label=CLASE_NO_PAGO, zero_division=0),
    }
    resultado_cv = cross_validate(pipeline, X_train, y_train, cv=validacion, scoring=metricas_cv)

    # Scalability: tiempo de entrenamiento y de prediccion
    inicio = time.perf_counter()
    pipeline.fit(X_train, y_train)
    tiempo_entrenamiento = time.perf_counter() - inicio

    inicio = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    prob_test = probabilidad_no_pago(pipeline, X_test)
    tiempo_prediccion = (time.perf_counter() - inicio) / len(X_test) * 1000

    # Performance: metricas en train y test
    metricas_test = summarize_classification(y_test, y_pred, prob_test)
    metricas_train = summarize_classification(
        y_train, pipeline.predict(X_train), probabilidad_no_pago(pipeline, X_train)
    )

    resumen = {"modelo": nombre}
    resumen.update(metricas_test)
    resumen.update({
        "roc_auc_train": metricas_train["roc_auc"],
        "diferencia_train_test": metricas_train["roc_auc"] - metricas_test["roc_auc"],
        "cv_roc_auc_media": resultado_cv["test_roc_auc"].mean(),
        "cv_roc_auc_desv": resultado_cv["test_roc_auc"].std(),
        "cv_recall_no_pago_media": resultado_cv["test_recall_no_pago"].mean(),
        "tiempo_entrenamiento_s": tiempo_entrenamiento,
        "tiempo_prediccion_ms_x1000": tiempo_prediccion * 1000,
    })

    return {
        "pipeline": pipeline,
        "resumen": resumen,
        "cv_roc_auc": resultado_cv["test_roc_auc"],
        "prob_test": prob_test,
        "y_pred": y_pred,
    }


def seleccionar_mejor_modelo(tabla):
    """Elige el mejor modelo segun performance, consistency y scalability.

    1. Se descarta la linea base.
    2. Consistency: se descartan modelos con sobreajuste (diferencia train/test alta)
       o inestables (desviacion alta en validacion cruzada).
    3. Performance: entre los que quedan, el de mayor ROC-AUC promedio en validacion cruzada.
    4. Scalability: en caso de empate (diferencia < 0.005), el de menor tiempo de entrenamiento.
    """
    candidatos = tabla[tabla["modelo"] != "Linea base (Dummy)"].copy()
    consistentes = candidatos[
        (candidatos["diferencia_train_test"] <= MAX_DIFERENCIA_TRAIN_TEST)
        & (candidatos["cv_roc_auc_desv"] <= MAX_DESVIACION_CV)
    ]
    if consistentes.empty:
        consistentes = candidatos

    mejor_auc = consistentes["cv_roc_auc_media"].max()
    empatados = consistentes[consistentes["cv_roc_auc_media"] >= mejor_auc - 0.005]
    return empatados.sort_values("tiempo_entrenamiento_s").iloc[0]["modelo"]


def graficar_comparacion(tabla, ruta):
    """Barras con las metricas principales de cada modelo en el conjunto de evaluacion."""
    metricas = ["roc_auc", "pr_auc", "recall_no_pago", "precision_no_pago", "f1_no_pago", "accuracy"]
    datos = tabla.set_index("modelo")[metricas]
    ax = datos.plot(kind="bar", figsize=(12, 5), width=0.8, colormap="tab10")
    ax.set_title("Comparacion de modelos (conjunto de evaluacion)")
    ax.set_ylabel("Valor de la metrica")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(ruta, dpi=110)
    plt.close()


def graficar_curvas(resultados, y_test, ruta):
    """Curvas ROC y Precision-Recall de la clase de no pago para cada modelo."""
    es_no_pago = (y_test == CLASE_NO_PAGO).astype(int)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5))
    for nombre, resultado in resultados.items():
        fpr, tpr, _ = roc_curve(es_no_pago, resultado["prob_test"])
        precision, recall, _ = precision_recall_curve(es_no_pago, resultado["prob_test"])
        auc = resultado["resumen"]["roc_auc"]
        ap = resultado["resumen"]["pr_auc"]
        ax_roc.plot(fpr, tpr, label=f"{nombre} (AUC={auc:.3f})")
        if nombre != "Linea base (Dummy)":
            ax_pr.plot(recall, precision, label=f"{nombre} (AP={ap:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
    ax_roc.set(title="Curva ROC", xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos")
    ax_pr.axhline(es_no_pago.mean(), color="k", ls="--", lw=1, label="Azar")
    ax_pr.set(title="Curva Precision-Recall (clase no pago)", xlabel="Recall", ylabel="Precision")
    ax_roc.legend(fontsize=8)
    ax_pr.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(ruta, dpi=110)
    plt.close()


def graficar_consistencia(resultados, ruta):
    """Distribucion del ROC-AUC en los folds de validacion cruzada (consistency)."""
    nombres = list(resultados)
    valores = [resultados[n]["cv_roc_auc"] for n in nombres]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.boxplot(valores, tick_labels=nombres)
    ax.set_title(f"ROC-AUC en validacion cruzada ({N_FOLDS} folds)")
    ax.set_ylabel("ROC-AUC")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(ruta, dpi=110)
    plt.close()


def graficar_matriz_confusion(y_test, y_pred, nombre, ruta):
    """Matriz de confusion del modelo seleccionado."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=[0, 1], display_labels=["No pago", "Pago"], cmap="Blues", ax=ax
    )
    ax.set_title(f"Matriz de confusion - {nombre}")
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Valor real")
    plt.tight_layout()
    plt.savefig(ruta, dpi=110)
    plt.close()


def graficar_importancia(pipeline, X_test, y_test, ruta, top=15):
    """Importancia por permutacion: cuanto cae el ROC-AUC al desordenar cada variable."""
    importancia = permutation_importance(
        pipeline, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=SEMILLA, n_jobs=-1
    )
    serie = pd.Series(importancia.importances_mean, index=X_test.columns).sort_values()
    serie = serie.tail(top)
    fig, ax = plt.subplots(figsize=(8, 6))
    serie.plot(kind="barh", color="#2E86C1", ax=ax)
    ax.set_title("Importancia de variables (permutacion, caida del ROC-AUC)")
    ax.set_xlabel("Caida promedio del ROC-AUC")
    plt.tight_layout()
    plt.savefig(ruta, dpi=110)
    plt.close()
    return serie.sort_values(ascending=False)


def main():
    CARPETA_ARTEFACTOS.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test, preprocesador = ejecutar_ft_engineering()

    print("=== Entrenamiento y evaluacion de modelos ===")
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,} | Clase de interes: {OBJETIVO} = 0\n")

    resultados = {}
    for nombre, estimador in obtener_modelos().items():
        print(f"Entrenando: {nombre} ...")
        resultados[nombre] = build_model(nombre, estimador, preprocesador, X_train, y_train, X_test, y_test)

    tabla = pd.DataFrame([r["resumen"] for r in resultados.values()])
    tabla = tabla.sort_values("cv_roc_auc_media", ascending=False).reset_index(drop=True)
    tabla.to_csv(CARPETA_ARTEFACTOS / "tabla_resumen.csv", index=False)

    columnas_vista = [
        "modelo", "roc_auc", "pr_auc", "recall_no_pago", "precision_no_pago", "f1_no_pago",
        "accuracy", "cv_roc_auc_media", "cv_roc_auc_desv", "diferencia_train_test", "tiempo_entrenamiento_s",
    ]
    print("\n=== Tabla resumen ===")
    print(tabla[columnas_vista].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    mejor = seleccionar_mejor_modelo(tabla)
    resultado_mejor = resultados[mejor]
    print(f"\nModelo seleccionado: {mejor}")

    graficar_comparacion(tabla, CARPETA_ARTEFACTOS / "comparacion_modelos.png")
    graficar_curvas(resultados, y_test, CARPETA_ARTEFACTOS / "curvas_roc_pr.png")
    graficar_consistencia(resultados, CARPETA_ARTEFACTOS / "consistencia_cv.png")
    graficar_matriz_confusion(y_test, resultado_mejor["y_pred"], mejor, CARPETA_ARTEFACTOS / "matriz_confusion.png")
    importancia = graficar_importancia(
        resultado_mejor["pipeline"], X_test, y_test, CARPETA_ARTEFACTOS / "importancia_variables.png"
    )
    print("\nVariables mas importantes del modelo seleccionado:")
    print(importancia.head(10).to_string(float_format=lambda x: f"{x:.4f}"))

    joblib.dump(resultado_mejor["pipeline"], CARPETA_ARTEFACTOS / "mejor_modelo.joblib")
    info = {
        "modelo": mejor,
        "metricas_test": {k: round(float(v), 4) for k, v in resultado_mejor["resumen"].items() if k != "modelo"},
        "variables_entrada": list(X_train.columns),
        "objetivo": OBJETIVO,
        "clase_interes": CLASE_NO_PAGO,
    }
    with open(CARPETA_ARTEFACTOS / "mejor_modelo_info.json", "w", encoding="utf-8") as archivo:
        json.dump(info, archivo, indent=2, ensure_ascii=False)

    print(f"\nModelo guardado en: {CARPETA_ARTEFACTOS / 'mejor_modelo.joblib'}")
    print(f"Graficos y tabla guardados en: {CARPETA_ARTEFACTOS}")


if __name__ == "__main__":
    main()
