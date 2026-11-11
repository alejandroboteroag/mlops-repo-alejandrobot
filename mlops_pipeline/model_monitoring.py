"""
model_monitoring.py
-------------------
Monitoreo del modelo y deteccion de data drift.

1. Construye la tabla de monitoreo: los datos de cada credito junto con el
   pronostico entregado por el modelo (probabilidad de no pago y prediccion).
2. Muestrea esa tabla con una periodicidad definida (mensual o trimestral).
3. Compara cada periodo contra la poblacion de referencia (datos de entrenamiento)
   con cuatro metricas de drift:
     * Kolmogorov-Smirnov (variables numericas continuas)
     * Chi-cuadrado (variables categoricas y discretas)
     * Population Stability Index (PSI)
     * Distancia de Jensen-Shannon
4. Clasifica cada variable en un semaforo (Verde, Amarillo, Rojo), genera alertas
   y recomendaciones (revisar variables o reentrenar el modelo).

Uso:
    python mlops_pipeline/model_monitoring.py            -> ejecuta el monitoreo y guarda resultados
    streamlit run mlops_pipeline/model_monitoring.py     -> abre la aplicacion de monitoreo
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_engineering import (  # noqa: E402
    OBJETIVO,
    VARIABLES_LOG,
    cargar_datos,
    crear_atributos,
    ejecutar_ft_engineering,
    limpiar_datos,
)

CARPETA_ARTEFACTOS = Path(__file__).resolve().parent / "artefactos"
RUTA_MODELO = CARPETA_ARTEFACTOS / "mejor_modelo.joblib"

COLUMNA_FECHA = "fecha_prestamo"
COLUMNA_PROBABILIDAD = "prob_no_pago"
COLUMNA_PREDICCION = "prediccion"
COLUMNA_PERIODO = "periodo"
COLUMNA_ENTRENAMIENTO = "en_entrenamiento"
CLASE_NO_PAGO = 0

FRECUENCIAS = {"Mensual": "M", "Trimestral": "Q"}
TAMANO_MINIMO = 100      # registros minimos para que un periodo genere alertas
MAX_CATEGORIAS = 10      # una variable con pocos valores distintos se trata como discreta
N_BINS = 10              # intervalos (deciles de la referencia) para PSI y Jensen-Shannon
EPSILON = 1e-4           # evita divisiones por cero y logaritmos de cero
P_VALOR_CRITICO = 0.05
SALTO_ABRUPTO_PSI = 0.10

# Umbrales (amarillo, rojo). PSI: estandar de la industria de riesgo de credito.
UMBRALES = {
    "psi": (0.10, 0.25),
    "jensen_shannon": (0.15, 0.25),
    "ks": (0.10, 0.20),
}
CAIDA_MAXIMA_AUC = 0.10  # caida de ROC-AUC frente a la referencia que activa reentrenamiento
MIN_CASOS_CLASE = 10     # casos minimos de cada clase para calcular un ROC-AUC confiable

VERDE = "Verde"
AMARILLO = "Amarillo"
ROJO = "Rojo"
SIN_DATOS = "Muestra insuficiente"
SEVERIDAD = {VERDE: 0, AMARILLO: 1, ROJO: 2, SIN_DATOS: -1}
ICONOS = {VERDE: "🟢", AMARILLO: "🟡", ROJO: "🔴", SIN_DATOS: "⚪"}
COLORES = {VERDE: "#C8E6C9", AMARILLO: "#FFF3C4", ROJO: "#F8C9C4", SIN_DATOS: "#E0E0E0"}

NOMBRE_PREDICCIONES = "prob_no_pago (predicciones)"


# ---------------------------------------------------------------------------
# 1. Tabla de monitoreo: datos + pronosticos
# ---------------------------------------------------------------------------
def cargar_modelo(ruta=None):
    """Carga el pipeline entrenado (preprocesador + modelo)."""
    ruta = Path(ruta) if ruta is not None else RUTA_MODELO
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo en {ruta}. Ejecute primero model_training_evaluation.py"
        )
    return joblib.load(ruta)


def probabilidad_no_pago(modelo, X):
    """Probabilidad de que el cliente NO pague a tiempo (clase 0)."""
    indice = list(modelo.classes_).index(CLASE_NO_PAGO)
    return modelo.predict_proba(X)[:, indice]


def construir_tabla_monitoreo(modelo=None, datos=None):
    """Une los datos de cada credito con el pronostico del modelo.

    Retorna
    -------
    referencia : DataFrame con la poblacion de entrenamiento (linea base) y sus pronosticos.
    tabla : DataFrame con todos los creditos, su fecha, el valor real, el pronostico y si el
        credito se uso para entrenar (el desempeno solo se mide con los que no se usaron).
    """
    modelo = modelo if modelo is not None else cargar_modelo()
    datos = datos if datos is not None else cargar_datos()

    limpios = limpiar_datos(datos)
    fechas = limpios[COLUMNA_FECHA]
    features = crear_atributos(limpios)
    X = features.drop(columns=[OBJETIVO])
    X["tendencia_ingresos"] = X["tendencia_ingresos"].astype(object)

    tabla = X.copy()
    tabla[COLUMNA_FECHA] = fechas
    tabla[OBJETIVO] = features[OBJETIVO].astype(int)
    tabla[COLUMNA_PROBABILIDAD] = probabilidad_no_pago(modelo, X)
    tabla[COLUMNA_PREDICCION] = modelo.predict(X)

    X_train = ejecutar_ft_engineering()[0]
    tabla[COLUMNA_ENTRENAMIENTO] = tabla.index.isin(X_train.index)
    referencia = tabla[tabla[COLUMNA_ENTRENAMIENTO]].copy()
    return referencia, tabla


def variables_monitoreadas(tabla):
    """Variables de entrada del modelo (sin fecha, objetivo ni pronosticos)."""
    excluir = {COLUMNA_FECHA, OBJETIVO, COLUMNA_PROBABILIDAD, COLUMNA_PREDICCION, COLUMNA_PERIODO,
               COLUMNA_ENTRENAMIENTO}
    return [c for c in tabla.columns if c not in excluir]


def muestreo_periodico(tabla, frecuencia="M", tamano_muestra=None, semilla=42):
    """Agrupa la tabla por periodo y, opcionalmente, toma una muestra aleatoria de cada uno.

    Retorna un diccionario {periodo: DataFrame} ordenado cronologicamente.
    """
    periodos = tabla[COLUMNA_FECHA].dt.to_period(frecuencia)
    muestras = {}
    for periodo, grupo in tabla.groupby(periodos, sort=True):
        if tamano_muestra is not None and len(grupo) > tamano_muestra:
            grupo = grupo.sample(n=tamano_muestra, random_state=semilla)
        muestras[str(periodo)] = grupo
    return muestras


# ---------------------------------------------------------------------------
# 2. Metricas de drift
# ---------------------------------------------------------------------------
def es_discreta(serie):
    """Una variable de texto o con pocos valores distintos se trata como categorica."""
    es_texto = serie.dtype == object or str(serie.dtype) in ("category", "str", "string")
    return es_texto or serie.nunique(dropna=True) <= MAX_CATEGORIAS


def _proporciones(conteos):
    """Convierte conteos en proporciones sin ceros (para PSI y Jensen-Shannon)."""
    proporciones = np.asarray(conteos, dtype=float)
    proporciones = proporciones / max(proporciones.sum(), 1.0)
    proporciones = np.clip(proporciones, EPSILON, None)
    return proporciones / proporciones.sum()


def _conteo_por_intervalos(serie, cortes, etiquetas):
    """Cuenta los valores de una serie numerica en intervalos fijos (los nulos aparte)."""
    intervalos = pd.cut(serie.astype(float), bins=cortes, labels=False, include_lowest=True)
    intervalos = intervalos.astype(object).where(serie.notna(), "Sin dato")
    return intervalos.value_counts().reindex(etiquetas, fill_value=0).to_numpy()


def conteos_comparables(referencia, actual, discreta):
    """Cuenta los valores de referencia y actual en los mismos intervalos o categorias.

    Los nulos se cuentan en su propio grupo: un cambio en la proporcion de nulos tambien es drift.
    """
    if discreta:
        ref = referencia.astype(object).where(referencia.notna(), "Sin dato").astype(str)
        act = actual.astype(object).where(actual.notna(), "Sin dato").astype(str)
        categorias = sorted(set(ref) | set(act))
        conteo_ref = ref.value_counts().reindex(categorias, fill_value=0)
        conteo_act = act.value_counts().reindex(categorias, fill_value=0)
        return conteo_ref.to_numpy(), conteo_act.to_numpy(), categorias

    cortes = np.unique(np.nanquantile(referencia.astype(float), np.linspace(0, 1, N_BINS + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    etiquetas = list(range(len(cortes) - 1)) + ["Sin dato"]
    return (_conteo_por_intervalos(referencia, cortes, etiquetas),
            _conteo_por_intervalos(actual, cortes, etiquetas), etiquetas)


def calcular_psi(conteo_referencia, conteo_actual):
    """Population Stability Index: suma de (actual - ref) * ln(actual / ref)."""
    p = _proporciones(conteo_referencia)
    q = _proporciones(conteo_actual)
    return float(np.sum((q - p) * np.log(q / p)))


def calcular_jensen_shannon(conteo_referencia, conteo_actual):
    """Distancia de Jensen-Shannon en base 2 (0 = identicas, 1 = totalmente distintas)."""
    return float(jensenshannon(_proporciones(conteo_referencia), _proporciones(conteo_actual), base=2))


def prueba_ks(referencia, actual):
    """Prueba de Kolmogorov-Smirnov de dos muestras (variables continuas)."""
    ref, act = referencia.dropna().astype(float), actual.dropna().astype(float)
    if len(ref) < 2 or len(act) < 2:
        return np.nan, np.nan
    resultado = stats.ks_2samp(ref, act)
    return float(resultado.statistic), float(resultado.pvalue)


def prueba_chi_cuadrado(conteo_referencia, conteo_actual):
    """Prueba Chi-cuadrado de homogeneidad entre la distribucion de referencia y la actual."""
    tabla = np.vstack([conteo_referencia, conteo_actual])
    tabla = tabla[:, tabla.sum(axis=0) > 0]
    if tabla.shape[1] < 2 or tabla[1].sum() == 0:
        return np.nan, np.nan
    estadistico, p_valor, _, _ = stats.chi2_contingency(tabla)
    return float(estadistico), float(p_valor)


def nivel_por_umbral(valor, umbrales):
    """Verde, Amarillo o Rojo segun los umbrales (amarillo, rojo)."""
    if pd.isna(valor) or valor < umbrales[0]:
        return VERDE
    return AMARILLO if valor < umbrales[1] else ROJO


def peor_estado(estados):
    """El estado mas severo de una lista."""
    return max(estados, key=lambda estado: SEVERIDAD[estado])


def _resultado_prueba(referencia, actual, conteo_ref, conteo_act, discreta):
    """Aplica Chi-cuadrado (discretas) o KS (continuas).

    Retorna el nombre de la prueba, el estadistico, el p-valor, si la diferencia es
    significativa y el estado que aporta la propia prueba.
    """
    if discreta:
        estadistico, p_valor = prueba_chi_cuadrado(conteo_ref, conteo_act)
        significativa = not pd.isna(p_valor) and p_valor < P_VALOR_CRITICO
        return "Chi-cuadrado", estadistico, p_valor, significativa, VERDE

    estadistico, p_valor = prueba_ks(referencia, actual)
    significativa = not pd.isna(p_valor) and p_valor < P_VALOR_CRITICO
    return "KS", estadistico, p_valor, significativa, nivel_por_umbral(estadistico, UMBRALES["ks"])


def evaluar_variable(nombre, referencia, actual):
    """Calcula todas las metricas de drift de una variable y su estado de semaforo.

    Una variable solo pasa a amarillo o rojo si la prueba estadistica confirma que el
    cambio es significativo (p < 0.05). Asi se evita que el ruido de las muestras
    pequenas, que infla el PSI y Jensen-Shannon, genere falsas alarmas.
    """
    discreta = es_discreta(referencia)
    conteo_ref, conteo_act, _ = conteos_comparables(referencia, actual, discreta)
    psi = calcular_psi(conteo_ref, conteo_act)
    js = calcular_jensen_shannon(conteo_ref, conteo_act)
    prueba, estadistico, p_valor, significativa, estado_prueba = _resultado_prueba(
        referencia, actual, conteo_ref, conteo_act, discreta
    )
    estado = VERDE
    if significativa:
        estado = peor_estado([
            nivel_por_umbral(psi, UMBRALES["psi"]),
            nivel_por_umbral(js, UMBRALES["jensen_shannon"]),
            estado_prueba,
        ])
    return {
        "variable": nombre,
        "tipo": "Categorica / discreta" if discreta else "Numerica continua",
        "prueba": prueba,
        "estadistico": estadistico,
        "p_valor": p_valor,
        "psi": psi,
        "jensen_shannon": js,
        "estado": estado,
    }


def calcular_drift_periodo(referencia, actual, variables):
    """Drift de todas las variables y de las predicciones para un periodo."""
    filas = [evaluar_variable(v, referencia[v], actual[v]) for v in variables]
    fila_pred = evaluar_variable(COLUMNA_PROBABILIDAD, referencia[COLUMNA_PROBABILIDAD],
                                 actual[COLUMNA_PROBABILIDAD])
    fila_pred["variable"] = NOMBRE_PREDICCIONES
    filas.append(fila_pred)
    resultado = pd.DataFrame(filas)
    resultado["registros"] = len(actual)
    if len(actual) < TAMANO_MINIMO:
        resultado["estado"] = SIN_DATOS
    return resultado


def calcular_desempeno(datos):
    """Desempeno del modelo en un grupo de creditos cuyo resultado real ya se conoce.

    Las tasas usan todos los creditos del grupo. El ROC-AUC y el recall se calculan solo con
    los creditos que no se usaron para entrenar (medirlos sobre datos de entrenamiento daria
    un resultado optimista) y solo si hay al menos MIN_CASOS_CLASE casos de cada clase.
    """
    y_real = datos[OBJETIVO]
    evaluables = datos[~datos[COLUMNA_ENTRENAMIENTO]] if COLUMNA_ENTRENAMIENTO in datos else datos
    y_eval = evaluables[OBJETIVO]
    casos_no_pago = int((y_eval == CLASE_NO_PAGO).sum())
    casos_pago = int((y_eval != CLASE_NO_PAGO).sum())

    roc_auc, recall = np.nan, np.nan
    if casos_no_pago >= MIN_CASOS_CLASE and casos_pago >= MIN_CASOS_CLASE:
        roc_auc = roc_auc_score((y_eval == CLASE_NO_PAGO).astype(int), evaluables[COLUMNA_PROBABILIDAD])
        recall = recall_score(y_eval, evaluables[COLUMNA_PREDICCION], pos_label=CLASE_NO_PAGO, zero_division=0)
    return {
        "registros": len(datos),
        "registros_evaluados": len(evaluables),
        "casos_no_pago_evaluados": casos_no_pago,
        "tasa_real_no_pago": float((y_real == CLASE_NO_PAGO).mean()),
        "tasa_predicha_no_pago": float((datos[COLUMNA_PREDICCION] == CLASE_NO_PAGO).mean()),
        "prob_media_no_pago": float(datos[COLUMNA_PROBABILIDAD].mean()),
        "roc_auc": roc_auc,
        "recall_no_pago": recall,
    }


def monitorear(referencia, tabla, frecuencia="M", tamano_muestra=None):
    """Ejecuta el monitoreo periodico completo.

    Retorna
    -------
    historial : metricas de drift por periodo y variable.
    desempeno : metricas de desempeno del modelo por periodo.
    """
    variables = variables_monitoreadas(tabla)
    historiales, desempenos = [], []
    for periodo, actual in muestreo_periodico(tabla, frecuencia, tamano_muestra).items():
        drift = calcular_drift_periodo(referencia, actual, variables)
        drift.insert(0, COLUMNA_PERIODO, periodo)
        historiales.append(drift)
        desempenos.append({COLUMNA_PERIODO: periodo, **calcular_desempeno(actual)})
    return pd.concat(historiales, ignore_index=True), pd.DataFrame(desempenos)


# ---------------------------------------------------------------------------
# 3. Analisis temporal, alertas y recomendaciones
# ---------------------------------------------------------------------------
def periodos_validos(historial):
    """Periodos con muestra suficiente para sacar conclusiones."""
    validos = historial.loc[historial["estado"] != SIN_DATOS, COLUMNA_PERIODO]
    return list(dict.fromkeys(validos))


def detectar_cambios_abruptos(historial, umbral=SALTO_ABRUPTO_PSI):
    """Variables cuyo PSI sube mas que el umbral de un periodo valido al siguiente."""
    validos = historial[historial["estado"] != SIN_DATOS].sort_values(COLUMNA_PERIODO)
    validos = validos.assign(cambio_psi=validos.groupby("variable")["psi"].diff())
    cambios = validos[validos["cambio_psi"] > umbral]
    return cambios[[COLUMNA_PERIODO, "variable", "psi", "cambio_psi", "estado"]].reset_index(drop=True)


def tendencia_psi(historial):
    """Pendiente del PSI por variable (cuanto sube o baja por periodo, en promedio)."""
    validos = historial[historial["estado"] != SIN_DATOS].sort_values(COLUMNA_PERIODO)
    filas = []
    for variable, grupo in validos.groupby("variable"):
        if len(grupo) < 3:
            continue
        pendiente = np.polyfit(np.arange(len(grupo)), grupo["psi"], 1)[0]
        filas.append({"variable": variable, "pendiente_psi": float(pendiente),
                      "psi_ultimo": float(grupo["psi"].iloc[-1])})
    if not filas:
        return pd.DataFrame(columns=["variable", "pendiente_psi", "psi_ultimo", "tendencia"])
    resultado = pd.DataFrame(filas)
    resultado["tendencia"] = np.select(
        [resultado["pendiente_psi"] > 0.01, resultado["pendiente_psi"] < -0.01],
        ["Creciente", "Decreciente"], default="Estable",
    )
    return resultado.sort_values("pendiente_psi", ascending=False).reset_index(drop=True)


def estado_general(drift_periodo):
    """Estado global de un periodo: el peor estado entre sus variables."""
    if (drift_periodo["estado"] == SIN_DATOS).all():
        return SIN_DATOS
    return peor_estado(drift_periodo["estado"].tolist())


def _caida_auc(desempeno_periodo, desempeno_referencia):
    return desempeno_referencia["roc_auc"] - desempeno_periodo["roc_auc"]


def generar_alertas(drift_periodo, desempeno_periodo, desempeno_referencia):
    """Mensajes de alerta automaticos para un periodo."""
    periodo = drift_periodo[COLUMNA_PERIODO].iloc[0]
    if (drift_periodo["estado"] == SIN_DATOS).all():
        return [{"nivel": SIN_DATOS, "mensaje":
                 f"{periodo}: solo {int(drift_periodo['registros'].iloc[0])} registros "
                 f"(mínimo {TAMANO_MINIMO}). No se generan alertas estadísticas."}]

    alertas = []
    for _, fila in drift_periodo[drift_periodo["estado"].isin([ROJO, AMARILLO])].iterrows():
        alertas.append({"nivel": fila["estado"], "mensaje":
                        f"{periodo}: drift {fila['estado'].lower()} en {fila['variable']} "
                        f"(PSI={fila['psi']:.3f}, JS={fila['jensen_shannon']:.3f}, "
                        f"{fila['prueba']} p={fila['p_valor']:.3g})."})

    caida = _caida_auc(desempeno_periodo, desempeno_referencia)
    if not pd.isna(caida) and caida > CAIDA_MAXIMA_AUC:
        alertas.append({"nivel": ROJO, "mensaje":
                        f"{periodo}: el ROC-AUC cayó {caida:.3f} frente a la referencia "
                        f"({desempeno_periodo['roc_auc']:.3f} vs {desempeno_referencia['roc_auc']:.3f})."})
    if not alertas:
        alertas.append({"nivel": VERDE, "mensaje": f"{periodo}: sin desviaciones significativas."})
    return sorted(alertas, key=lambda a: SEVERIDAD[a["nivel"]], reverse=True)


def requiere_reentrenamiento(drift_periodo, desempeno_periodo, desempeno_referencia):
    """Reentrenar si las predicciones cambian mucho, si 3 o mas variables estan en rojo o si cae el AUC."""
    rojas = drift_periodo.loc[drift_periodo["estado"] == ROJO, "variable"].tolist()
    variables_rojas = [v for v in rojas if v != NOMBRE_PREDICCIONES]
    caida = _caida_auc(desempeno_periodo, desempeno_referencia)
    return (
        NOMBRE_PREDICCIONES in rojas
        or len(variables_rojas) >= 3
        or (not pd.isna(caida) and caida > CAIDA_MAXIMA_AUC)
    )


def generar_recomendaciones(drift_periodo, desempeno_periodo, desempeno_referencia):
    """Sugerencias de accion segun el drift y el desempeno del periodo."""
    if (drift_periodo["estado"] == SIN_DATOS).all():
        return ["Acumular más registros antes de evaluar este período o usar una frecuencia trimestral."]

    rojas = drift_periodo.loc[drift_periodo["estado"] == ROJO, "variable"].tolist()
    amarillas = drift_periodo.loc[drift_periodo["estado"] == AMARILLO, "variable"].tolist()

    recomendaciones = []
    if requiere_reentrenamiento(drift_periodo, desempeno_periodo, desempeno_referencia):
        recomendaciones.append("Reentrenar el modelo incluyendo los datos más recientes: el drift "
                               "o la caída de desempeño pueden comprometer la precisión.")
    for variable in rojas:
        recomendaciones.append(f"Revisar {variable}: verificar cambios en la fuente de datos, en la "
                               "política de crédito o en el perfil de los clientes.")
    if amarillas:
        recomendaciones.append("Vigilar en los próximos períodos: " + ", ".join(amarillas) + ".")
    if not recomendaciones:
        recomendaciones.append("No se requiere acción: la población es estable frente a la referencia.")
    return recomendaciones


def guardar_resultados(historial, desempeno, alertas, carpeta=None):
    """Guarda el historial de monitoreo para consultarlo despues."""
    carpeta = Path(carpeta) if carpeta is not None else CARPETA_ARTEFACTOS
    carpeta.mkdir(exist_ok=True)
    historial.to_csv(carpeta / "monitoreo_historial.csv", index=False)
    desempeno.to_csv(carpeta / "monitoreo_desempeno.csv", index=False)
    pd.DataFrame(alertas).to_csv(carpeta / "monitoreo_alertas.csv", index=False)


def ejecutar_monitoreo(frecuencia="M", tamano_muestra=None, modelo=None):
    """Corre el trabajo de monitoreo completo y retorna todos sus resultados."""
    referencia, tabla = construir_tabla_monitoreo(modelo)
    historial, desempeno = monitorear(referencia, tabla, frecuencia, tamano_muestra)
    desempeno_referencia = calcular_desempeno(tabla)  # todo el conjunto de evaluacion
    alertas = []
    for periodo, drift in historial.groupby(COLUMNA_PERIODO, sort=True):
        fila_desempeno = desempeno[desempeno[COLUMNA_PERIODO] == periodo].iloc[0]
        for alerta in generar_alertas(drift, fila_desempeno, desempeno_referencia):
            alertas.append({COLUMNA_PERIODO: periodo, **alerta})
    return {
        "referencia": referencia,
        "tabla": tabla,
        "historial": historial,
        "desempeno": desempeno,
        "desempeno_referencia": desempeno_referencia,
        "alertas": alertas,
    }


def main():
    resultados = ejecutar_monitoreo(frecuencia="M")
    historial = resultados["historial"]
    guardar_resultados(historial, resultados["desempeno"], resultados["alertas"])

    print("=== Monitoreo del modelo (frecuencia mensual) ===")
    for periodo, drift in historial.groupby(COLUMNA_PERIODO, sort=True):
        print(f"{periodo}: {estado_general(drift)} ({int(drift['registros'].iloc[0])} registros)")

    ultimo = periodos_validos(historial)[-1]
    drift_ultimo = historial[historial[COLUMNA_PERIODO] == ultimo]
    desempeno_ultimo = resultados["desempeno"].set_index(COLUMNA_PERIODO).loc[ultimo]
    print(f"\nUltimo periodo con muestra suficiente: {ultimo}")
    for recomendacion in generar_recomendaciones(drift_ultimo, desempeno_ultimo,
                                                 resultados["desempeno_referencia"]):
        print(f"- {recomendacion}")
    print(f"\nResultados guardados en: {CARPETA_ARTEFACTOS}")


# ---------------------------------------------------------------------------
# 4. Aplicacion en Streamlit
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Calculando métricas de drift...")
def _resultados_app(frecuencia, tamano_muestra):
    return ejecutar_monitoreo(frecuencia=frecuencia, tamano_muestra=tamano_muestra)


def _colorear_estado(valor):
    return f"background-color: {COLORES.get(valor, '')}"


def _grafico_distribucion(referencia, actual, variable, periodo):
    """Histograma (numericas) o barras (categoricas) de la referencia frente al periodo."""
    discreta = es_discreta(referencia[variable])
    fig, ax = plt.subplots(figsize=(9, 3.8))
    if discreta:
        conteo_ref, conteo_act, etiquetas = conteos_comparables(referencia[variable], actual[variable], True)
        posiciones = np.arange(len(etiquetas))
        ax.bar(posiciones - 0.2, conteo_ref / max(conteo_ref.sum(), 1), width=0.4,
               label="Histórico (referencia)", color="#2E86C1")
        ax.bar(posiciones + 0.2, conteo_act / max(conteo_act.sum(), 1), width=0.4,
               label=f"Actual ({periodo})", color="#E67E22")
        ax.set_xticks(posiciones, [str(e) for e in etiquetas], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Proporción")
    else:
        minimo, maximo = np.nanquantile(referencia[variable].astype(float), [0.005, 0.995])
        bins = np.linspace(minimo, maximo, 31)
        ax.set_xlabel(f"log(1 + {variable})" if variable in VARIABLES_LOG else variable)
        ax.hist(referencia[variable].dropna(), bins=bins, density=True, alpha=0.55,
                label="Histórico (referencia)", color="#2E86C1")
        ax.hist(actual[variable].dropna(), bins=bins, density=True, alpha=0.55,
                label=f"Actual ({periodo})", color="#E67E22")
        ax.set_ylabel("Densidad")
    ax.set_title(f"Distribución de {variable}: histórico vs actual")
    ax.legend()
    fig.tight_layout()
    return fig


def _grafico_barras_psi(drift_periodo):
    """Barras de riesgo: PSI por variable con las zonas del semaforo."""
    datos = drift_periodo.sort_values("psi")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(datos))))
    ax.barh(datos["variable"], datos["psi"], color=datos["estado"].map(COLORES), edgecolor="#555555")
    ax.axvline(UMBRALES["psi"][0], color="#C9A400", ls="--", label="Umbral amarillo (0.10)")
    ax.axvline(UMBRALES["psi"][1], color="#C0392B", ls="--", label="Umbral rojo (0.25)")
    ax.set_xlabel("PSI")
    ax.set_title("Barras de riesgo: PSI por variable")
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def _grafico_evolucion(historial, variables):
    """Evolucion del PSI en el tiempo para las variables elegidas.

    Solo se grafican los periodos con muestra suficiente: con pocos registros el PSI se infla
    por ruido y deformaria la escala.
    """
    validos = historial[historial["estado"] != SIN_DATOS]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for variable in variables:
        datos = validos[validos["variable"] == variable].sort_values(COLUMNA_PERIODO)
        ax.plot(datos[COLUMNA_PERIODO], datos["psi"], marker="o", label=variable)
    techo = max(0.5, float(validos[validos["variable"].isin(variables)]["psi"].max()) * 1.1)
    ax.axhspan(UMBRALES["psi"][0], UMBRALES["psi"][1], color="#FFF3C4", alpha=0.6, label="Zona amarilla")
    ax.axhspan(UMBRALES["psi"][1], techo, color="#F8C9C4", alpha=0.6, label="Zona roja")
    ax.set_ylabel("PSI")
    ax.set_title("Evolución del drift (PSI) por período")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def _mapa_estados(historial):
    """Tabla variable x periodo con el icono del semaforo."""
    mapa = historial.pivot_table(index="variable", columns=COLUMNA_PERIODO, values="estado", aggfunc="first")
    return mapa.apply(lambda columna: columna.map(lambda estado: ICONOS.get(estado, "")))


def _seccion_kpis(drift_periodo, desempeno_periodo):
    conteo = drift_periodo["estado"].value_counts()
    estado = estado_general(drift_periodo)
    fila_pred = drift_periodo[drift_periodo["variable"] == NOMBRE_PREDICCIONES].iloc[0]
    auc = desempeno_periodo["roc_auc"]
    columnas = st.columns(5)
    columnas[0].metric("Estado general", f"{ICONOS[estado]} {estado}")
    columnas[1].metric("Registros del período", f"{int(desempeno_periodo['registros']):,}")
    columnas[2].metric("Variables en rojo / amarillo", f"{conteo.get(ROJO, 0)} / {conteo.get(AMARILLO, 0)}")
    columnas[3].metric("PSI de las predicciones", f"{fila_pred['psi']:.3f}")
    columnas[4].metric("ROC-AUC del período", "N/D" if pd.isna(auc) else f"{auc:.3f}")


def _seccion_alertas(drift_periodo, desempeno_periodo, desempeno_referencia):
    st.subheader("Alertas y recomendaciones")
    mostrar = {ROJO: st.error, AMARILLO: st.warning, VERDE: st.success, SIN_DATOS: st.info}
    for alerta in generar_alertas(drift_periodo, desempeno_periodo, desempeno_referencia):
        mostrar[alerta["nivel"]](f"{ICONOS[alerta['nivel']]} {alerta['mensaje']}")
    st.markdown("**Recomendaciones**")
    for recomendacion in generar_recomendaciones(drift_periodo, desempeno_periodo, desempeno_referencia):
        st.markdown(f"- {recomendacion}")


def _barra_lateral():
    """Controles de la aplicacion. Retorna (codigo de frecuencia, tamano de muestra)."""
    with st.sidebar:
        st.header("Configuración")
        nombre_frecuencia = st.radio("Periodicidad del muestreo", list(FRECUENCIAS), index=0)
        usar_muestra = st.checkbox("Muestrear un máximo de registros por período", value=False)
        tamano = st.slider("Registros por período", 100, 2000, 500, step=100) if usar_muestra else None
        with st.expander("Umbrales del semáforo"):
            st.markdown(
                f"- **PSI**: verde < {UMBRALES['psi'][0]}, amarillo < {UMBRALES['psi'][1]}, "
                f"rojo >= {UMBRALES['psi'][1]}\n"
                f"- **Jensen-Shannon**: verde < {UMBRALES['jensen_shannon'][0]}, "
                f"rojo >= {UMBRALES['jensen_shannon'][1]}\n"
                f"- **KS**: amarillo >= {UMBRALES['ks'][0]}, rojo >= {UMBRALES['ks'][1]}\n"
                f"- Una variable solo cambia de verde si la prueba KS o Chi-cuadrado da p < {P_VALOR_CRITICO}\n"
                f"- Los períodos con menos de {TAMANO_MINIMO} registros no generan alertas"
            )
    return FRECUENCIAS[nombre_frecuencia], tamano


def _pestana_metricas(drift_periodo):
    tabla = drift_periodo.drop(columns=[COLUMNA_PERIODO]).copy()
    tabla.insert(0, "semaforo", tabla["estado"].map(ICONOS))
    formato = {"estadistico": "{:.3f}", "p_valor": "{:.3g}", "psi": "{:.3f}", "jensen_shannon": "{:.3f}"}
    st.dataframe(tabla.style.map(_colorear_estado, subset=["estado"]).format(formato),
                 width="stretch", hide_index=True)
    st.pyplot(_grafico_barras_psi(drift_periodo))


def _pestana_temporal(historial):
    validos = historial[historial["estado"] != SIN_DATOS]
    top = validos.groupby("variable")["psi"].max().sort_values(ascending=False).head(5).index.tolist()
    elegidas = st.multiselect("Variables a graficar", sorted(historial["variable"].unique()), default=top)
    if elegidas:
        st.pyplot(_grafico_evolucion(historial, elegidas))
    st.markdown("**Semáforo por variable y período**")
    st.dataframe(_mapa_estados(historial), width="stretch")
    st.markdown(f"**Cambios abruptos** (el PSI sube más de {SALTO_ABRUPTO_PSI} de un período al siguiente)")
    st.dataframe(detectar_cambios_abruptos(historial), width="stretch", hide_index=True)
    st.markdown("**Tendencia del drift** (pendiente del PSI por período)")
    st.dataframe(tendencia_psi(historial), width="stretch", hide_index=True)


def _pestana_desempeno(desempeno, desempeno_referencia):
    st.markdown(f"ROC-AUC de referencia (conjunto de evaluación): **{desempeno_referencia['roc_auc']:.3f}** · "
                f"tasa real de no pago: **{desempeno_referencia['tasa_real_no_pago']:.2%}**")
    st.caption(f"El ROC-AUC de cada período usa solo créditos que no se usaron para entrenar, y se muestra "
               f"si hay al menos {MIN_CASOS_CLASE} casos de cada clase.")
    st.dataframe(desempeno, width="stretch", hide_index=True)
    st.line_chart(desempeno.set_index(COLUMNA_PERIODO)[["tasa_real_no_pago", "tasa_predicha_no_pago"]])


def ejecutar_app():
    """Aplicacion de monitoreo en Streamlit."""
    st.set_page_config(page_title="Monitoreo del modelo de crédito", layout="wide")
    st.title("Monitoreo del modelo de pago a tiempo")
    st.caption("Compara cada período con la población de entrenamiento (referencia) para detectar data drift.")

    frecuencia, tamano = _barra_lateral()
    resultados = _resultados_app(frecuencia, tamano)
    historial, desempeno = resultados["historial"], resultados["desempeno"]
    periodos = list(dict.fromkeys(historial[COLUMNA_PERIODO]))
    validos = periodos_validos(historial)
    indice = periodos.index(validos[-1]) if validos else len(periodos) - 1
    periodo = st.sidebar.selectbox("Período a analizar", periodos, index=indice)

    drift_periodo = historial[historial[COLUMNA_PERIODO] == periodo]
    desempeno_periodo = desempeno.set_index(COLUMNA_PERIODO).loc[periodo]
    _seccion_kpis(drift_periodo, desempeno_periodo)
    _seccion_alertas(drift_periodo, desempeno_periodo, resultados["desempeno_referencia"])

    tab_metricas, tab_distribucion, tab_tiempo, tab_desempeno = st.tabs(
        ["Métricas por variable", "Histórico vs actual", "Análisis temporal", "Desempeño del modelo"]
    )
    with tab_metricas:
        _pestana_metricas(drift_periodo)
    with tab_distribucion:
        opciones = variables_monitoreadas(resultados["tabla"]) + [COLUMNA_PROBABILIDAD]
        mayor_drift = drift_periodo.sort_values("psi", ascending=False)["variable"].iloc[0]
        mayor_drift = COLUMNA_PROBABILIDAD if mayor_drift == NOMBRE_PREDICCIONES else mayor_drift
        variable = st.selectbox("Variable (por defecto, la de mayor PSI en el período)", opciones,
                                index=opciones.index(mayor_drift))
        actual = muestreo_periodico(resultados["tabla"], frecuencia, tamano)[periodo]
        st.pyplot(_grafico_distribucion(resultados["referencia"], actual, variable, periodo))
    with tab_tiempo:
        _pestana_temporal(historial)
    with tab_desempeno:
        _pestana_desempeno(desempeno, resultados["desempeno_referencia"])


if __name__ == "__main__":
    if st.runtime.exists():
        ejecutar_app()
    else:
        main()
