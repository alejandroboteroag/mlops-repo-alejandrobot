"""
ft_engineering.py
-----------------
Primera componente del flujo de creacion de modelos.

A partir de la base cruda:
1. Aplica las reglas de limpieza definidas en comprension_eda.ipynb
   (nulos disfrazados -> NaN y eliminacion de variables irrelevantes).
2. Crea los atributos derivados identificados en el EDA.
3. Separa los conjuntos de entrenamiento y evaluacion (estratificados).
4. Construye el preprocesador (ColumnTransformer) con tres ramas:
   numericas, categoricas y categoricas ordinales.

El preprocesador se entrega SIN ajustar: se ajusta solo con los datos de
entrenamiento dentro del pipeline de cada modelo, para evitar fuga de informacion.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

RUTA_DATOS = Path(__file__).resolve().parent.parent / "Base_de_datos.csv"
OBJETIVO = "Pago_atiempo"
SEMILLA = 42
TAMANO_TEST = 0.2

# Variables que se eliminan (ver comprension_eda.ipynb, secciones 2.5 y 3.3)
#  - puntaje: fuga de informacion
#  - saldo_mora_codeudor: casi constante
#  - saldo_principal: correlacion 0.95 con saldo_total
#  - saldo_mora: se reemplaza por la bandera tiene_mora
#  - fecha_prestamo: se reemplaza por dia_semana y es_fin_de_semana
VARIABLES_ELIMINAR = ["puntaje", "saldo_mora_codeudor", "saldo_principal", "saldo_mora", "fecha_prestamo"]

# Variables monetarias con asimetria fuerte: se transforman con log(1 + x)
VARIABLES_LOG = [
    "capital_prestado", "salario_cliente", "total_otros_prestamos",
    "cuota_pactada", "saldo_total", "promedio_ingresos_datacredito",
]

SIN_INFORMACION = "Sin informacion"

# Orden de riesgo observado en el EDA (de mayor a menor tasa de no pago)
ORDEN_TENDENCIA = ["Decreciente", SIN_INFORMACION, "Estable", "Creciente"]

VARIABLES_CATEGORICAS = ["tipo_credito", "tipo_laboral"]
VARIABLES_ORDINALES = ["tendencia_ingresos"]


def cargar_datos(ruta=RUTA_DATOS):
    """Lee la base de datos cruda desde CSV."""
    return pd.read_csv(ruta, parse_dates=["fecha_prestamo"], encoding="utf-8")


def limpiar_datos(datos):
    """Convierte los valores imposibles o mal codificados en NaN (reglas del EDA)."""
    datos = datos.copy()

    tendencia_valida = datos["tendencia_ingresos"].isin(["Decreciente", "Estable", "Creciente"])
    datos["tendencia_ingresos"] = datos["tendencia_ingresos"].where(tendencia_valida, np.nan)

    reglas = {
        "edad_cliente": (18, 100),
        "salario_cliente": (100_000, 100_000_000),
        "total_otros_prestamos": (0, 1_000_000_000),
        "puntaje_datacredito": (150, 950),
        "plazo_meses": (1, 60),
    }
    for columna, (minimo, maximo) in reglas.items():
        datos[columna] = datos[columna].astype(float)
        fuera_de_rango = ~datos[columna].between(minimo, maximo)
        datos.loc[fuera_de_rango, columna] = np.nan

    datos["tipo_credito"] = datos["tipo_credito"].astype(str)
    return datos


def crear_atributos(datos):
    """Crea los atributos derivados propuestos en el EDA (seccion 4)."""
    datos = datos.copy()

    datos["tiene_mora"] = (datos["saldo_mora"] > 0).astype(int)
    datos["sin_info_ingresos_dc"] = datos["promedio_ingresos_datacredito"].isna().astype(int)
    datos["ratio_cuota_salario"] = datos["cuota_pactada"] / datos["salario_cliente"]
    datos["ratio_capital_salario"] = datos["capital_prestado"] / datos["salario_cliente"]
    datos["ratio_endeudamiento"] = datos["total_otros_prestamos"] / datos["salario_cliente"]
    datos["dia_semana"] = datos["fecha_prestamo"].dt.dayofweek
    datos["es_fin_de_semana"] = (datos["dia_semana"] >= 5).astype(int)

    for columna in VARIABLES_LOG:
        datos[columna] = np.log1p(datos[columna])

    return datos.drop(columns=VARIABLES_ELIMINAR)


def construir_preprocesador(variables_numericas):
    """ColumnTransformer con las ramas numerica, categorica y categorica ordinal."""
    numericas = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categoricas = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # Las categorias con menos del 1 % de los registros se agrupan en una sola
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                                 sparse_output=False)),
    ])

    ordinales = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=SIN_INFORMACION)),
        ("ordinal", OrdinalEncoder(categories=[ORDEN_TENDENCIA], handle_unknown="use_encoded_value",
                                   unknown_value=-1)),
    ])

    return ColumnTransformer(transformers=[
        ("numeric", numericas, variables_numericas),
        ("categoric", categoricas, VARIABLES_CATEGORICAS),
        ("categoric_ordinales", ordinales, VARIABLES_ORDINALES),
    ])


def preparar_features(datos):
    """Aplica limpieza y atributos derivados y separa X (features) de y (objetivo)."""
    datos = crear_atributos(limpiar_datos(datos))
    X = datos.drop(columns=[OBJETIVO])
    y = datos[OBJETIVO].astype(int)
    # tendencia_ingresos como texto para que el imputer constante funcione
    X["tendencia_ingresos"] = X["tendencia_ingresos"].astype(object)
    return X, y


def ejecutar_ft_engineering(ruta=RUTA_DATOS):
    """Ejecuta el componente completo.

    Retorna
    -------
    X_train, X_test, y_train, y_test, preprocesador
    """
    X, y = preparar_features(cargar_datos(ruta))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TAMANO_TEST, stratify=y, random_state=SEMILLA
    )

    variables_numericas = [c for c in X.columns if c not in VARIABLES_CATEGORICAS + VARIABLES_ORDINALES]
    preprocesador = construir_preprocesador(variables_numericas)

    return X_train, X_test, y_train, y_test, preprocesador


def main():
    X_train, X_test, y_train, y_test, preprocesador = ejecutar_ft_engineering()

    print("=== Ingenieria de caracteristicas ===")
    print(f"Entrenamiento: {X_train.shape[0]:,} registros | Evaluacion: {X_test.shape[0]:,} registros")
    print(f"Proporcion de no pago -> train: {1 - y_train.mean():.2%} | test: {1 - y_test.mean():.2%}")
    print(f"Variables de entrada: {X_train.shape[1]}")

    transformado = preprocesador.fit_transform(X_train)
    print(f"Columnas despues del ColumnTransformer: {transformado.shape[1]}")
    for nombre in preprocesador.get_feature_names_out():
        print(f"  - {nombre}")


if __name__ == "__main__":
    main()
