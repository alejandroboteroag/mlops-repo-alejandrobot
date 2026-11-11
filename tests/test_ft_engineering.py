"""Pruebas de la ingenieria de caracteristicas."""

import numpy as np
import pandas as pd

import ft_engineering as fe


def _registro_base(**cambios):
    registro = {
        "tipo_credito": 4, "fecha_prestamo": pd.Timestamp("2025-01-11 10:00"), "capital_prestado": 2_000_000.0,
        "plazo_meses": 12, "edad_cliente": 35, "tipo_laboral": "Empleado", "salario_cliente": 3_000_000,
        "total_otros_prestamos": 1_000_000, "cuota_pactada": 200_000, "puntaje": 95.2, "puntaje_datacredito": 780.0,
        "cant_creditosvigentes": 2, "huella_consulta": 3, "saldo_mora": 0.0, "saldo_total": 100.0,
        "saldo_principal": 90.0, "saldo_mora_codeudor": 0.0, "creditos_sectorFinanciero": 1,
        "creditos_sectorCooperativo": 0, "creditos_sectorReal": 1, "promedio_ingresos_datacredito": 2_500_000.0,
        "tendencia_ingresos": "Creciente", "Pago_atiempo": 1,
    }
    registro.update(cambios)
    return pd.DataFrame([registro])


def test_limpiar_datos_convierte_valores_imposibles_en_nulos():
    datos = _registro_base(edad_cliente=122, salario_cliente=0, puntaje_datacredito=-7,
                           plazo_meses=90, total_otros_prestamos=5e9, tendencia_ingresos="123")
    limpios = fe.limpiar_datos(datos)
    for columna in ["edad_cliente", "salario_cliente", "puntaje_datacredito", "plazo_meses",
                    "total_otros_prestamos", "tendencia_ingresos"]:
        assert pd.isna(limpios.loc[0, columna]), columna
    assert limpios.loc[0, "tipo_credito"] == "4"


def test_limpiar_datos_conserva_valores_validos():
    limpios = fe.limpiar_datos(_registro_base())
    assert limpios.loc[0, "edad_cliente"] == 35
    assert limpios.loc[0, "tendencia_ingresos"] == "Creciente"


def test_crear_atributos_genera_derivadas_y_elimina_variables():
    atributos = fe.crear_atributos(fe.limpiar_datos(_registro_base(saldo_mora=50.0)))
    assert atributos.loc[0, "tiene_mora"] == 1
    assert atributos.loc[0, "sin_info_ingresos_dc"] == 0
    assert atributos.loc[0, "es_fin_de_semana"] == 1  # 2025-01-11 fue sabado
    assert np.isclose(atributos.loc[0, "ratio_cuota_salario"], 200_000 / 3_000_000)
    assert np.isclose(atributos.loc[0, "capital_prestado"], np.log1p(2_000_000))
    for columna in fe.VARIABLES_ELIMINAR:
        assert columna not in atributos


def test_ejecutar_ft_engineering_divide_de_forma_estratificada(datos_divididos):
    X_train, X_test, y_train, y_test, _ = datos_divididos
    assert len(X_train) + len(X_test) == 10_763
    assert abs(len(X_test) / (len(X_train) + len(X_test)) - fe.TAMANO_TEST) < 0.01
    assert abs((1 - y_train.mean()) - (1 - y_test.mean())) < 0.005
    assert "puntaje" not in X_train.columns


def test_preprocesador_no_deja_nulos(datos_divididos):
    X_train, X_test, _, _, preprocesador = datos_divididos
    transformado = preprocesador.fit_transform(X_train)
    assert not np.isnan(transformado).any()
    assert preprocesador.transform(X_test).shape[1] == transformado.shape[1]


def test_main_imprime_resumen(capsys):
    fe.main()
    salida = capsys.readouterr().out
    assert "Entrenamiento: 8,610" in salida
    assert "Columnas despues del ColumnTransformer" in salida
