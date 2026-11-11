"""Pruebas del monitoreo y la deteccion de data drift."""

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import model_monitoring as mm


@pytest.fixture(scope="module")
def resultados(modelo_prueba):
    return mm.ejecutar_monitoreo(frecuencia="Q", modelo=modelo_prueba)


def test_psi_es_cero_para_distribuciones_iguales():
    conteo = np.array([10, 20, 30, 40])
    assert mm.calcular_psi(conteo, conteo) == pytest.approx(0.0, abs=1e-9)


def test_psi_detecta_un_cambio_fuerte():
    assert mm.calcular_psi(np.array([90, 10]), np.array([10, 90])) > mm.UMBRALES["psi"][1]


def test_jensen_shannon_esta_entre_cero_y_uno():
    assert mm.calcular_jensen_shannon(np.array([5, 5]), np.array([5, 5])) == pytest.approx(0.0, abs=1e-6)
    assert 0.9 < mm.calcular_jensen_shannon(np.array([100, 0]), np.array([0, 100])) <= 1.0


def test_prueba_ks_detecta_desplazamiento():
    rng = np.random.default_rng(0)
    referencia = pd.Series(rng.normal(0, 1, 2000))
    _, p_igual = mm.prueba_ks(referencia, pd.Series(rng.normal(0, 1, 2000)))
    estadistico, p_distinta = mm.prueba_ks(referencia, pd.Series(rng.normal(1, 1, 2000)))
    assert p_igual > 0.01
    assert p_distinta < 0.001 and estadistico > 0.3
    assert np.isnan(mm.prueba_ks(referencia, pd.Series([1.0]))[0])


def test_prueba_chi_cuadrado():
    _, p_valor = mm.prueba_chi_cuadrado(np.array([500, 500]), np.array([100, 900]))
    assert p_valor < 0.001
    assert np.isnan(mm.prueba_chi_cuadrado(np.array([10, 0]), np.array([5, 0]))[1])


def test_nivel_por_umbral_y_peor_estado():
    assert mm.nivel_por_umbral(0.05, (0.1, 0.25)) == mm.VERDE
    assert mm.nivel_por_umbral(0.15, (0.1, 0.25)) == mm.AMARILLO
    assert mm.nivel_por_umbral(0.30, (0.1, 0.25)) == mm.ROJO
    assert mm.nivel_por_umbral(np.nan, (0.1, 0.25)) == mm.VERDE
    assert mm.peor_estado([mm.VERDE, mm.ROJO, mm.AMARILLO]) == mm.ROJO


def test_evaluar_variable_numerica_y_categorica():
    rng = np.random.default_rng(1)
    estable = mm.evaluar_variable("x", pd.Series(rng.normal(0, 1, 3000)), pd.Series(rng.normal(0, 1, 1000)))
    cambio = mm.evaluar_variable("x", pd.Series(rng.normal(0, 1, 3000)), pd.Series(rng.normal(2, 1, 1000)))
    assert estable["prueba"] == "KS" and estable["estado"] == mm.VERDE
    assert cambio["estado"] == mm.ROJO
    categorica = mm.evaluar_variable("c", pd.Series(["a"] * 500 + ["b"] * 500), pd.Series(["a"] * 100 + ["b"] * 900))
    assert categorica["prueba"] == "Chi-cuadrado" and categorica["estado"] == mm.ROJO


def test_conteos_incluyen_nulos_como_grupo():
    referencia = pd.Series([1.0, 2.0, 3.0, np.nan] * 50)
    _, conteo_act, etiquetas = mm.conteos_comparables(referencia, pd.Series([np.nan] * 10), discreta=False)
    assert etiquetas[-1] == "Sin dato" and conteo_act[-1] == 10


def test_tabla_de_monitoreo_une_datos_y_pronosticos(resultados):
    tabla, referencia = resultados["tabla"], resultados["referencia"]
    assert len(tabla) == 10_763
    assert {mm.COLUMNA_PROBABILIDAD, mm.COLUMNA_PREDICCION, mm.COLUMNA_FECHA}.issubset(tabla.columns)
    assert tabla[mm.COLUMNA_PROBABILIDAD].between(0, 1).all()
    assert referencia[mm.COLUMNA_ENTRENAMIENTO].all()
    assert mm.COLUMNA_ENTRENAMIENTO not in mm.variables_monitoreadas(tabla)


def test_muestreo_periodico_limita_el_tamano(resultados):
    muestras = mm.muestreo_periodico(resultados["tabla"], "Q", tamano_muestra=50)
    assert list(muestras) == sorted(muestras)
    assert all(len(m) <= 50 for m in muestras.values())


def test_historial_marca_periodos_con_muestra_insuficiente(resultados):
    historial = resultados["historial"]
    ultimo = historial[historial[mm.COLUMNA_PERIODO] == "2026Q2"]
    assert (ultimo["estado"] == mm.SIN_DATOS).all()
    assert mm.estado_general(ultimo) == mm.SIN_DATOS
    assert "2026Q2" not in mm.periodos_validos(historial)


def test_alertas_y_recomendaciones(resultados):
    historial, desempeno = resultados["historial"], resultados["desempeno"].set_index(mm.COLUMNA_PERIODO)
    referencia = resultados["desempeno_referencia"]
    for periodo in ["2025Q1", "2026Q2"]:
        drift = historial[historial[mm.COLUMNA_PERIODO] == periodo]
        alertas = mm.generar_alertas(drift, desempeno.loc[periodo], referencia)
        recomendaciones = mm.generar_recomendaciones(drift, desempeno.loc[periodo], referencia)
        assert alertas and recomendaciones
        assert all(alerta["nivel"] in mm.SEVERIDAD for alerta in alertas)


def test_recomienda_reentrenar_si_cae_el_auc():
    drift = pd.DataFrame({mm.COLUMNA_PERIODO: ["p"], "variable": ["x"], "estado": [mm.VERDE], "psi": [0.0],
                          "jensen_shannon": [0.0], "prueba": ["KS"], "p_valor": [0.9], "registros": [500]})
    alertas = mm.generar_alertas(drift, {"roc_auc": 0.50}, {"roc_auc": 0.70})
    assert alertas[0]["nivel"] == mm.ROJO
    assert "Reentrenar" in mm.generar_recomendaciones(drift, {"roc_auc": 0.50}, {"roc_auc": 0.70})[0]
    estable = mm.generar_recomendaciones(drift, {"roc_auc": 0.70}, {"roc_auc": 0.70})
    assert estable[0].startswith("No se requiere acción")


def test_analisis_temporal(resultados):
    historial = resultados["historial"]
    tendencia = mm.tendencia_psi(historial)
    assert set(tendencia["tendencia"]).issubset({"Creciente", "Decreciente", "Estable"})
    cambios = mm.detectar_cambios_abruptos(historial)
    assert (cambios["cambio_psi"] > mm.SALTO_ABRUPTO_PSI).all()
    assert mm.tendencia_psi(historial.head(0)).empty


def test_desempeno_usa_solo_creditos_no_vistos(resultados):
    referencia = resultados["desempeno_referencia"]
    assert referencia["registros_evaluados"] == 2_153
    assert 0.5 < referencia["roc_auc"] < 0.9


def test_guardar_resultados(resultados, tmp_path):
    mm.guardar_resultados(resultados["historial"], resultados["desempeno"], resultados["alertas"], tmp_path)
    for archivo in ["monitoreo_historial.csv", "monitoreo_desempeno.csv", "monitoreo_alertas.csv"]:
        assert (tmp_path / archivo).exists()


def test_cargar_modelo_sin_archivo(tmp_path):
    with pytest.raises(FileNotFoundError):
        mm.cargar_modelo(tmp_path / "no_existe.joblib")


def test_main_genera_resultados(ruta_modelo_prueba, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mm, "RUTA_MODELO", ruta_modelo_prueba)
    monkeypatch.setattr(mm, "CARPETA_ARTEFACTOS", tmp_path)
    mm.main()
    assert "Ultimo periodo con muestra suficiente" in capsys.readouterr().out
    assert (tmp_path / "monitoreo_historial.csv").exists()


def _script_app():
    from model_monitoring import ejecutar_app

    ejecutar_app()


def test_aplicacion_streamlit(ruta_modelo_prueba, monkeypatch):
    monkeypatch.setattr(mm, "RUTA_MODELO", ruta_modelo_prueba)
    app = AppTest.from_function(_script_app, default_timeout=180)
    app.run()
    assert not app.exception
    assert app.title[0].value == "Monitoreo del modelo de pago a tiempo"
    assert len(app.metric) == 5
    app.sidebar.radio[0].set_value("Trimestral").run()
    app.sidebar.checkbox[0].check().run()
    assert not app.exception
    app.sidebar.selectbox[0].set_value("2026Q2").run()
    assert not app.exception
    assert any("No se generan alertas" in info.value for info in app.info)
