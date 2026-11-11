"""Pruebas de la API de despliegue."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import model_deploy as md
from conftest import RAIZ


@pytest.fixture(scope="module")
def cliente(ruta_modelo_prueba):
    original = md.RUTA_MODELO
    md.RUTA_MODELO = ruta_modelo_prueba
    with TestClient(md.app) as cliente:
        yield cliente
    md.RUTA_MODELO = original


@pytest.fixture(scope="module")
def muestra():
    return pd.read_csv(RAIZ / "Base_de_datos.csv").sample(50, random_state=7)


def test_inicio_y_salud(cliente):
    assert "/predict" in cliente.get("/").json()["endpoints"].values()
    salud = cliente.get("/health").json()
    assert salud == {"estado": "ok", "modelo_cargado": True, "modelo": "Modelo de prueba"}


def test_prediccion_json_por_lotes(cliente, muestra):
    registros = muestra.drop(columns=["Pago_atiempo"]).astype(object).where(muestra.notna(), None)
    respuesta = cliente.post("/predict", json={"registros": registros.to_dict(orient="records")})
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["total_registros"] == 50
    assert len(cuerpo["predicciones"]) == 50
    primera = cuerpo["predicciones"][0]
    assert 0 <= primera["probabilidad_no_pago"] <= 1
    assert primera["nivel_riesgo"] in {"Bajo", "Medio", "Alto"}
    assert primera["prediccion"] in {0, 1}


def test_prediccion_coincide_con_el_modelo(cliente, modelo_prueba):
    respuesta = cliente.post("/predict", json={"registros": [md.EJEMPLO_REGISTRO]}).json()
    X = md.preparar_entrada([md.SolicitudCredito(**md.EJEMPLO_REGISTRO)])
    esperado = modelo_prueba.predict_proba(X[list(modelo_prueba.feature_names_in_)])[0, 0]
    assert respuesta["predicciones"][0]["probabilidad_no_pago"] == pytest.approx(esperado, abs=1e-4)


def test_campos_opcionales_y_fecha_por_defecto(cliente):
    minimo = {k: v for k, v in md.EJEMPLO_REGISTRO.items()
              if k not in {"fecha_prestamo", "salario_cliente", "puntaje_datacredito", "tendencia_ingresos"}}
    assert cliente.post("/predict", json={"registros": [minimo]}).status_code == 200


@pytest.mark.parametrize("cuerpo", [
    {"registros": []},
    {"registros": [{**md.EJEMPLO_REGISTRO, "capital_prestado": -1}]},
    {"registros": [{"tipo_credito": 4}]},
])
def test_datos_invalidos_se_rechazan(cliente, cuerpo):
    assert cliente.post("/predict", json=cuerpo).status_code == 422


def test_prediccion_csv(cliente, muestra):
    contenido = muestra.to_csv(index=False).encode()
    respuesta = cliente.post("/predict/csv", files={"archivo": ("lote.csv", contenido, "text/csv")})
    assert respuesta.status_code == 200
    assert respuesta.json()["total_registros"] == 50


def test_prediccion_csv_punto_y_coma_con_salida_csv(cliente, muestra):
    contenido = muestra.to_csv(index=False, sep=";").encode()
    respuesta = cliente.post("/predict/csv?formato=csv", files={"archivo": ("lote.csv", contenido, "text/csv")})
    assert respuesta.status_code == 200
    encabezado = respuesta.text.splitlines()[0]
    assert encabezado.endswith("prediccion,probabilidad_no_pago,nivel_riesgo")


@pytest.mark.parametrize("contenido, codigo", [
    (b"", 400),
    (b"a,b\n1,2\n", 422),
    (b"\xff\xfe\x00", 400),
])
def test_csv_invalido(cliente, contenido, codigo):
    respuesta = cliente.post("/predict/csv", files={"archivo": ("lote.csv", contenido, "text/csv")})
    assert respuesta.status_code == codigo


def test_nivel_de_riesgo():
    assert md.nivel_riesgo(0.10) == "Bajo"
    assert md.nivel_riesgo(0.35) == "Medio"
    assert md.nivel_riesgo(0.80) == "Alto"


def test_servicio_sin_modelo(tmp_path):
    original = md.RUTA_MODELO
    md.RUTA_MODELO = tmp_path / "no_existe.joblib"
    try:
        with TestClient(md.app) as cliente:
            assert cliente.get("/health").json()["modelo_cargado"] is False
            assert cliente.post("/predict", json={"registros": [md.EJEMPLO_REGISTRO]}).status_code == 503
    finally:
        md.RUTA_MODELO = original


def test_main_levanta_uvicorn(monkeypatch):
    import uvicorn

    llamadas = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: llamadas.update(host=host, port=port))
    md.main()
    assert llamadas == {"host": "127.0.0.1", "port": 8000}
