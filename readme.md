# mlops-repo-alejandrobot

Proyecto final del curso de Machine Learning - Universidad Católica Luis Amigó.

## Caso de negocio

Una entidad de crédito de consumo necesita decidir, al momento de la solicitud, si aprueba un crédito. El proyecto construye un modelo que predice si un cliente va a pagar a tiempo y lo lleva a un flujo de MLOps completo: preparación de datos, entrenamiento, monitoreo y despliegue como API.

La variable objetivo es `Pago_atiempo` (1 = pagó a tiempo, 0 = no pagó a tiempo). El error más costoso es aprobar un crédito a quien no paga, por eso la clase que interesa detectar es la 0.

## Datos

`Base_de_datos.csv` tiene 10.763 créditos otorgados entre noviembre de 2024 y abril de 2026, con 23 variables del cliente, del crédito y de Datacrédito. Solo el 4,75 % de los créditos no se pagó a tiempo, así que la base está muy desbalanceada.

## Estructura

```
mlops_pipeline/
    Cargar_datos.ipynb              carga y validación de la base
    comprension_eda.ipynb           análisis exploratorio
    ft_engineering.py               limpieza, atributos derivados y preprocesador
    model_training_evaluation.py    entrenamiento, comparación y selección del modelo
    model_monitoring.py             monitoreo de data drift y app en Streamlit
    model_deploy.py                 API con FastAPI
    artefactos/                     modelo, gráficos y resultados generados
tests/                              pruebas unitarias (pytest)
jenkins/docker-compose.yml          Jenkins local
.github/workflows/sonarcloud.yml    pruebas y análisis en SonarCloud
Base_de_datos.csv
requirements.txt
setup.bat
Dockerfile
.dockerignore
Jenkinsfile
sonar-project.properties
pytest.ini
.gitignore
readme.md
```

## Ramas

- `developer`: desarrollo
- `certification`: validación
- `master`: producción

## Instalación

```
git clone https://github.com/alejandroboteroag/mlops-repo-alejandrobot.git
cd mlops-repo-alejandrobot
setup.bat
venv\Scripts\activate
```

`setup.bat` crea el entorno virtual e instala las librerías de `requirements.txt`.

## Cómo ejecutar cada parte

| Paso | Comando |
|---|---|
| Entrenar y comparar modelos | `python mlops_pipeline\model_training_evaluation.py` |
| Monitoreo (guarda resultados) | `python mlops_pipeline\model_monitoring.py` |
| App de monitoreo | `streamlit run mlops_pipeline\model_monitoring.py` |
| API local | `uvicorn model_deploy:app --app-dir mlops_pipeline --port 8000` |
| Pruebas | `python -m pytest --cov=mlops_pipeline` |

Los notebooks se abren en VS Code o Jupyter con el entorno `venv`: primero `Cargar_datos.ipynb` y luego `comprension_eda.ipynb`.

## Proceso y hallazgos

### 1. Análisis exploratorio

- La variable `puntaje` separaba casi perfecto la objetivo: con una sola regla se acertaba el 99,8 %. Es una fuga de información, porque ese puntaje no existiría así al momento de aprobar el crédito, y se eliminó.
- Había nulos disfrazados que se convirtieron en `NaN`: edades de 121 a 123 años, salarios en 0 o de miles de millones, puntajes Datacrédito fuera de 150 – 950 y números dentro de `tendencia_ingresos`.
- Las variables con más señal son tener saldo en mora (36,4 % de no pago frente a 4,6 %), puntaje Datacrédito bajo, muchas consultas recientes, clientes jóvenes, montos altos, plazos mayores a 12 meses e ingresos decrecientes.
- Las relaciones con la objetivo son débiles y en parte no lineales.

### 2. Ingeniería de características

`ft_engineering.py` aplica las reglas de limpieza del EDA, crea los atributos derivados (`tiene_mora`, `sin_info_ingresos_dc`, razones cuota/salario, capital/salario y endeudamiento, día de la semana), aplica logaritmo a las variables de dinero y separa 80 % para entrenamiento y 20 % para evaluación de forma estratificada. El `ColumnTransformer` tiene tres ramas: numéricas, categóricas y ordinales.

### 3. Modelos

Se compararon cinco modelos. Como la clase que interesa es la de no pago, no se usa la accuracy como métrica principal.

| Modelo | ROC-AUC test | ROC-AUC CV | Recall no pago | Diferencia train/test |
|---|---|---|---|---|
| Gradient Boosting | 0.692 | 0.679 ± 0.013 | 0.539 | 0.095 |
| Random Forest | 0.692 | 0.676 ± 0.012 | 0.441 | 0.139 |
| Regresión Logística | 0.695 | 0.671 ± 0.025 | 0.618 | 0.005 |
| Árbol de Decisión | 0.606 | 0.599 ± 0.024 | 0.578 | 0.119 |
| Línea base (Dummy) | 0.500 | 0.500 | 0.000 | 0.000 |

Se eligió **Gradient Boosting**: se descartaron los modelos con sobreajuste (diferencia train/test mayor a 0.10) y entre los demás tuvo el mejor ROC-AUC en validación cruzada y la menor variación entre folds. Detecta el 54 % de los clientes que no pagan, así que sirve como filtro de alerta para revisar solicitudes, no como rechazo automático.

### 4. Monitoreo

`model_monitoring.py` une los datos con los pronósticos del modelo, los agrupa por mes o trimestre y compara cada período con los datos de entrenamiento usando KS, Chi-cuadrado, PSI y Jensen-Shannon. Cada variable queda en verde, amarillo o rojo. Para evitar falsas alarmas, una variable solo cambia de verde si la prueba estadística es significativa, y los períodos con menos de 100 créditos no generan alertas.

Hallazgos con frecuencia trimestral:

- El primer trimestre de 2025 está estable.
- Desde el tercer trimestre de 2025 la población cambia: suben el plazo, el capital y el endeudamiento, y cambian los ingresos reportados en Datacrédito.
- En el primer trimestre de 2026 hay 6 variables en rojo y la proporción de clientes marcados como riesgosos sube del 27 % al 40 %. El sistema recomienda reentrenar el modelo con datos recientes.

La app de Streamlit muestra el semáforo, las alertas y recomendaciones, la tabla de métricas por variable, las barras de riesgo, la comparación de distribuciones histórica vs actual, la evolución del drift en el tiempo y el desempeño del modelo por período.

### 5. Despliegue

`model_deploy.py` expone el modelo con FastAPI:

- `GET /health`: estado del servicio.
- `POST /predict`: predicción por lotes en JSON (`{"registros": [ {...}, {...} ]}`).
- `POST /predict/csv`: predicción por lotes desde un archivo CSV. Con `?formato=csv` devuelve un CSV con las predicciones.

La API recibe los datos crudos, les aplica la misma limpieza del entrenamiento y devuelve la predicción, la probabilidad de no pago y el nivel de riesgo (Bajo, Medio, Alto). La documentación interactiva queda en `/docs`.

Imagen Docker:

```
docker build -t mlops-creditos-api .
docker run -d -p 8000:8000 --name api-creditos mlops-creditos-api
```

La imagen instala las dependencias, copia el código y entrena el modelo dentro de la imagen para que coincida con las versiones instaladas. El servicio corre con Uvicorn y con un usuario sin privilegios.

## Calidad del código

- **Pruebas:** 47 pruebas unitarias y de integración en `tests/`, con una cobertura del 98 % de `mlops_pipeline`.
- **SonarCloud:** el workflow `.github/workflows/sonarcloud.yml` corre las pruebas en cada push a las tres ramas y envía el código y la cobertura a SonarCloud, que evalúa calidad, seguridad, cobertura y estilo.

## Jenkins

El `Jenkinsfile` clona el repositorio, valida la estructura de carpetas, instala dependencias, corre las pruebas, construye la imagen Docker y envía una notificación al terminar. Revisa el repositorio cada 5 minutos, así que un push o un merge a la rama configurada dispara el pipeline. Jenkins se levanta localmente con `jenkins/docker-compose.yml`.
