# mlops-repo-alejandrobot

Proyecto final del curso de Machine Learning - Universidad Católica Luis Amigó.

## Descripción

El proyecto busca predecir si un cliente va a pagar a tiempo su crédito. Se trabaja con la base `Base_de_datos.csv`, que tiene 10.763 créditos y 23 variables con información del cliente, del crédito y de Datacrédito.

La variable objetivo es `Pago_atiempo` (1 = pagó a tiempo, 0 = no pagó a tiempo).

## Estructura

```
mlops_pipeline/
    Cargar_datos.ipynb
    comprension_eda.ipynb
    ft_engineering.py
    model_training_evaluation.py
    model_deploy.py
    model_monitoring.py
Base_de_datos.csv
requirements.txt
.gitignore
setup.bat
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
```

`setup.bat` crea el entorno virtual e instala las librerías de `requirements.txt`. Después se abren los notebooks de `mlops_pipeline` con ese entorno, primero `Cargar_datos.ipynb` y luego `comprension_eda.ipynb`.

Para entrenar los modelos:

```
venv\Scripts\activate
python mlops_pipeline\model_training_evaluation.py
```

El script usa `ft_engineering.py` para preparar los datos, entrena los modelos y guarda en `mlops_pipeline/artefactos/` la tabla de resultados, los gráficos y el mejor modelo (`mejor_modelo.joblib`).

## Modelos

Se compararon cinco modelos. La clase que interesa detectar es la de no pago (`Pago_atiempo = 0`), por eso no se usa la accuracy como métrica principal.

| Modelo | ROC-AUC test | ROC-AUC CV | Recall no pago | Diferencia train/test |
|---|---|---|---|---|
| Gradient Boosting | 0.692 | 0.679 ± 0.013 | 0.539 | 0.095 |
| Random Forest | 0.692 | 0.676 ± 0.012 | 0.441 | 0.139 |
| Regresión Logística | 0.695 | 0.671 ± 0.025 | 0.618 | 0.005 |
| Árbol de Decisión | 0.606 | 0.599 ± 0.024 | 0.578 | 0.119 |
| Línea base (Dummy) | 0.500 | 0.500 | 0.000 | 0.000 |

Se eligió **Gradient Boosting** por tener el mejor ROC-AUC en validación cruzada, la menor variación entre folds y una diferencia train/test aceptable. La regresión logística queda muy cerca y es una buena alternativa por ser más simple y rápida.

## Estado

Están la carga de datos, el análisis exploratorio, la ingeniería de características y el entrenamiento de modelos. Faltan el monitoreo y el despliegue.
