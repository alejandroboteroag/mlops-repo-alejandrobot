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

## Estado

Por ahora están la carga de datos y el análisis exploratorio. Los demás archivos se van completando en las siguientes entregas.
