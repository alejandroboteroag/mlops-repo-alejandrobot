@echo off
REM ==========================================================
REM  Configura el entorno virtual del proyecto (Windows)
REM  Uso: doble clic o ejecutar "setup.bat" desde la raiz del repo
REM ==========================================================

echo.
echo === 1. Verificando Python ===
python --version
if errorlevel 1 (
    echo [ERROR] Python no esta instalado o no esta en el PATH.
    exit /b 1
)

echo.
echo === 2. Creando entorno virtual (venv) ===
if not exist venv (
    python -m venv venv
) else (
    echo El entorno virtual ya existe, se reutiliza.
)

echo.
echo === 3. Activando entorno virtual ===
call venv\Scripts\activate.bat

echo.
echo === 4. Instalando dependencias de requirements.txt ===
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Fallo la instalacion de dependencias.
    exit /b 1
)

echo.
echo === 5. Registrando el kernel para Jupyter / VS Code ===
python -m ipykernel install --user --name=mlops-creditos --display-name "Python (mlops-creditos)"

echo.
echo === Entorno listo. Para activarlo en una nueva terminal: venv\Scripts\activate ===
