@echo off
setlocal enabledelayedexpansion
title Prometeo PACS v2.0 - Estacion Radiologica Neuro-Simbolica

echo ======================================================================
echo    PROMETEO PACS v2.0 - ESTACION RADIOLOGICA NEURO-SIMBOLICA
echo    Facultad de Ciencias - Universidad El Bosque
echo ======================================================================
echo.

cd /d "%~dp0"

:: 1. Detectar entorno virtual Python
set "PYTHON_EXE="
if exist "..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=..\.venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] No se encontro el interprete Python en .venv\Scripts\python.exe.
    echo Cree el entorno en la raiz del repositorio: python -m venv .venv  y  pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: 2. Detectar Node.js y npm
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Node.js no esta instalado o no se encuentra en el PATH del sistema.
    echo Node.js v18+ es necesario para ejecutar el visor Electron.
    echo.
    pause
    exit /b 1
)

where npm >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] npm no esta disponible en el PATH del sistema.
    echo.
    pause
    exit /b 1
)

:: 3. Verificar dependencias de Electron
if not exist "electron\node_modules" (
    echo [INFO] Instalando dependencias de Electron en electron...
    pushd electron
    call npm install
    popd
)

:: 4. Iniciar aplicacion Electron
echo [INFO] Iniciando Prometeo PACS v2.0...
cd electron
call npm start

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [AVISO] La aplicacion se cerro con codigo de salida %ERRORLEVEL%.
    pause
)
