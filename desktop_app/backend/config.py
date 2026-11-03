"""
Configuración central del backend de Prometeo PACS v2.0.
Resuelve rutas base, entorno de ejecución y selección de dispositivo de cómputo.
"""
import os
import sys
from pathlib import Path
import torch

APP_TITLE = "Prometeo PACS v2.0"
VERSION = "2.0.0"
HOST = "127.0.0.1"
PORT = 8000

# Resolución de directorios
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path(__file__).resolve().parents[2]


def _resolve_codigos_dir() -> Path:
    """Raíz que contiene `src/`, `reports/` y `artifacts/`.

    Orden de búsqueda: variable TESIS_CODIGOS; la raíz del repositorio (este
    paquete vive en `<repo>/desktop_app`); y, por compatibilidad con el árbol
    de trabajo original, la carpeta hermana `Códigos`.
    """
    env = os.environ.get("TESIS_CODIGOS")
    if env and Path(env).is_dir():
        return Path(env).resolve()
    if (ROOT_DIR / "src").is_dir() and (ROOT_DIR / "reports").is_dir():
        return ROOT_DIR.resolve()
    candidate = ROOT_DIR / "Códigos"
    if candidate.is_dir():
        return candidate.resolve()
    return ROOT_DIR.resolve()


CODIGOS_DIR = _resolve_codigos_dir()

# Asegurar directorios en sys.path para importación limpia de src.*
if CODIGOS_DIR.exists() and str(CODIGOS_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGOS_DIR))

if ROOT_DIR.exists() and str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Detección y asignación del dispositivo de cómputo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
