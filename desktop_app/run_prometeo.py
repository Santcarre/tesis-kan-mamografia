#!/usr/bin/env python3
"""
Prometeo PACS v2.0 Launcher
Estación Radiológica Neuro-Simbólica
Facultad de Ciencias — Universidad El Bosque
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path


def get_app_dir() -> Path:
    return Path(__file__).resolve().parent


def get_root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_python_executable(root_dir: Path) -> Path:
    win_venv = root_dir / ".venv" / "Scripts" / "python.exe"
    unix_venv = root_dir / ".venv" / "bin" / "python"

    if sys.platform == "win32" and win_venv.exists():
        return win_venv
    if unix_venv.exists():
        return unix_venv
    return Path(sys.executable)


def run_backend_only(root_dir: Path, host: str = "127.0.0.1", port: int = 8000) -> int:
    python_exe = resolve_python_executable(root_dir)
    print(f"[Prometeo Launcher] Ejecutando backend FastAPI en http://{host}:{port}...")
    cmd = [
        str(python_exe),
        "-m", "uvicorn",
        "desktop_app.backend.app:app",
        "--host", host,
        "--port", str(port),
    ]
    try:
        proc = subprocess.run(cmd, cwd=str(root_dir))
        return proc.returncode
    except KeyboardInterrupt:
        print("\n[Prometeo Launcher] Servidor detenido por el usuario.")
        return 0


def run_electron_app(root_dir: Path) -> int:
    electron_dir = get_app_dir() / "electron"
    node_exe = shutil.which("node")
    npm_exe = shutil.which("npm")

    if not electron_dir.exists():
        print(f"[ERROR] Directorio Electron no encontrado: {electron_dir}")
        return 1

    if not node_exe or not npm_exe:
        print("[ADVERTENCIA] Node.js o npm no se encuentran instalados en el sistema.")
        print("[ADVERTENCIA] Cambiando automáticamente al modo backend directo...")
        return run_backend_only(root_dir)

    node_modules = electron_dir / "node_modules"
    if not node_modules.exists():
        print("[INFO] Instalando dependencias de Electron...")
        npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
        subprocess.run([npm_cmd, "install"], cwd=str(electron_dir), check=True)

    print("[Prometeo Launcher] Iniciando Shell Electron de Prometeo PACS...")
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    try:
        proc = subprocess.run([npm_cmd, "start"], cwd=str(electron_dir))
        return proc.returncode
    except KeyboardInterrupt:
        print("\n[Prometeo Launcher] Aplicación detenida.")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lanzador de Prometeo PACS v2.0 — Estación Radiológica Neuro-Simbólica"
    )
    parser.add_argument(
        "--backend-only",
        "--no-electron",
        action="store_true",
        help="Ejecutar solo el servidor backend FastAPI (sin la ventana de Electron)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Dirección IP de escucha para FastAPI (por defecto: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Puerto TCP para FastAPI (por defecto: 8000)",
    )

    args = parser.parse_args()
    root_dir = get_root_dir()

    if args.backend_only:
        return run_backend_only(root_dir, host=args.host, port=args.port)
    return run_electron_app(root_dir)


if __name__ == "__main__":
    sys.exit(main())
