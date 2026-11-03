import ast
import json
import subprocess
import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ELECTRON_DIR = ROOT_DIR / "desktop_app" / "electron"


def test_electron_package_json():
    pkg_file = ELECTRON_DIR / "package.json"
    assert pkg_file.exists(), f"package.json not found at {pkg_file}"

    with open(pkg_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("name") == "prometeo-pacs"
    assert data.get("version") == "2.0.0"
    assert data.get("main") == "main.js"
    assert "scripts" in data and "start" in data["scripts"]
    assert "electron" in data["scripts"]["start"]
    assert "devDependencies" in data
    assert "electron" in data["devDependencies"]


def test_electron_main_js_structure():
    main_file = ELECTRON_DIR / "main.js"
    assert main_file.exists(), f"main.js not found at {main_file}"

    content = main_file.read_text(encoding="utf-8")
    assert "uvicorn" in content, "main.js must spawn uvicorn backend"
    assert "taskkill" in content, "main.js must use taskkill for clean termination on Windows"
    assert "127.0.0.1:8000" in content, "main.js must target 127.0.0.1:8000"
    assert "api/health" in content, "main.js must poll api/health"
    assert "splash" in content.lower(), "main.js must coordinate splash window"


def test_electron_splash_html():
    splash_file = ELECTRON_DIR / "splash.html"
    assert splash_file.exists(), f"splash.html not found at {splash_file}"

    content = splash_file.read_text(encoding="utf-8")
    assert "500px" in content
    assert "320px" in content
    assert "Iniciando motor neuro-simbólico y backend FastAPI..." in content
    assert "PROMETEO PACS" in content


def test_electron_preload_js():
    preload_file = ELECTRON_DIR / "preload.js"
    assert preload_file.exists(), f"preload.js not found at {preload_file}"

    content = preload_file.read_text(encoding="utf-8")
    assert "contextBridge" in content
    assert "prometeo" in content.lower()


def test_run_prometeo_bat():
    bat_file = ROOT_DIR / "desktop_app" / "run_prometeo.bat"
    assert bat_file.exists(), f"run_prometeo.bat not found at {bat_file}"

    content = bat_file.read_text(encoding="utf-8")
    assert "node" in content.lower()
    assert "electron" in content.lower()
    assert ".venv" in content


def test_run_prometeo_py():
    py_file = ROOT_DIR / "desktop_app" / "run_prometeo.py"
    assert py_file.exists(), f"run_prometeo.py not found at {py_file}"

    content = py_file.read_text(encoding="utf-8")
    parsed = ast.parse(content)
    assert isinstance(parsed, ast.Module)

    res = subprocess.run(
        [sys.executable, str(py_file), "--help"],
        capture_output=True,
        text=True,
        cwd=str(ROOT_DIR),
    )
    assert res.returncode == 0
    assert "Prometeo PACS" in res.stdout
