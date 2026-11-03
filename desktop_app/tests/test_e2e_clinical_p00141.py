"""
Pruebas de integración clínica End-to-End para el Caso P_00141 y verificación
de garantías arquitectónicas de cero procesos huérfanos en Prometeo PACS v2.0.

Referencia: estudio de caso P_00141 del Capítulo 4 de la tesis (Resultados), §Traslación a la práctica asistencial.
"""
import os
from pathlib import Path
import pytest
from httpx import AsyncClient, ASGITransport

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


@pytest.mark.asyncio
async def test_case_p00141_exact_clinical_metrics():
    """Verifica el caso P_00141 (primera fila de la cohorte) contra la tesis (Cap. 4):
    - u0 = +0,55; g(u0) = -0,3395; p_sym = 41,6 %
    - g queda bajo el umbral de Youden (+0,1296; 53,2 % en probabilidad): veredicto BENIGN
    - categoría BI-RADS de la herramienta: 3 (p_sym < 42 %), caso limítrofe
    - la decisión sale de la fórmula (decision_source == 'formula'), no de la red
    - histopatología benigna; el radiólogo había asignado BI-RADS 4
    """
    from desktop_app.backend.app import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res_cohort = await client.get("/api/cohort?pathology=all&type=all")
        assert res_cohort.status_code == 200
        cohort = res_cohort.json()
        assert len(cohort) > 0

        target = next((c for c in cohort if "P_00141" in c["patient_id"]), None)
        assert target is not None, "El caso clínico P_00141 debe existir en la cohorte CBIS-DDSM"
        assert "BENIGN" in target["pathology"].upper()

        res_detail = await client.get(f"/api/cases/{target['index']}")
        assert res_detail.status_code == 200
        case = res_detail.json()
        pred = case["prediction"]

        assert pred["decision_source"] == "formula"
        assert pred["u0"] == pytest.approx(0.55, abs=0.01)
        assert pred["g_logit"] == pytest.approx(-0.3395, abs=5e-3)
        assert pred["p_sym"] == pytest.approx(0.416, abs=5e-3)
        assert pred["prob_malignant"] == pytest.approx(pred["p_sym"], abs=1e-9)
        assert pred["threshold_logit"] == pytest.approx(0.1296, abs=1e-4)
        assert pred["g_logit"] < pred["threshold_logit"]
        assert pred["verdict"] == "BENIGN"
        assert pred["birads_pred"] == 3
        assert pred["birads_note"], "debe advertir que BI-RADS es una convención de la herramienta"

        # La componente auxiliar x2 es muy negativa en este caso (tesis: -2,18)
        assert case["canonical_variables"]["x2"]["value"] < -0.30

        meta = case["metadata"]
        assert "P_00141" in meta["patient_id"]
        assert "BENIGN" in meta["pathology_gt"].upper()


@pytest.mark.asyncio
async def test_case_p00141_render_layers():
    """Verifica que las 5 capas de visualización radiológica (clean, wsss, cam, zoom, psi)
    se rendericen en Base64 sin anomalías para el caso clínico P_00141.
    """
    from desktop_app.backend.app import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res_cohort = await client.get("/api/cohort?pathology=all&type=all")
        cohort = res_cohort.json()
        target_index = next(
            item["index"] for item in cohort if "P_00141" in item["patient_id"]
        )

        for layer in ["clean", "wsss", "cam", "zoom", "psi"]:
            res = await client.get(f"/api/cases/{target_index}/render?layer={layer}")
            assert res.status_code == 200, f"Error al renderizar capa {layer}: {res.text}"
            data = res.json()
            assert data["layer"] == layer
            assert data["image_b64"].startswith("data:image/png;base64,")
            assert len(data["image_b64"]) > 200


@pytest.mark.asyncio
async def test_case_p00141_report_and_pdf_generation():
    """Verifica la generación de informe factual y exportación a PDF para el caso P_00141,
    asegurando que el informe contenga evidencia neuro-simbólica y el archivo PDF se guarde en disco.
    """
    from desktop_app.backend.app import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res_cohort = await client.get("/api/cohort?pathology=all&type=all")
        cohort = res_cohort.json()
        target_index = next(
            item["index"] for item in cohort if "P_00141" in item["patient_id"]
        )

        # 1. Generar informe determinista fundamentado
        res_report = await client.post(
            "/api/reports/generate",
            json={"case_index": target_index, "use_llm": False}
        )
        assert res_report.status_code == 200
        report_data = res_report.json()
        assert "report_text" in report_data
        report_text = report_data["report_text"]

        # El informe debe contener la categoría BI-RADS asignada y la terminología de benignidad
        assert any(term in report_text for term in ["BI-RADS", "BENIGN", "Benigno"])
        assert len(report_text) > 100

        # 2. Generar PDF estandarizado
        res_pdf = await client.post(
            "/api/reports/pdf",
            json={"case_index": target_index, "report_text": report_text}
        )
        assert res_pdf.status_code == 200
        pdf_data = res_pdf.json()
        assert "pdf_path" in pdf_data
        assert "filename" in pdf_data
        assert "P_00141" in pdf_data["filename"]
        pdf_path = pdf_data["pdf_path"]
        assert os.path.exists(pdf_path), f"El archivo PDF generado debe existir en disco: {pdf_path}"
        assert os.path.getsize(pdf_path) > 1000, "El tamaño del PDF debe ser superior a 1 KB"


@pytest.mark.asyncio
async def test_case_p00141_caso_limitrofe():
    """P_00141 es un caso ilustrativo y limítrofe (Cap. 4): el radiólogo asignó BI-RADS 4, la biopsia
    fue benigna y la fórmula queda por debajo del umbral, pero cerca (p_sym 41,6 % frente a 53,2 %).
    No es evidencia de eficacia: un solo caso no permite estimar el beneficio en la práctica.
    """
    from desktop_app.backend.app import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res_cohort = await client.get("/api/cohort?pathology=all&type=all")
        cohort = res_cohort.json()
        target_index = next(
            item["index"] for item in cohort if "P_00141" in item["patient_id"]
        )

        res_detail = await client.get(f"/api/cases/{target_index}")
        assert res_detail.status_code == 200
        case = res_detail.json()

        assert case["metadata"]["assessment_gt"] == 4
        assert case["metadata"]["pathology_gt"].upper() == "BENIGN"
        pred = case["prediction"]
        assert pred["verdict"] == "BENIGN"
        assert pred["g_logit"] < pred["threshold_logit"]
        # Limítrofe: a menos de 15 puntos porcentuales del umbral
        assert abs(pred["p_sym"] - pred["threshold_prob"]) < 0.15


def test_zero_orphan_processes_architectural_contracts():
    """Verifica que el ciclo de vida del subproceso Python esté blindado tanto en
    el shell de Electron (main.js) como en el launcher (run_prometeo.py y run_prometeo.bat)
    para garantizar cero procesos huérfanos al cerrar la aplicación.
    """
    # 1. Verificar electron/main.js
    main_js_path = ROOT_DIR / "desktop_app" / "electron" / "main.js"
    assert main_js_path.exists()
    main_js = main_js_path.read_text(encoding="utf-8")

    assert "killPythonProcess" in main_js
    assert "taskkill /PID" in main_js
    assert "/T /F" in main_js, "taskkill debe forzar terminación del árbol de procesos (/T /F)"
    assert "will-quit" in main_js
    assert "mainWindow.on('closed'" in main_js

    # 2. Verificar run_prometeo.py
    launcher_py_path = ROOT_DIR / "desktop_app" / "run_prometeo.py"
    assert launcher_py_path.exists()
    launcher_py = launcher_py_path.read_text(encoding="utf-8")
    assert "KeyboardInterrupt" in launcher_py
    assert "run_backend_only" in launcher_py
    assert "run_electron_app" in launcher_py

    # 3. Verificar run_prometeo.bat
    launcher_bat_path = ROOT_DIR / "desktop_app" / "run_prometeo.bat"
    assert launcher_bat_path.exists()
    launcher_bat = launcher_bat_path.read_text(encoding="utf-8")
    assert ".venv" in launcher_bat
    assert "electron" in launcher_bat.lower()
    assert "npm start" in launcher_bat
