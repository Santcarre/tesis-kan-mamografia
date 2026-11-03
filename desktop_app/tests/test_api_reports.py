"""
Pruebas automatizadas de la API de informes clínicos y generación de PDF.
Verifica endpoints de generación determinista, fallback ante desconexión de LLM
y compilación de documentos PDF con trazabilidad neuro-simbólica.
"""
import os
import pytest
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_generate_report_and_pdf():
    """Prueba básica de contrato según especificación del plan."""
    from desktop_app.backend.app import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Generar texto determinista
        res_text = await client.post("/api/reports/generate", json={"case_index": 0, "use_llm": False})
        assert res_text.status_code == 200
        text_data = res_text.json()
        assert "report_text" in text_data
        assert len(text_data["report_text"]) > 50
        assert text_data.get("engine") == "deterministic-grounded"

        # Generar PDF
        res_pdf = await client.post("/api/reports/pdf", json={"case_index": 0, "report_text": text_data["report_text"]})
        assert res_pdf.status_code == 200
        pdf_data = res_pdf.json()
        assert os.path.exists(pdf_data["pdf_path"])
        assert pdf_data["filename"].endswith(".pdf")
        assert os.path.getsize(pdf_data["pdf_path"]) > 1000


@pytest.mark.asyncio
async def test_generate_report_llm_fallback():
    """Verifica que solicitar generación con LLM cuando Ollama no está disponible
    haga fallback gracioso a generación determinista sin fallar ni arrojar 500.
    """
    from desktop_app.backend.app import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post("/api/reports/generate", json={"case_index": 0, "use_llm": True})
        assert res.status_code == 200
        data = res.json()
        assert "report_text" in data
        assert len(data["report_text"]) > 50
        # Debe haber realizado fallback si Ollama no está activo o ejecutado Qwen si está activo
        assert data["engine"] in ["deterministic-grounded", "llm-local"]
        assert "BI-RADS" in data["report_text"]


@pytest.mark.asyncio
async def test_generate_report_invalid_case():
    """Verifica que un índice de caso inválido sin custom_evidence retorne 404."""
    from desktop_app.backend.app import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post("/api/reports/generate", json={"case_index": 999999, "use_llm": False})
        assert res.status_code == 404
        assert "no encontrado" in res.json()["detail"].lower() or "fuera de rango" in res.json()["detail"].lower()


@pytest.mark.asyncio
async def test_generate_report_custom_evidence():
    """Verifica que la generación acepte un diccionario de evidencia personalizada."""
    from desktop_app.backend.app import app
    custom_ev = {
        "patient_id": "P_CUSTOM_TEST",
        "laterality": "RIGHT",
        "view": "CC",
        "abnormality_type": "mass",
        "predicted_quadrant": "Cuadrante Superior Externo (CSE)",
        "pred_area_px": 420,
        "compactness": 0.85,
        "equiv_diameter_px": 23.1,
        "kan64_birads": "BI-RADS 4C",
        "kan64_g": 0.45,
        "kan64_p": 0.61,
        "kan64_pred": "MALIGNANT",
        "kan64_th": 0.1296,
        "kan64_AUC_D": 0.84,
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/api/reports/generate",
            json={"case_index": -1, "use_llm": False, "custom_evidence": custom_ev}
        )
        assert res.status_code == 200
        data = res.json()
        assert "P_CUSTOM_TEST" in data["report_text"] or "BI-RADS 4C" in data["report_text"]
