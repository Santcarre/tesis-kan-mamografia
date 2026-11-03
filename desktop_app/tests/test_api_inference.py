"""
Pruebas de integración para endpoints de inferencia externa (DICOM / JPG).
"""
import io
from pathlib import Path
import pytest
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_get_demos_and_infer():
    """Verifica el listado de casos demo y la inferencia sobre el primer caso demo."""
    from desktop_app.backend.app import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res_demos = await client.get("/api/inference/demos")
        assert res_demos.status_code == 200
        demos = res_demos.json()
        assert len(demos) >= 2
        assert "id" in demos[0]
        assert "filename" in demos[0]

        # Inferencia con demo 0
        demo_id = demos[0]["id"]
        res_infer = await client.post(f"/api/inference/demo/{demo_id}")
        assert res_infer.status_code == 200
        result = res_infer.json()
        assert "prob_malignant" in result
        assert "canonical_variables" in result
        assert "image_b64" in result
        assert 0.0 <= result["prob_malignant"] <= 1.0
        assert "x0" in result["canonical_variables"]
        assert "x1" in result["canonical_variables"]
        assert "x2" in result["canonical_variables"]
        assert result["image_b64"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_infer_all_demos():
    """Verifica que todos los casos demo (DICOM y JPG) puedan ejecutarse correctamente."""
    from desktop_app.backend.app import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res_demos = await client.get("/api/inference/demos")
        assert res_demos.status_code == 200
        demos = res_demos.json()
        assert len(demos) == 4

        for demo in demos:
            d_id = demo["id"]
            res = await client.post(f"/api/inference/demo/{d_id}")
            assert res.status_code == 200
            data = res.json()
            assert "prob_malignant" in data
            assert "birads_pred" in data
            assert "canonical_variables" in data
            assert data["image_b64"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_demo_not_found():
    """Verifica que solicitar un demo inexistente retorne 404 Not Found."""
    from desktop_app.backend.app import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post("/api/inference/demo/demo_inexistente_999")
        assert res.status_code == 404
        data = res.json()
        assert "detail" in data


@pytest.mark.asyncio
async def test_upload_image_inference():
    """Verifica la subida de un archivo real de imagen/DICOM mediante multipart upload."""
    from desktop_app.backend.app import app
    sample_path = Path("desktop_app/demo_samples/sample_calc_benign.jpg")
    assert sample_path.exists()
    content = sample_path.read_bytes()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        files = {"file": ("sample_calc_benign.jpg", io.BytesIO(content), "image/jpeg")}
        res = await client.post("/api/inference/upload", files=files)
        assert res.status_code == 200
        data = res.json()
        assert "prob_malignant" in data
        assert "canonical_variables" in data
        assert "image_b64" in data
        assert data["image_b64"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_upload_corrupt_file():
    """Verifica que un archivo corrupto retorne HTTP 422 con mensaje descriptivo en español."""
    from desktop_app.backend.app import app
    corrupt_bytes = b"Este no es un archivo DICOM ni una imagen grafica valida."

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        files = {"file": ("corrupto.dcm", io.BytesIO(corrupt_bytes), "application/octet-stream")}
        res = await client.post("/api/inference/upload", files=files)
        assert res.status_code == 422
        data = res.json()
        assert "detail" in data
        # Verificar que el mensaje descriptivo esté en español
        detail_msg = data["detail"].lower()
        assert any(word in detail_msg for word in ["archivo", "formato", "imagen", "corrupto", "válido", "valido"])
