"""
Pruebas de integración para el servicio y endpoint de renderizado analítico de imágenes.
Valida la generación de mapas Base64 en memoria para clean, cam, wsss, zoom y psi.
"""
import pytest
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_render_layers():
    from desktop_app.backend.app import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for layer in ["clean", "cam", "wsss", "zoom", "psi"]:
            res = await client.get(f"/api/cases/0/render?layer={layer}")
            assert res.status_code == 200
            data = res.json()
            assert data["layer"] == layer
            assert data["image_b64"].startswith("data:image/png;base64,")
            # Verificar que el contenido base64 no sea trivialmente corto
            assert len(data["image_b64"]) > 200


@pytest.mark.asyncio
async def test_render_invalid_layer():
    from desktop_app.backend.app import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/api/cases/0/render?layer=invalid_layer")
        assert res.status_code == 400
        assert "detail" in res.json()


@pytest.mark.asyncio
async def test_render_case_not_found():
    from desktop_app.backend.app import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/api/cases/999999/render?layer=clean")
        assert res.status_code == 404
        assert "detail" in res.json()
