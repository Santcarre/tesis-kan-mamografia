# desktop_app/tests/test_api_health.py
import pytest
from httpx import AsyncClient, ASGITransport

@pytest.mark.asyncio
async def test_health_check_returns_ready():
    from desktop_app.backend.app import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["version"] == "2.0.0"
        assert "device" in data
