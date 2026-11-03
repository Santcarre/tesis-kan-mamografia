"""
Router para endpoints de estado, diagnóstico y verificación de salud (health check).
"""
from fastapi import APIRouter
from desktop_app.backend.config import DEVICE, VERSION
from desktop_app.backend.schemas import HealthResponse

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """Verifica el estado de salud del backend y disponibilidad del dispositivo."""
    return HealthResponse(
        status="ready",
        version=VERSION,
        device=DEVICE,
        models_loaded=False,
    )
