"""
Router para exploración, filtrado y catálogo de la cohorte CBIS-DDSM.
"""
from typing import List
from fastapi import APIRouter, Query

from desktop_app.backend.schemas import CaseSummary
from desktop_app.backend.services.cases_service import get_cases_service

router = APIRouter(prefix="/api/cohort", tags=["Cohorte"])


@router.get("", response_model=List[CaseSummary], summary="Catálogo filtrado de la cohorte CBIS-DDSM")
async def get_cohort(
    pathology: str = Query(
        default="all",
        description="Filtro por patología: 'all', 'benign' o 'malignant'"
    ),
    type: str = Query(
        default="all",
        description="Filtro por tipo de lesión: 'all', 'mass' o 'calc'"
    ),
) -> List[CaseSummary]:
    """Retorna la lista de casos disponibles con sus metadatos principales,
    permitiendo filtrado dinámico para la barra de navegación clínica.
    """
    service = get_cases_service()
    return service.get_cohort(pathology=pathology, abnormality_type=type)
