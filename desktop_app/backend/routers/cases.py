"""
Router para consulta de casos individuales, predicciones analíticas y variables canónicas.
"""
import asyncio
from fastapi import APIRouter, HTTPException, Path, Query

from desktop_app.backend.schemas import CaseDetail, RenderResponse
from desktop_app.backend.services.cases_service import get_cases_service
from desktop_app.backend.services.render_service import get_render_service

router = APIRouter(prefix="/api/cases", tags=["Casos"])



@router.get(
    "/{case_index}",
    response_model=CaseDetail,
    summary="Detalle analítico y clínico completo de un caso CBIS-DDSM"
)
async def get_case_by_index(
    case_index: int = Path(..., ge=0, description="Índice numérico del caso en el dataset")
) -> CaseDetail:
    """Extrae y retorna los metadatos clínicos, la predicción del modelo robusto R3,
    las tres variables canónicas (x0, x1, x2), el centroide focal WSSS y el cuadrante anatómico.
    """
    service = get_cases_service()
    try:
        return service.get_case_detail(case_index)
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante el procesamiento del caso {case_index}: {str(e)}"
        )


@router.get(
    "/{case_index}/render",
    response_model=RenderResponse,
    summary="Renderizado analítico en Base64 de capas mamográficas"
)
async def render_case_layer(
    case_index: int = Path(..., ge=0, description="Índice numérico del caso en el dataset"),
    layer: str = Query("clean", description="Capa a renderizar: clean, cam, wsss, zoom, psi, psikan")
) -> RenderResponse:
    """Genera en memoria una imagen PNG codificada en Base64 para la capa analítica seleccionada:
    - clean: mamografía desnormalizada en escala de grises médico.
    - cam: mapa continuo de activación espectral S-PCA / Grad-CAM con centroide focal.
    - wsss: segmentación débil focal (tau >= 60%) superpuesta a la silueta tisular.
    - zoom: lupa focal de aumento 2.5x con realce adaptativo de contraste local.
    - psi: campo continuo de densidad simbólica psi_2(u, v) para análisis de márgenes/espiculación.
    """
    render_service = get_render_service()
    try:
        image_b64 = await asyncio.to_thread(render_service.render_layer, case_index, layer)
        return RenderResponse(image_b64=image_b64, layer=layer.strip().lower())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante el renderizado del caso {case_index}: {str(e)}"
        )

