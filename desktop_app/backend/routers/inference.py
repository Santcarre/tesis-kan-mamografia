"""
Router para inferencia y análisis prospectivo de estudios externos (DICOM / JPG / PNG)
y ejecución de casos de demostración preconfigurados.
"""
import asyncio
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File

from desktop_app.backend.schemas import DemoCaseSummary, ExternalInferenceResponse
from desktop_app.backend.services.inference_service import get_inference_service

router = APIRouter(prefix="/api/inference", tags=["Inferencia Externa"])


@router.get(
    "/demos",
    response_model=List[DemoCaseSummary],
    summary="Listar catálogo de casos demo preconfigurados"
)
async def get_demo_cases() -> List[DemoCaseSummary]:
    """Retorna la lista de los 4 casos de demostración (DICOM nativos y JPG)
    preconfigurados en la estación radiológica.
    """
    service = get_inference_service()
    return service.get_demo_cases()


@router.post(
    "/demo/{demo_id}",
    response_model=ExternalInferenceResponse,
    summary="Ejecutar inferencia analítica sobre un caso demo"
)
async def infer_demo_case(demo_id: str) -> ExternalInferenceResponse:
    """Ejecuta el pipeline completo de inferencia neuro-simbólica (extracción R3,
    proyección canónica S-PCA, clasificación BI-RADS y segmentación WSSS) sobre
    el caso de demostración solicitado de forma no bloqueante.
    """
    service = get_inference_service()
    try:
        return await asyncio.to_thread(service.process_demo_case, demo_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Caso de demostración '{demo_id}' no encontrado en el catálogo."
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Archivo físico de muestra no encontrado: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno durante la inferencia del caso demo: {str(e)}"
        )


@router.post(
    "/upload",
    response_model=ExternalInferenceResponse,
    summary="Cargar mamografía externa (DICOM / JPG / PNG) y ejecutar inferencia"
)
async def upload_and_infer(
    file: UploadFile = File(..., description="Archivo de mamografía (.dcm, .dicom, .jpg, .jpeg, .png)")
) -> ExternalInferenceResponse:
    """Recibe un archivo médico mediante multipart/form-data, decodifica la matriz
    de píxeles, la estandariza a 224×224 y ejecuta la inferencia neuro-simbólica en
    un hilo de trabajo separado sin bloquear el bucle de eventos.
    Si el archivo está corrupto o tiene un formato no soportado, retorna HTTP 422.
    """
    service = get_inference_service()
    try:
        content = await file.read()
        return await asyncio.to_thread(service.process_image, content, file.filename or "")
    except ValueError as e:
        # Formato no soportado, matriz vacía o archivo corrupto
        raise HTTPException(
            status_code=422,
            detail=f"El archivo subido está corrupto o no tiene un formato válido (DICOM, JPG, PNG): {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno durante el procesamiento del archivo: {str(e)}"
        )
