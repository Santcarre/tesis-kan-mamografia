"""
Router para generación de informes clínicos estructurados y exportación en PDF.
Expone endpoints asíncronos y no bloqueantes delegando a ReportService.
"""
import asyncio
from fastapi import APIRouter, HTTPException

from desktop_app.backend.schemas import (
    ReportGenerateRequest,
    ReportGenerateResponse,
    ReportPDFRequest,
    ReportPDFResponse,
)
from desktop_app.backend.services.report_service import get_report_service

router = APIRouter(prefix="/api/reports", tags=["Informes Clínicos"])


@router.post(
    "/generate",
    response_model=ReportGenerateResponse,
    summary="Generar informe radiológico estructurado grounded"
)
async def generate_report_endpoint(request: ReportGenerateRequest) -> ReportGenerateResponse:
    """Sintetiza un informe clínico estructurado con respaldo factual (BI-RADS,
    fórmula analítica R3 y centroide WSSS).
    Si se solicita use_llm=True, consulta Qwen-2.5 local vía Ollama con fallback
    automático determinista si no está disponible.
    """
    service = get_report_service()
    try:
        report_text, engine = await asyncio.to_thread(
            service.generate_report,
            case_index=request.case_index,
            use_llm=request.use_llm,
            custom_evidence=request.custom_evidence,
        )
        return ReportGenerateResponse(report_text=report_text, engine=engine)
    except (IndexError, KeyError) as e:
        raise HTTPException(
            status_code=404,
            detail=f"Caso clínico no encontrado o índice fuera de rango: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno durante la generación del informe clínico: {str(e)}"
        )


@router.post(
    "/pdf",
    response_model=ReportPDFResponse,
    summary="Compilar y exportar informe clínico formal en formato PDF"
)
async def generate_pdf_endpoint(request: ReportPDFRequest) -> ReportPDFResponse:
    """Compila un documento PDF formal de 2 páginas con ReportLab que integra
    el informe estructurado, la ficha del paciente, la evidencia matemática
    y el anexo visual de trazabilidad espacial (WSSS y curva sigmoide).
    """
    service = get_report_service()
    try:
        pdf_path, filename = await asyncio.to_thread(
            service.generate_pdf,
            case_index=request.case_index,
            report_text=request.report_text,
            custom_evidence=request.custom_evidence,
        )
        return ReportPDFResponse(pdf_path=pdf_path, filename=filename)
    except (IndexError, KeyError) as e:
        raise HTTPException(
            status_code=404,
            detail=f"Caso clínico no encontrado o índice fuera de rango: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno durante la compilación del documento PDF: {str(e)}"
        )
