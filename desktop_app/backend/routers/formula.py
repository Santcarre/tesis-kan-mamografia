"""
Router para evaluación puntual y barrido de la fórmula canónica g(u0).
"""
from fastapi import APIRouter, HTTPException, Query

from desktop_app.backend.schemas import (
    FormulaEvalRequest,
    FormulaEvalResponse,
    FormulaSweepResponse,
)
from desktop_app.backend.services.formula_service import get_formula_service

router = APIRouter(prefix="/api/formula", tags=["Fórmula canónica"])


@router.post(
    "/eval",
    response_model=FormulaEvalResponse,
    summary="Evalúa g(u0), p_sym y la decisión en un punto u0"
)
async def evaluate_formula(request: FormulaEvalRequest) -> FormulaEvalResponse:
    """Evalúa f_ben(u0), f_mal(u0), g(u0) = f_mal - f_ben y p_sym = sigmoide(g).
    La decisión compara g con el umbral de Youden calibrado en validación.
    """
    service = get_formula_service()
    try:
        return service.evaluate(request)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante la evaluación de la fórmula: {str(e)}"
        )


@router.get(
    "/sweep",
    response_model=FormulaSweepResponse,
    summary="Barrido de la fórmula a lo largo de u0"
)
async def sweep_formula(
    var_name: str = Query("u0", description="Variable a barrer; la fórmula solo depende de u0"),
    val_min: float = Query(-4.0, description="Valor mínimo de u0"),
    val_max: float = Query(4.0, description="Valor máximo de u0"),
    points: int = Query(81, ge=2, le=500, description="Número de puntos del barrido"),
) -> FormulaSweepResponse:
    """Devuelve f_ben, f_mal, g y p_sym sobre una malla uniforme de u0."""
    service = get_formula_service()
    try:
        return service.sweep(var_name=var_name, val_min=val_min, val_max=val_max, points=points)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante el barrido: {str(e)}"
        )
