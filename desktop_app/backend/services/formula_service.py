"""
formula_service.py: evaluación de la fórmula canónica g(u0) = f_mal(u0) - f_ben(u0).

La fórmula se lee de `reports/experiments/kan_robust64_canonical.json` (el mismo
artefacto del Anexo A) y se evalúa con un intérprete aritmético de lista blanca:
no se usa eval() ni sympy. La variable de entrada es la coordenada canónica u0
(primera componente de S-PCA64); no existe un puente desde las componentes
auxiliares (x0, x1, x2) hacia u0.

    p_sym = sigmoide(g(u0));   decisión: MALIGNANT si g(u0) >= tau* (Youden, validación)

La categoría BI-RADS es la orientativa de la herramienta, anclada en tau* (src/reporting/birads_herramienta.py).
"""
import json
import os
from typing import List, Optional

from desktop_app.backend.config import CODIGOS_DIR
from desktop_app.backend.schemas import (
    FormulaEvalRequest,
    FormulaEvalResponse,
    FormulaSweepResponse,
)
from desktop_app.backend.services.decision import birads_from_p, birads_int
from src.reporting.formula_eval import compile_expr, sigmoid

# Respaldo solo si faltara el artefacto canónico (son los mismos coeficientes del JSON).
_FALLBACK_BEN = (
    "0.0471891401526251*x_1 + 0.328273892402649*cos(0.214169642965341*x_1 - 2.33565497672423) "
    "+ 0.336114823818207*cos(0.274644437716577*x_1 + 3.59239355625278) "
    "+ 0.349621593952179*cos(0.314384453987867*x_1 - 6.70736043138517) "
    "+ 0.385232597589493*cos(0.350673697391121*x_1 + 3.69334876906847) "
    "+ 0.381856471300125*cos(0.354226147711424*x_1 + 5.92455551636943) "
    "+ 0.259875884173518"
)
_FALLBACK_MAL = (
    "-0.320272325390727*x_1 + 0.34225144982338*sin(0.32903093975628*x_1 - 2.08868830700881) "
    "+ 0.175986140966415*sin(0.370365415273934*x_1 + 8.44352602817863) "
    "+ 0.283501774072647*cos(0.300271759027153*x_1 + 7.10500282872347) "
    "+ 0.362019270658493*cos(0.332756501225072*x_1 - 3.46045717846679) "
    "+ 0.306373566389084*cos(0.353347587136186*x_1 - 9.53764778569112) "
    "+ 0.786598895731083"
)
_FALLBACK_TH = 0.1296

_U_MIN, _U_MAX = -10.0, 10.0
_sigmoid = sigmoid


class SymbolicFormulaEvaluator:
    """Evaluador de f_ben, f_mal y g sobre u0, con el umbral de Youden de validación."""

    def __init__(self, codigos_dir: Optional[str] = None):
        self.codigos_dir = codigos_dir or str(CODIGOS_DIR)
        self.expr_ben, self.expr_mal = _FALLBACK_BEN, _FALLBACK_MAL
        self.youden_threshold = _FALLBACK_TH
        self.source = "respaldo incrustado"
        self._load_canonical_json()
        self._f_ben = compile_expr(self.expr_ben)
        self._f_mal = compile_expr(self.expr_mal)

    def _load_canonical_json(self) -> None:
        path = os.path.join(self.codigos_dir, "reports", "experiments", "kan_robust64_canonical.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "fben" in data and "fmal" in data:
                self.expr_ben, self.expr_mal = data["fben"], data["fmal"]
                self.source = "kan_robust64_canonical.json"
            if "th_youden_val" in data:
                self.youden_threshold = float(data["th_youden_val"])
        except Exception:
            pass

    @property
    def formula_expr(self) -> str:
        return f"({self.expr_mal}) - ({self.expr_ben})"

    @property
    def youden_prob(self) -> float:
        return _sigmoid(self.youden_threshold)

    def eval_g(self, u0: float):
        u = min(max(float(u0), _U_MIN), _U_MAX)
        fb, fm = self._f_ben(u), self._f_mal(u)
        return u, fb, fm, fm - fb


class FormulaService:
    """Evaluación puntual y barrido de la fórmula sobre u0."""

    def __init__(self, evaluator: Optional[SymbolicFormulaEvaluator] = None):
        self.evaluator = evaluator or SymbolicFormulaEvaluator()

    def evaluate(self, req: FormulaEvalRequest) -> FormulaEvalResponse:
        ev = self.evaluator
        u, fb, fm, g = ev.eval_g(req.u0)
        p = _sigmoid(g)
        label = birads_from_p(p, ev.youden_prob)   # anclada en el umbral de la fórmula
        return FormulaEvalResponse(
            u0=round(u, 5),
            f_ben=round(fb, 5),
            f_mal=round(fm, 5),
            logit=round(g, 5),
            prob=round(p, 5),
            birads=birads_int(label),
            birads_label=label,
            formula_expr=ev.formula_expr,
            decision="MALIGNANT" if g >= ev.youden_threshold else "BENIGN",
            youden_threshold=ev.youden_threshold,
            youden_threshold_prob=round(ev.youden_prob, 5),
        )

    def sweep(self, var_name: str = "u0", val_min: float = -4.0, val_max: float = 4.0,
              points: int = 81, **_ignored) -> FormulaSweepResponse:
        if var_name.strip().lower() != "u0":
            raise ValueError(
                f"Variable '{var_name}' no soportada: la fórmula canónica depende solo de u0."
            )
        if val_min >= val_max:
            raise ValueError(f"val_min ({val_min}) debe ser menor que val_max ({val_max}).")
        if points < 2:
            raise ValueError(f"points ({points}) debe ser al menos 2.")

        step = (val_max - val_min) / (points - 1)
        xs: List[float] = [val_min + i * step for i in range(points)]
        probs, logits, fbs, fms = [], [], [], []
        for x in xs:
            _, fb, fm, g = self.evaluator.eval_g(x)
            fbs.append(round(fb, 5))
            fms.append(round(fm, 5))
            logits.append(round(g, 5))
            probs.append(round(_sigmoid(g), 5))
        return FormulaSweepResponse(
            var_name="u0",
            x_values=[round(x, 4) for x in xs],
            prob_values=probs,
            logit_values=logits,
            fben_values=fbs,
            fmal_values=fms,
            youden_threshold=self.evaluator.youden_threshold,
        )


_instance: Optional[FormulaService] = None


def get_formula_service() -> FormulaService:
    global _instance
    if _instance is None:
        _instance = FormulaService()
    return _instance
