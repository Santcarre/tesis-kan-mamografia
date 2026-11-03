"""
decision.py: Decisión diagnóstica de la estación a partir de la fórmula canónica.

La tesis define la decisión como g(u0) >= tau*, con g(u0) = f_mal(u0) - f_ben(u0),
p_sym = sigma(g(u0)) y tau* el umbral de Youden calibrado en validación. Las
claves `kan64_*` de la evidencia (producidas por ClinicalEvidenceExtractor)
traen esos valores. Este módulo:

  * decide el veredicto con la fórmula (no con la probabilidad de la CNN);
  * asigna la categoría BI-RADS orientativa de src/reporting/birads_herramienta.py:
    anclada en la decisión (2-3 si es benigna, 4-5 si es sospechosa), con cortes
    ajustados en validación a la valoración del radiólogo; no es un riesgo calibrado;
  * si la evidencia no trae la fórmula (artefactos ausentes), cae a la
    probabilidad de la red y lo declara en `decision_source`.

No importa torch ni numpy: se puede probar de forma aislada (solo src.reporting puro).
"""
import math
import re
from typing import Any, Dict, Optional

from src.reporting import birads_herramienta as _bh

NOTA_CONVENCION = _bh.NOTA


def birads_from_p(p: float, p_tau: Optional[float] = None) -> str:
    """Etiqueta BI-RADS orientativa para un puntaje p_sym (fuente única: src/reporting/birads_herramienta.py).

    p_tau es el umbral de decisión expresado como puntaje, sigma(tau*); por defecto, el de la fórmula canónica.
    """
    return _bh.etiqueta_desde_p(p, _P_TAU if p_tau is None else p_tau)


def birads_int(label: str) -> int:
    return _bh.a_entero(label)


def recommendation_for(label: str) -> str:
    return _bh.conducta(label)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _tau_canonico() -> float:
    """tau* de la fórmula canónica (escala de g); 0,1296 si el artefacto no está disponible."""
    import os
    from pathlib import Path
    try:
        from src.reporting.formula_eval import load_canonical
    except Exception:
        return 0.1296
    raiz = Path(__file__).resolve().parents[3]
    for base in (os.environ.get("TESIS_CODIGOS"), str(raiz), str(raiz / "Códigos"), None):
        d = load_canonical(base) if (base is None or os.path.isdir(base)) else None
        if d and "th_youden_val" in d:
            return float(d["th_youden_val"])
    return 0.1296


_P_TAU = _sigmoid(_tau_canonico())


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def apply_formula_decision(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Añade a `evidence` el bloque de decisión basado en la fórmula.

    Claves añadidas (y sobrescritas: `birads_suggested`, `clinical_recommendation`):
      decision_source   'formula' | 'cnn_fallback'
      decision          'MALIGNANT' | 'BENIGN'
      decision_u0, decision_g, decision_p, decision_th_logit, decision_th_prob
      prob_net_robust   probabilidad de la red robusta (sigmoide del logit)
      birads_suggested, clinical_recommendation
    Es idempotente.
    """
    g = _num(evidence.get("kan64_g"))
    th = _num(evidence.get("kan64_th"))
    p_net = _num(evidence.get("prob_cnn"))
    evidence["prob_net_robust"] = p_net

    if g is not None and th is not None:
        p = _sigmoid(g)
        evidence["decision_source"] = "formula"
        evidence["decision"] = "MALIGNANT" if g >= th else "BENIGN"
        evidence["decision_u0"] = _num(evidence.get("kan64_z64_0"))
        evidence["decision_g"] = g
        evidence["decision_p"] = p
        evidence["decision_th_logit"] = th
        evidence["decision_th_prob"] = _sigmoid(th)
    else:
        p = p_net if p_net is not None else 0.0
        evidence["decision_source"] = "cnn_fallback"
        evidence["decision"] = "MALIGNANT" if p >= 0.5 else "BENIGN"
        evidence["decision_u0"] = None
        evidence["decision_g"] = None
        evidence["decision_p"] = p
        evidence["decision_th_logit"] = None
        evidence["decision_th_prob"] = 0.5

    if evidence["decision_source"] == "formula":
        label = _bh.etiqueta(g, th)          # anclada en la decisión de la fórmula
    else:
        label = birads_from_p(p, 0.5)        # respaldo declarado: la red con corte 0,5
    evidence["birads_suggested"] = label
    evidence["clinical_recommendation"] = recommendation_for(label)
    return evidence
