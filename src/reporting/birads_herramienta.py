"""
birads_herramienta.py: categoría BI-RADS orientativa de la herramienta (fuente única).

La categoría se deriva de la decisión de la fórmula y de su puntaje p_sym = sigma(g(u0)):

  * decisión benigna   (g < tau*)  -> BI-RADS 2 si p_sym < CORTE_2_3, si no BI-RADS 3;
  * decisión sospechosa (g >= tau*) -> BI-RADS 4 si p_sym < CORTE_4_5, si no BI-RADS 5.

El corte entre 3 y 4 es el propio umbral de decisión, de modo que la categoría nunca contradice
el veredicto. Los cortes 2|3 y 4|5 se fijaron SOLO en validación, maximizando el kappa ponderado
cuadrático con la valoración BI-RADS del radiólogo (columna 'assessment' de CBIS-DDSM, escala
entera 2-5; CBIS no subdivide la categoría 4), con scripts/u0_birads_radiologo.py
(reports/auditoria/u0_birads_radiologo.json). En prueba: kappa = 0,44 [0,35; 0,51].

Es una categoría ORIENTATIVA que imita la valoración del radiólogo con acuerdo moderado. No es
una probabilidad de malignidad: en CBIS-DDSM, cohorte enriquecida con lesiones biopsiadas, la
proporción de malignas por categoría no sigue los rangos de riesgo del ACR (tampoco la del
radiólogo). Módulo puro (biblioteca estándar).
"""
import math
import re

CORTE_2_3 = 0.3572  # p_sym; fijado en validación (u0_birads_radiologo.json)
CORTE_4_5 = 0.6761  # p_sym; fijado en validación (u0_birads_radiologo.json)

ETIQUETAS = {
    2: "BI-RADS 2 (Hallazgo benigno)",
    3: "BI-RADS 3 (Probablemente benigno)",
    4: "BI-RADS 4 (Sospechoso)",
    5: "BI-RADS 5 (Altamente sugestivo de malignidad)",
}

# Conducta que BI-RADS asocia a cada categoría (ACR, 5.a ed.); no es una indicación para la paciente.
CONDUCTA = {
    2: "Control mamográfico de tamizaje habitual.",
    3: "Seguimiento a corto plazo (6 meses) para verificar estabilidad.",
    4: "Valoración de biopsia.",
    5: "Biopsia y manejo según el resultado histopatológico.",
}

NOTA = ("Categoría orientativa de la herramienta: cortes sobre el puntaje anclados en el umbral de "
        "decisión y ajustados en validación a la valoración del radiólogo (kappa ponderado 0,44 en "
        "prueba). No es una probabilidad de malignidad ni una recomendación médica.")


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def categoria(g: float, tau: float) -> int:
    """Categoría entera (2-5) a partir de g(u0) y del umbral de decisión tau* (en la escala de g)."""
    p = _sigmoid(float(g))
    if float(g) >= float(tau):
        return 5 if p >= CORTE_4_5 else 4
    return 2 if p < CORTE_2_3 else 3


def categoria_desde_p(p: float, p_tau: float) -> int:
    """Igual que categoria(), con el puntaje p_sym y el umbral expresado como puntaje sigma(tau*)."""
    p = float(p)
    if p >= float(p_tau):
        return 5 if p >= CORTE_4_5 else 4
    return 2 if p < CORTE_2_3 else 3


def etiqueta(g: float, tau: float) -> str:
    return ETIQUETAS[categoria(g, tau)]


def etiqueta_desde_p(p: float, p_tau: float) -> str:
    return ETIQUETAS[categoria_desde_p(p, p_tau)]


def a_entero(label) -> int:
    """'BI-RADS 4 (Sospechoso)' -> 4; 0 si no hay categoría."""
    m = re.search(r"BI-RADS\s*(\d)", str(label))
    return int(m.group(1)) if m else 0


def conducta(label) -> str:
    return CONDUCTA.get(a_entero(label), "")
