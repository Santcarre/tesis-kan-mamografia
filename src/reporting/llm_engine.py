"""
llm_engine.py: Motor unificado de generación de informes clínicos estructurados
para modelos de lenguaje pequeños (Small LLMs), con soporte para Ollama local (Qwen 2.5)
y adaptadores para modelos avanzados (Bonsai 2 27B / OpenRouter).
"""
import os
import re
import json
import requests
from typing import Dict, Any, Optional

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


def _proyeccion(evidence: Dict[str, Any]) -> str:
    """'CC de la mama izquierda' a partir de view y laterality (LEFT/RIGHT de CBIS-DDSM)."""
    lado = {"LEFT": "izquierda", "RIGHT": "derecha"}.get(str(evidence.get("laterality", "")).upper())
    return f"{evidence.get('view')} de la mama {lado}" if lado else f"{evidence.get('view')} ({evidence.get('laterality')})"


def _clasificacion(evidence: Dict[str, Any]) -> str:
    """Línea de la clasificación del modelo, idéntica en las dos condiciones del estudio."""
    g = float(evidence["kan64_g"]); tau = float(evidence["kan64_th"]); p = float(evidence["kan64_p"])
    dec = "sospechosa de malignidad" if g >= tau else "probablemente benigna"
    return f"- Con el umbral del modelo, la lesión se clasifica como {dec} (puntaje p = {p:.3f}; no es una probabilidad calibrada)."


def _birads_herramienta(evidence: Dict[str, Any]) -> str:
    """Categoría orientativa anclada en la decisión (fuente única: birads_herramienta.py)."""
    try:
        from src.reporting import birads_herramienta as _bh
    except ImportError:  # ejecución desde src/reporting
        import birads_herramienta as _bh
    return _bh.etiqueta(float(evidence["kan64_g"]), float(evidence["kan64_th"]))


def format_control_prompt(evidence: Dict[str, Any]) -> str:
    """Texto de entrada de la condición de control del estudio de informes (Anexo A, «Condición sin evidencia»).

    Tiene las MISMAS instrucciones que format_evidence_prompt (usar solo la información dada, no inventar
    hallazgos, el mismo esquema de cuatro secciones) y la MISMA clasificación: la decisión de la fórmula
    con su umbral y su puntaje. Solo le falta la evidencia (u0 y g, categoría, región y tamaño de la zona
    del mapa), de modo que la diferencia entre condiciones se debe a la evidencia y no a las instrucciones
    ni a cambiar de modelo.
    """
    return f"""Eres un radiólogo especialista en imágenes mamarias. Redacta un borrador de informe a partir de la información de un modelo de apoyo. Usa solo esa información: no inventes hallazgos, medidas ni características que no aparezcan abajo.

INFORMACIÓN DEL MODELO
- Paciente: {evidence['patient_id']} | Proyección: {_proyeccion(evidence)}
{_clasificacion(evidence)}

INSTRUCCIÓN:
Redacta un informe radiológico formal y estructurado en español siguiendo estrictamente el siguiente esquema:
1. MOTIVO DE ESTUDIO Y TÉCNICA
2. HALLAZGOS Y DESCRIPCIÓN MAMOGRÁFICA
3. IMPRESIÓN DIAGNÓSTICA Y CATEGORÍA BI-RADS (escribe explícitamente una categoría).
4. RECOMENDACIÓN Y CONDUCTA CLÍNICA
"""


# Regiones de la imagen (regla geométrica de la Sección «Regiones de la imagen»); no son cuadrantes anatómicos.
_REGION_TXT = {"Cuadrante Superior Externo (CSE)": "región superior externa de la imagen (CSE)",
               "Cuadrante Superior Interno (CSI)": "región superior interna de la imagen (CSI)",
               "Cuadrante Inferior Externo (CIE)": "región inferior externa de la imagen (CIE)",
               "Cuadrante Inferior Interno (CII)": "región inferior interna de la imagen (CII)",
               "Región Retroareolar / Central": "región central de la imagen"}


def region_imagen(etiqueta: Any, mayuscula: bool = True) -> str:
    """Texto para mostrar la región del núcleo focal: región de la imagen, no cuadrante anatómico."""
    t = _REGION_TXT.get(str(etiqueta), str(etiqueta))
    return t[:1].upper() + t[1:] if (mayuscula and t) else t


def format_evidence_prompt(evidence: Dict[str, Any]) -> str:
    """Texto de entrada con evidencia de la tesis (Anexo A, «Condición con evidencia»).

    Solo la evidencia que la tesis respalda: u0, g(u0) y el umbral; la misma línea de clasificación que el
    control; la categoría BI-RADS orientativa de la herramienta (anclada en la decisión); la región de la
    imagen y el tamaño de la zona marcada por el núcleo focal sobre el mapa de la fórmula. No incluye la
    compacidad del núcleo (describe la zona del mapa, no la lesión), el tipo de hallazgo, la morfología
    ni la patología.
    """
    u0 = float(evidence["kan64_z64_0"]); g = float(evidence["kan64_g"]); tau = float(evidence["kan64_th"])
    birads = _birads_herramienta(evidence); region = evidence.get("predicted_quadrant", "N/A")
    area = int(evidence.get("pred_area_px", 0))
    return f"""Eres un radiólogo especialista en imágenes mamarias. Redacta un borrador de informe a partir de la evidencia numérica de un modelo de apoyo. Usa solo esa evidencia: no inventes hallazgos, medidas ni características que no aparezcan abajo.

EVIDENCIA DEL MODELO
- Paciente: {evidence['patient_id']} | Proyección: {_proyeccion(evidence)}
{_clasificacion(evidence)}
- Fórmula del modelo: u0 = {u0:+.3f}; g(u0) = {g:+.4f}; umbral g = {tau:.4f}.
- Categoría BI-RADS orientativa de la herramienta (sigue a la clasificación y se ajustó para parecerse a la que asigna el radiólogo; no es un riesgo de malignidad): {birads}.
- Región de la imagen donde se concentra el mapa del modelo: {_REGION_TXT.get(region, region)}. Es una región de la imagen según una regla geométrica, no un cuadrante anatómico de la mama, y es una estimación aproximada.
- Tamaño de la zona marcada por el mapa (no es el contorno de la lesión): {area} píxeles sobre una imagen de 224 x 224.

INSTRUCCIÓN:
Redacta un informe radiológico formal y estructurado en español siguiendo estrictamente el siguiente esquema:
1. MOTIVO DE ESTUDIO Y TÉCNICA
2. HALLAZGOS Y DESCRIPCIÓN MAMOGRÁFICA (ubica el hallazgo en la región de la imagen indicada arriba).
3. IMPRESIÓN DIAGNÓSTICA Y CATEGORÍA BI-RADS (escribe explícitamente la categoría: {birads.split(' (')[0]}).
4. RECOMENDACIÓN Y CONDUCTA CLÍNICA
"""


def generate_deterministic_grounded_report(evidence: Dict[str, Any]) -> str:
    """Informe determinista de la estación: solo la evidencia que la tesis respalda.

    Usa la decisión de la fórmula (claves decision_* de desktop_app/backend/services/decision.py
    si existen; si no, las kan64_*), la categoría BI-RADS orientativa de la herramienta y la región
    y el área de la zona marcada por el núcleo focal sobre el mapa de la fórmula. No usa la
    compacidad del núcleo, el tipo de hallazgo ni la morfología anotados en CBIS-DDSM.
    """
    def _f(v, fmt):
        try:
            return format(float(v), fmt)
        except (TypeError, ValueError):
            return "N/D"

    p_id = evidence.get("patient_id", "P_UNKNOWN")
    lat = evidence.get("laterality", "N/D"); view = evidence.get("view", "N/D")
    u0 = evidence.get("decision_u0", evidence.get("kan64_z64_0"))
    g = evidence.get("decision_g", evidence.get("kan64_g"))
    p = evidence.get("decision_p", evidence.get("kan64_p"))
    th = evidence.get("decision_th_logit", evidence.get("kan64_th"))
    decision = evidence.get("decision") or evidence.get("kan64_pred")
    if decision is None and g is not None and th is not None:
        decision = "MALIGNANT" if float(g) >= float(th) else "BENIGN"
    dec_txt = {"MALIGNANT": "sospechosa de malignidad", "BENIGN": "probablemente benigna"}.get(decision, "sin decisión disponible")
    birads = evidence.get("birads_suggested") or evidence.get("kan64_birads") or "N/D"
    region = evidence.get("predicted_quadrant", "N/D")
    area = evidence.get("pred_area_px", 0)
    rec = evidence.get("clinical_recommendation") or "Correlación clínica y radiológica según la categoría."

    return f"""**1. MOTIVO DE ESTUDIO Y TÉCNICA**
Mamografía de la paciente {p_id}, proyección {_proyeccion(evidence)}, evaluada con el modelo de la tesis (extractor ResNet-18, coordenada u0 y fórmula cerrada g). Borrador generado automáticamente, sin validez clínica.

**2. HALLAZGOS**
- Fórmula: u0 = {_f(u0, '+.3f')}; g(u0) = {_f(g, '+.4f')}; puntaje p = {_f(p, '.3f')} (no es una probabilidad calibrada).
- Con el umbral del modelo (g = {_f(th, '.4f')}), la imagen se clasifica como {dec_txt}.
- El mapa de la fórmula concentra su energía en la {_REGION_TXT.get(region, region)}, en una zona de {area} píxeles sobre 224 x 224 (no es el contorno de la lesión). Es una región de la imagen, no un cuadrante anatómico, y una estimación aproximada: en la evaluación de la tesis coincide con la región de la lesión en menos de la mitad de las mamografías.

**3. IMPRESIÓN DIAGNÓSTICA Y CATEGORÍA BI-RADS**
Categoría orientativa de la herramienta: {birads}. Sigue a la clasificación y se ajustó en validación para parecerse a la que asigna el radiólogo; no es un riesgo de malignidad.

**4. RECOMENDACIÓN Y CONDUCTA CLÍNICA**
{rec} Este borrador requiere la revisión de un radiólogo."""


class LLMEngine:
    """Motor que ejecuta la inferencia con LLMs locales o APIs remotas con fallback determinista."""

    def __init__(self, model_name: str = "qwen2.5-coder:7b-instruct", backend: str = "ollama"):
        self.model_name = model_name
        self.backend = backend.lower()
        self.api_key = os.environ.get("OPENROUTER_API_KEY", None)

    def generate_report(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512, evidence: Dict[str, Any] = None) -> str:
        """Genera el reporte usando el backend configurado con fallback robusto."""
        if self.backend == "ollama":
            out = self._generate_ollama(prompt, temperature, max_tokens)
            if ("Error" in out or "Excepción" in out) and evidence is not None:
                return generate_deterministic_grounded_report(evidence)
            return out
        elif self.backend == "openrouter":
            out = self._generate_openrouter(prompt, temperature, max_tokens)
            if ("Error" in out or "Excepción" in out) and evidence is not None:
                return generate_deterministic_grounded_report(evidence)
            return out
        else:
            if evidence is not None:
                return generate_deterministic_grounded_report(evidence)
            raise ValueError(f"Backend no soportado: {self.backend}")

    def _generate_ollama(self, prompt: str, temperature: float, max_tokens: int) -> str:
        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        try:
            resp = requests.post(url, json=payload, timeout=120)
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
            else:
                return f"Error en Ollama ({resp.status_code}): {resp.text}"
        except Exception as e:
            return f"Excepción de conexión con Ollama: {str(e)}"

    def _generate_openrouter(self, prompt: str, temperature: float, max_tokens: int) -> str:
        if not self.api_key:
            return "[OpenRouter Error: Se requiere OPENROUTER_API_KEY para consultar Bonsai 2 27B remotamente]"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        try:
            resp = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=90)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                return f"Error OpenRouter ({resp.status_code}): {resp.text}"
        except Exception as e:
            return f"Excepción OpenRouter: {str(e)}"


def parse_clinical_report(report_text: str) -> Dict[str, Any]:
    """Extrae las entidades clínicas y diagnósticas del texto del reporte generado."""
    # 1. Extraer cuadrante reportado con límites de palabra estrictos
    quadrant = "No especificado"
    if re.search(r"cuadrante\s+superior\s+externo|\bCSE\b", report_text, re.IGNORECASE):
        quadrant = "Cuadrante Superior Externo (CSE)"
    elif re.search(r"cuadrante\s+superior\s+interno|\bCSI\b", report_text, re.IGNORECASE):
        quadrant = "Cuadrante Superior Interno (CSI)"
    elif re.search(r"cuadrante\s+inferior\s+externo|\bCIE\b", report_text, re.IGNORECASE):
        quadrant = "Cuadrante Inferior Externo (CIE)"
    elif re.search(r"cuadrante\s+inferior\s+interno|\bCII\b", report_text, re.IGNORECASE):
        quadrant = "Cuadrante Inferior Interno (CII)"
    elif re.search(r"retroareolar|regi[oó]n\s+central", report_text, re.IGNORECASE):
        quadrant = "Región Retroareolar / Central"

    # 2. Extraer categoría BI-RADS
    birads_cat = None
    m_birads = re.search(r"BI-?RADS\s*([1-5])", report_text, re.IGNORECASE)
    if m_birads:
        birads_cat = int(m_birads.group(1))

    # 3. Detectar si menciona masas o calcificaciones
    mentions_mass = bool(re.search(r"masa|nódulo|lesión sólida|distorsión", report_text, re.IGNORECASE))
    mentions_calc = bool(re.search(r"calcificaci|microcalcificaci", report_text, re.IGNORECASE))

    return {
        "reported_quadrant": quadrant,
        "reported_birads": birads_cat,
        "mentions_mass": mentions_mass,
        "mentions_calc": mentions_calc,
        "full_text": report_text
    }
