"""
Esquemas Pydantic para validación y serialización de peticiones y respuestas.
"""
from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Respuesta del endpoint de diagnóstico y salud del sistema."""
    status: str = Field(default="ready", description="Estado del backend")
    version: str = Field(default="2.0.0", description="Versión de Prometeo PACS")
    device: str = Field(description="Dispositivo de cómputo activo (cuda o cpu)")
    models_loaded: bool = Field(default=False, description="Indica si los modelos PyTorch están cargados en memoria")


class CaseSummary(BaseModel):
    """Resumen de caso para el catálogo y filtrado de la cohorte CBIS-DDSM."""
    index: int = Field(description="Índice secuencial del caso en la cohorte")
    patient_id: str = Field(description="Identificador del paciente (ej: P_00141)")
    laterality: str = Field(default="UNKNOWN", description="Lateralidad mamaria (LEFT / RIGHT)")
    view: str = Field(default="UNKNOWN", description="Proyección radiológica (CC / MLO)")
    pathology: str = Field(description="Patología histopatológica de referencia (BENIGN / MALIGNANT)")
    type: str = Field(description="Tipo de alteración (mass / calc)")
    birads_gt: int = Field(default=0, description="Evaluación BI-RADS ground truth asignada por radiólogo")
    density: Optional[int] = Field(default=None, description="Densidad mamaria ACR (1-4)")


class CanonicalVariableItem(BaseModel):
    """Valor, percentil e interpretación contextual de una variable canónica."""
    value: float = Field(description="Valor numérico en el subespacio S-PCA")
    percentile: Optional[float] = Field(default=None, description="Percentil poblacional en el conjunto de prueba")
    interpretation: Optional[str] = Field(default=None, description="Significado clínico estructurado del biomarcador")
    alert: Optional[bool] = Field(default=None, description="Bandera de alerta clínica (ej: infiltración invasiva)")


class CanonicalVariables(BaseModel):
    """Tríada de componentes canónicos fundamentales derivados de la proyección S-PCA."""
    x0: CanonicalVariableItem = Field(description="Componente auxiliar x0 (asociada a la densidad global)")
    x1: CanonicalVariableItem = Field(description="Componente auxiliar x1 (asociada a la severidad)")
    x2: CanonicalVariableItem = Field(description="Componente auxiliar x2 (asociada al margen de la lesión)")


class PredictionDetail(BaseModel):
    """Resultados de predicción neuro-simbólica y consenso diagnóstico."""
    prob_malignant: float = Field(description="Probabilidad de malignidad calculada por el modelo canónico")
    birads_pred: int = Field(description="Categoría BI-RADS sugerida (1 a 5)")
    birads_label: str = Field(description="Etiqueta clínica completa (ej: BI-RADS 2 (Hallazgo benigno))")
    verdict: str = Field(description="Veredicto diagnóstico (BENIGN o MALIGNANT)")
    prob_resnet: Optional[float] = Field(default=None, description="Probabilidad estimada por la ResNet-18 base")
    pred_resnet: Optional[str] = Field(default=None, description="Predicción binaria de la ResNet-18 base")
    prob_kan_symb: Optional[float] = Field(default=None, description="Probabilidad calculada por la cabeza analítica")
    pred_kan_symb: Optional[str] = Field(default=None, description="Predicción binaria de la cabeza analítica")
    consensus_es: Optional[str] = Field(default=None, description="Veredicto en español (Benigno o Maligno)")
    clinical_recommendation: Optional[str] = Field(default=None, description="Conducta asistencial recomendada")
    margin_status: Optional[str] = Field(default=None, description="Estado de márgenes según x2")
    margin_desc: Optional[str] = Field(default=None, description="Descripción textual del patrón de bordes")
    density_status: Optional[str] = Field(default=None, description="Estado de densidad según x1")
    density_desc: Optional[str] = Field(default=None, description="Descripción del patrón parenquimatoso")
    decision_source: Optional[str] = Field(default=None, description="'formula' si la decisión sale de g(u0); 'cnn_fallback' si faltan los artefactos")
    u0: Optional[float] = Field(default=None, description="Coordenada canónica u0 (primera componente de S-PCA64)")
    g_logit: Optional[float] = Field(default=None, description="g(u0) = f_mal(u0) - f_ben(u0)")
    p_sym: Optional[float] = Field(default=None, description="p_sym = sigmoide(g(u0))")
    threshold_logit: Optional[float] = Field(default=None, description="Umbral de Youden (logit) calibrado en validación")
    threshold_prob: Optional[float] = Field(default=None, description="Umbral de Youden expresado como probabilidad")
    prob_net_robust: Optional[float] = Field(default=None, description="Probabilidad de malignidad de la red ResNet-18 robusta completa")
    prob_net_base: Optional[float] = Field(default=None, description="Probabilidad de la ResNet-18 de referencia (baseline)")
    birads_note: Optional[str] = Field(default=None, description="Aviso: la categoría BI-RADS es orientativa de la herramienta, no un riesgo de malignidad")


class ClinicalMetadata(BaseModel):
    """Metadatos clínicos y radiológicos completos del estudio mamográfico."""
    patient_id: str = Field(description="Identificador del paciente")
    laterality: str = Field(default="UNKNOWN", description="Lateralidad de la mama examinada")
    view: str = Field(default="UNKNOWN", description="Proyección mamográfica")
    pathology_gt: str = Field(description="Diagnóstico histológico patológico de referencia")
    assessment_gt: int = Field(default=0, description="Evaluación BI-RADS clínica inicial")
    birads_gt: int = Field(default=0, description="BI-RADS de referencia estandarizado")
    abnormality_type: str = Field(description="Tipo de lesión (mass o calc)")
    lesion_shape: Optional[str] = Field(default=None, description="Forma de la masa (si aplica)")
    lesion_margins: Optional[str] = Field(default=None, description="Márgenes de la masa (si aplica)")
    calc_type: Optional[str] = Field(default=None, description="Morfología de microcalcificaciones (si aplica)")
    calc_distribution: Optional[str] = Field(default=None, description="Distribución de microcalcificaciones")
    density: Optional[int] = Field(default=None, description="Densidad mamaria ACR")
    subtlety: Optional[int] = Field(default=None, description="Grado de sutileza radiológica (1-5)")


class SpatialLocationDetail(BaseModel):
    """Localización espacial, centroides y métricas de segmentación débil WSSS."""
    quadrant_pred: str = Field(description="Región de la imagen del centroide del núcleo focal (no es un cuadrante anatómico)")
    quadrant_gt: Optional[str] = Field(default=None, description="Región de la imagen del centroide de la lesión real (misma regla)")
    is_quadrant_exact: Optional[bool] = Field(default=None, description="Coincidencia exacta entre cuadrantes")
    predicted_centroid: Optional[List[float]] = Field(default=None, description="Coordenadas [y, x] del centroide predicho")
    gt_centroid: Optional[List[float]] = Field(default=None, description="Coordenadas [y, x] del centroide ground truth")
    centroid_dist_px: Optional[float] = Field(default=None, description="Distancia euclidiana en píxeles entre centroides")
    pred_area_px: Optional[float] = Field(default=None, description="Área en píxeles del núcleo focal predicho")
    gt_area_px: Optional[float] = Field(default=None, description="Área en píxeles de la máscara ROI ground truth")
    equiv_diameter_px: Optional[float] = Field(default=None, description="Diámetro equivalente en píxeles")
    compactness: Optional[float] = Field(default=None, description="Compacidad isoperimétrica de la lesión")
    eccentricity: Optional[float] = Field(default=None, description="Excentricidad del tensor de inercia")
    dice: Optional[float] = Field(default=None, description="Coeficiente de similitud Dice frente a ROI ground truth")
    iou: Optional[float] = Field(default=None, description="Intersección sobre Unión (IoU)")
    ebpg_score: Optional[float] = Field(default=None, description="Puntaje Energy-based Pointing Game (EBPG)")


class CaseDetail(BaseModel):
    """Detalle analítico integral de un caso clínico para visualización e informe."""
    index: int = Field(description="Índice secuencial del caso en el dataset")
    patient_id: str = Field(description="Identificador del paciente")
    metadata: ClinicalMetadata = Field(description="Metadatos clínicos y radiológicos")
    prediction: PredictionDetail = Field(description="Predicciones analíticas y veredicto BI-RADS")
    canonical_variables: CanonicalVariables = Field(description="Variables canónicas x0, x1, x2")
    spatial_location: SpatialLocationDetail = Field(description="Localización en cuadrante y métricas espaciales")
    quadrant: str = Field(description="Región de la imagen predicha (acceso directo)")
    pathology: str = Field(description="Patología histológica (acceso directo)")
    abnormality_type: str = Field(description="Tipo de anomalía (acceso directo)")
    x0_canonical: Optional[float] = Field(default=None, description="Valor numérico directo de x0")
    x1_canonical: Optional[float] = Field(default=None, description="Valor numérico directo de x1")
    x2_canonical: Optional[float] = Field(default=None, description="Valor numérico directo de x2")


class RenderResponse(BaseModel):
    """Respuesta que contiene la imagen renderizada en Base64 y el identificador de la capa."""
    image_b64: str = Field(description="Cadena de imagen PNG codificada en Base64 (data:image/png;base64,...)")
    layer: str = Field(description="Identificador de la capa renderizada (clean, cam, wsss, zoom, psi)")


class FormulaEvalRequest(BaseModel):
    """Petición de evaluación de la fórmula canónica en el punto u0."""
    u0: float = Field(default=0.0, description="Coordenada canónica u0 en la que se evalúa g")


class FormulaEvalResponse(BaseModel):
    """Evaluación de la fórmula: f_ben, f_mal, g, p_sym, decisión y categoría de la herramienta."""
    u0: float = Field(description="Punto evaluado (tras recortar a [-10, 10])")
    f_ben: float = Field(description="Pierna benigna f_ben(u0)")
    f_mal: float = Field(description="Pierna maligna f_mal(u0)")
    logit: float = Field(description="g(u0) = f_mal(u0) - f_ben(u0)")
    prob: float = Field(description="p_sym = sigmoide(g(u0)) en [0, 1]")
    birads: int = Field(description="Categoría BI-RADS orientativa de la herramienta (2 a 5, anclada en tau*)")
    birads_label: str = Field(description="Etiqueta BI-RADS completa de la herramienta")
    formula_expr: str = Field(description="Expresión simbólica g = f_mal - f_ben")
    decision: Optional[str] = Field(default=None, description="MALIGNANT si g >= umbral de Youden, BENIGN en otro caso")
    youden_threshold: Optional[float] = Field(default=None, description="Umbral de Youden (logit) calibrado en validación")
    youden_threshold_prob: Optional[float] = Field(default=None, description="Umbral de Youden expresado como probabilidad")


class FormulaSweepResponse(BaseModel):
    """Barrido de la fórmula a lo largo de u0."""
    var_name: str = Field(description="Variable barrida (u0)")
    x_values: List[float] = Field(description="Muestras de u0")
    prob_values: List[float] = Field(description="p_sym en cada muestra")
    logit_values: Optional[List[float]] = Field(default=None, description="g(u0) en cada muestra")
    fben_values: Optional[List[float]] = Field(default=None, description="f_ben(u0) en cada muestra")
    fmal_values: Optional[List[float]] = Field(default=None, description="f_mal(u0) en cada muestra")
    youden_threshold: Optional[float] = Field(default=None, description="Umbral de Youden (logit)")


class DemoCaseSummary(BaseModel):
    """Resumen de un caso de estudio externo de demostración preconfigurado."""
    id: str = Field(description="Identificador único del caso demo")
    title: str = Field(description="Título clínico descriptivo")
    filename: str = Field(description="Nombre del archivo en demo_samples")
    format: str = Field(description="Formato del archivo (DICOM / JPEG)")
    type: str = Field(description="Tipo clínico de hallazgo o alteración")
    description: str = Field(description="Descripción del caso clínico")
    path: Optional[str] = Field(default=None, description="Ruta al archivo en disco")


class ExternalInferenceResponse(BaseModel):
    """Respuesta completa del análisis e inferencia prospectiva de una mamografía externa."""
    # Métricas principales de predicción
    prob_malignant: float = Field(description="Probabilidad de malignidad predicha [0.0, 1.0]")
    birads_pred: int = Field(description="Categorización BI-RADS estimada (1 a 5)")
    birads_label: str = Field(description="Etiqueta clínica BI-RADS completa")
    verdict: str = Field(description="Veredicto diagnóstico (BENIGN o MALIGNANT)")
    consensus_es: Optional[str] = Field(default=None, description="Veredicto diagnóstico en español")
    clinical_recommendation: Optional[str] = Field(default=None, description="Recomendación asistencial sugerida")

    decision_source: Optional[str] = Field(default=None, description="'formula' o 'cnn_fallback'")
    u0: Optional[float] = Field(default=None, description="Coordenada canónica u0")
    g_logit: Optional[float] = Field(default=None, description="g(u0)")
    p_sym: Optional[float] = Field(default=None, description="p_sym = sigmoide(g(u0))")
    threshold_logit: Optional[float] = Field(default=None, description="Umbral de Youden (logit)")
    prob_net_robust: Optional[float] = Field(default=None, description="Probabilidad de la red robusta completa")
    birads_note: Optional[str] = Field(default=None, description="Aviso: la categoría BI-RADS es orientativa de la herramienta")

    # Variables canónicas fundamentales y descriptores S-PCA
    canonical_variables: CanonicalVariables = Field(description="Componentes auxiliares x0, x1, x2")
    x0_canonical: Optional[float] = Field(default=None, description="Valor canónico x0 (Densidad)")
    x1_canonical: Optional[float] = Field(default=None, description="Valor canónico x1 (Severidad)")
    x2_canonical: Optional[float] = Field(default=None, description="Valor canónico x2 (Márgenes / Espiculación)")
    margin_status: Optional[str] = Field(default=None, description="Estado de márgenes según x2")
    margin_desc: Optional[str] = Field(default=None, description="Descripción textual de los contornos")
    density_status: Optional[str] = Field(default=None, description="Estado de densidad según x1")
    density_desc: Optional[str] = Field(default=None, description="Descripción del parénquima mamario")

    # Localización espacial y segmentación débil WSSS
    spatial_location: Optional[SpatialLocationDetail] = Field(default=None, description="Localización espacial y centroide WSSS")
    quadrant: Optional[str] = Field(default=None, description="Región de la imagen predicha (no es un cuadrante anatómico)")

    # Metadatos del estudio externo
    patient_id: Optional[str] = Field(default="EXT_MAMMO", description="Identificador del paciente o estudio")
    laterality: Optional[str] = Field(default="DESCONOCIDA", description="Lateralidad mamaria")
    view: Optional[str] = Field(default="DR", description="Proyección mamográfica")
    file_format: Optional[str] = Field(default="DESCONOCIDO", description="Formato del archivo procesado")
    orig_resolution: Optional[str] = Field(default=None, description="Resolución original de la imagen")
    bit_depth: Optional[str] = Field(default=None, description="Profundidad de bits")
    file_size_kb: Optional[float] = Field(default=None, description="Tamaño del archivo en KB")
    is_external: bool = Field(default=True, description="Indica que se trata de un caso externo")

    # Renderizaciones en Base64
    image_b64: str = Field(description="Imagen mamográfica desnormalizada en escala de grises médico (PNG Base64)")
    cam_b64: Optional[str] = Field(default=None, description="Mapa espectral S-PCA / Grad-CAM en Base64")
    wsss_b64: Optional[str] = Field(default=None, description="Superposición WSSS de núcleo focal y cuadrante en Base64")
    zoom_b64: Optional[str] = Field(default=None, description="Lupa focal ampliada 2.5x en Base64")
    psi1_b64: Optional[str] = Field(default=None, description="Campo continuo psi_1 (Severidad) en Base64")
    psi2_b64: Optional[str] = Field(default=None, description="Campo continuo psi_2 (Márgenes) en Base64")
    plots_b64: Optional[str] = Field(default=None, description="Gráficos duales de enlace analítico y x2 en Base64")

    # Campos adicionales de soporte e informe
    report_text: Optional[str] = Field(default=None, description="Informe clínico preliminar en Markdown")
    birads_suggested: Optional[str] = Field(default=None, description="Alias para birads_label")
    delta_logit: Optional[float] = Field(default=None, description="Margen logit de la cabeza lineal")
    kan_score: Optional[float] = Field(default=None, description="Puntaje analítico")
    kan_prediction: Optional[str] = Field(default=None, description="Predicción analítica")
    img_orig_b64: Optional[str] = Field(default=None, description="Alias de compatibilidad para image_b64")
    img_cam_b64: Optional[str] = Field(default=None, description="Alias de compatibilidad para cam_b64")
    img_wsss_b64: Optional[str] = Field(default=None, description="Alias de compatibilidad para wsss_b64")
    img_zoom_b64: Optional[str] = Field(default=None, description="Alias de compatibilidad para zoom_b64")
    img_plots_b64: Optional[str] = Field(default=None, description="Alias de compatibilidad para plots_b64")
    img_psi1_b64: Optional[str] = Field(default=None, description="Alias de compatibilidad para psi1_b64")
    img_psi2_b64: Optional[str] = Field(default=None, description="Alias de compatibilidad para psi2_b64")
    pred_area_px: Optional[int] = Field(default=None, description="Área en píxeles del núcleo focal")
    equiv_diameter_px: Optional[float] = Field(default=None, description="Diámetro equivalente en píxeles")
    compactness: Optional[float] = Field(default=None, description="Compacidad de la lesión")


class ReportGenerateRequest(BaseModel):
    """Petición para generación de informe clínico estructurado."""
    case_index: int = Field(default=0, description="Índice del caso en la cohorte CBIS-DDSM")
    use_llm: bool = Field(default=False, description="Activar síntesis asistida por LLM local o usar modo determinista")
    custom_evidence: Optional[Dict[str, Any]] = Field(default=None, description="Evidencia factual personalizada o externa")


class ReportGenerateResponse(BaseModel):
    """Respuesta con el texto del informe clínico generado."""
    report_text: str = Field(description="Texto del informe clínico estructurado en formato Markdown")
    engine: str = Field(description="Motor de generación utilizado: 'llm-local' (modelo de lenguaje local, por defecto Phi-3.5) o 'deterministic-grounded'")


class ReportPDFRequest(BaseModel):
    """Petición para compilación y exportación de informe clínico en formato PDF."""
    case_index: int = Field(default=0, description="Índice del caso en la cohorte CBIS-DDSM")
    report_text: str = Field(description="Texto del informe clínico estructurado a incluir en el documento")
    custom_evidence: Optional[Dict[str, Any]] = Field(default=None, description="Evidencia factual personalizada o externa")


class ReportPDFResponse(BaseModel):
    """Respuesta con las referencias al archivo PDF generado."""
    pdf_path: str = Field(description="Ruta absoluta al archivo PDF compilado en disco")
    filename: str = Field(description="Nombre del archivo PDF generado")




