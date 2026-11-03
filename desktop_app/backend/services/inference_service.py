"""
Servicio de inferencia y procesamiento para estudios externos (DICOM / JPG / PNG).
Orquesta la decodificación de imagen médica, estandarización tensorial,
extracción convolucional con ResNet-18 robusto, proyección S-PCA y renderizado WSSS.
"""
import io
import os
import re
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from desktop_app.backend.config import CODIGOS_DIR, DEVICE
from desktop_app.backend.schemas import (
    DemoCaseSummary,
    ExternalInferenceResponse,
    CanonicalVariables,
    CanonicalVariableItem,
    SpatialLocationDetail,
)
from desktop_app.backend.services.cases_service import CasesService, get_cases_service
from desktop_app.backend.services.decision import apply_formula_decision, birads_int, NOTA_CONVENCION
from src.reporting.llm_engine import generate_deterministic_grounded_report, region_imagen


def _resolve_demo_dir() -> Path:
    """Resuelve la ubicación del directorio de casos demo."""
    candidates = [
        Path(__file__).resolve().parents[2] / "demo_samples",
        Path(__file__).resolve().parents[1] / "demo_samples",
        CODIGOS_DIR.parent / "desktop_app" / "demo_samples",
        Path.cwd() / "desktop_app" / "demo_samples",
    ]
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    return candidates[0]


class InferenceService:
    """Gestiona el preprocesamiento, inferencia y generación de artefactos visuales
    para mamografías prospectivas externas y casos de demostración preconfigurados.
    """

    def __init__(self, cases_service: Optional[CasesService] = None):
        self.cases_service = cases_service or get_cases_service()
        self.demo_dir = _resolve_demo_dir()

        # Último caso procesado en memoria para persistencia de reportes
        self._last_external_item: Optional[Dict[str, Any]] = None
        self._last_external_evidence: Optional[Dict[str, Any]] = None

        # Catálogo canónico de los 4 casos demo
        self._demo_definitions: List[Dict[str, Any]] = [
            {
                "id": "demo_dicom_mal",
                "title": "Masa Maligna Espiculada (DICOM Nativo)",
                "filename": "sample_dicom_malignant.dcm",
                "format": "DICOM (.dcm)",
                "type": "Masa Espiculada (Alta Sospecha)",
                "description": "Estudio DICOM digital directo con lesión sospechosa y espiculación estromal.",
            },
            {
                "id": "demo_dicom_ben",
                "title": "Nódulo Benigno Circunscrito (DICOM Nativo)",
                "filename": "sample_dicom_benign.dcm",
                "format": "DICOM (.dcm)",
                "type": "Nódulo Circunscrito (Benigno)",
                "description": "Estudio de tamizaje DICOM con bordes regulares y patrón benigno.",
            },
            {
                "id": "demo_calc_mal",
                "title": "Microcalcificaciones Pleomórficas (JPG)",
                "filename": "sample_calc_malignant.jpg",
                "format": "JPEG (.jpg)",
                "type": "Microcalcificaciones Malignas",
                "description": "Agrupación pleomórfica de microcalcificaciones de alta sospecha.",
            },
            {
                "id": "demo_calc_ben",
                "title": "Calcificaciones Benignas Habituales (JPG)",
                "filename": "sample_calc_benign.jpg",
                "format": "JPEG (.jpg)",
                "type": "Calcificaciones Benignas",
                "description": "Calcificaciones dispersas de morfología típicamente benigna.",
            },
        ]

    def get_demo_cases(self) -> List[DemoCaseSummary]:
        """Devuelve el catálogo de los 4 casos de demostración con su ruta y metadatos."""
        summaries: List[DemoCaseSummary] = []
        for d in self._demo_definitions:
            file_path = self.demo_dir / d["filename"]
            summaries.append(
                DemoCaseSummary(
                    id=d["id"],
                    title=d["title"],
                    filename=d["filename"],
                    format=d["format"],
                    type=d["type"],
                    description=d["description"],
                    path=str(file_path) if file_path.exists() else None,
                )
            )
        return summaries

    def get_demo_case_by_id(self, demo_id: str) -> Optional[Dict[str, Any]]:
        """Busca un caso demo por ID único, índice o nombre de archivo."""
        demo_clean = str(demo_id).strip().lower()

        # Búsqueda por ID directo
        for d in self._demo_definitions:
            if d["id"].lower() == demo_clean:
                return d

        # Búsqueda por nombre de archivo
        for d in self._demo_definitions:
            if d["filename"].lower() == demo_clean:
                return d

        # Búsqueda por índice (ej: "0", "1", "demo_0", "demo_1")
        m = re.search(r"(\d+)", demo_clean)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < len(self._demo_definitions):
                return self._demo_definitions[idx]

        return None

    def process_demo_case(self, demo_id: str) -> ExternalInferenceResponse:
        """Carga y procesa un caso de demostración preconfigurado."""
        demo_info = self.get_demo_case_by_id(demo_id)
        if not demo_info:
            raise KeyError(f"Caso de demostración no encontrado: '{demo_id}'")

        file_path = self.demo_dir / demo_info["filename"]
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo de muestra no encontrado en el disco: {file_path}")

        image_bytes = file_path.read_bytes()
        return self.process_image(image_bytes, filename=demo_info["filename"])

    def process_image(self, image_bytes: bytes, filename: str = "") -> ExternalInferenceResponse:
        """Procesa una mamografía externa (DICOM o imagen digital) a partir de bytes crudos."""
        if not image_bytes or len(image_bytes) < 16:
            raise ValueError("El archivo recibido está vacío o corrupto (menos de 16 bytes).")

        img_disp, img_t, meta = self._load_and_preprocess(image_bytes, filename=filename)

        item = {
            "image_tensor": img_t,
            "mask_np": np.zeros((224, 224), dtype=np.uint8),
            "patient_id": meta.get("patient_id", "EXT_MAMMO"),
            "pathology": "UNKNOWN",
            "abnormality_type": "mass",
            "density": 2,
            "image_pil": meta.pop("_pil_completa", None),
        }

        # Ejecutar extracción analítica completa
        evidence = self.cases_service.extractor.extract_evidence(item, None)
        evidence["patient_id"] = meta.get("patient_id", item["patient_id"])
        evidence["pathology_gt"] = "PROSPECTIVO (SIN HISTOPATOLOGÍA)"
        evidence["assessment_gt"] = 0
        evidence["laterality"] = meta.get("laterality", "DESCONOCIDA")
        evidence["view"] = meta.get("view", "DR")
        apply_formula_decision(evidence)

        # Persistir en memoria para trazabilidad y exportación posterior
        self._last_external_item = item
        self._last_external_evidence = evidence

        # Métricas de predicción: la decisión sale de la fórmula g(u0) (ver decision.py)
        prob_mal = float(evidence.get("decision_p", 0.0))
        verdict_str = str(evidence.get("decision", "BENIGN"))
        consensus_es_str = "Maligno" if verdict_str == "MALIGNANT" else "Benigno"

        birads_label = str(evidence.get("birads_suggested", ""))
        birads_pred_int = birads_int(birads_label)

        # Variables canónicas
        x0_val = float(evidence.get("x0_canonical", 0.0))
        x1_val = float(evidence.get("x1_canonical", 0.0))
        x2_val = float(evidence.get("x2_canonical", 0.0))

        canonical_obj = CanonicalVariables(
            x0=CanonicalVariableItem(
                value=x0_val,
                percentile=50.0,
                interpretation="Componente auxiliar asociada a la densidad global del tejido (interpretación estadística, no una medición).",
                alert=False,
            ),
            x1=CanonicalVariableItem(
                value=x1_val,
                percentile=None,
                interpretation=evidence.get("density_desc", "Componente auxiliar asociada a la severidad (interpretación estadística)."),
                alert=bool(x1_val > 0.25),
            ),
            x2=CanonicalVariableItem(
                value=x2_val,
                percentile=None,
                interpretation=evidence.get("margin_desc", "Componente auxiliar asociada al margen de la lesión (interpretación estadística)."),
                alert=bool(x2_val > 0.35),
            ),
        )

        # Localización espacial
        pred_centroid = evidence.get("predicted_centroid")
        pred_cent_list = [float(pred_centroid[0]), float(pred_centroid[1])] if pred_centroid else None

        spatial_obj = SpatialLocationDetail(
            quadrant_pred=str(evidence.get("predicted_quadrant", "Región Retroareolar / Central")),
            quadrant_gt=None,
            is_quadrant_exact=None,
            predicted_centroid=pred_cent_list,
            gt_centroid=None,
            centroid_dist_px=None,
            pred_area_px=evidence.get("pred_area_px"),
            gt_area_px=None,
            equiv_diameter_px=evidence.get("equiv_diameter_px"),
            compactness=evidence.get("compactness"),
            eccentricity=evidence.get("eccentricity"),
            dice=None,
            iou=None,
            ebpg_score=evidence.get("psi_kan_ebpg"),
        )

        # Renderizado de capas en Base64
        b64_clean = self._render_clean(img_disp)
        b64_cam = self._render_cam(img_disp, evidence)
        b64_wsss = self._render_wsss(img_disp, evidence)
        b64_zoom = self._render_zoom(img_disp, evidence)
        b64_psi1, b64_psi2 = self._render_psi_fields(img_disp, item, evidence)
        b64_plots = self._render_dual_plot(evidence)

        # Generación del informe clínico grounded
        report_text = generate_deterministic_grounded_report(evidence)

        return ExternalInferenceResponse(
            prob_malignant=prob_mal,
            birads_pred=birads_pred_int,
            birads_label=birads_label,
            verdict=verdict_str,
            consensus_es=consensus_es_str,
            clinical_recommendation=evidence.get("clinical_recommendation"),
            decision_source=evidence.get("decision_source"),
            u0=evidence.get("decision_u0"),
            g_logit=evidence.get("decision_g"),
            p_sym=evidence.get("decision_p"),
            threshold_logit=evidence.get("decision_th_logit"),
            prob_net_robust=evidence.get("prob_net_robust"),
            birads_note=NOTA_CONVENCION,
            canonical_variables=canonical_obj,
            x0_canonical=x0_val,
            x1_canonical=x1_val,
            x2_canonical=x2_val,
            margin_status=evidence.get("margin_status"),
            margin_desc=evidence.get("margin_desc"),
            density_status=evidence.get("density_status"),
            density_desc=evidence.get("density_desc"),
            spatial_location=spatial_obj,
            quadrant=spatial_obj.quadrant_pred,
            patient_id=meta.get("patient_id", "EXT_MAMMO"),
            laterality=meta.get("laterality", "DESCONOCIDA"),
            view=meta.get("view", "DR"),
            file_format=meta.get("format", "DESCONOCIDO"),
            orig_resolution=f"{meta['orig_shape'][1]} × {meta['orig_shape'][0]} px" if meta.get("orig_shape") else None,
            bit_depth=f"{meta['bits']} bits" if meta.get("bits") else None,
            file_size_kb=float(meta.get("filesize_kb", 0.0)),
            is_external=True,
            image_b64=b64_clean,
            cam_b64=b64_cam,
            wsss_b64=b64_wsss,
            zoom_b64=b64_zoom,
            psi1_b64=b64_psi1,
            psi2_b64=b64_psi2,
            plots_b64=b64_plots,
            report_text=report_text,
            birads_suggested=birads_label,
            delta_logit=float(evidence.get("delta_logit", 0.0)),
            kan_score=float(evidence.get("kan_score", prob_mal)),
            kan_prediction=str(evidence.get("kan_prediction", verdict_str)),
            img_orig_b64=b64_clean,
            img_cam_b64=b64_cam,
            img_wsss_b64=b64_wsss,
            img_zoom_b64=b64_zoom,
            img_plots_b64=b64_plots,
            img_psi1_b64=b64_psi1,
            img_psi2_b64=b64_psi2,
            pred_area_px=int(evidence.get("pred_area_px", 0)) if evidence.get("pred_area_px") is not None else None,
            equiv_diameter_px=float(evidence.get("equiv_diameter_px", 0.0)) if evidence.get("equiv_diameter_px") is not None else None,
            compactness=float(evidence.get("compactness", 0.0)) if evidence.get("compactness") is not None else None,
        )

    def _load_and_preprocess(self, image_bytes: bytes, filename: str = "") -> Tuple[np.ndarray, torch.Tensor, Dict[str, Any]]:
        """Decodifica un flujo de bytes en formato DICOM o imagen estándar (JPG/PNG)."""
        meta: Dict[str, Any] = {
            "format": "DESCONOCIDO",
            "orig_shape": [0, 0],
            "patient_id": "EXT_MAMMO",
            "laterality": "DESCONOCIDA",
            "view": "DR",
            "bits": 8,
            "filesize_kb": round(len(image_bytes) / 1024, 1),
        }
        arr: Optional[np.ndarray] = None

        # 1. Intentar decodificación DICOM
        try:
            import pydicom
            ds = pydicom.dcmread(io.BytesIO(image_bytes))
            if hasattr(ds, "pixel_array"):
                arr = ds.pixel_array.astype(np.float32)
                meta["format"] = "DICOM Médico (DICOM 3.0)"
                p_id = getattr(ds, "PatientID", None)
                if p_id:
                    meta["patient_id"] = str(p_id).strip()
                else:
                    base_name = os.path.splitext(filename)[0] if filename else "EXT_DICOM"
                    meta["patient_id"] = f"EXT_{base_name}"

                lat = str(getattr(ds, "ImageLaterality", getattr(ds, "Laterality", "DESCONOCIDA"))).upper()
                meta["laterality"] = lat if lat in ["R", "L", "RIGHT", "LEFT"] else "DESCONOCIDA"
                view = str(getattr(ds, "ViewPosition", "DR")).upper()
                meta["view"] = view if view in ["CC", "MLO", "ML", "LM", "DR"] else "DR"
                meta["bits"] = int(getattr(ds, "BitsStored", getattr(ds, "BitsAllocated", 16)))

                photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
                if photometric == "MONOCHROME1":
                    arr = arr.max() - arr
        except Exception:
            arr = None

        # 2. Si no es DICOM, intentar con PIL (JPG, PNG, TIFF, BMP)
        if arr is None:
            try:
                img_pil = Image.open(io.BytesIO(image_bytes))
                img_pil.load()  # Forzar carga completa de bytes para detectar archivos truncados o corruptos
                fmt = (img_pil.format or "JPEG").upper()
                meta["format"] = f"Imagen Digital ({fmt})"
                base_name = os.path.splitext(filename)[0] if filename else "EXT_IMG"
                meta["patient_id"] = f"EXT_{base_name[:14]}"
                img_gray = img_pil.convert("L")
                arr = np.array(img_gray, dtype=np.float32)
                meta["bits"] = 8
            except Exception as e:
                raise ValueError(
                    f"El archivo no es una imagen médica DICOM ni un formato gráfico válido (JPEG/PNG) o está corrupto: {str(e)}"
                )

        if arr is None or arr.size == 0 or arr.ndim < 2:
            raise ValueError("El archivo no contiene una matriz bidimensional de píxeles válida.")

        if arr.ndim > 2:
            arr = arr.mean(axis=-1)

        meta["orig_shape"] = [int(arr.shape[0]), int(arr.shape[1])]

        # Normalización de rango dinámico a [0.0, 1.0]
        a_min, a_max = float(arr.min()), float(arr.max())
        if a_max > a_min:
            arr_01 = (arr - a_min) / (a_max - a_min)
        else:
            arr_01 = np.zeros_like(arr, dtype=np.float32)

        # Imagen completa (8 bits, RGB) para el localizador a 448 px; la decisión usa la de 224
        meta["_pil_completa"] = Image.fromarray((arr_01 * 255.0).astype(np.uint8)).convert("RGB")

        # Redimensionado a resolución canónica (224, 224)
        pil_norm = Image.fromarray((arr_01 * 255.0).astype(np.uint8)).resize(
            (224, 224), Image.Resampling.BILINEAR
        )
        img_disp = np.array(pil_norm, dtype=np.float32) / 255.0

        # Normalización ImageNet para backbone convolucional
        rgb = np.stack([img_disp, img_disp, img_disp], axis=-1)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (rgb - mean) / std
        img_t = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()

        return img_disp, img_t, meta

    @staticmethod
    def _fig_to_b64(fig: Figure) -> str:
        """Serializa una figura Matplotlib directamente a Base64 sin tocar disco."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.04, dpi=160)
        fig.clear()
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def _render_clean(self, img_disp: np.ndarray) -> str:
        """Renderiza la mamografía original en escala de grises médico."""
        fig = Figure(figsize=(4.2, 4.2), dpi=160)
        ax = fig.subplots()
        ax.imshow(img_disp, cmap="gray")
        ax.axis("off")
        return self._fig_to_b64(fig)

    def _render_cam(self, img_disp: np.ndarray, evidence: Dict[str, Any]) -> str:
        """Renderiza la saliencia espectral continua S-PCA / Grad-CAM."""
        fig = Figure(figsize=(4.2, 4.2), dpi=160)
        ax = fig.subplots()
        ax.imshow(img_disp, cmap="gray")

        cam_map = evidence.get("cam_map")
        tissue_mask = evidence.get("tissue_mask")
        centroid = evidence.get("predicted_centroid")

        if cam_map is not None:
            cam_show = cam_map * tissue_mask if tissue_mask is not None else cam_map
            alpha_map = np.clip((cam_show - 0.04) / 0.96, 0.0, 0.65)
            if tissue_mask is not None:
                alpha_map = alpha_map * (tissue_mask > 0.5)
            cmap = plt.cm.jet
            cam_rgba = cmap(cam_show)
            cam_rgba[..., 3] = alpha_map
            ax.imshow(cam_rgba)

            if centroid is not None:
                cy, cx = centroid
                ax.plot(cx, cy, "w*", markersize=13, markeredgecolor="#EA580C", markeredgewidth=1.2)

        ax.axis("off")
        return self._fig_to_b64(fig)

    def _render_wsss(self, img_disp: np.ndarray, evidence: Dict[str, Any]) -> str:
        """Renderiza el contorno focal débilmente supervisado (tau >= 60%) y el cuadrante."""
        fig = Figure(figsize=(4.2, 4.2), dpi=160)
        ax = fig.subplots()
        ax.imshow(img_disp, cmap="gray")

        mask_pred = evidence.get("mask_pred")
        tissue_mask = evidence.get("tissue_mask")
        centroid = evidence.get("predicted_centroid")
        quadrant_str = region_imagen(evidence.get("predicted_quadrant", "")) if evidence.get("predicted_quadrant") else ""

        if tissue_mask is not None and tissue_mask.sum() > 0:
            ax.contour(tissue_mask, levels=[0.5], colors=['#3B82F6'], linewidths=1.2, alpha=0.6, linestyles='dashed')

        if mask_pred is not None and mask_pred.sum() > 0:
            ax.contour(mask_pred, levels=[0.5], colors=['#EA580C'], linewidths=2.0)

        if centroid is not None:
            cy, cx = centroid
            ax.plot(cx, cy, "+", color="#F97316", markersize=15, markeredgewidth=2.2)

        if quadrant_str:
            ax.text(0.04, 0.93, quadrant_str, transform=ax.transAxes,
                    fontsize=8.2, color="#FFFFFF", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#9A3412", ec="#EA580C", alpha=0.92))

        ax.axis("off")
        return self._fig_to_b64(fig)

    def _render_zoom(self, img_disp: np.ndarray, evidence: Dict[str, Any], zoom_factor: float = 2.5) -> str:
        """Renderiza una vista ampliada (lupa 2.5x) con realce adaptativo de contraste."""
        centroid = evidence.get("predicted_centroid")
        if centroid is None:
            centroid = (img_disp.shape[0] / 2.0, img_disp.shape[1] / 2.0)

        cy, cx = centroid
        H, W = img_disp.shape
        R = int(H / (zoom_factor * 2))
        y0 = int(max(0, min(H - 2 * R, cy - R)))
        y1 = int(min(H, y0 + 2 * R))
        x0 = int(max(0, min(W - 2 * R, cx - R)))
        x1 = int(min(W, x0 + 2 * R))

        crop = img_disp[y0:y1, x0:x1]
        p1, p99 = np.percentile(crop, 1), np.percentile(crop, 99)
        crop_norm = np.clip((crop - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)

        fig = Figure(figsize=(4.2, 4.2), dpi=160)
        ax = fig.subplots()
        ax.imshow(crop_norm, cmap="gray")

        mask_pred = evidence.get("mask_pred")
        if mask_pred is not None and mask_pred.sum() > 0:
            crop_mask = mask_pred[y0:y1, x0:x1]
            if crop_mask.sum() > 0:
                ax.contour(crop_mask, levels=[0.5], colors=['#EA580C'], linewidths=1.8, alpha=0.9)

        crop_cy = cy - y0
        crop_cx = cx - x0
        ax.plot(crop_cx, crop_cy, "+", color="#F97316", markersize=14, markeredgewidth=2.2)
        ax.text(0.04, 0.93, f"Lupa Focal {zoom_factor:.1f}x", transform=ax.transAxes,
                fontsize=8.2, color="#FFFFFF", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="#C2410C", ec="#EA580C", alpha=0.92))

        ax.axis("off")
        return self._fig_to_b64(fig)

    def _render_psi_fields(self, img_disp: np.ndarray, item: Dict[str, Any], evidence: Dict[str, Any]) -> Tuple[str, str]:
        """Renderiza los campos continuos psi_1 (Severidad) y psi_2 (Márgenes)."""
        tissue_mask = evidence.get("tissue_mask")
        psi_fields = self.cases_service.extractor.compute_psi_fields(
            item["image_tensor"], tissue_mask
        )

        def _render_one(psi_field: Any, title: str) -> str:
            fig = Figure(figsize=(4.2, 4.2), dpi=160)
            ax = fig.subplots()
            ax.imshow(img_disp, cmap="gray")
            if psi_field is not None:
                field = np.asarray(psi_field, dtype=np.float32)
                if tissue_mask is not None:
                    tm = np.asarray(tissue_mask, dtype=np.float32)
                    if tm.shape == field.shape:
                        field = field * (tm > 0.5)
                mu = float(field.mean())
                sd = float(field.std())
                if sd > 1e-8:
                    z = np.clip((field - mu) / (2.0 * sd), -1.0, 1.0)
                    alpha = np.clip(np.abs(z) * 0.78, 0.0, 0.78)
                    rgba = plt.cm.RdBu_r(z)
                    rgba[..., 3] = alpha
                    ax.imshow(rgba)
            ax.text(0.04, 0.93, title, transform=ax.transAxes,
                    fontsize=8.2, color="#FFFFFF", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#1E3A8A", ec="#3B82F6", alpha=0.92))
            ax.axis("off")
            return self._fig_to_b64(fig)

        b64_psi1 = _render_one(
            psi_fields.get("psi1_tissue", psi_fields.get("psi1")),
            r"$\psi_1(u,v)$ — Severidad ($x_1$)"
        )
        b64_psi2 = _render_one(
            psi_fields.get("psi2_tissue", psi_fields.get("psi2")),
            r"$\psi_2(u,v)$ — Márgenes ($x_2$)"
        )
        return b64_psi1, b64_psi2

    def _render_dual_plot(self, evidence: Dict[str, Any]) -> str:
        """Panel doble: (1) la fórmula g(u0) con el punto del caso y el umbral de Youden;
        (2) probabilidades de la fórmula y de la red completa frente al umbral."""
        from desktop_app.backend.services.formula_service import get_formula_service
        ev = get_formula_service().evaluator
        u0 = evidence.get("decision_u0")
        g = evidence.get("decision_g")
        th = evidence.get("decision_th_logit", ev.youden_threshold)
        p_sym = evidence.get("decision_p")
        p_net = evidence.get("prob_net_robust")

        fig = Figure(figsize=(10.5, 2.9), dpi=160)
        (ax_g, ax_p) = fig.subplots(1, 2)
        fig.subplots_adjust(wspace=0.28, bottom=0.2)

        xs = np.linspace(-4.0, 4.0, 200)
        gs = np.array([ev.eval_g(x)[3] for x in xs])
        ax_g.plot(xs, gs, color="#EA580C", linewidth=2.4, label=r"$g(u_0)=f_{mal}-f_{ben}$")
        if th is not None:
            ax_g.axhline(th, color="#16A34A", linestyle="--", linewidth=1.2, label=f"Umbral de Youden ({th:+.3f})")
        if u0 is not None and g is not None:
            ax_g.plot(u0, g, marker="o", color="#C2410C", markersize=8, markeredgecolor="#FFFFFF",
                      markeredgewidth=1.5, label=f"Caso: u₀={u0:+.2f}, g={g:+.2f}")
        ax_g.set_xlabel(r"$u_0$", fontsize=8.2, fontweight="bold", color="#334155")
        ax_g.set_ylabel(r"$g(u_0)$", fontsize=8.2, fontweight="bold", color="#334155")
        ax_g.set_title("1. La fórmula evaluada en el caso", fontsize=9.2, fontweight="bold", color="#9A3412", pad=6)
        ax_g.grid(True, linestyle="--", alpha=0.45, color="#CBD5E1")
        ax_g.legend(fontsize=7.2, loc="upper right")
        ax_g.set_facecolor("#FAFAFA")

        labels, vals, cols = [], [], []
        if p_sym is not None:
            labels.append("Fórmula\np_sym"); vals.append(float(p_sym)); cols.append("#EA580C")
        if p_net is not None:
            labels.append("Red robusta\ncompleta"); vals.append(float(p_net)); cols.append("#64748B")
        ax_p.bar(range(len(vals)), vals, color=cols, width=0.5)
        for k, v in enumerate(vals):
            ax_p.text(k, v + 0.02, f"{v*100:.1f}%", ha="center", fontsize=8, fontweight="bold", color="#334155")
        thp = evidence.get("decision_th_prob")
        if thp is not None:
            ax_p.axhline(thp, color="#16A34A", linestyle="--", linewidth=1.2,
                         label=f"Umbral de la fórmula ({thp*100:.1f}%)")
            ax_p.legend(fontsize=7.2, loc="upper right")
        ax_p.set_xticks(range(len(labels)))
        ax_p.set_xticklabels(labels, fontsize=8)
        ax_p.set_ylim(0, 1.08)
        ax_p.set_ylabel("Probabilidad de malignidad", fontsize=8.2, fontweight="bold", color="#334155")
        ax_p.set_title("2. Fórmula y red completa", fontsize=9.2, fontweight="bold", color="#9A3412", pad=6)
        ax_p.grid(True, axis="y", linestyle="--", alpha=0.45, color="#CBD5E1")
        ax_p.set_facecolor("#FAFAFA")

        return self._fig_to_b64(fig)


# Instancia Singleton perezosa para el ciclo de vida del backend
_inference_service_instance: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    """Provee la instancia compartida de InferenceService."""
    global _inference_service_instance
    if _inference_service_instance is None:
        _inference_service_instance = InferenceService()
    return _inference_service_instance
