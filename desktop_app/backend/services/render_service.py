"""
render_service.py: Servicio de renderizado analítico en memoria para mamografías CBIS-DDSM.
Genera figuras científicas en modo Agg (headless) y las serializa directamente a Base64 sin
escribir archivos temporales en disco.

Capas analíticas soportadas:
- clean: imagen mamográfica desnormalizada en escala de grises médico.
- cam: mapa continuo de activación espectral S-PCA / Grad-CAM con centroide focal.
- wsss: segmentación débil focal (tau >= 60%) superpuesta a la máscara tisular Omega_breast.
- zoom: lupa focal de aumento 2.5x con realce adaptativo de contraste local.
- psi: campo continuo de densidad simbólica psi_2(u, v) para análisis de espiculación.
"""
import io
import base64
from typing import Optional, Dict, Any, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from desktop_app.backend.services.cases_service import CasesService, get_cases_service
from src.reporting.llm_engine import region_imagen


VALID_LAYERS = {
    "clean": "clean",
    "original": "clean",
    "raw": "clean",
    "cam": "cam",
    "gradcam": "cam",
    "saliencia": "cam",
    "wsss": "wsss",
    "focal": "wsss",
    "core": "wsss",
    "zoom": "zoom",
    "lupa": "zoom",
    "psi": "psi",
    "psi2": "psi",
    "spca": "psi",
    "psikan": "psikan",
    "formula": "psikan",
    "mapa": "psikan",
}


class RenderService:
    """Genera visualizaciones de capas analíticas y las codifica en cadenas Base64 PNG."""

    def __init__(self, cases_service: Optional[CasesService] = None):
        self.cases_service = cases_service or get_cases_service()
        self._cache: Dict[Tuple[int, str], str] = {}

    @staticmethod
    def _fig_to_b64(fig: Figure) -> str:
        """Convierte una figura Matplotlib en string base64 listo para HTML <img>."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.04, dpi=160)
        fig.clear()
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def render_layer(self, case_index: int, layer: str = "clean") -> str:
        """Renderiza una capa para el caso especificado y devuelve la cadena Base64 PNG."""
        norm_layer = layer.strip().lower()
        if norm_layer not in VALID_LAYERS:
            raise ValueError(
                f"Capa no reconocida: '{layer}'. Capas válidas: clean, cam, wsss, zoom, psi, psikan"
            )
        canonical_layer = VALID_LAYERS[norm_layer]

        cache_key = (case_index, canonical_layer)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Validar índice y extraer insumos del dataset y evidencia
        raw_item = self.cases_service.get_raw_item(case_index)
        evidence = self.cases_service.extract_evidence(case_index)

        # Imagen normalizada [0, 1] para visualización en escala de grises médico
        img_np = raw_item["image_tensor"].cpu().numpy()
        if img_np.ndim == 3:
            img_np = img_np[0] if img_np.shape[0] in [1, 3] else img_np[:, :, 0]
        img_disp = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

        if canonical_layer == "clean":
            b64 = self._render_clean(img_disp)
        elif canonical_layer == "cam":
            b64 = self._render_cam(img_disp, evidence)
        elif canonical_layer == "wsss":
            b64 = self._render_wsss(img_disp, evidence)
        elif canonical_layer == "zoom":
            b64 = self._render_zoom(img_disp, evidence)
        elif canonical_layer == "psi":
            b64 = self._render_psi(img_disp, raw_item, evidence)
        elif canonical_layer == "psikan":
            b64 = self._render_psikan(img_disp, evidence)
        else:
            raise ValueError(f"Capa no implementada: {canonical_layer}")

        self._cache[cache_key] = b64
        return b64

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

        # Contorno tisular Omega_breast
        if tissue_mask is not None and tissue_mask.sum() > 0:
            ax.contour(tissue_mask, levels=[0.5], colors=['#3B82F6'], linewidths=1.2, alpha=0.6, linestyles='dashed')

        # Contorno focal predicho
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

    def _render_psikan(self, img_disp: np.ndarray, evidence: Dict[str, Any]) -> str:
        """Mapa espacial de la fórmula: psi_KAN(u, v) = sum_c (dg/dgap_c) A_c(u, v)."""
        fig = Figure(figsize=(4.2, 4.2), dpi=160)
        ax = fig.subplots()
        ax.imshow(img_disp, cmap="gray")
        field = evidence.get("psi_kan_map")
        tissue_mask = evidence.get("tissue_mask")
        peak = evidence.get("psi_kan_peak")
        if field is not None:
            f = np.asarray(field, dtype=np.float32)
            if tissue_mask is not None and np.asarray(tissue_mask).shape == f.shape:
                f = f * (np.asarray(tissue_mask) > 0.5)
            mx = float(f.max())
            if mx > 1e-8:
                z = f / mx
                alpha = np.clip((z - 0.04) / 0.96, 0.0, 0.65)
                rgba = plt.cm.jet(z)
                rgba[..., 3] = alpha
                ax.imshow(rgba)
            if peak is not None:
                ax.plot(peak[1], peak[0], "w*", markersize=13, markeredgecolor="#EA580C", markeredgewidth=1.2)
        ax.text(0.04, 0.93, "Mapa de la fórmula ψ_KAN", transform=ax.transAxes,
                fontsize=8.2, color="#FFFFFF", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="#9A3412", ec="#EA580C", alpha=0.92))
        ax.axis("off")
        return self._fig_to_b64(fig)

    def _render_psi(self, img_disp: np.ndarray, raw_item: Dict[str, Any], evidence: Dict[str, Any]) -> str:
        """Renderiza el campo continuo de densidad simbólica psi_2(u, v) para márgenes/espiculación."""
        tissue_mask = evidence.get("tissue_mask")
        psi_fields = self.cases_service.extractor.compute_psi_fields(
            raw_item["image_tensor"], tissue_mask
        )
        psi = psi_fields.get("psi2_tissue", psi_fields.get("psi2"))

        fig = Figure(figsize=(4.2, 4.2), dpi=160)
        ax = fig.subplots()
        ax.imshow(img_disp, cmap="gray")

        stats = {}
        if psi is not None:
            field = np.asarray(psi, dtype=np.float32)
            if tissue_mask is not None:
                tm = np.asarray(tissue_mask, dtype=np.float32)
                if tm.shape == field.shape:
                    field = field * (tm > 0.5)
            mu = float(field.mean())
            sd = float(field.std())
            stats = {"psi_mean": mu, "psi_std": sd}
            if sd > 1e-8:
                z = np.clip((field - mu) / (2.0 * sd), -1.0, 1.0)
                # Rojo = sobre la media (infiltración), Azul = bajo la media; centro transparente
                alpha = np.clip(np.abs(z) * 0.78, 0.0, 0.78)
                rgba = plt.cm.RdBu_r(z)
                rgba[..., 3] = alpha
                ax.imshow(rgba)

        ax.text(0.04, 0.93, r"$\psi_2(u,v)$ — componente auxiliar $x_2$", transform=ax.transAxes,
                fontsize=8.2, color="#FFFFFF", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="#1E3A8A", ec="#3B82F6", alpha=0.92))
        if stats:
            ax.text(0.96, 0.07, r"$\overline{\psi}=%.2f$, $\sigma_\psi=%.2f$" % (
                stats["psi_mean"], stats["psi_std"]), transform=ax.transAxes,
                fontsize=7.4, color="#E2E8F0", ha="right",
                bbox=dict(boxstyle="round,pad=0.22", fc="#0F172A", ec="#3B82F6", alpha=0.85))

        ax.axis("off")
        return self._fig_to_b64(fig)


# Instancia Singleton perezosa para el ciclo de vida del backend
_render_service_instance: Optional[RenderService] = None


def get_render_service() -> RenderService:
    """Provee la instancia compartida de RenderService."""
    global _render_service_instance
    if _render_service_instance is None:
        _render_service_instance = RenderService()
    return _render_service_instance
