"""
pdf_report_generator.py: Generador de Informes Radiológicos Formales en PDF
utilizando ReportLab. Integra la evidencia matemática Neuro-Simbólica (x1, F(x1)),
la segmentación espacial WSSS, el mosaico de imágenes y el texto clínico estructurado BI-RADS.
"""
import os
import io
import re
import time
import numpy as np
import matplotlib.pyplot as plt

try:
    from src.reporting.llm_engine import region_imagen as _region_imagen
except Exception:  # pragma: no cover
    def _region_imagen(etiqueta, mayuscula=True):
        return str(etiqueta)
try:
    from src.reporting.formula_eval import g_function as _g_function
except Exception:  # pragma: no cover
    _g_function = None

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)

# --------------------------------------------------------------------------
# Fuentes
# Las base-14 (Helvetica) no cubren los subindices (x1, x2), la letra griega
# sigma ni los superindices de exponentes, por lo que se registrran las fuentes
# DejaVu que acompanan a matplotlib. Si no estan disponibles se conserva
# Helvetica como respaldo.
# --------------------------------------------------------------------------
_FONT_CANDIDATES = {
    "PrometeoSans": "DejaVuSans.ttf",
    "PrometeoSans-Bold": "DejaVuSans-Bold.ttf",
    "PrometeoSans-Italic": "DejaVuSans-Oblique.ttf",
    "PrometeoSans-BoldItalic": "DejaVuSans-BoldOblique.ttf",
}
_BASE14_FALLBACK = {
    "PrometeoSans": "Helvetica",
    "PrometeoSans-Bold": "Helvetica-Bold",
    "PrometeoSans-Italic": "Helvetica-Oblique",
    "PrometeoSans-BoldItalic": "Helvetica-BoldOblique",
}
_REGISTERED_FONTS = None


def _register_fonts():
    """Registra DejaVu Sans (cubre subindices, griegas y superindices)."""
    global _REGISTERED_FONTS
    if _REGISTERED_FONTS is not None:
        return _REGISTERED_FONTS

    try:
        import matplotlib
        tdir = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    except Exception:
        tdir = ""

    found = {}
    for alias, fname in _FONT_CANDIDATES.items():
        path = os.path.join(tdir, fname) if tdir else ""
        if path and os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(alias, path))
                found[alias] = alias
                continue
            except Exception:
                pass
        found[alias] = _BASE14_FALLBACK[alias]

    if "PrometeoSans-Bold" in found and "PrometeoSans" in found:
        try:
            pdfmetrics.registerFontFamily(
                "PrometeoSans", normal="PrometeoSans", bold="PrometeoSans-Bold",
                italic="PrometeoSans-Italic", boldItalic="PrometeoSans-BoldItalic")
        except Exception:
            pass

    _REGISTERED_FONTS = found
    return found


class ClinicalPDFReportGenerator:
    """Generador hospitalario de informes mamográficos en PDF con respaldo neuro-simbólico."""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = os.environ.get("TESIS_PDF_DIR") or os.path.join(
                os.environ.get("TESIS_CODIGOS", os.getcwd()), "reports", "pdf")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Define estilos tipográficos limpios y profesionales para el reporte médico."""
        F = _register_fonts()
        self.styles.add(ParagraphStyle(
            name="HospitalHeader",
            fontName=F.get("PrometeoSans-Bold"),
            fontSize=15,
            leading=18,
            textColor=colors.HexColor("#1A365D"),
            alignment=1  # Centrado
        ))
        self.styles.add(ParagraphStyle(
            name="HospitalSubHeader",
            fontName=F.get("PrometeoSans"),
            fontSize=9.5,
            leading=12,
            textColor=colors.HexColor("#4A5568"),
            alignment=1
        ))
        self.styles.add(ParagraphStyle(
            name="SectionHeading",
            fontName=F.get("PrometeoSans-Bold"),
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#2B6CB0"),
            spaceBefore=6,
            spaceAfter=4
        ))
        self.styles.add(ParagraphStyle(
            name="ReportBody",
            fontName=F.get("PrometeoSans"),
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#2D3748")
        ))
        self.styles.add(ParagraphStyle(
            name="TableLabel",
            fontName=F.get("PrometeoSans-Bold"),
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#1A202C")
        ))
        self.styles.add(ParagraphStyle(
            name="TableValue",
            fontName=F.get("PrometeoSans"),
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#2D3748")
        ))
        self.styles.add(ParagraphStyle(
            name="Footnote",
            fontName=F.get("PrometeoSans-Italic"),
            fontSize=7.5,
            leading=9.5,
            textColor=colors.HexColor("#718096"),
            alignment=1
        ))

    def _create_visual_triptych(self, item: dict, evidence: dict) -> io.BytesIO:
        """Genera el panel visual cuádruple de alta resolución:
        1. Mamografía Panorámica DR (con contorno de referencia solo si existe)
        2. Mapa de la fórmula (psi) en el tejido mamario
        3. Delimitación Focal WSSS (Núcleo Amarillo) y región de la imagen
        4. Lupa Diagnóstica Focal (Zoom 2.8x del núcleo tumoral)
        """
        img_np = item["image_tensor"].cpu().numpy()
        if img_np.ndim == 3:
            if img_np.shape[0] in [1, 3]:
                img_np = img_np[0]
            elif img_np.shape[-1] in [1, 3]:
                img_np = img_np[:, :, 0]
        img_disp = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

        gt_mask = item.get("mask_np", None)
        cam_map = evidence.get("cam_map", None)
        tissue_mask = evidence.get("tissue_mask", None)
        cy, cx = evidence.get("predicted_centroid", (112, 112))
        mask_pred = evidence.get("mask_pred", None)

        fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.9), dpi=200)
        plt.subplots_adjust(wspace=0.14)

        # 1. Mamografía Panorámica DR
        axes[0].imshow(img_disp, cmap="gray")
        if gt_mask is not None and gt_mask.sum() > 0:
            axes[0].contour(gt_mask, levels=[0.5], colors=['#F1C40F'], linewidths=1.5)
        axes[0].set_title("1. Mamografía", fontsize=9.0, fontweight="bold")
        axes[0].axis("off")

        # 2. Saliencia Espectral en Parénquima
        axes[1].imshow(img_disp, cmap="gray")
        if cam_map is not None:
            cam_show = cam_map * tissue_mask if tissue_mask is not None else cam_map
            alpha_map = np.clip((cam_show - 0.04) / 0.96, 0.0, 0.65)
            if tissue_mask is not None:
                alpha_map = alpha_map * (tissue_mask > 0.5)
            cam_rgba = plt.cm.jet(cam_show)
            cam_rgba[..., 3] = alpha_map
            axes[1].imshow(cam_rgba)
            axes[1].plot(cx, cy, "w*", markersize=9, markeredgecolor="black")
            axes[1].set_title("2. Mapa de la fórmula (tejido)", fontsize=9.0, fontweight="bold")
        else:
            axes[1].set_title("2. Atención Convolucional", fontsize=9.0, fontweight="bold")
        axes[1].axis("off")

        # 3. Delimitación Focal WSSS y región de la imagen
        axes[2].imshow(img_disp, cmap="gray")
        if mask_pred is not None and mask_pred.sum() > 0:
            axes[2].contour(mask_pred, levels=[0.5], colors=['#00FFCC'], linewidths=1.8)
        axes[2].plot(cx, cy, "r+", markersize=11, markeredgewidth=2.0)
        quad_txt = _region_imagen(evidence.get("predicted_quadrant", "N/A"))
        axes[2].text(0.04, 0.92, f"{quad_txt}", transform=axes[2].transAxes,
                     fontsize=8.0, color="white", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.22", fc="#1A365D", ec="none", alpha=0.9))
        axes[2].set_title("3. Núcleo focal y región", fontsize=9.0, fontweight="bold")
        axes[2].axis("off")

        # 4. Lupa Diagnóstica Focal (Zoom 2.8x)
        H, W = img_disp.shape
        R = int(H / 5.6)
        y0 = int(max(0, min(H - 2*R, cy - R)))
        y1 = int(min(H, y0 + 2*R))
        x0 = int(max(0, min(W - 2*R, cx - R)))
        x1 = int(min(W, x0 + 2*R))
        crop = img_disp[y0:y1, x0:x1]
        p1, p99 = np.percentile(crop, 1), np.percentile(crop, 99)
        crop_norm = np.clip((crop - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)
        axes[3].imshow(crop_norm, cmap="gray")
        if mask_pred is not None and mask_pred.sum() > 0:
            crop_mask = mask_pred[y0:y1, x0:x1]
            if crop_mask.sum() > 0:
                axes[3].contour(crop_mask, levels=[0.5], colors=['#00FFCC'], linewidths=1.4, alpha=0.85)
        crop_cy = cy - y0
        crop_cx = cx - x0
        axes[3].plot(crop_cx, crop_cy, "r+", markersize=11, markeredgewidth=1.8)
        axes[3].text(0.04, 0.92, "Zoom 2.8x", transform=axes[3].transAxes,
                     fontsize=8.0, color="white", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.22", fc="#0F172A", ec="#38BDF8", alpha=0.9))
        axes[3].set_title("4. Lupa Focal (Zoom)", fontsize=9.0, fontweight="bold")
        axes[3].axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)
        buf.seek(0)
        return buf

    def _create_kan_curve_plot(self, evidence: dict) -> io.BytesIO:
        """Panel doble: (1) la fórmula g(u0) con el punto de la paciente y el umbral de Youden;
        (2) puntaje de la fórmula frente al de la red completa (ninguno es una probabilidad calibrada)."""
        u0 = evidence.get("decision_u0", evidence.get("kan64_z64_0"))
        g_val = evidence.get("decision_g", evidence.get("kan64_g"))
        th = evidence.get("decision_th_logit", evidence.get("kan64_th"))
        p_sym = evidence.get("decision_p")
        p_net = evidence.get("prob_net_robust", evidence.get("prob_cnn"))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 2.5), dpi=180)
        plt.subplots_adjust(wspace=0.28)

        gf = _g_function() if _g_function is not None else None
        if gf is not None:
            g_fn, th_json = gf
            xs = np.linspace(-4.0, 4.0, 200)
            ax1.plot(xs, [g_fn(x) for x in xs], color="#2B6CB0", linewidth=2.2,
                     label=r"$g(u_0)=f_{mal}(u_0)-f_{ben}(u_0)$")
            if th is None:
                th = th_json
        if th is not None:
            ax1.axhline(th, color="#38A169", linestyle="--", linewidth=1.0,
                        label=f"Umbral de Youden ({float(th):+.3f})")
        if u0 is not None and g_val is not None:
            ax1.plot(float(u0), float(g_val), marker="o", color="#E53E3E", markersize=8, markeredgecolor="black",
                     label=f"Paciente: $u_0={float(u0):+.2f}$, $g={float(g_val):+.2f}$")
        ax1.set_xlabel(r"Coordenada canónica $u_0$", fontsize=8.0, fontweight="bold")
        ax1.set_ylabel(r"$g(u_0)$", fontsize=8.0, fontweight="bold")
        ax1.set_title("1. La fórmula evaluada en la paciente", fontsize=9.0, fontweight="bold", pad=6)
        ax1.grid(True, linestyle="--", alpha=0.45)
        ax1.legend(fontsize=7.0, loc="upper right")

        labels, vals, cols = [], [], []
        if p_sym is not None:
            labels.append("Fórmula p_sym"); vals.append(float(p_sym)); cols.append("#2B6CB0")
        if p_net is not None:
            labels.append("Red completa"); vals.append(float(p_net)); cols.append("#718096")
        ax2.bar(range(len(vals)), vals, color=cols, width=0.5)
        for k, v in enumerate(vals):
            ax2.text(k, v + 0.02, f"{v * 100:.1f}%", ha="center", fontsize=8, fontweight="bold")
        thp = evidence.get("decision_th_prob")
        if thp is not None:
            ax2.axhline(float(thp), color="#38A169", linestyle="--", linewidth=1.0,
                        label=f"Umbral de la fórmula ({float(thp) * 100:.1f}%)")
            ax2.legend(fontsize=7.0, loc="upper right")
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, fontsize=8)
        ax2.set_ylim(0.0, 1.08)
        ax2.set_ylabel("Puntaje (no calibrado)", fontsize=8.0, fontweight="bold")
        ax2.set_title("2. Fórmula y red completa", fontsize=9.0, fontweight="bold", pad=6)
        ax2.grid(True, axis="y", linestyle="--", alpha=0.45)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)
        buf.seek(0)
        return buf

    def generate_pdf(self, item: dict, evidence: dict, report_text: str, filename: str = None) -> str:
        """Construye y compila el informe completo en PDF de 2 páginas."""
        patient_id = evidence.get("patient_id", "P_UNKNOWN")
        if filename is None:
            filename = f"Informe_Radiologico_{patient_id}.pdf"
        pdf_path = os.path.join(self.output_dir, filename)

        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        story = []

        # ==================== PÁGINA 1: INFORME CLÍNICO FORMAL ====================

        # 1. Encabezado Institucional
        story.append(Paragraph("UNIVERSIDAD EL BOSQUE — PROTOTIPO ACADÉMICO (NO ES UN DISPOSITIVO MÉDICO)", self.styles["HospitalHeader"]))
        story.append(Paragraph("SISTEMA NEURO-SIMBÓLICO PROMETEO | BORRADOR PARA REVISIÓN POR UN RADIÓLOGO", self.styles["HospitalSubHeader"]))
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A365D"), spaceAfter=10))

        # 2. Ficha del Paciente y Estudio
        today_str = time.strftime("%d/%m/%Y")
        patient_table_data = [
            [
                Paragraph("<b>ID PACIENTE:</b>", self.styles["TableLabel"]),
                Paragraph(str(patient_id), self.styles["TableValue"]),
                Paragraph("<b>FECHA DEL INFORME:</b>", self.styles["TableLabel"]),
                Paragraph(today_str, self.styles["TableValue"]),
            ],
            [
                Paragraph("<b>MODALIDAD:</b>", self.styles["TableLabel"]),
                Paragraph("Mamografía de película digitalizada (CBIS-DDSM)" if str(patient_id).startswith("P_")
                          else "Mamografía (imagen cargada en el estudio externo)", self.styles["TableValue"]),
                Paragraph("<b>LATERALIDAD / VISTA:</b>", self.styles["TableLabel"]),
                Paragraph(f"Mama {dict(LEFT='izquierda', RIGHT='derecha').get(str(evidence.get('laterality', '')).upper(), evidence.get('laterality', 'N/A'))}"
                          f" — Proyección {evidence.get('view', 'N/A')}", self.styles["TableValue"]),
            ]
        ]
        t_patient = Table(patient_table_data, colWidths=[110, 160, 130, 140])
        t_patient.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#CBD5E0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t_patient)
        story.append(Spacer(1, 10))

        # 3. Resumen de Evidencia Matemática y Espacial (Grounding)
        x0_val = evidence.get("x0_canonical", 0.0)
        x1_val = evidence.get("x1_canonical", 0.0)
        x2_val = evidence.get("x2_canonical", 0.0)
        u0_val = evidence.get("decision_u0", evidence.get("kan64_z64_0"))
        g_val = evidence.get("decision_g", evidence.get("kan64_g"))
        kan_prob = float(evidence.get("decision_p", evidence.get("kan_score", 0.0)) or 0.0)
        is_malignant = (evidence.get("decision") == "MALIGNANT") if evidence.get("decision") else (kan_prob >= 0.50)
        birads_sug = evidence.get("birads_suggested", "N/A")
        quad_pred = _region_imagen(evidence.get("predicted_quadrant", "N/A"))
        area_px = evidence.get("pred_area_px", 0)
        compact_val = evidence.get("compactness", 0.0)
        d_eq = evidence.get("equiv_diameter_px", 0.0)

        # Color de alerta según severidad
        color_badge = "#C53030" if is_malignant else "#276749"
        decision_label = "g(u₀) sobre el umbral: SOSPECHA" if is_malignant else "g(u₀) bajo el umbral: SIN SOSPECHA"
        try:
            u0_txt = f"{float(u0_val):+.3f}"
            g_txt = f"{float(g_val):+.3f}"
        except (TypeError, ValueError):
            u0_txt, g_txt = "N/D", "N/D"

        math_table_data = [
            [
                Paragraph("<b>Coordenada u₀:</b>", self.styles["TableLabel"]),
                Paragraph(f"<b>{u0_txt}</b> (proyección lineal de los 512 descriptores)", self.styles["TableValue"]),
                Paragraph("<b>Fórmula g(u₀):</b>", self.styles["TableLabel"]),
                Paragraph(f"<b>{g_txt}</b> (umbral de Youden: {evidence.get('kan64_th', 'N/D')})", self.styles["TableValue"]),
            ],
            [
                Paragraph(f"<b>p_sym = σ(g(u₀)) [u₀ = {u0_txt}; g = {g_txt}]:</b>", self.styles["TableLabel"]),
                Paragraph(f"<b>{kan_prob * 100:.1f}%</b> ({decision_label})", self.styles["TableValue"]),
                Paragraph("<b>Categoría BI-RADS orientativa de la herramienta:</b>", self.styles["TableLabel"]),
                Paragraph(f"<font color='{color_badge}'><b>{birads_sug}</b></font>", self.styles["TableValue"]),
            ],
            [
                Paragraph("<b>Región del foco del mapa de la fórmula (aprox.):</b>", self.styles["TableLabel"]),
                Paragraph(f"<b>{quad_pred}</b>", self.styles["TableValue"]),
                Paragraph("<b>Área de la zona marcada por el mapa:</b>", self.styles["TableLabel"]),
                Paragraph(f"{area_px} px² en la imagen de 224 px (no es el contorno de la lesión)", self.styles["TableValue"]),
            ]
        ]
        t_math = Table(math_table_data, colWidths=[145, 135, 140, 120])
        t_math.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EBF8FF")),
            ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#3182CE")),
            ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BEE3F8")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(Paragraph("<b>EVIDENCIA CUANTITATIVA EXTRAÍDA POR EL PIPELINE NEURO-SIMBÓLICO:</b>", self.styles["SectionHeading"]))
        story.append(t_math)
        story.append(Spacer(1, 12))

        # 4. Texto Estructurado del Informe Clínico
        story.append(Paragraph("<b>BORRADOR DE INFORME ESTRUCTURADO (SIN VALIDEZ CLÍNICA):</b>", self.styles["SectionHeading"]))
        story.append(Spacer(1, 4))
        
        # Parsear párrafos del reporte
        raw_paragraphs = report_text.split("\n\n")
        for p_raw in raw_paragraphs:
            p_raw = p_raw.strip()
            if not p_raw:
                continue
            # Ignorar encabezados redundantes o firmas duplicadas
            if "Firma del Radiólogo" in p_raw or "[Nombre del Radiólogo]" in p_raw:
                continue
            if p_raw.startswith("**Informe Radiológico") or p_raw.startswith("Informe Radiológico"):
                continue

            # Convertir markdown a tags HTML válidos de ReportLab
            p_formatted = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", p_raw)
            p_formatted = re.sub(r"\*(.*?)\*", r"<i>\1</i>", p_formatted)
            p_formatted = p_formatted.replace("\n", "<br/>")

            # Estilo especial para los encabezados de sección
            if any(p_formatted.startswith(f"<b>{i}.") for i in [1, 2, 3, 4]) or "MOTIVO DE ESTUDIO" in p_formatted or "HALLAZGOS" in p_formatted or "IMPRESIÓN" in p_formatted or "RECOMENDACIÓN" in p_formatted:
                story.append(Paragraph(f"<font color='#1A365D'>{p_formatted}</font>", self.styles["ReportBody"]))
            else:
                story.append(Paragraph(p_formatted, self.styles["ReportBody"]))
            story.append(Spacer(1, 5))

        story.append(Spacer(1, 10))

        # 5. Firma y Validación
        sig_data = [
            [
                Paragraph("___________________________________________<br/><b>Revisión por radiólogo (pendiente)</b><br/>Nombre y registro profesional del revisor<br/>Este borrador no tiene validez clínica", self.styles["TableValue"]),
                Paragraph("___________________________________________<br/><b>Auditoría automática de anclaje factual</b><br/>Consistencia con la evidencia numérica extraída<br/>Verificación exploratoria: no descarta todos los errores", self.styles["TableValue"]),
            ]
        ]
        t_sig = Table(sig_data, colWidths=[270, 270])
        t_sig.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(t_sig)

        # ==================== PÁGINA 2: ANEXO DE TRAZABILIDAD ESPACIAL ====================
        story.append(PageBreak())

        story.append(Paragraph("ANEXO TÉCNICO: TRAZABILIDAD ESPACIAL Y AUDITORÍA ALGORÍTMICA", self.styles["HospitalHeader"]))
        story.append(Paragraph(f"PACIENTE ID: {patient_id} | EVALUACIÓN DEL CAMPO RECEPTIVO Y ANÁLISIS ESPECTRAL", self.styles["HospitalSubHeader"]))
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#2B6CB0"), spaceAfter=10))

        # Tríptico de imágenes
        story.append(Paragraph("<b>1. TRÍPTICO DE LOCALIZACIÓN Y SEGMENTACIÓN DÉBILMENTE SUPERVISADA (WSSS)</b>", self.styles["SectionHeading"]))
        img_buf = self._create_visual_triptych(item, evidence)
        triptych_img = Image(img_buf, width=7.2 * inch, height=2.35 * inch)
        story.append(triptych_img)
        story.append(Spacer(1, 10))

        # Gráfico del enlace logístico
        story.append(Paragraph("<b>2. POSICIÓN DE LA PACIENTE EN LA FÓRMULA g(u₀)</b>", self.styles["SectionHeading"]))
        kan_buf = self._create_kan_curve_plot(evidence)
        kan_img = Image(kan_buf, width=6.8 * inch, height=2.4 * inch)
        story.append(kan_img)
        story.append(Spacer(1, 12))

        # Certificado de Consistencia Factual
        if str(evidence.get("localizer_source", "")).endswith("448px"):
            loc_txt = ("La localización es aproximada: el mapa se calcula con la imagen a 448 píxeles (14×14 celdas) y, en la "
                       "evaluación de la tesis, la región de su foco coincide con la de la lesión en el 47,8 % de las lesiones "
                       "malignas, frente al 34,0 % del centro del tejido. ")
        else:
            loc_txt = ("La localización es aproximada: el mapa se calculó a la resolución de la red (7×7 celdas), con la que, "
                       "en la evaluación de la tesis, el foco no localiza la lesión mejor que el centro del tejido. ")
        cert_text = (
            "<b>NOTA DE TRAZABILIDAD (verificación automática exploratoria):</b><br/>"
            "El borrador se redactó a partir de la evidencia numérica que entrega el pipeline: la coordenada canónica u₀, "
            "la fórmula g(u₀) y p_sym = σ(g(u₀)), la región de la imagen del centroide del núcleo focal sobre el mapa de la fórmula y el área de ese núcleo. "
            "La categoría BI-RADS es la orientativa de la herramienta: una regla sobre p_sym anclada en el umbral de decisión, "
            "de modo que nunca contradice el veredicto; no es una calibración clínica ni un riesgo de malignidad. "
            + loc_txt +
            "La región es de la imagen, no un cuadrante anatómico. "
            "El sistema es un prototipo de investigación no validado clínicamente: el borrador debe ser revisado por un especialista."
        )
        p_cert = Paragraph(cert_text, self.styles["ReportBody"])
        t_cert = Table([[p_cert]], colWidths=[540])
        t_cert.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0FFF4")),
            ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#38A169")),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(t_cert)
        story.append(Spacer(1, 14))

        # Pie de página técnico
        story.append(Paragraph(
            "Documento generado automáticamente por el Pipeline Neuro-Simbólico Prometeo. "
            "Trabajo de Grado — Pregrado en Matemáticas — Universidad El Bosque. "
            f"Marca de tiempo: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            self.styles["Footnote"]
        ))

        # Compilar documento
        doc.build(story)
        print(f"Informe PDF generado exitosamente en: {pdf_path}")
        return pdf_path
