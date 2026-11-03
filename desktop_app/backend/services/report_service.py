"""
Servicio de generación de informes clínicos neuro-simbólicos y exportación en PDF.
Orquesta el anclaje factual con la coordenada canónica u0 y la fórmula g(u0), el motor LLM
local (Qwen-2.5) con fallback determinista robusto y la compilación formal con ReportLab.
"""
import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

from desktop_app.backend.config import ROOT_DIR, CODIGOS_DIR
from desktop_app.backend.services.cases_service import CasesService, get_cases_service
from desktop_app.backend.services.inference_service import get_inference_service
from src.reporting.llm_engine import (
    LLMEngine,
    format_evidence_prompt,
    generate_deterministic_grounded_report,
)
from src.reporting.pdf_report_generator import ClinicalPDFReportGenerator


class ReportService:
    """Gestiona la síntesis de informes clínicos con evidencia neuro-simbólica
    y su renderizado en documentos PDF hospitalarios de alta fidelidad.
    """

    def __init__(
        self,
        cases_service: Optional[CasesService] = None,
        export_dir: Optional[Path] = None
    ):
        self.cases_service = cases_service or get_cases_service()
        self.export_dir = export_dir or (ROOT_DIR / "desktop_app" / "reports")
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_evidence_and_item(
        self,
        case_index: int,
        custom_evidence: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Recupera el diccionario de evidencia y el tensor de imagen sin procesar."""
        evidence: Optional[Dict[str, Any]] = None
        item: Optional[Dict[str, Any]] = None

        if custom_evidence is not None and isinstance(custom_evidence, dict) and len(custom_evidence) > 0:
            evidence = dict(custom_evidence)
            # Intentar obtener el tensor del item si fue incrustado
            if "item" in evidence and isinstance(evidence["item"], dict):
                item = evidence["item"]
            elif "image_tensor" in evidence:
                item = evidence
            else:
                # Comprobar si proviene de la última inferencia externa
                inf_service = get_inference_service()
                if inf_service._last_external_item is not None:
                    item = inf_service._last_external_item

        if evidence is None:
            # Caso del dataset CBIS-DDSM
            total_cases = self.cases_service.get_total_cases()
            if 0 <= case_index < total_cases:
                evidence = self.cases_service.extract_evidence(case_index)
                item = self.cases_service.get_raw_item(case_index)
            elif case_index == -1:
                # Caso de inferencia externa más reciente
                inf_service = get_inference_service()
                if inf_service._last_external_evidence is not None:
                    evidence = inf_service._last_external_evidence
                    item = inf_service._last_external_item
                else:
                    raise IndexError("No hay ninguna inferencia externa previa en memoria.")
            else:
                raise IndexError(f"Caso {case_index} no encontrado o fuera de rango (0 a {total_cases - 1}).")

        # Asegurar un item válido con tensor de imagen para el generador de PDF
        if item is None or "image_tensor" not in item:
            item = {
                "image_tensor": torch.zeros((1, 224, 224), dtype=torch.float32),
                "mask_np": np.zeros((224, 224), dtype=np.uint8),
            }

        return evidence, item

    def generate_report(
        self,
        case_index: int = 0,
        use_llm: bool = False,
        custom_evidence: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """Genera el informe clínico estructurado en Markdown.
        
        Si use_llm es True, intenta llamar al modelo local Qwen-2.5 vía Ollama.
        Si Ollama no está activo o genera error, efectúa fallback determinista sin fallar.
        
        Retorna:
            (report_text, engine) donde engine es 'llm-local' o 'deterministic-grounded'.
        """
        evidence, _ = self._resolve_evidence_and_item(case_index, custom_evidence)

        if use_llm:
            deterministic_text = generate_deterministic_grounded_report(evidence)
            try:
                prompt = format_evidence_prompt(evidence)   # el texto con evidencia del estudio de informes (Anexo A)
                llm = LLMEngine(model_name=os.environ.get("PROMETEO_LLM", "phi3.5"), backend="ollama")   # el de mejor equilibrio en el estudio de informes
                llm_output = llm.generate_report(prompt=prompt, evidence=evidence)

                # Verificar si LLMEngine retornó error o aplicó fallback interno
                if (
                    not llm_output
                    or "Error" in llm_output
                    or "Excepción" in llm_output
                    or llm_output.strip() == deterministic_text.strip()
                ):
                    return deterministic_text, "deterministic-grounded"
                return llm_output, "llm-local"
            except Exception:
                return deterministic_text, "deterministic-grounded"
        else:
            report_text = generate_deterministic_grounded_report(evidence)
            return report_text, "deterministic-grounded"

    def generate_pdf(
        self,
        case_index: int,
        report_text: str,
        custom_evidence: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """Compila y genera el documento formal hospitalario en PDF de 2 páginas.
        
        Retorna:
            (pdf_path, filename) con la ruta absoluta y nombre del archivo generado.
        """
        evidence, item = self._resolve_evidence_and_item(case_index, custom_evidence)

        patient_id = evidence.get("patient_id", f"case_{case_index}")
        safe_id = re.sub(r"[^\w\-]", "_", str(patient_id))
        filename = f"Informe_Radiologico_{safe_id}.pdf"

        generator = ClinicalPDFReportGenerator(output_dir=str(self.export_dir))
        pdf_path = generator.generate_pdf(
            item=item,
            evidence=evidence,
            report_text=report_text,
            filename=filename
        )

        return str(pdf_path), filename


# Instancia Singleton compartida para el backend
_report_service_instance: Optional[ReportService] = None


def get_report_service() -> ReportService:
    """Provee la instancia singleton compartida de ReportService."""
    global _report_service_instance
    if _report_service_instance is None:
        _report_service_instance = ReportService()
    return _report_service_instance
