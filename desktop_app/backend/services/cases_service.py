"""
Servicio central de acceso a datos y extracción de evidencia clínica CBIS-DDSM.
Orquesta el cargador de ROI, metadatos dicom y el extractor neuro-simbólico.
"""
import os
import re
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd

from desktop_app.backend.config import CODIGOS_DIR, DEVICE
from desktop_app.backend.schemas import (
    CaseSummary,
    CaseDetail,
    ClinicalMetadata,
    PredictionDetail,
    CanonicalVariables,
    CanonicalVariableItem,
    SpatialLocationDetail,
)
from desktop_app.backend.services.decision import apply_formula_decision, birads_int, NOTA_CONVENCION
from src.data.roi_loader import get_roi_manifest, CBISROIDataset
from src.reporting.clinical_evidence_extractor import ClinicalEvidenceExtractor


class CasesService:
    """Gestiona el catálogo de mamografías de prueba de CBIS-DDSM y ejecuta la
    extracción de evidencia factual, mapas CAM/WSSS y variables canónicas.
    """

    def __init__(self, codigos_dir: str = str(CODIGOS_DIR), device: str = DEVICE):
        self.codigos_dir = codigos_dir
        self.device = device
        
        # Carga del catálogo de mamografías emparejadas con ROI
        self.df_roi: pd.DataFrame = get_roi_manifest(self.codigos_dir)
        self.dataset: CBISROIDataset = CBISROIDataset(self.df_roi, image_size=224)
        
        # Inicialización del extractor unificado neuro-simbólico
        self.extractor: ClinicalEvidenceExtractor = ClinicalEvidenceExtractor(
            codigos_dir=self.codigos_dir,
            device=self.device
        )
        
        # Caché en memoria para evitar recomputaciones tensoriales costosas
        self._evidence_cache: Dict[int, Dict[str, Any]] = {}

    def get_total_cases(self) -> int:
        """Devuelve el total de casos disponibles en el dataset."""
        return len(self.df_roi)

    def get_cohort(
        self,
        pathology: str = "all",
        abnormality_type: str = "all"
    ) -> List[CaseSummary]:
        """Devuelve la lista filtrada de resúmenes de casos de la cohorte."""
        filtered_df = self.df_roi.copy()
        
        # Normalización y filtrado por patología
        pat_clean = pathology.strip().lower()
        if pat_clean in ["benign", "benigno"]:
            filtered_df = filtered_df[
                filtered_df["pathology"].str.upper().isin(["BENIGN", "BENIGN_WITHOUT_CALLBACK"])
            ]
        elif pat_clean in ["malignant", "maligno"]:
            filtered_df = filtered_df[
                filtered_df["pathology"].str.upper() == "MALIGNANT"
            ]

        # Normalización y filtrado por tipo de alteración
        type_clean = abnormality_type.strip().lower()
        if type_clean in ["mass", "masas", "masa"]:
            filtered_df = filtered_df[filtered_df["abnormality_type"] == "mass"]
        elif type_clean in ["calc", "calcificaciones", "calcificacion"]:
            filtered_df = filtered_df[filtered_df["abnormality_type"] == "calc"]

        summaries: List[CaseSummary] = []
        for idx, row in filtered_df.iterrows():
            rel_img = row.get("rel_img_path", "")
            p_id = str(row.get("patient_id", ""))
            
            # Recuperar metadatos clínicos adicionales del catálogo del extractor
            meta = self.extractor.meta_by_relpath.get(rel_img, {})
            if not meta and rel_img:
                meta = self.extractor.meta_by_relpath.get(os.path.normpath(rel_img), {})
            if not meta and p_id:
                meta = self.extractor.meta_by_patient_id.get(p_id, {})

            birads_gt_val = 0
            if meta and "assessment" in meta and meta["assessment"] is not None:
                try:
                    birads_gt_val = int(meta["assessment"])
                except (ValueError, TypeError):
                    birads_gt_val = 0

            density_val = None
            if pd.notna(row.get("density")):
                try:
                    density_val = int(row["density"])
                except (ValueError, TypeError):
                    density_val = None

            summaries.append(
                CaseSummary(
                    index=int(idx),
                    patient_id=p_id,
                    laterality=str(meta.get("laterality", "UNKNOWN")),
                    view=str(meta.get("view", "UNKNOWN")),
                    pathology=str(row.get("pathology", "UNKNOWN")),
                    type=str(row.get("abnormality_type", "UNKNOWN")),
                    birads_gt=birads_gt_val,
                    density=density_val,
                )
            )

        return summaries

    def extract_evidence(self, case_index: int) -> Dict[str, Any]:
        """Calcula o recupera de la caché la evidencia analítica de un caso."""
        if case_index < 0 or case_index >= len(self.df_roi):
            raise IndexError(f"Índice de caso fuera de rango: {case_index} (máx {len(self.df_roi) - 1})")

        if case_index not in self._evidence_cache:
            item = self.dataset[case_index]
            rel_path = self.df_roi.iloc[case_index]["rel_img_path"]
            evidence = self.extractor.extract_evidence(item, rel_path)
            apply_formula_decision(evidence)
            self._evidence_cache[case_index] = evidence

        return self._evidence_cache[case_index]

    def get_raw_item(self, case_index: int) -> Dict[str, Any]:
        """Devuelve el elemento sin procesar del dataset PyTorch (imágenes y máscaras)."""
        if case_index < 0 or case_index >= len(self.df_roi):
            raise IndexError(f"Índice de caso fuera de rango: {case_index}")
        return self.dataset[case_index]

    def get_case_detail(self, case_index: int) -> CaseDetail:
        """Construye el detalle clínico y analítico serializado para un caso."""
        evidence = self.extract_evidence(case_index)
        row = self.df_roi.iloc[case_index]
        p_id = str(evidence.get("patient_id", row.get("patient_id", "")))
        rel_img = str(row.get("rel_img_path", ""))

        meta = self.extractor.meta_by_relpath.get(rel_img, {})
        if not meta and rel_img:
            meta = self.extractor.meta_by_relpath.get(os.path.normpath(rel_img), {})
        if not meta and p_id:
            meta = self.extractor.meta_by_patient_id.get(p_id, {})

        density_val = None
        if pd.notna(row.get("density")):
            try:
                density_val = int(row["density"])
            except (ValueError, TypeError):
                density_val = None

        subtlety_val = None
        if meta and meta.get("subtlety") is not None:
            try:
                subtlety_val = int(meta["subtlety"])
            except (ValueError, TypeError):
                subtlety_val = None

        assessment_val = int(evidence.get("assessment_gt", meta.get("assessment", 0)) or 0)

        metadata_obj = ClinicalMetadata(
            patient_id=p_id,
            laterality=str(evidence.get("laterality", meta.get("laterality", "UNKNOWN"))),
            view=str(evidence.get("view", meta.get("view", "UNKNOWN"))),
            pathology_gt=str(evidence.get("pathology_gt", row.get("pathology", "UNKNOWN"))),
            assessment_gt=assessment_val,
            birads_gt=assessment_val,
            abnormality_type=str(evidence.get("abnormality_type", row.get("abnormality_type", "UNKNOWN"))),
            lesion_shape=meta.get("mass_shape") if meta else None,
            lesion_margins=meta.get("mass_margins") if meta else None,
            calc_type=meta.get("calc_type") if meta else None,
            calc_distribution=meta.get("calc_distribution") if meta else None,
            density=density_val,
            subtlety=subtlety_val,
        )

        # Decisión: la fórmula g(u0) con el umbral de Youden (ver decision.py)
        birads_label = str(evidence.get("birads_suggested", ""))
        birads_pred_int = birads_int(birads_label)
        prob_mal = float(evidence.get("decision_p", 0.0))
        verdict_str = str(evidence.get("decision", "BENIGN"))
        consensus_es_str = "Maligno" if verdict_str == "MALIGNANT" else "Benigno"

        prediction_obj = PredictionDetail(
            prob_malignant=prob_mal,
            birads_pred=birads_pred_int,
            birads_label=birads_label,
            verdict=verdict_str,
            prob_resnet=evidence.get("prob_resnet"),
            pred_resnet=evidence.get("pred_resnet"),
            prob_kan_symb=evidence.get("prob_kan_symb"),
            pred_kan_symb=evidence.get("pred_kan_symb"),
            consensus_es=consensus_es_str,
            clinical_recommendation=evidence.get("clinical_recommendation"),
            margin_status=evidence.get("margin_status"),
            margin_desc=evidence.get("margin_desc"),
            density_status=evidence.get("density_status"),
            density_desc=evidence.get("density_desc"),
            decision_source=evidence.get("decision_source"),
            u0=evidence.get("decision_u0"),
            g_logit=evidence.get("decision_g"),
            p_sym=evidence.get("decision_p"),
            threshold_logit=evidence.get("decision_th_logit"),
            threshold_prob=evidence.get("decision_th_prob"),
            prob_net_robust=evidence.get("prob_net_robust"),
            prob_net_base=evidence.get("prob_resnet"),
            birads_note=NOTA_CONVENCION,
        )

        # Variables canónicas fundamentales
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

        # Métricas de localización espacial y segmentación WSSS
        pred_centroid = evidence.get("predicted_centroid")
        gt_centroid = evidence.get("gt_centroid")

        pred_cent_list = [float(pred_centroid[0]), float(pred_centroid[1])] if pred_centroid else None
        gt_cent_list = [float(gt_centroid[0]), float(gt_centroid[1])] if gt_centroid else None

        spatial_obj = SpatialLocationDetail(
            quadrant_pred=str(evidence.get("predicted_quadrant", "Región Retroareolar / Central")),
            quadrant_gt=str(evidence.get("gt_quadrant", "Región Retroareolar / Central")),
            is_quadrant_exact=bool(evidence.get("is_quadrant_exact", False)),
            predicted_centroid=pred_cent_list,
            gt_centroid=gt_cent_list,
            centroid_dist_px=evidence.get("centroid_dist_px"),
            pred_area_px=evidence.get("pred_area_px"),
            gt_area_px=evidence.get("gt_area_px"),
            equiv_diameter_px=evidence.get("equiv_diameter_px"),
            compactness=evidence.get("compactness"),
            eccentricity=evidence.get("eccentricity"),
            dice=evidence.get("dice"),
            iou=evidence.get("iou"),
            ebpg_score=evidence.get("psi_kan_ebpg"),
        )

        return CaseDetail(
            index=case_index,
            patient_id=p_id,
            metadata=metadata_obj,
            prediction=prediction_obj,
            canonical_variables=canonical_obj,
            spatial_location=spatial_obj,
            quadrant=spatial_obj.quadrant_pred,
            pathology=metadata_obj.pathology_gt,
            abnormality_type=metadata_obj.abnormality_type,
            x0_canonical=x0_val,
            x1_canonical=x1_val,
            x2_canonical=x2_val,
        )


# Instancia Singleton perezosa para el ciclo de vida de la aplicación
_cases_service_instance: Optional[CasesService] = None


def get_cases_service() -> CasesService:
    """Provee la instancia compartida de CasesService (lazy initialization)."""
    global _cases_service_instance
    if _cases_service_instance is None:
        _cases_service_instance = CasesService()
    return _cases_service_instance
