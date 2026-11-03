"""
roi_loader.py: Cargador de mamografías de prueba de CBIS-DDSM emparejadas con sus
máscaras de segmentación patológica (ROI) ground-truth delineadas por radiólogos clínicos.
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def get_roi_manifest(codigos_dir: str = None):
    """Construye un DataFrame con todas las imágenes de manifest_test.csv que
    poseen máscara ROI ground-truth en CBIS-DDSM.
    """
    if not codigos_dir:
        codigos_dir = os.environ.get("TESIS_CODIGOS") or os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    manifest_p = os.path.join(codigos_dir, "src", "data", "processed", "manifest_test.csv")
    dicom_info_p = os.path.join(codigos_dir, "database", "archive", "csv", "dicom_info.csv")
    mass_test_p = os.path.join(codigos_dir, "database", "archive", "csv", "mass_case_description_test_set.csv")
    calc_test_p = os.path.join(codigos_dir, "database", "archive", "csv", "calc_case_description_test_set.csv")

    manifest = pd.read_csv(manifest_p)
    dicom_info = pd.read_csv(dicom_info_p)
    mass_test = pd.read_csv(mass_test_p)
    calc_test = pd.read_csv(calc_test_p)

    def extract_series_uid(p):
        parts = str(p).strip().replace("\\", "/").split("/")
        if len(parts) >= 3:
            return parts[-2]
        return None

    dicom_info["area"] = dicom_info["Rows"] * dicom_info["Columns"]

    # Separar con precisión quirúrgica el mapa de imágenes completas vs máscaras ROI
    # En CBIS-DDSM cada serie contiene tanto el recorte (crop) como la máscara de tamaño completo.
    series_roi_map = {}
    series_full_map = {}
    for uid, grp in dicom_info.groupby("SeriesInstanceUID"):
        uid_str = str(uid).strip()
        roi_rows = grp[grp["SeriesDescription"] == "ROI mask images"]
        if len(roi_rows) > 0:
            best_roi = roi_rows.iloc[0]
        else:
            best_roi = grp.sort_values("area", ascending=False).iloc[0]
        series_roi_map[uid_str] = str(best_roi["image_path"]).replace("CBIS-DDSM/", "").replace("/", "\\")

        full_rows = grp[grp["SeriesDescription"] == "full mammogram images"]
        if len(full_rows) > 0:
            best_full = full_rows.iloc[0]
        else:
            best_full = grp.sort_values("area", ascending=False).iloc[0]
        series_full_map[uid_str] = str(best_full["image_path"]).replace("CBIS-DDSM/", "").replace("/", "\\")

    img_to_roi_meta = {}
    for df_meta, ab_type in [(mass_test, "mass"), (calc_test, "calc")]:
        for _, row in df_meta.iterrows():
            s_img = extract_series_uid(row["image file path"])
            s_roi = extract_series_uid(row["ROI mask file path"])
            if s_img in series_full_map and s_roi in series_roi_map:
                jpg_img = os.path.join("database", "archive", series_full_map[s_img])
                jpg_roi = os.path.join("database", "archive", series_roi_map[s_roi])
                img_to_roi_meta[jpg_img] = {
                    "roi_path": jpg_roi,
                    "pathology": str(row.get("pathology", "UNKNOWN")).strip(),
                    "abnormality_type": ab_type,
                    "density": row.get("breast_density" if "breast_density" in row else "breast density", None),
                    "patient_id": row.get("patient_id", "")
                }

    records = []
    for _, row in manifest.iterrows():
        fpath = row["filepath"]
        if fpath in img_to_roi_meta:
            meta = img_to_roi_meta[fpath]
            full_img = os.path.join(codigos_dir, fpath)
            full_roi = os.path.join(codigos_dir, meta["roi_path"])
            if os.path.exists(full_img) and os.path.exists(full_roi):
                records.append({
                    "image_path": full_img,
                    "roi_path": full_roi,
                    "rel_img_path": fpath,
                    "rel_roi_path": meta["roi_path"],
                    "label": int(row["label"]),
                    "patient_id": row["patient_id"],
                    "pathology": meta["pathology"],
                    "abnormality_type": meta["abnormality_type"],
                    "density": meta["density"]
                })

    df_roi = pd.DataFrame(records)
    return df_roi


class CBISROIDataset(Dataset):
    """Dataset de parejas (mamografía, máscara ROI) para CBIS-DDSM."""

    def __init__(self, df: pd.DataFrame, image_size: int = 224):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.norm = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_raw = Image.open(row["image_path"]).convert("RGB")
        roi_raw = Image.open(row["roi_path"]).convert("L")

        # Tensor normalizado para PyTorch
        img_tensor = self.norm(img_raw)

        # Máscara binaria redimensionada
        roi_resized = roi_raw.resize((self.image_size, self.image_size), Image.NEAREST)
        roi_np = (np.array(roi_resized) > 127).astype(np.float32)

        return {
            "image_tensor": img_tensor,
            "mask_np": roi_np,
            "label": row["label"],
            "pathology": row["pathology"],
            "abnormality_type": row["abnormality_type"],
            "patient_id": row["patient_id"],
            "rel_img_path": row["rel_img_path"],
            "rel_roi_path": row["rel_roi_path"],
            "image_path": str(row["image_path"]),   # el localizador la usa a 448 px
        }
