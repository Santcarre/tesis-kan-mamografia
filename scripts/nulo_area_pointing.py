# -*- coding: utf-8 -*-
"""
nulo_area_pointing.py -- Nivel de azar del Pointing Game por area de tejido (Cap. 4, Pointing Game)

El EBPG necesita un valor de referencia. Este script construye el nulo por area con las mismas 368
mascaras clinicas de CBIS-DDSM:

  - null_image : fraccion del area del tumor sobre la imagen completa (mapa uniforme sobre el dominio).
  - null_breast: fraccion del area del tumor sobre el tejido mamario (nulo realista: el mapa solo puede
                 caer dentro de la mama). La mascara de tejido es la del pipeline WSSS
                 (`extract_breast_tissue_mask`), de modo que nulo y metrica comparten dominio.

Se reportan medias con IC95 % bootstrap (B=5000, semilla 42) para el conjunto global, malignos y benignos.

Salida: reports/experiments/pointing_game_nulo_area.json
Uso (desde la raiz del repositorio, requiere las imagenes CBIS-DDSM):  python scripts/nulo_area_pointing.py
"""
import json
import os
import sys

import numpy as np

CODIGOS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (CODIGOS, os.path.join(CODIGOS, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data.roi_loader import CBISROIDataset, get_roi_manifest
from src.explainability.weak_segmentation import extract_breast_tissue_mask

OUT_JSON = os.path.join(CODIGOS, "reports", "experiments", "pointing_game_nulo_area.json")
SEED = 42
H, W = 224, 224


def bootstrap_mean_ci(x, B=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    draws = rng.integers(0, len(x), size=(B, len(x)))
    means = x[draws].mean(axis=1)
    return float(means.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    df_roi = get_roi_manifest()
    assert len(df_roi) == 368, f"Se esperaban 368 mamografias con mascara, hay {len(df_roi)}"
    ds = CBISROIDataset(df_roi, image_size=224)
    area_mask, area_tissue, label = [], [], []
    for i in range(len(ds)):
        item = ds[i]
        mask = np.asarray(item["mask_np"]) > 0.5
        img = np.asarray(item["image_tensor"].numpy()[0], dtype=float)
        disp = (img - img.min()) / (img.max() - img.min() + 1e-8)
        tissue = extract_breast_tissue_mask(disp) > 0.5
        area_mask.append(int(mask.sum()))
        area_tissue.append(int(tissue.sum()))
        label.append(int(item["label"]))
    area_mask, area_tissue, label = map(np.asarray, (area_mask, area_tissue, label))
    null_image = area_mask / float(H * W)
    null_breast = area_mask / np.maximum(area_tissue, 1).astype(float)
    mal = label == 1

    out = {"n": int(len(label)), "n_malignant": int(mal.sum()), "n_benign": int((~mal).sum()), "nulls": {}}
    for name, sel in [("global", np.ones_like(mal)), ("malignant", mal), ("benign", ~mal)]:
        m_img, lo_img, hi_img = bootstrap_mean_ci(null_image[sel])
        m_tis, lo_tis, hi_tis = bootstrap_mean_ci(null_breast[sel])
        m_area, _, _ = bootstrap_mean_ci(area_mask[sel])
        out["nulls"][name] = {
            "mean_area_mask_px": round(m_area, 1),
            "null_image_mean": round(m_img, 6), "null_image_ci95": [round(lo_img, 6), round(hi_img, 6)],
            "null_breast_mean": round(m_tis, 6), "null_breast_ci95": [round(lo_tis, 6), round(hi_tis, 6)],
        }
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    for k, v in out["nulls"].items():
        print(f"{k:10s} azar/imagen {v['null_image_mean']*100:.3f}% | azar/tejido {v['null_breast_mean']*100:.3f}% {v['null_breast_ci95']}")
    print("Guardado:", OUT_JSON)


if __name__ == "__main__":
    main()
