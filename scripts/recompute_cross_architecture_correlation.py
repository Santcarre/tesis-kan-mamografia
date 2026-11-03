"""
recompute_cross_architecture_correlation.py

Reproduce el análisis de invarianza inter-arquitectura del Notebook 30 (P1 #11)
con matrices de correlación COMPLETAS y emparejamiento húngaro exhaustivo.

Replica exactamente `get_spca_coords` del NB30:
  - embeddings train/test por prefijo, z-score con stats de train,
  - ponderación por |corr| con la etiqueta de train,
  - PCA(12) ajustado en train ponderado, coordenadas de test ponderadas.

Para cada par (referencia ResNet-18 Robusto vs. cada uno de los otros 4):
  - matriz de correlación cruzada 12x12 sobre prueba,
  - asignación óptima bipartita (Húngaro) sobre |R| para las 12 parejas,
  - estadísticos: |r| ordenados, media/mediana, conteos por umbral.

Salida:
    Códigos/reports/experiments/cross_architecture_correlation.json

Uso:
    python scripts/recompute_cross_architecture_correlation.py
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA

CODIGOS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP_DIR = os.environ.get("TESIS_REP") or os.path.join(CODIGOS, "artifacts")
EMB_DIR = os.path.join(REP_DIR, "output", "embeddings")
OUT_JSON = os.path.join(CODIGOS, "reports", "experiments", "cross_architecture_correlation.json")

MODELS = {
    "ResNet-18 Robusto": "f2robust",
    "ResNet-18 Base": "f2base",
    "DenseNet-121": "f2_densenet121",
    "ResNet-50": "f2_resnet50",
    "EfficientNet-B0": "f2_efficientnet_b0",
}
REFERENCE = "ResNet-18 Robusto"
N_COMP = 12


def get_spca_coords(prefix, n_components=N_COMP):
    X_tr = np.load(os.path.join(EMB_DIR, f"{prefix}_train_feats.npy")).astype(np.float64)
    y_tr = np.load(os.path.join(EMB_DIR, f"{prefix}_train_labels.npy")).astype(int)
    X_te = np.load(os.path.join(EMB_DIR, f"{prefix}_test_feats.npy")).astype(np.float64)

    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0) + 1e-8
    X_tr_n = (X_tr - mu) / sd
    X_te_n = (X_te - mu) / sd

    corrs = np.array([np.corrcoef(X_tr_n[:, i], y_tr)[0, 1] for i in range(X_tr_n.shape[1])])
    weights = np.abs(np.nan_to_num(corrs, 0.0))
    weights = weights / (weights.max() + 1e-8)

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_tr_n * weights)
    Z_te = pca.transform(X_te_n * weights)
    return Z_te


def main():
    t0 = time.time()
    coords = {}
    for name, prefix in MODELS.items():
        coords[name] = get_spca_coords(prefix)
        print(f"{name:18s}: {coords[name].shape}")

    Z_ref = coords[REFERENCE]
    comparisons = {}
    for target in MODELS:
        if target == REFERENCE:
            continue
        Z_tgt = coords[target]
        R = np.zeros((N_COMP, N_COMP))
        for i in range(N_COMP):
            for j in range(N_COMP):
                R[i, j] = float(np.corrcoef(Z_ref[:, i], Z_tgt[:, j])[0, 1])
        A = np.abs(R)
        row, col = linear_sum_assignment(-A)
        pairs = [
            {"ref_var": int(i), "tgt_var": int(j),
             "r": round(float(R[i, j]), 4), "abs_r": round(float(A[i, j]), 4)}
            for i, j in zip(row, col)
        ]
        pairs.sort(key=lambda d: d["abs_r"], reverse=True)
        mags = np.array([p["abs_r"] for p in pairs])
        comparisons[target] = {
            "matrix": [[round(float(v), 4) for v in fila] for fila in R.tolist()],
            "assignment": pairs,
            "mean_abs_r": round(float(mags.mean()), 4),
            "median_abs_r": round(float(np.median(mags)), 4),
            "n_ge_090": int((mags >= 0.90).sum()),
            "n_ge_070": int((mags >= 0.70).sum()),
            "n_ge_050": int((mags >= 0.50).sum()),
        }
        top = pairs[0]
        print(
            f"{REFERENCE} vs {target}: top (x{top['ref_var']} <-> x{top['tgt_var']}) "
            f"r={top['r']:+.4f} | media| r|={mags.mean():.4f} mediana={np.median(mags):.4f} | "
            f">=0.90: {(mags >= 0.90).sum()}/12 >=0.70: {(mags >= 0.70).sum()}/12 >=0.50: {(mags >= 0.50).sum()}/12"
        )

    out = {
        "reference": REFERENCE,
        "n_components": N_COMP,
        "method": "S-PCA ponderado por correlación (réplica exacta de get_spca_coords del NB30) + Húngaro sobre |R|",
        "comparisons": comparisons,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nGuardado: {OUT_JSON}")
    print(f"Tiempo total: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()