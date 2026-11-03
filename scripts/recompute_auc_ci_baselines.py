"""
recompute_auc_ci_baselines.py

Cuantifica la incertidumbre y el contexto de los AUC reportados (P1 #3):

  1. IC95% del AUC por bootstrap estratificado (B=2000, semilla 42) para los
     6 modelos del Frente 1, usando las probabilidades guardadas en
     reports/experiments/youden_val_probs/.
  2. Comparacion por pares con la prueba de DeLong (covarianza para curvas
     ROC correlacionadas sobre la misma muestra), con estad->stico z,
     valor p bilateral e IC95% de la diferencia.
  3. Baselines de regresion logistica (control lineal trivial) entrenados
     sobre los embeddings GAP de entrenamiento y evaluados en prueba, para
     los cuatro backbones del estudio multi-backbone.

Salida:
    Códigos/reports/experiments/auc_ci_comparison.json

Uso:
    python scripts/recompute_auc_ci_baselines.py
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

CODIGOS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP_DIR = os.environ.get("TESIS_REP") or os.path.join(CODIGOS, "artifacts")
PROBS_DIR = os.path.join(CODIGOS, "reports", "experiments", "youden_val_probs")
EMB_DIR = os.path.join(REP_DIR, "output", "embeddings")
OUT_JSON = os.path.join(CODIGOS, "reports", "experiments", "auc_ci_comparison.json")

MODELS = ["Baseline_FC", "1A", "1C", "1A-Base", "1C-Base", "Robusto", "R3_formula"]
PAIRS = [
    ("Baseline_FC", "1A"),
    ("Baseline_FC", "1C"),
    ("Baseline_FC", "1A-Base"),
    ("Baseline_FC", "1C-Base"),
    ("Baseline_FC", "Robusto"),
    ("1A", "1C"),
    ("1A", "1A-Base"),
    ("1A", "1C-Base"),
    ("1A", "Robusto"),
    ("1C", "1A-Base"),
    ("1C", "1C-Base"),
    ("1C", "Robusto"),
    ("1A-Base", "1C-Base"),
    ("1A-Base", "Robusto"),
    ("1C-Base", "Robusto"),
    ("Baseline_FC", "R3_formula"),
    ("Robusto", "R3_formula"),
]
LR_BACKBONES = {
    "resnet18": "f2base",
    "densenet121": "f2_densenet121",
    "efficientnet_b0": "f2_efficientnet_b0",
    "resnet50": "f2_resnet50",
}
B_BOOT = 2000
SEED = 42


def stratified_bootstrap_ci(y, p, B=B_BOOT, seed=SEED):
    """IC95% del AUC por bootstrap estratificado por clase."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    n0, n1 = len(idx0), len(idx1)
    aucs = np.empty(B)
    for b in range(B):
        s0 = rng.choice(idx0, size=n0, replace=True)
        s1 = rng.choice(idx1, size=n1, replace=True)
        ii = np.concatenate([s0, s1])
        aucs[b] = roc_auc_score(y[ii], p[ii])
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)), aucs


def delong_auc_pair(y, pa, pb):
    """Prueba de DeLong para dos curvas ROC correlacionadas.

    Devuelve (auc_a, auc_b, delta, var_delta, z, p) con p bilateral.
    """
    y = np.asarray(y)
    pa = np.asarray(pa, dtype=float)
    pb = np.asarray(pb, dtype=float)
    pos = y == 1
    xa, ya = pa[pos], pa[~pos]
    xb, yb = pb[pos], pb[~pos]
    m, n = len(xa), len(ya)

    def comps(x, y_):
        gt = (x[:, None] > y_[None, :]).astype(float)
        eq = (x[:, None] == y_[None, :]).astype(float)
        psi = gt + 0.5 * eq
        v10 = psi.mean(axis=1)
        v01 = psi.mean(axis=0)
        return v10, v01

    v10a, v01a = comps(xa, ya)
    v10b, v01b = comps(xb, yb)
    auc_a = float(v10a.mean())
    auc_b = float(v10b.mean())
    cov10 = np.cov(v10a, v10b, ddof=1)
    cov01 = np.cov(v01a, v01b, ddof=1)
    var_a = cov10[0, 0] / m + cov01[0, 0] / n
    var_b = cov10[1, 1] / m + cov01[1, 1] / n
    cov_ab = cov10[0, 1] / m + cov01[0, 1] / n
    var_d = float(max(var_a + var_b - 2.0 * cov_ab, 1e-12))
    delta = auc_a - auc_b
    z = float(delta / np.sqrt(var_d)) if var_d > 0 else 0.0
    p = float(2.0 * norm.sf(abs(z)))
    return auc_a, auc_b, delta, var_d, z, p


def main():
    t0 = time.time()
    data = {}
    for name in MODELS:
        y = np.load(os.path.join(PROBS_DIR, f"{name}_test_labels.npy")).astype(int)
        p = np.load(os.path.join(PROBS_DIR, f"{name}_test_probs.npy")).astype(float)
        data[name] = (y, p)

    summary = {"models": {}, "delong": [], "logistic_baselines": {}}
    print("=== IC95%% bootstrap del AUC (B=%d) ===" % B_BOOT)
    for name in MODELS:
        y, p = data[name]
        auc = float(roc_auc_score(y, p))
        lo, hi, _ = stratified_bootstrap_ci(y, p)
        summary["models"][name] = {
            "auc": round(auc, 4),
            "ci95_low": round(lo, 4),
            "ci95_high": round(hi, 4),
            "n": len(y),
            "n_pos": int((y == 1).sum()),
            "n_neg": int((y == 0).sum()),
            "B": B_BOOT,
            "seed": SEED,
        }
        print(f"  {name:12s} AUC={auc:.4f}  IC95%=[{lo:.4f}, {hi:.4f}]")

    print("\n=== DeLong por pares (misma muestra de prueba) ===")
    for a, b in PAIRS:
        y = data[a][0]
        auc_a, auc_b, delta, var_d, z, p = delong_auc_pair(y, data[a][1], data[b][1])
        sd = float(np.sqrt(var_d))
        summary["delong"].append({
            "a": a, "b": b,
            "auc_a": round(auc_a, 4), "auc_b": round(auc_b, 4),
            "delta": round(delta, 4),
            "ci95_low_delta": round(delta - 1.96 * sd, 4),
            "ci95_high_delta": round(delta + 1.96 * sd, 4),
            "z": round(z, 4), "p": p,
        })
        print(
            f"  {a} vs {b}: AUC {auc_a:.4f} vs {auc_b:.4f} | "
            f"delta={delta:+.4f} IC=[{delta - 1.96 * sd:+.4f}, {delta + 1.96 * sd:+.4f}] | "
            f"z={z:.3f} p={p:.4g}"
        )

    print("\n=== Baselines de regresión logística (embeddings GAP) ===")
    for name, prefix in LR_BACKBONES.items():
        Xtr = np.load(os.path.join(EMB_DIR, f"{prefix}_train_feats.npy")).astype(np.float32)
        ytr = np.load(os.path.join(EMB_DIR, f"{prefix}_train_labels.npy")).astype(int)
        Xte = np.load(os.path.join(EMB_DIR, f"{prefix}_test_feats.npy")).astype(np.float32)
        yte = np.load(os.path.join(EMB_DIR, f"{prefix}_test_labels.npy")).astype(int)
        mu = Xtr.mean(0, keepdims=True)
        sd = Xtr.std(0, keepdims=True) + 1e-8
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED)
        clf.fit((Xtr - mu) / sd, ytr)
        p = clf.predict_proba((Xte - mu) / sd)[:, 1]
        auc = float(roc_auc_score(yte, p))
        lo, hi, _ = stratified_bootstrap_ci(yte, p, B=1000, seed=SEED)
        summary["logistic_baselines"][name] = {
            "prefix": prefix,
            "n_features": int(Xtr.shape[1]),
            "auc": round(auc, 4),
            "ci95_low": round(lo, 4),
            "ci95_high": round(hi, 4),
        }
        print(f"  {name:16s} D={Xtr.shape[1]:4d} AUC={auc:.4f}  IC95%=[{lo:.4f}, {hi:.4f}]")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nGuardado: {OUT_JSON}")
    print(f"Tiempo total: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()