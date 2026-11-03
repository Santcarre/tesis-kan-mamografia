# -*- coding: utf-8 -*-
"""
credito_por_caso.py -- ¿Que componente canonica recibe el mayor credito, caso a caso, en el logit de la CNN?

Para cada imagen de prueba, el valor de Shapley exacto de la CNN (logit afin en GAP) es phi_c = a_c (GAP_c - E[GAP_c]);
LIME lo recupera con rho = 0,9998 (attribution_coherence.json). Expresado en la base canonica del reductor de la formula,
la contribucion de u_j al logit centrado es  c_j = beta_j (u_j - media(u_j)),  con beta de minimos cuadrados en train.
Se mide en cuantos casos |c_0| es la mayor, que fraccion del credito absoluto recibe u0, y la concordancia
caso a caso entre c_0 y el logit centrado.

Uso (desde la raiz del repo):  python scripts/credito_por_caso.py [RAIZ] [SALIDA.json]
"""
import sys, os, json
import numpy as np, torch
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = sys.argv[1] if len(sys.argv) > 1 else REPO   # raiz del repositorio
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "reports", "auditoria", "credito_por_caso.json")
COD = ROOT; EMB = os.path.join(ROOT, "artifacts", "output", "embeddings", "f2robust_")
EXP = os.path.join(COD, "reports", "experiments")
X = {s: np.load(EMB + f"{s}_feats.npy").astype(np.float64) for s in ("train", "test")}
r = np.load(os.path.join(EXP, "kan_robust64_reducer.npz"))
sd = torch.load(os.path.join(COD, "reports", "models", "resnet18_robust_best.pt"), map_location="cpu")["model_state_dict"]
a = (sd["fc.weight"][1] - sd["fc.weight"][0]).numpy().astype(np.float64); c0 = float((sd["fc.bias"][1] - sd["fc.bias"][0]))


def z64(Xs):
    g = (Xs - r["gap_mu"]) / r["gap_sd"]; z = g @ r["proj_w"].T + r["proj_b"]
    return ((z - r["pca_mean"]) @ r["pca_components"].T - r["mu_z"]) / r["sd_z"]


Z = {k: z64(X[k]) for k in X}; s = {k: X[k] @ a + c0 for k in X}
A = np.c_[np.ones(len(Z["train"])), Z["train"]]
beta = np.linalg.lstsq(A, s["train"], rcond=None)[0][1:]
zbar = Z["train"].mean(0)
C = (Z["test"] - zbar) * beta                      # contribuciones por caso (n x 64)
s_c = s["test"] - s["train"].mean()
top = np.abs(C).argmax(1)
share0 = np.abs(C[:, 0]) / np.abs(C).sum(1)
res = {
    "n_test": int(len(C)),
    "casos_con_u0_como_mayor_credito": float((top == 0).mean()),
    "fraccion_credito_absoluto_u0": {"media": float(share0.mean()), "mediana": float(np.median(share0)),
                                      "p10": float(np.percentile(share0, 10)), "p90": float(np.percentile(share0, 90))},
    "spearman_c0_vs_logit_centrado": float(spearmanr(C[:, 0], s_c)[0]),
    "pearson_c0_vs_logit_centrado": float(np.corrcoef(C[:, 0], s_c)[0, 1]),
    "segunda_componente_mas_frecuente": {str(int(j)): float((top == j).mean()) for j in np.unique(top) if j != 0},
}
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2); print(json.dumps(res, ensure_ascii=False, indent=2))
