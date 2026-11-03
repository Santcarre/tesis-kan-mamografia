# -*- coding: utf-8 -*-
"""
beta_canonico.py -- Credito de cada componente CANONICA (reductor de 64 d de la formula) en el logit de la CNN robusta.

Corrige la Tabla beta_canonico del Cap. 4: attribution_coherence.json uso el S-PCA de 12 componentes de la estacion,
que no es el reductor kan_robust64_reducer.npz de la formula. Aqui se usa el mismo reductor que la formula.

Definiciones. s(x) = (w1 - w0) . GAP(x) + (b1 - b0)  (logit de la CNN robusta; Z = coordenadas canonicas u_0..u_63).
beta = minimos cuadrados de s sobre Z (con intercepto) AJUSTADOS EN TRAIN. Como Z es ~ortonormal en train (PCA, estandarizada),
beta_j ~ cov(s, u_j) y la fraccion de varianza de la parte lineal atribuible a u_j es beta_j^2 / sum_k beta_k^2.

Uso (desde la raiz del repo):  python scripts/beta_canonico.py [RAIZ] [SALIDA.json]       (requiere torch, numpy, scikit-learn, scipy)
"""
import sys, os, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = sys.argv[1] if len(sys.argv) > 1 else REPO   # raiz del repositorio
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "reports", "auditoria", "beta_canonico.json")
COD = ROOT; EMB = os.path.join(ROOT, "artifacts", "output", "embeddings", "f2robust_")
EXP = os.path.join(COD, "reports", "experiments"); PROB = os.path.join(EXP, "youden_val_probs")
X = {s: np.load(EMB + f"{s}_feats.npy").astype(np.float64) for s in ("train", "val", "test")}
y = {s: np.load(EMB + f"{s}_labels.npy").astype(int) for s in X}
r = np.load(os.path.join(EXP, "kan_robust64_reducer.npz"))
ck = torch.load(os.path.join(COD, "reports", "models", "resnet18_robust_best.pt"), map_location="cpu")
sd = ck["model_state_dict"]; Wf = sd["fc.weight"].numpy().astype(np.float64); bf = sd["fc.bias"].numpy().astype(np.float64)
a = Wf[1] - Wf[0]; c = bf[1] - bf[0]


def z64(Xs):
    g = (Xs - r["gap_mu"]) / r["gap_sd"]; z = g @ r["proj_w"].T + r["proj_b"]
    return ((z - r["pca_mean"]) @ r["pca_components"].T - r["mu_z"]) / r["sd_z"]


Z = {k: z64(X[k]) for k in X}
s = {k: X[k] @ a + c for k in X}

# comprobacion: s reproduce las probabilidades guardadas de la red robusta (test)
chk = None
f = os.path.join(PROB, "Robusto_test_probs.npy")
if os.path.exists(f):
    p = np.load(f).ravel(); pm = 1 / (1 + np.exp(-s["test"]))
    chk = {"max_abs_dif_prob": float(np.abs(p - pm).max()), "media": float(np.abs(p - pm).mean())}
print("comprobacion s vs probs guardadas:", chk)

A = np.c_[np.ones(len(Z["train"])), Z["train"]]
coef = np.linalg.lstsq(A, s["train"], rcond=None)[0]; b0, beta = coef[0], coef[1:]
var_u = Z["train"].var(axis=0)
contrib = beta ** 2 * var_u; share = contrib / contrib.sum()


def r2(split, k):
    pred = b0 + Z[split][:, :k] @ beta[:k]; return float(1 - ((s[split] - pred) ** 2).sum() / ((s[split] - s[split].mean()) ** 2).sum())


def auc_k(split, k, idx=None):
    cols = list(range(k)) if idx is None else idx
    return float(roc_auc_score(y[split], b0 + Z[split][:, cols] @ beta[cols]))


order = np.argsort(-np.abs(beta * np.sqrt(var_u)))
res = {
    "comprobacion_logit": chk,
    "var_Z_train_media": float(var_u.mean()),
    "r2_logit_k": {str(k): {"train": r2("train", k), "test": r2("test", k)} for k in (1, 2, 3, 6, 12, 64)},
    "auc_logit_completo": {"train": float(roc_auc_score(y["train"], s["train"])), "test": float(roc_auc_score(y["test"], s["test"]))},
    "auc_logit_reconstruido_12": {"train": auc_k("train", 12), "test": auc_k("test", 12)},
    "auc_logit_reconstruido_64": {"train": auc_k("train", 64), "test": auc_k("test", 64)},
    "u0": {"beta": float(beta[0]), "fraccion_varianza_logit": float(share[0]),
           "corr_pearson_con_s_test": float(np.corrcoef(Z["test"][:, 0], s["test"])[0, 1]),
           "spearman_con_s_test": float(spearmanr(Z["test"][:, 0], s["test"])[0])},
    "top_componentes": [{"j": int(j), "beta": float(beta[j]), "fraccion_varianza": float(share[j]),
                         "auc_test_componente": float(roc_auc_score(y["test"], Z["test"][:, j])),
                         "corr_con_s_test": float(np.corrcoef(Z["test"][:, j], s["test"])[0, 1])} for j in order[:8]],
    "suma_fraccion_primeras_12": float(share[:12].sum()),
}
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(res, ensure_ascii=False, indent=2))
