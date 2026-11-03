"""
Recalculos baratos de la revision de octubre 2026 (sin reentrenar).
Uso (desde la raiz del repo):  python scripts/recalculos_auditoria.py [RAIZ] [SALIDA.json]
RAIZ = raiz del repositorio (por defecto, la carpeta padre de scripts/).
Produce: logistica sobre u0 vs formula, sensibilidad a especificidad fija, pAUC y
bootstrap agrupado por paciente (B=2000, semilla 0). Requiere numpy, pandas, scikit-learn, scipy.
"""
import sys, os, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = sys.argv[1] if len(sys.argv) > 1 else REPO   # raiz del repositorio
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "reports", "auditoria", "recalculos_auditoria.json")
COD = ROOT; EMB = os.path.join(ROOT, "artifacts", "output", "embeddings", "f2robust_")
EXP = os.path.join(COD, "reports", "experiments"); PROB = os.path.join(EXP, "youden_val_probs")
SEED = 0; B = 2000
X = {s: np.load(EMB + f"{s}_feats.npy").astype(np.float64) for s in ("train", "val", "test")}
y = {s: np.load(EMB + f"{s}_labels.npy").astype(int) for s in X}
r = np.load(os.path.join(EXP, "kan_robust64_reducer.npz"))

def z64(Xs):
    g = (Xs - r["gap_mu"]) / r["gap_sd"]; z = g @ r["proj_w"].T + r["proj_b"]
    return ((z - r["pca_mean"]) @ r["pca_components"].T - r["mu_z"]) / r["sd_z"]
Z = {s: z64(X[s]) for s in X}; u0 = {s: Z[s][:, 0] for s in X}

def youden(yv, sv):
    f, t, th = roc_curve(yv, sv); return th[(t - f).argmax()]
def op(yt, s, th):
    p = s >= th
    return float((p & (yt == 1)).sum() / (yt == 1).sum()), float(((~p) & (yt == 0)).sum() / (yt == 0).sum())

res = {"semilla_bootstrap": SEED, "B": B, "n_test": int(len(y["test"]))}
res["auc_u0"] = {s: round(float(roc_auc_score(y[s], u0[s])), 4) for s in X}   # <0.5: u0 menor = mas maligno
log = {}
def fit(name, Ftr, Fva, Fte, C=1.0):
    m = LogisticRegression(C=C, max_iter=5000).fit(Ftr, y["train"])
    sv, st = m.decision_function(Fva), m.decision_function(Fte)
    th = youden(y["val"], sv); se, sp = op(y["test"], st, th)
    log[name] = dict(auc_val=round(float(roc_auc_score(y["val"], sv)), 4), auc_test=round(float(roc_auc_score(y["test"], st)), 4),
                     sens_test=round(se, 4), spec_test=round(sp, 4), umbral="Youden en validacion")
    return st
st_u0 = fit("logistica_u0", u0["train"][:, None], u0["val"][:, None], u0["test"][:, None])
fit("logistica_z64", Z["train"], Z["val"], Z["test"])
fit("logistica_u0_u1", Z["train"][:, :2], Z["val"][:, :2], Z["test"][:, :2])
mu, sd = X["train"].mean(0), X["train"].std(0) + 1e-8
for C in (1.0, 0.01):
    fit(f"logistica_512_C{C}", (X["train"] - mu) / sd, (X["val"] - mu) / sd, (X["test"] - mu) / sd, C=C)
res["logistica"] = log

s_te = -u0["test"]; th = youden(y["val"], -u0["val"]); se, sp = op(y["test"], s_te, th)
res["formula_equivalente_u0"] = dict(sens=round(se, 4), spec=round(sp, 4), J=round(se + sp - 1, 4))
f, t, _ = roc_curve(y["test"], s_te)
res["sens_a_espec_fija_test"] = {str(k): round(float(np.interp(1 - k, f, t)), 4) for k in (0.6, 0.663, 0.8, 0.9)}
res["pAUC_estandarizado_FPR<=0.2"] = round(float(roc_auc_score(y["test"], s_te, max_fpr=0.2)), 4)
p = np.load(os.path.join(PROB, "R3_formula_test_probs.npy")).ravel()
res["spearman_formula_vs_menos_u0"] = round(float(spearmanr(p, s_te)[0]), 4)
res["auc_formula_test"] = round(float(roc_auc_score(y["test"], p)), 4)

# bootstrap agrupado por paciente
m = pd.read_csv(os.path.join(COD, "src", "data", "processed", "manifest_test.csv"))
yt = np.load(os.path.join(PROB, "Baseline_FC_test_labels.npy"))
assert (yt == m.label.values).all()
P = {k: np.load(os.path.join(PROB, f"{k}_test_probs.npy")).ravel() for k in ("Baseline_FC", "Robusto", "R3_formula")}
up, inv = np.unique(m.patient_id.values, return_inverse=True); idx = [np.where(inv == i)[0] for i in range(len(up))]
rng = np.random.default_rng(SEED); acc = {k: [] for k in P}; dif = []
for _ in range(B):
    ii = np.concatenate([idx[j] for j in rng.integers(0, len(up), len(up))]); yy = yt[ii]
    if yy.min() == yy.max(): continue
    a = {k: roc_auc_score(yy, P[k][ii]) for k in P}
    for k in P: acc[k].append(a[k])
    dif.append(a["Baseline_FC"] - a["R3_formula"])
dif = np.array(dif)
res["bootstrap_por_paciente"] = {"n_pacientes": int(len(up)),
    "auc_ic95": {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for k, v in acc.items()},
    "auc_punto": {k: round(float(roc_auc_score(yt, P[k])), 4) for k in P},
    "delta_Baseline_menos_R3": dict(media=round(float(dif.mean()), 4), ic95=[round(float(np.percentile(dif, 2.5)), 4), round(float(np.percentile(dif, 97.5)), 4)],
                                    p_bilateral=round(float(2 * min((dif <= 0).mean(), (dif >= 0).mean())), 4))}
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2); print(json.dumps(res, ensure_ascii=False, indent=2))
