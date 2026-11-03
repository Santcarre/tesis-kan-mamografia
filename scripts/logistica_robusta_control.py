"""
Control lineal de la Tabla 4.1: regresion logistica (C=0.01) sobre los 512 descriptores
GAP del extractor ROBUSTO, el mismo del que se destila la formula.

Reemplaza a la logistica de recompute_auc_ci_baselines.py (descriptores del extractor base,
C=1, class_weight="balanced"), que no era comparable con la formula.

Uso (desde la raiz del repo):  python scripts/logistica_robusta_control.py [RAIZ] [SALIDA.json]
Produce: AUC, IC95 % por bootstrap de pacientes (B=2000, semilla 0, mismo esquema que
recalculos_auditoria.py), umbral de Youden en validacion, sensibilidad y especificidad en prueba,
DeLong y bootstrap pareado por paciente frente a la formula, y calibracion (Brier, ECE por
deciles y ECE uniforme de 10 intervalos). Tambien recalcula esas metricas para el baseline,
la red robusta y la formula como verificacion. Requiere numpy, pandas, scikit-learn, scipy.
"""
import sys, os, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = sys.argv[1] if len(sys.argv) > 1 else REPO
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "reports", "auditoria", "logistica_robusta_control.json")
EMB = os.path.join(ROOT, "artifacts", "output", "embeddings", "f2robust_")
PROB = os.path.join(ROOT, "reports", "experiments", "youden_val_probs")
SEED, B, C = 0, 2000, 0.01

X = {s: np.load(EMB + f"{s}_feats.npy").astype(np.float64) for s in ("train", "val", "test")}
y = {s: np.load(EMB + f"{s}_labels.npy").astype(int) for s in X}
mu, sd = X["train"].mean(0), X["train"].std(0) + 1e-8
F = {s: (X[s] - mu) / sd for s in X}
m = LogisticRegression(C=C, max_iter=5000).fit(F["train"], y["train"])
p_val, p_test = m.predict_proba(F["val"])[:, 1], m.predict_proba(F["test"])[:, 1]


def youden(yv, sv):
    f, t, th = roc_curve(yv, sv)
    return float(th[(t - f).argmax()])


def op(yt, s, th):
    p = s >= th
    return float((p & (yt == 1)).sum() / (yt == 1).sum()), float(((~p) & (yt == 0)).sum() / (yt == 0).sum())


def delong(yv, pa, pb):
    pos = yv == 1

    def comps(x, z):
        psi = (x[:, None] > z[None, :]) + 0.5 * (x[:, None] == z[None, :])
        return psi.mean(1), psi.mean(0)
    a10, a01 = comps(pa[pos], pa[~pos]); b10, b01 = comps(pb[pos], pb[~pos])
    mm, nn = pos.sum(), (~pos).sum()
    c10, c01 = np.cov(a10, b10, ddof=1), np.cov(a01, b01, ddof=1)
    va, vb = c10[0, 0] / mm + c01[0, 0] / nn, c10[1, 1] / mm + c01[1, 1] / nn
    cab = c10[0, 1] / mm + c01[0, 1] / nn
    vd = va + vb - 2 * cab; d = a10.mean() - b10.mean(); z = d / np.sqrt(vd)
    return dict(auc_a=round(float(a10.mean()), 4), auc_b=round(float(b10.mean()), 4), delta=round(float(d), 4),
                var_a=float(va), var_b=float(vb), cov_ab=float(cab), var_delta=float(vd), se_delta=round(float(np.sqrt(vd)), 5),
                ic95_delta=[round(float(d - 1.96 * np.sqrt(vd)), 4), round(float(d + 1.96 * np.sqrt(vd)), 4)],
                z=round(float(z), 3), p=round(float(2 * norm.sf(abs(z))), 4))


def calib(yv, p, n_bins=10):
    brier = float(np.mean((p - yv) ** 2))
    def ece(edges):
        b = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1); e = 0.0
        for k in range(n_bins):
            s = b == k
            if s.any(): e += s.mean() * abs(p[s].mean() - yv[s].mean())
        return float(e)
    return dict(brier=round(brier, 4), ece_deciles=round(ece(np.quantile(p, np.linspace(0, 1, n_bins + 1))), 4),
                ece_uniforme=round(ece(np.linspace(0, 1, n_bins + 1)), 4))


th = youden(y["val"], p_val); se, sp = op(y["test"], p_test, th)
res = {"modelo": f"LogisticRegression(C={C}) sobre GAP-512 estandarizado del extractor robusto (f2robust), sin pesos de clase",
       "semilla_bootstrap": SEED, "B": B, "n_test": int(len(y["test"])),
       "logistica": dict(auc_val=round(float(roc_auc_score(y["val"], p_val)), 4), auc_test=round(float(roc_auc_score(y["test"], p_test)), 4),
                         umbral_youden_prob=round(th, 4), sens_test=round(se, 4), spec_test=round(sp, 4), n_coef=int(m.coef_.size + 1))}

man = pd.read_csv(os.path.join(ROOT, "src", "data", "processed", "manifest_test.csv"))
yt = np.load(os.path.join(PROB, "Baseline_FC_test_labels.npy")).astype(int)
assert (yt == man.label.values).all() and (yt == y["test"]).all()
P = {k: np.load(os.path.join(PROB, f"{k}_test_probs.npy")).ravel().astype(float) for k in ("Baseline_FC", "Robusto", "R3_formula")}
P["Logistica_robusta_C0.01"] = p_test

res["delong_frente_a_formula"] = {k: delong(yt, P[k], P["R3_formula"]) for k in ("Logistica_robusta_C0.01", "Robusto", "Baseline_FC")}
res["calibracion"] = {k: calib(yt, P[k]) for k in P}

up, inv = np.unique(man.patient_id.values, return_inverse=True); idx = [np.where(inv == i)[0] for i in range(len(up))]
rng = np.random.default_rng(SEED); acc = {k: [] for k in P}; dif = {k: [] for k in P if k != "R3_formula"}
for _ in range(B):
    ii = np.concatenate([idx[j] for j in rng.integers(0, len(up), len(up))]); yy = yt[ii]
    if yy.min() == yy.max():
        continue
    a = {k: roc_auc_score(yy, P[k][ii]) for k in P}
    for k in P: acc[k].append(a[k])
    for k in dif: dif[k].append(a[k] - a["R3_formula"])
pct = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
res["bootstrap_por_paciente"] = {
    "n_pacientes": int(len(up)),
    "auc_ic95": {k: pct(v) for k, v in acc.items()},
    "delta_menos_formula": {k: dict(media=round(float(np.mean(v)), 4), ic95=pct(v),
                                    p_bilateral=round(float(2 * min((np.array(v) <= 0).mean(), (np.array(v) >= 0).mean())), 4))
                            for k, v in dif.items()}}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(res, ensure_ascii=False, indent=2))
