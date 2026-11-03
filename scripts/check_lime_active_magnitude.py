"""Magnitud activa entre casos: coef_LIME[u0] vs dg/du0 analitico (200 casos)."""
import json, os, sys
import numpy as np
CODIGOS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (CODIGOS, os.path.join(CODIGOS, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
from scipy.stats import pearsonr
from src.explainability.attribution_baselines import lime_surrogate
from src.reporting.clinical_evidence_extractor import _resolve_embeddings_dir

SEED = 42; N = 200; LIME_SAMPLES = 3000
emb = _resolve_embeddings_dir()
Xtr = np.load(os.path.join(emb, "f2robust_train_feats.npy")).astype(np.float64)
Xte = np.load(os.path.join(emb, "f2robust_test_feats.npy")).astype(np.float64)
nz = np.load(os.path.join(CODIGOS, "reports", "experiments", "kan_robust64_reducer.npz"))
cj = json.load(open(os.path.join(CODIGOS, "reports", "experiments", "kan_robust64_canonical.json"), encoding="utf-8"))
gap_mu, gap_sd = nz["gap_mu"].astype(np.float64), nz["gap_sd"].astype(np.float64)
proj_w, proj_b = nz["proj_w"].astype(np.float64), nz["proj_b"].astype(np.float64)
pc, pm = nz["pca_components"].astype(np.float64), nz["pca_mean"].astype(np.float64)
mu_z, sd_z = nz["mu_z"].astype(np.float64), nz["sd_z"].astype(np.float64)
import sympy as sp, re
e_ben, e_mal = sp.sympify(cj["fben"]), sp.sympify(cj["fmal"])
used = sorted({int(x) for x in re.findall(r"x_(\d+)", cj["fmal"] + cj["fben"])})
syms = [sp.Symbol(f"x_{i}") for i in used]
fn_b, fn_m = sp.lambdify(syms, e_ben, "numpy"), sp.lambdify(syms, e_mal, "numpy")
fn_d = sp.lambdify(syms, sp.diff(e_mal - e_ben, syms[0]), "numpy")

def to_z64(X):
    gz = (X - gap_mu) / gap_sd
    return (((gz @ proj_w.T + proj_b) - pm) @ pc.T - mu_z) / sd_z

rng = np.random.default_rng(SEED)
Zte = to_z64(Xte[rng.choice(Xte.shape[0], size=N, replace=False)])
Ztr = to_z64(Xtr)
bg = Ztr[rng.choice(Ztr.shape[0], size=200, replace=False)]
def g_vec(Z):
    Z = np.asarray(Z, float)
    if Z.ndim == 1: Z = Z[None, :]
    Xk = np.clip(Z[:, [i - 1 for i in used]], -10, 10)
    return np.array([float(fn_m(*r) - fn_b(*r)) for r in Xk])
ana_mag, lime_mag, null_noise = [], [], []
for k in range(N):
    xi = Zte[k]
    s = float(np.clip(xi[used[0] - 1], -10, 10))
    ana_mag.append(float(fn_d(s)))
    lm = lime_surrogate(g_vec, xi, background=bg, n_samples=LIME_SAMPLES, seed=SEED + k)
    lime_mag.append(float(lm["coef"][used[0] - 1]))
    null = np.delete(np.abs(lm["coef"]), used[0] - 1)
    null_noise.append(float(null.mean()))
ana_mag, lime_mag = np.array(ana_mag), np.array(lime_mag)
r = float(pearsonr(ana_mag, lime_mag)[0])
print(f"Pearson coef_LIME[u0] vs dg/du0 entre {N} casos: r={r:.4f}")
print(f"magnitud media |coef| en 63 dims nulas (piso de ruido LIME): {np.mean(null_noise):.4f}")
print(f"magnitud media |coef| en u0: {np.mean(np.abs(lime_mag)):.4f}")
json.dump({"pearson_activo": r, "ruido_nulo_medio": float(np.mean(null_noise)),
           "activo_medio": float(np.mean(np.abs(lime_mag)))},
          open(os.path.join(CODIGOS, "reports", "experiments", "coherence_kan_self_active.json"), "w"), indent=2)
