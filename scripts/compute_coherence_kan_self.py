"""
Autocoherencia: LIME y KernelSHAP explicando a la propia formula KAN (R3).
Si la formula es autoconsistente, el surrogate local recupera dg/du y los
valores SHAP recuperan el credito analitico por coordenada.
Salida: reports/experiments/coherence_kan_self.json
"""
import json, os, sys, time
import numpy as np

CODIGOS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (CODIGOS, os.path.join(CODIGOS, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
from scipy.stats import spearmanr
from src.explainability.attribution_baselines import (
    lime_surrogate, kernel_shap, coherence_metrics)
from src.reporting.clinical_evidence_extractor import _resolve_embeddings_dir

SEED = 42
N_CASES = 200
LIME_SAMPLES = 3000
KS_SAMPLES = 600
KS_BACKGROUND = 32
OUT = os.path.join(CODIGOS, "reports", "experiments", "coherence_kan_self.json")

def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
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
    d_expr = sp.diff(e_mal - e_ben, syms[0])
    fn_d = sp.lambdify(syms, d_expr, "numpy")

    def to_z64(X):
        gz = (X - gap_mu) / gap_sd
        return (((gz @ proj_w.T + proj_b) - pm) @ pc.T - mu_z) / sd_z

    Ztr = to_z64(Xtr)
    Zte = to_z64(Xte[np.random.default_rng(SEED).choice(Xte.shape[0], size=min(N_CASES, Xte.shape[0]), replace=False)])
    bg = Ztr[rng.choice(Ztr.shape[0], size=200, replace=False)]
    mu_bg = bg.mean(axis=0)

    def g_vec(Z):
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim == 1:
            Z = Z[None, :]
        Xk = np.clip(Z[:, [i - 1 for i in used]], -10, 10)
        return np.array([float(fn_m(*row) - fn_b(*row)) for row in Xk])

    def grad_vec(Z):
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim == 1:
            Z = Z[None, :]
        Xk = np.clip(Z[:, [i - 1 for i in used]], -10, 10)
        g64 = np.zeros((Z.shape[0], 64))
        g64[:, used[0] - 1] = [float(fn_d(*row)) for row in Xk]
        return g64

    f_kan = lambda U: g_vec(U)
    r_lime_top1 = r_ks_top1 = 0
    rhos_lime, rhos_ks, r2_lime = [], [], []
    n = Zte.shape[0]
    for k in range(n):
        xi = Zte[k]
        ana = grad_vec(xi)[0]
        ana_shap = ana * (xi - mu_bg)
        lm = lime_surrogate(f_kan, xi, background=bg, n_samples=LIME_SAMPLES, seed=SEED + k)
        ks = kernel_shap(f_kan, xi, background=bg, n_samples=KS_SAMPLES,
                         n_background=KS_BACKGROUND, seed=SEED + k)
        r2_lime.append(float(lm["r2"]))
        rhos_lime.append(float(coherence_metrics(lm["coef"], ana)["spearman"]))
        rhos_ks.append(float(coherence_metrics(ks["shap"], ana_shap)["spearman"]))
        if int(np.argmax(np.abs(lm["coef"]))) == int(np.argmax(np.abs(ana))):
            r_lime_top1 += 1
        if int(np.argmax(np.abs(ks["shap"]))) == int(np.argmax(np.abs(ana_shap))):
            r_ks_top1 += 1
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{n}", flush=True)
    res = {"n": n, "seed": SEED, "nota": (
        "LIME y KernelSHAP explican a la propia g(z64) de la canonica R3; el credito "
        "analitico es dg/du por coordenada (disperso: solo u0). Top-1 = acuerdo en la "
        "variable dominante."),
        "lime_vs_analytic": {"rho_mean": float(np.mean(rhos_lime)), "rho_median": float(np.median(rhos_lime)),
                             "top1": f"{r_lime_top1}/{n}", "r2_surrogate_mean": float(np.mean(r2_lime))},
        "shap_vs_analytic": {"rho_mean": float(np.mean(rhos_ks)), "rho_median": float(np.median(rhos_ks)),
                             "top1": f"{r_ks_top1}/{n}"},
        "segundos": round(time.time() - t0, 1)}
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"LIME rho={res['lime_vs_analytic']['rho_mean']:.4f} top1={res['lime_vs_analytic']['top1']} R2={res['lime_vs_analytic']['r2_surrogate_mean']:.4f}")
    print(f"SHAP rho={res['shap_vs_analytic']['rho_mean']:.4f} top1={res['shap_vs_analytic']['top1']}")
    print(f"Guardado: {OUT} ({res['segundos']}s)")

if __name__ == "__main__":
    main()
