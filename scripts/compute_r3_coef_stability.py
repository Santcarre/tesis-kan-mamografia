"""
Fase 2: estabilidad de coeficientes/forma (inter-semilla e inter-backbone) +
valores acompañantes (g/p_sym por patologia, calibracion) de la canonica.
Salida: reports/experiments/r3_coef_stability.json + reports/figures/33_r3_g_calibration.png
"""
import json, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

CODIGOS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP_DIR = os.environ.get("TESIS_REP") or os.path.join(CODIGOS, "artifacts")
EMB_DIR = os.path.join(REP_DIR, "output", "embeddings")
REPORTS_DIR = os.path.join(CODIGOS, "reports", "experiments")
FIGS_DIR = os.path.join(CODIGOS, "reports", "figures")

def hist(form):
    return {"sin": form.count("sin("), "cos": form.count("cos("), "exp": form.count("exp("),
            "tanh": form.count("tanh("), "log": form.count("log("),
            "n_terms": form.count("+") + form.count("-"),
            "vars": sorted(set(re.findall(r"x_(\d+)", form)))}

def main():
    multi = json.load(open(os.path.join(REPORTS_DIR, "r3_spca64_multiseed.json"), encoding="utf-8"))
    rows = []
    for r in multi:
        if "error" in r:
            continue
        h = hist(r["fmal"] + r["fben"])
        rows.append({"backbone": r["backbone"], "seed": r["seed"], "AUC_D": r["AUC_D"],
                     "n_vars": r["n_vars"], "chars": r["chars"], **{f"mal_{k}": v for k, v in hist(r['fmal']).items()},
                     "fam_mal": max(["sin", "cos", "exp", "tanh", "log", "lin"],
                                    key=lambda k: hist(r["fmal"]).get(k, 0) if k != "lin" else (0 if re.search(r"sin|cos|exp|tanh|log", r["fmal"]) else 1))})
    df = pd.DataFrame(rows)
    uni = float((df["n_vars"] == 1).mean())
    print(f"univariables: {uni*100:.1f}% ({int((df['n_vars']==1).sum())}/{len(df)})")
    print(df.groupby("backbone")[["AUC_D", "n_vars", "chars"]].agg(["mean", "std", "min", "max"]).round(4).to_string())
    fam = df["fam_mal"].value_counts().to_dict()
    print("familia dominante pierna maligna:", fam)

    # valores acompañantes: g/p_sym por patologia + calibracion (canonica, test)
    import sympy as sp
    cj = json.load(open(os.path.join(REPORTS_DIR, "kan_robust64_canonical.json"), encoding="utf-8"))
    nz = np.load(os.path.join(REPORTS_DIR, "kan_robust64_reducer.npz"))
    Xte = np.load(os.path.join(EMB_DIR, "f2robust_test_feats.npy")).astype(np.float64)
    yte = np.load(os.path.join(EMB_DIR, "f2robust_test_labels.npy")).astype(int)
    gz = (Xte - nz["gap_mu"]) / nz["gap_sd"]
    z64 = (((gz @ nz["proj_w"].T + nz["proj_b"]) - nz["pca_mean"]) @ nz["pca_components"].T - nz["mu_z"]) / nz["sd_z"]
    e_ben, e_mal = sp.sympify(cj["fben"]), sp.sympify(cj["fmal"])
    used = sorted({int(x) for x in re.findall(r"x_(\d+)", cj["fmal"] + cj["fben"])})
    syms = [sp.Symbol(f"x_{i}") for i in used]
    fn_b, fn_m = sp.lambdify(syms, e_ben, "numpy"), sp.lambdify(syms, e_mal, "numpy")
    Xk = np.clip(z64[:, [i - 1 for i in used]], -10, 10)
    g = np.array([float(fn_m(*row) - fn_b(*row)) for row in Xk])
    p = 1.0 / (1.0 + np.exp(-g))
    mw = stats.mannwhitneyu(g[yte == 1], g[yte == 0], alternative="two-sided")
    qs = np.quantile(p, np.linspace(0, 1, 11))
    cal = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (p >= lo) & (p <= hi)
        if m.sum() > 0:
            cal.append({"p_media": round(float(p[m].mean()), 3), "tasa_maligna": round(float(yte[m].mean()), 3), "n": int(m.sum())})
    out = {"univariable_frac": round(uni, 3), "familia_mal": fam,
           "auc_d_por_backbone": df.groupby("backbone")["AUC_D"].agg(["mean", "std"]).round(4).to_dict(),
           "g_mw": {"U": float(mw.statistic), "p": float(mw.pvalue),
                    "g_mal": round(float(g[yte == 1].mean()), 3), "g_ben": round(float(g[yte == 0].mean()), 3)},
           "th_youden_val": cj.get("th_youden_val"), "calibracion": cal}
    json.dump(out, open(os.path.join(REPORTS_DIR, "r3_coef_stability.json"), "w"), indent=2)
    print(f"g: MW p={mw.pvalue:.3g} mal={g[yte==1].mean():+.3f} ben={g[yte==0].mean():+.3f} th_val={cj.get('th_youden_val')}")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=pd.DataFrame({"Patología": np.where(yte == 1, "Maligno", "Benigno"), "g": g}),
                x="Patología", y="g", hue="Patología", order=["Benigno", "Maligno"],
                palette=["#2ECC71", "#E74C3C"], ax=axes[0], legend=False)
    axes[0].axhline(cj.get("th_youden_val", 0), color="k", ls="--", label="Umbral de Youden (val)")
    axes[0].set_title(r"$g(u_0)$ canónica por patología (prueba)", fontweight="bold")
    axes[0].set_xlabel("Patología")
    axes[0].set_ylabel(r"Score analítico $g(u_0)$")
    axes[0].legend(); axes[0].grid(True, ls=":", alpha=0.5)
    cc = pd.DataFrame(cal)
    axes[1].plot(cc["p_media"], cc["tasa_maligna"], "o-", label="Observado")
    axes[1].plot([0, 1], [0, 1], "k--", label="Ideal")
    axes[1].set_title(r"Calibración de $p_{\mathrm{sym}}$ (deciles)", fontweight="bold")
    axes[1].set_xlabel(r"Probabilidad predicha ($p_{\mathrm{sym}}$)")
    axes[1].set_ylabel("Frecuencia empírica observada")
    axes[1].legend(); axes[1].grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    for d in (FIGS_DIR,):
        os.makedirs(d, exist_ok=True)
        if os.path.exists(d):
            fig.savefig(os.path.join(d, "33_r3_g_calibration.png"), dpi=300, bbox_inches="tight")
    print("Guardado: r3_coef_stability.json + 33_r3_g_calibration.png")

if __name__ == "__main__":
    main()
