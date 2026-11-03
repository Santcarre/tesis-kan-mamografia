# Figura 4.3 (g(u0) por patología y calibración de p_sym) con rótulos en español.
# Uso: python scripts/figura_calibracion_formula.py . reports/figures/33_r3_g_calibration.png
# Mismo cálculo que scripts/compute_r3_coef_stability.py, con el evaluador de la fórmula de la estación.
import json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
REPO, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)
from src.reporting.formula_eval import g_function
REP = os.path.join(REPO, "reports", "experiments"); EMB = os.path.join(REPO, "artifacts", "output", "embeddings")
nz = np.load(os.path.join(REP, "kan_robust64_reducer.npz"))
X = np.load(os.path.join(EMB, "f2robust_test_feats.npy")).astype(np.float64)
y = np.load(os.path.join(EMB, "f2robust_test_labels.npy")).astype(int)
gz = (X - nz["gap_mu"]) / nz["gap_sd"]
z64 = (((gz @ nz["proj_w"].T + nz["proj_b"]) - nz["pca_mean"]) @ nz["pca_components"].T - nz["mu_z"]) / nz["sd_z"]
gf, tau = g_function(REPO)
g = np.array([gf(float(np.clip(u, -10, 10))) for u in z64[:, 0]])
p = 1 / (1 + np.exp(-g))
cal = json.load(open(os.path.join(REP, "r3_coef_stability.json"), encoding="utf-8"))["calibracion"]
# comprobación: los deciles recalculados deben coincidir con los guardados
qs = np.quantile(p, np.linspace(0, 1, 11))
pm = [round(float(p[(p >= lo) & (p <= hi)].mean()), 3) for lo, hi in zip(qs[:-1], qs[1:])]
assert max(abs(a - c["p_media"]) for a, c in zip(pm, cal)) < 2e-3, (pm, cal)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=pd.DataFrame({"Patología": np.where(y == 1, "Maligna", "Benigna"), "g": g}),
            x="Patología", y="g", hue="Patología", order=["Benigna", "Maligna"],
            palette=["#2ECC71", "#E74C3C"], ax=axes[0], legend=False)
axes[0].axhline(tau, color="k", ls="--", label="Umbral de Youden (validación)")
axes[0].set_title(r"$g(u_0)$ por patología (prueba)", fontweight="bold")
axes[0].set_xlabel("Patología"); axes[0].set_ylabel(r"$g(u_0)$")
axes[0].legend(); axes[0].grid(True, ls=":", alpha=0.5)
cc = pd.DataFrame(cal)
axes[1].plot(cc["p_media"], cc["tasa_maligna"], "o-", label="Observado")
axes[1].plot([0, 1], [0, 1], "k--", label="Calibración perfecta")
axes[1].set_title(r"Calibración de $p_{\mathrm{sym}}$ por deciles (prueba)", fontweight="bold")
axes[1].set_xlabel(r"$p_{\mathrm{sym}}$ medio del decil")
axes[1].set_ylabel("Fracción de lesiones malignas")
axes[1].legend(); axes[1].grid(True, ls=":", alpha=0.5)
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("guardado", OUT, "tau", tau, "n", len(y))
