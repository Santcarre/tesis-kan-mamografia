# Figura 4.4 (S-PCA frente a PCA y al extractor base) con rótulos en español; datos de reports/27_spca_robust_comparison.json.
# Uso: python scripts/figura_spca_robusto.py reports/27_spca_robust_comparison.json reports/figures/27_spca_robust_comparison.png
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ENT, OUT = sys.argv[1], sys.argv[2]
data = json.load(open(ENT, encoding="utf-8"))
roc = data["roc_curves"]; results = {r["experiment"]: r for r in data["results"]}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5), dpi=300)
cfg = [("ResNet-18 Baseline (f2base) + S-PCA", "ResNet-18 base + S-PCA", "#7F7F7F", "--", 2.2),
       ("ResNet-18 Robusto (f2robust) + PCA Ciego", "ResNet-18 robusto + PCA", "#DC143C", ":", 2.2),
       ("ResNet-18 Robusto (f2robust) + S-PCA", "ResNet-18 robusto + S-PCA (RS-KAN)", "#4169E1", "-", 2.5)]
for key, lab, col, ls, lw in cfg:
    c = roc[key]
    ax1.plot(c["fpr"], c["tpr"], color=col, linestyle=ls, linewidth=lw, label=f"{lab} (AUC = {c['auc']:.4f})")
ax1.plot([0, 1], [0, 1], "--", color="#B0B0B0", lw=1.2)
ax1.set_title("A) Curvas ROC de la fórmula (semilla 42)", fontweight="bold", fontsize=11.5)
ax1.set_xlabel("1 − especificidad", fontsize=10.5); ax1.set_ylabel("Sensibilidad", fontsize=10.5)
ax1.grid(True, ls="--", alpha=0.45); ax1.legend(loc="lower right", fontsize=9.2)
labs = ["ResNet-18 base\n+ S-PCA", "ResNet-18 robusto\n+ PCA", "ResNet-18 robusto\n+ S-PCA (RS-KAN)"]
means = [results[k[0]]["auc_d_mean"] for k in cfg]; stds = [results[k[0]]["auc_d_std"] for k in cfg]
x = np.arange(3)
bars = ax2.bar(x, means, yerr=stds, capsize=5, color=["#8C8C8C", "#E03556", "#587CE8"], edgecolor="black", linewidth=1.0, width=0.62)
ax2.set_xticks(x); ax2.set_xticklabels(labs, fontsize=9.8); ax2.set_ylim(0.55, 0.80)
ax2.set_ylabel("AUC medio en prueba (5 semillas)", fontsize=10.5)
ax2.set_title("B) AUC medio y desviación entre semillas", fontweight="bold", fontsize=11.5)
ax2.grid(True, axis="y", ls="--", alpha=0.45)
for b, m, s in zip(bars, means, stds):
    ax2.text(b.get_x() + b.get_width() / 2, m + s + 0.012, f"{m:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10.5)
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.12)
print("guardado", OUT, [round(m, 4) for m in means])
