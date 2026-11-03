# Figura 4.1 (curvas del extractor robusto) con rótulos en español. Lee el historial del entrenamiento.
# Uso: python scripts/figura_curvas_extractor.py reports/experiments reports/figures/26_resnet18_robust_training_curves.png
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
REP, OUT = sys.argv[1], sys.argv[2]
h = json.load(open(os.path.join(REP, "26_resnet18_robust.json"), encoding="utf-8"))
hist = h["history"]
ep = [r["epoch"] for r in hist]; loss = [r["loss"] for r in hist]; auc = [r["val_auc"] for r in hist]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(ep, loss, "-o", color="#DC143C", label="Pérdida ponderada (entrenamiento)")
axes[0].axvline(5.5, color="gray", ls="--", label="Inicio del ajuste fino (layer3 y layer4)")
axes[0].set_title("A) Pérdida de entrenamiento", fontweight="bold")
axes[0].set_xlabel("Época"); axes[0].set_ylabel("Pérdida")
axes[0].legend(); axes[0].grid(True, ls="--", alpha=0.5)
axes[1].plot(ep, auc, "-s", color="#4169E1", label="AUC en validación")
axes[1].axhline(h["best_val_auc"], color="green", ls=":", label=f"Mejor AUC en validación = {h['best_val_auc']:.4f}")
axes[1].axhline(h["auc_test"], color="darkorange", ls="-.", label=f"AUC en prueba = {h['auc_test']:.4f}")
axes[1].axvline(5.5, color="gray", ls="--")
axes[1].set_title("B) AUC por época", fontweight="bold")
axes[1].set_xlabel("Época"); axes[1].set_ylabel("AUC")
axes[1].legend(); axes[1].grid(True, ls="--", alpha=0.5)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("guardado", OUT)
