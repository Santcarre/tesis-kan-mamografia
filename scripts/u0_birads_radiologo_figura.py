"""
Figura de la Sección «Asociación con la valoración del radiólogo» (Cap. 4).

Panel izquierdo: p_sym según la categoría BI-RADS del radiólogo (2-5), separado por patología, con los
cortes de la categoría orientativa de la herramienta. Panel derecho: proporción de lesiones malignas por
categoría, del radiólogo y de la herramienta (IC de Wilson), en las mismas imágenes de prueba.

Lee reports/auditoria/u0_birads_radiologo_prueba.csv (de scripts/u0_birads_radiologo.py).
Uso: python scripts/u0_birads_radiologo_figura.py [SALIDA.png]
"""
import math, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO = os.environ.get("TESIS_REPO") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.reporting import birads_herramienta as BH  # noqa: E402

ENT = os.path.join(REPO, "reports", "auditoria", "u0_birads_radiologo_prueba.csv")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "Figuras", "33_u0_birads_radiologo.png")
TAU = 0.1296
P_TAU = 1 / (1 + math.exp(-TAU))


def wilson(k, n, z=1.96):
    ph = k / n; d = 1 + z * z / n; c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


m = pd.read_csv(ENT)
m = m[m["birads_rad"].between(2, 5)].copy()
m["Patología"] = m["y"].map({0: "Benigna", 1: "Maligna"})
m["herr"] = [BH.categoria_desde_p(p, P_TAU) for p in m["p"]]

fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
plt.subplots_adjust(wspace=0.22)

ax = axes[0]
orden = [2, 3, 4, 5]
sns.boxplot(data=m, x="birads_rad", y="p", hue="Patología", order=orden, hue_order=["Benigna", "Maligna"],
            palette=["#2ECC71", "#E74C3C"], ax=ax, fliersize=3)
cnt = m.groupby(["birads_rad", "y"]).size()
ax.set_xticks(range(4))
ax.set_xticklabels([f"BI-RADS {c}\n{int(cnt.get((c, 0), 0))} benignas\n{int(cnt.get((c, 1), 0))} malignas" for c in orden], fontsize=10)
for y, et in ((BH.CORTE_2_3, "2 | 3"), (P_TAU, "3 | 4 (umbral)"), (BH.CORTE_4_5, "4 | 5")):
    ax.axhline(y, color="#555555", ls="--", lw=1)
    ax.text(3.5, y + 0.012, et, va="bottom", ha="left", fontsize=9.5, color="#333333")
ax.set_xlim(-0.6, 4.4)
ax.set_xlabel("Categoría asignada por el radiólogo (CBIS-DDSM)", fontsize=11.5)
ax.set_ylabel(r"$p_{\mathrm{sym}} = \sigma(g(u_0))$", fontsize=12)
ax.set_title(r"Puntaje de la fórmula según la categoría del radiólogo", fontsize=13, fontweight="bold")
ax.legend(title=None, loc="lower right", fontsize=10)  # arriba a la izquierda tapaba la caja maligna de BI-RADS 2
ax.grid(True, ls=":", alpha=0.5)

ax = axes[1]
ancho = 0.38
for j, (col, nombre, color) in enumerate((("birads_rad", "Radiólogo", "#5D6D7E"), ("herr", "Herramienta", "#3498DB"))):
    xs, ys, lo, hi, ns = [], [], [], [], []
    for i, c in enumerate(orden):
        s = m[m[col] == c]
        k, nn = int(s["y"].sum()), len(s)
        a, b = wilson(k, nn)
        xs.append(i + (j - 0.5) * ancho); ys.append(k / nn); lo.append(k / nn - a); hi.append(b - k / nn); ns.append(nn)
    ax.bar(xs, ys, width=ancho, color=color, label=nombre, yerr=[lo, hi], capsize=3, error_kw={"lw": 1})
    for x, y, h, nn in zip(xs, ys, hi, ns):
        ax.text(x, y + h + 0.015, f"{nn}", ha="center", va="bottom", fontsize=8.5, color="#333333")
ax.set_xticks(range(4)); ax.set_xticklabels([f"BI-RADS {c}" for c in orden], fontsize=11)
ax.set_ylim(0, 1.08)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_ylabel("Lesiones malignas", fontsize=12)
ax.set_title("Malignidad por categoría en las mismas imágenes", fontsize=13, fontweight="bold")
ax.text(0.99, 0.97, "sobre cada barra: n de imágenes", transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#555555")
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, axis="y", ls=":", alpha=0.5)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("Guardado:", OUT)
