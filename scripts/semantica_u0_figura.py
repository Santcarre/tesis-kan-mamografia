"""
Figura de la Sección BI-RADS del Cap. 4: distribución de u0 según la patología, el margen y la forma
de la masa, con el MISMO vínculo por imagen que semantica_u0_por_imagen.py (reutiliza construir_tabla).

Sustituye la figura 33_r3_semantics_boxplots.png de compute_r3_semantics.py (versión anterior, no incluida en este repositorio), que asignaba los
descriptores por paciente.

Uso (desde la raíz del repositorio):
  python scripts/semantica_u0_figura.py [CSV_DIR] [SALIDA.png]
SALIDA por defecto: Figuras/33_u0_birads_boxplots.png
Requiere numpy, pandas, scipy, matplotlib y seaborn.
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from semantica_u0_por_imagen import construir_tabla, CSV_DIR, REPO, MG, SG

csv_dir = sys.argv[1] if len(sys.argv) > 1 else CSV_DIR
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "Figuras", "33_u0_birads_boxplots.png")

m, _, _, _, _ = construir_tabla(csv_dir)
ES_M = {"CIRCUMSCRIBED": "Circunscrito", "ILL_DEFINED": "Mal definido", "SPICULATED": "Espiculado"}
ES_S = {"ROUND": "Redonda", "OVAL": "Ovalada", "IRREGULAR": "Irregular"}

pat = m.assign(cat=m["label"].map({0: "Benigna", 1: "Maligna"}))
mar = m[m["margen"].isin(MG)].assign(cat=lambda d: d["margen"].map(ES_M))
sha = m[m["forma"].isin(SG)].assign(cat=lambda d: d["forma"].map(ES_S))

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
plt.subplots_adjust(wspace=0.28)
paneles = [
    (pat, ["Benigna", "Maligna"], ["#2ECC71", "#E74C3C"], r"$u_0$ frente a la patología"),
    (mar, [ES_M[g] for g in MG], ["#2ECC71", "#F39C12", "#E74C3C"], r"$u_0$ frente al margen de la masa"),
    (sha, [ES_S[g] for g in SG], ["#2ECC71", "#3498DB", "#E74C3C"], r"$u_0$ frente a la forma de la masa"),
]
for ax, (d, orden, colores, titulo) in zip(axes, paneles):
    sns.boxplot(data=d, x="cat", y="u0", hue="cat", order=orden, hue_order=orden,
                palette=colores, ax=ax, legend=False)
    n = d["cat"].value_counts()
    ax.set_xticks(range(len(orden)))
    ax.set_xticklabels([f"{c}\n(n = {int(n.get(c, 0))})" for c in orden], fontsize=11)
    ax.set_title(titulo, fontsize=13, fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel(r"$u_0$", fontsize=12)
    ax.grid(True, ls=":", alpha=0.5)

os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")
print("Guardado:", out)
