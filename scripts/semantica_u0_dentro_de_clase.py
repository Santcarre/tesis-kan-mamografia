"""
Asociacion de u0 con el margen y la forma de la masa DENTRO de cada clase (Cap. 4, Sec. BI-RADS).

semantica_u0_por_imagen.py encuentra que u0 varia con el margen y con la forma de la masa. Como los
margenes espiculados y las formas irregulares son mas frecuentes en lesiones malignas, parte de esa
asociacion puede reflejar solo la malignidad. Aqui se repiten las pruebas por separado en las imagenes
malignas y en las benignas, con el mismo vinculo por imagen (construir_tabla), y se ajusta ademas un
modelo lineal sobre los rangos de u0 (rango ~ patologia + margen) para ver si el margen anade algo a la
patologia (prueba F de modelos anidados).

Pruebas por clase: Kruskal-Wallis de u0 ~ margen {CIRCUMSCRIBED, ILL_DEFINED, SPICULATED} y u0 ~ forma
{ROUND, OVAL, IRREGULAR}, con epsilon^2 = H / (n - 1). Holm sobre las 4 pruebas por clase y las 2 de
modelos anidados (familia de 6). Las pruebas son a nivel de imagen (318 pacientes; varias imagenes por
paciente), como en el resto del capitulo. Como sensibilidad, las pruebas por clase se repiten con una
sola observacion por paciente y descriptor (la media de u0 de sus imagenes), fuera de la familia de Holm.

Uso (desde la raiz del repositorio):
  python scripts/semantica_u0_dentro_de_clase.py [CSV_DIR] [SALIDA.json]
Requiere numpy, pandas, scipy.
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import semantica_u0_por_imagen as base  # noqa: E402

REPO = base.REPO
CSV_DIR = sys.argv[1] if len(sys.argv) > 1 else base.CSV_DIR
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "reports", "auditoria", "semantica_u0_dentro_de_clase.json")


def kw(sub, col, cats):
    g = [sub.loc[sub[col] == c, "u0"].values for c in cats]
    g_ok = [x for x in g if len(x) > 0]
    h = stats.kruskal(*g_ok); n = int(sum(len(x) for x in g_ok))
    return {"H": round(float(h.statistic), 4), "p_raw": float(h.pvalue), "n": n,
            "n_por_grupo": dict(zip(cats, [int(len(x)) for x in g])),
            "mediana_u0_por_grupo": dict(zip(cats, [round(float(np.median(x)), 4) if len(x) else None for x in g])),
            "epsilon2": round(float(h.statistic) / (n - 1), 4)}


def anidado(sub, col, cats):
    """F de rango(u0) ~ patologia + col frente a rango(u0) ~ patologia (OLS sobre rangos)."""
    s = sub[sub[col].isin(cats)].copy()
    r = stats.rankdata(s["u0"].values)
    X0 = np.column_stack([np.ones(len(s)), s["label"].values])
    D = np.column_stack([(s[col] == c).astype(float).values for c in cats[1:]])
    X1 = np.column_stack([X0, D])
    rss = lambda X: float(np.sum((r - X @ np.linalg.lstsq(X, r, rcond=None)[0]) ** 2))
    r0, r1 = rss(X0), rss(X1)
    df1, df2 = X1.shape[1] - X0.shape[1], len(s) - X1.shape[1]
    F = ((r0 - r1) / df1) / (r1 / df2)
    return {"F": round(F, 4), "gl": [int(df1), int(df2)], "p_raw": float(stats.f.sf(F, df1, df2)), "n": int(len(s)),
            "R2_parcial": round((r0 - r1) / r0, 4)}


def main():
    m, _, _, _, _ = base.construir_tabla(CSV_DIR)
    tests = {}
    for clase, lab in (("malignas", 1), ("benignas", 0)):
        sub = m[m["label"] == lab]
        tests[f"margen_{clase}"] = kw(sub, "margen", base.MG)
        tests[f"forma_{clase}"] = kw(sub, "forma", base.SG)
    tests["margen_ajustado_por_patologia"] = anidado(m, "margen", base.MG)
    tests["forma_ajustada_por_patologia"] = anidado(m, "forma", base.SG)
    sens = {}
    for col, cats in (("margen", base.MG), ("forma", base.SG)):
        for clase, lab in (("malignas", 1), ("benignas", 0)):
            sub = m[(m["label"] == lab) & m[col].isin(cats)]
            agg = sub.groupby(["patient_id", col], as_index=False)["u0"].mean()
            sens[f"{col}_{clase}_por_paciente"] = kw(agg, col, cats)
    names = list(tests); adj = base.holm(np.array([tests[k]["p_raw"] for k in names]))
    for k, a in zip(names, adj):
        tests[k]["p_holm"] = float(a)
    out = {"descripcion": "u0 frente a margen y forma dentro de cada clase (ver docstring)",
           "n_imagenes": int(len(m)), "holm_familia": names, "tests": tests,
           "sensibilidad_una_observacion_por_paciente": sens}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    for k, v in {**tests, **sens}.items():
        print(k, {kk: vv for kk, vv in v.items() if kk not in ("n_por_grupo",)})


if __name__ == "__main__":
    main()
