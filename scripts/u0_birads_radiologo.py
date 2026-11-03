"""
u0 y p_sym frente a la categoría BI-RADS del radiólogo (columna 'assessment' de CBIS-DDSM).

Pregunta: ¿la puntuación de la fórmula se relaciona con la valoración BI-RADS del radiólogo, y
qué cortes sobre p_sym serían defendibles para la "categoría de la herramienta"?

Vínculo imagen -> lesión: el mismo de semantica_u0_por_imagen.py (manifest -> dicom_info -> PatientID
'Mass-Test_P_00038_LEFT_CC[_k]' -> filas de *_case_description_*.csv). Validación sale del conjunto
de entrenamiento de CBIS (CSV *_train_set) y prueba del de prueba (*_test_set). Un recorte se vincula
con su lesión; una mamografía completa con todas las lesiones de esa proyección. Si estas tienen
valoraciones distintas, la imagen es ambigua: se excluye (análisis principal) o se toma la máxima
(análisis de sensibilidad, la valoración de la mama).

Análisis (validación y prueba; IC 95 % por bootstrap de pacientes, 2000 réplicas):
  1. Distribución de la valoración del radiólogo, tasa de malignidad y mediana de u0 y p_sym por categoría.
  2. Spearman entre p_sym y la valoración (categorías 2-5), global y dentro de cada patología.
  3. AUC de la valoración del radiólogo frente al AUC de p_sym en las mismas imágenes, y AUC de
     la combinación (logística valoración + u0 ajustada en validación).
  4. Tasa de malignidad por categoría de la herramienta (cortes actuales) y contradicciones con la decisión.
  5. Acuerdo herramienta-radiólogo (kappa ponderado cuadrático, acuerdo exacto) con los cortes
     actuales y con cortes alternativos fijados SOLO en validación.

Uso: python scripts/u0_birads_radiologo.py [CSV_DIR] [SALIDA.json]
"""
import json, math, os, sys
import numpy as np
import pandas as pd
from scipy import stats

REPO = os.environ.get("TESIS_REPO") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "database", "archive", "csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "reports", "auditoria", "u0_birads_radiologo.json")
EMB = os.path.join(os.environ.get("TESIS_REP") or os.path.join(REPO, "artifacts"), "output", "embeddings")
EXP = os.path.join(REPO, "reports", "experiments")
sys.path.insert(0, REPO)
from src.reporting.formula_eval import compile_expr, sigmoid  # noqa: E402

B = 2000
RNG = np.random.default_rng(0)
# "actual" = la convención anterior al 1-nov-2026 (0,32/0,42/0,52/0,65/0,78), que este análisis evaluó y reemplazó
CORTES_ACTUALES = [(0.32, "2"), (0.42, "3"), (0.52, "4A"), (0.65, "4B"), (0.78, "4C")]
ACR = {"2": "0 %", "3": "<= 2 %", "4A": "2-10 %", "4B": "10-50 %", "4C": "50-95 %", "5": ">= 95 %", "4": "2-95 %"}


def norm_path(s):
    return s.replace("\\", "/")


def cat_actual(p):
    for c, e in CORTES_ACTUALES:
        if p < c:
            return e
    return "5"


def a_entero(c):
    return int(str(c)[0])


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def kappa_cuadratico(a, b, cats):
    idx = {c: i for i, c in enumerate(cats)}; k = len(cats)
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], float) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return float(1 - (W * O).sum() / (W * E).sum())


def boot(df, f):
    """IC 95 % por bootstrap de pacientes de un estadístico f(df)."""
    pac = df["patient_id"].unique(); grupos = {p: np.where(df["patient_id"].values == p)[0] for p in pac}
    vals = []
    for _ in range(B):
        s = RNG.choice(pac, len(pac), replace=True)
        ii = np.concatenate([grupos[p] for p in s])
        try:
            v = f(df.iloc[ii])
        except Exception:
            continue
        if v is not None and np.isfinite(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def r4(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def tabla(split):
    nz = np.load(os.path.join(EXP, "kan_robust64_reducer.npz"))
    can = json.load(open(os.path.join(EXP, "kan_robust64_canonical.json"), encoding="utf-8"))
    fb, fm = compile_expr(can["fben"]), compile_expr(can["fmal"]); tau = float(can["th_youden_val"])
    X = np.load(os.path.join(EMB, f"f2robust_{split}_feats.npy")).astype(np.float64)
    y = np.load(os.path.join(EMB, f"f2robust_{split}_labels.npy")).astype(int)
    gz = (X - nz["gap_mu"]) / nz["gap_sd"]
    z64 = (((gz @ nz["proj_w"].T + nz["proj_b"]) - nz["pca_mean"]) @ nz["pca_components"].T - nz["mu_z"]) / nz["sd_z"]
    u0 = z64[:, 0]
    g = np.array([fm(v) - fb(v) for v in u0]); p = 1 / (1 + np.exp(-g))

    man = pd.read_csv(os.path.join(REPO, "src", "data", "processed", f"manifest_{split}.csv"))
    assert len(man) == len(y) and (man["label"].values == y).all()
    cbis = "train" if split in ("train", "val") else "test"
    di = pd.read_csv(os.path.join(CSV_DIR, "dicom_info.csv"))
    dm = pd.read_csv(os.path.join(CSV_DIR, f"mass_case_description_{cbis}_set.csv"))
    dc = pd.read_csv(os.path.join(CSV_DIR, f"calc_case_description_{cbis}_set.csv"))
    di["key"] = di["image_path"].map(lambda s: "/".join(norm_path(s).split("/")[-2:]))
    man["key"] = man["filepath"].map(lambda s: "/".join(norm_path(s).split("/")[-2:]))
    m = man.merge(di[["key", "PatientID", "SeriesDescription"]], on="key", how="left", validate="one_to_one")
    assert m["PatientID"].notna().all()

    def parse(pid):
        a = pid.split("_")
        return a[0].split("-")[0], "_".join(a[1:3]), a[3], a[4], (int(a[5]) if len(a) > 5 else None)
    parts = m["PatientID"].map(parse)
    m["tipo"] = [t[0] for t in parts]; m["lado"] = [t[2] for t in parts]; m["vista"] = [t[3] for t in parts]
    m["anom"] = [t[4] for t in parts]
    ev_unica, ev_max, n_les, pat = [], [], [], []
    for r in m.itertuples():
        src = dm if r.tipo == "Mass" else dc
        s = src[(src["patient_id"] == r.patient_id) & (src["left or right breast"] == r.lado) & (src["image view"] == r.vista)]
        if r.anom is not None and not pd.isna(r.anom):
            s = s[s["abnormality id"] == int(r.anom)]
        vals = sorted(set(int(v) for v in s["assessment"].dropna()))
        n_les.append(len(s))
        ps = sorted(set(str(v).strip() for v in s["pathology"].dropna()))
        pat.append(ps[0] if len(ps) == 1 else ("AMBIGUA" if ps else "N/A"))
        ev_unica.append(vals[0] if len(vals) == 1 else (-1 if vals else -9))
        ev_max.append(max(vals) if vals else -9)
    m["birads_rad"] = ev_unica; m["birads_rad_max"] = ev_max; m["n_lesiones"] = n_les; m["patologia_cbis"] = pat
    m["u0"] = u0; m["g"] = g; m["p"] = p; m["y"] = y; m["dec"] = (g >= tau).astype(int)
    m["cat_actual"] = [cat_actual(v) for v in p]
    m["completa"] = (m["SeriesDescription"] == "full mammogram images").astype(int)
    return m, tau


def describir(m, col):
    filas = []
    for c in sorted(m[col].unique()):
        s = m[m[col] == c]
        filas.append({"birads_radiologo": int(c), "n": int(len(s)), "malignas": int(s["y"].sum()),
                      "tasa_malignidad": r4(s["y"].mean()), "mediana_u0": r4(s["u0"].median()),
                      "mediana_p_sym": r4(s["p"].median()), "decision_maligna": r4(s["dec"].mean())})
    return filas


def por_categoria(m, col, decision=True):
    filas = []
    orden = ["2", "3", "4", "4A", "4B", "4C", "5"]
    for c in [o for o in orden if o in set(m[col])]:
        s = m[m[col] == c]
        tasa = s["y"].mean(); lo, hi = wilson(int(s["y"].sum()), len(s))
        d = {"categoria": c, "n": int(len(s)), "tasa_malignidad": r4(tasa), "ic95": [r4(lo), r4(hi)],
             "rango_ACR": ACR[c]}
        if decision:
            d["decision_benigna"] = int((s["dec"] == 0).sum()); d["decision_maligna"] = int((s["dec"] == 1).sum())
        filas.append(d)
    return filas


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n; den = 1 + z * z / n; c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def contradicciones(m, col):
    benigna_sosp = int(((m["dec"] == 0) & m[col].isin(["4", "4A", "4B", "4C", "5"])).sum())
    maligna_benig = int(((m["dec"] == 1) & m[col].isin(["2", "3"])).sum())
    return {"decision_benigna_con_categoria_4_o_mas": benigna_sosp,
            "decision_maligna_con_categoria_3_o_menos": maligna_benig, "n": int(len(m))}


def asignar(p, cortes, etiquetas):
    idx = np.searchsorted(np.asarray(cortes), np.asarray(p), side="right"); return [etiquetas[i] for i in idx]


def ajustar_cortes_anclados(val, ptau):
    """Cortes 2|3 (<ptau) y 4|5 (>ptau) que maximizan el kappa cuadrático con el radiólogo en validación."""
    pv = val["p"].values; rad = val["birads_rad"].values
    qb = np.unique(np.quantile(pv[pv < ptau], np.linspace(0, 1, 41)))
    qs = np.unique(np.quantile(pv[pv >= ptau], np.linspace(0, 1, 41)))
    mejor = (-9, None)
    for a in np.r_[qb, ptau]:
        for b in np.r_[qs, 1.01]:
            if not (a <= ptau <= b):
                continue
            cat = [int(c) for c in asignar(pv, [a, ptau, b], ["2", "3", "4", "5"])]
            k = kappa_cuadratico(cat, rad, [2, 3, 4, 5])
            if k > mejor[0]:
                mejor = (k, (float(a), float(ptau), float(b)))
    return mejor


def ajustar_cortes_libres(val):
    pv = val["p"].values; rad = val["birads_rad"].values
    q = np.unique(np.quantile(pv, np.linspace(0, 1, 41)))
    mejor = (-9, None)
    for i, a in enumerate(q):
        for j in range(i, len(q)):
            for k in range(j, len(q) + 1):
                b = q[j]; c = q[k] if k < len(q) else 1.01
                cat = [int(x) for x in asignar(pv, [a, b, c], ["2", "3", "4", "5"])]
                kk = kappa_cuadratico(cat, rad, [2, 3, 4, 5])
                if kk > mejor[0]:
                    mejor = (kk, (float(a), float(b), float(c)))
    return mejor


def acuerdo(m, col):
    a = [a_entero(c) for c in m[col]]; b = list(m["birads_rad"])
    tab = pd.crosstab(pd.Series(a, name="herramienta"), pd.Series(b, name="radiologo"))
    return {"kappa_cuadratico": r4(kappa_cuadratico(a, b, [2, 3, 4, 5])),
            "ic95": [r4(x) for x in boot(m.assign(_a=a), lambda d: kappa_cuadratico(list(d["_a"]), list(d["birads_rad"]), [2, 3, 4, 5]))],
            "acuerdo_exacto": r4(float(np.mean(np.array(a) == np.array(b)))),
            "tabla": {f"herr_{i}": {f"rad_{j}": int(tab.loc[i, j]) if (i in tab.index and j in tab.columns) else 0
                                    for j in [2, 3, 4, 5]} for i in [2, 3, 4, 5]}}


def main():
    val, tau = tabla("val"); te, _ = tabla("test"); ptau = sigmoid(tau)
    out = {"tau_logit": tau, "tau_p_sym": r4(ptau), "B_bootstrap": B,
           "nota": "birads_rad = valoración BI-RADS del radiólogo en CBIS-DDSM (assessment, entero 0-5).",
           "vinculo": {}}
    for nombre, m in (("validacion", val), ("prueba", te)):
        out["vinculo"][nombre] = {"imagenes": int(len(m)), "pacientes": int(m["patient_id"].nunique()),
                                  "completas": int(m["completa"].sum()), "recortes": int((1 - m["completa"]).sum()),
                                  "ambiguas_excluidas": int((m["birads_rad"] == -1).sum()),
                                  "sin_vinculo": int((m["birads_rad"] == -9).sum()),
                                  "distribucion_todas": describir(m[m["birads_rad"] >= 0], "birads_rad")}
    res = {}
    for nombre, m0 in (("validacion", val), ("prueba", te)):
        m = m0[m0["birads_rad"].between(2, 5)].copy()
        d = {"n_imagenes_2a5": int(len(m)), "n_pacientes": int(m["patient_id"].nunique())}
        rho = stats.spearmanr(m["p"], m["birads_rad"]).correlation
        d["spearman_p_vs_radiologo"] = {"rho": r4(rho), "ic95": [r4(x) for x in boot(m, lambda s: stats.spearmanr(s["p"], s["birads_rad"]).correlation)]}
        for etq, cl in (("benignas", 0), ("malignas", 1)):
            s = m[m["y"] == cl]
            d[f"spearman_{etq}"] = {"n": int(len(s)), "rho": r4(stats.spearmanr(s["p"], s["birads_rad"]).correlation),
                                    "ic95": [r4(x) for x in boot(s, lambda q: stats.spearmanr(q["p"], q["birads_rad"]).correlation)]}
        a_rad = auc(m["y"], m["birads_rad"]); a_p = auc(m["y"], m["p"])
        d["auc_radiologo"] = {"auc": r4(a_rad), "ic95": [r4(x) for x in boot(m, lambda s: auc(s["y"], s["birads_rad"]))]}
        d["auc_formula_mismas_imagenes"] = {"auc": r4(a_p), "ic95": [r4(x) for x in boot(m, lambda s: auc(s["y"], s["p"]))]}
        d["auc_radiologo_menos_formula"] = {"dif": r4(a_rad - a_p), "ic95": [r4(x) for x in boot(m, lambda s: auc(s["y"], s["birads_rad"]) - auc(s["y"], s["p"]))]}
        d["por_categoria_actual"] = por_categoria(m, "cat_actual")
        d["contradicciones_actual"] = contradicciones(m0, "cat_actual")
        d["acuerdo_actual"] = acuerdo(m, "cat_actual")
        d["radiologo_por_decision"] = {k: describir(m[m["dec"] == v], "birads_rad") for k, v in (("benigna", 0), ("maligna", 1))}
        res[nombre] = d
    # combinación valoración + u0 (ajuste en validación, evaluación en prueba)
    from sklearn.linear_model import LogisticRegression
    v = val[val["birads_rad"].between(2, 5)]; t = te[te["birads_rad"].between(2, 5)].copy()
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(v[["birads_rad", "u0"]].values, v["y"].values)
    t["comb"] = lr.decision_function(t[["birads_rad", "u0"]].values)
    res["combinacion_radiologo_u0"] = {
        "coef_birads": r4(lr.coef_[0][0]), "coef_u0": r4(lr.coef_[0][1]),
        "auc_prueba": r4(auc(t["y"], t["comb"])), "ic95": [r4(x) for x in boot(t, lambda s: auc(s["y"], s["comb"]))],
        "dif_vs_radiologo": r4(auc(t["y"], t["comb"]) - auc(t["y"], t["birads_rad"])),
        "dif_ic95": [r4(x) for x in boot(t, lambda s: auc(s["y"], s["comb"]) - auc(s["y"], s["birads_rad"]))]}
    # cortes alternativos fijados en validación
    alt = {}
    k_a, c_a = ajustar_cortes_anclados(v, ptau)
    k_l, c_l = ajustar_cortes_libres(v)
    for nombre, cortes, kv in (("anclado_en_tau_kappa_max", c_a, k_a), ("libre_kappa_max", c_l, k_l)):
        t["_c"] = asignar(t["p"].values, list(cortes), ["2", "3", "4", "5"])
        te_c = te.copy(); te_c["_c"] = asignar(te_c["p"].values, list(cortes), ["2", "3", "4", "5"])
        alt[nombre] = {"cortes_p_sym": [r4(x) for x in cortes], "kappa_validacion": r4(kv),
                       "acuerdo_prueba": acuerdo(t, "_c"), "por_categoria_prueba": por_categoria(t, "_c"),
                       "contradicciones_prueba": contradicciones(te_c, "_c")}
    # calibración isotónica en validación -> rangos de malignidad del ACR
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip").fit(val["p"].values, val["y"].values)
    def acr(q):
        return "2" if q <= 0 else "3" if q <= 0.02 else "4A" if q <= 0.10 else "4B" if q <= 0.50 else "4C" if q < 0.95 else "5"
    te_i = te.copy(); te_i["q"] = iso.predict(te_i["p"].values); te_i["_c"] = [acr(q) for q in te_i["q"]]
    alt["calibracion_isotonica_rangos_ACR"] = {
        "prob_calibrada_min_max_validacion": [r4(float(iso.predict([val["p"].min()])[0])), r4(float(iso.predict([val["p"].max()])[0]))],
        "por_categoria_prueba": por_categoria(te_i, "_c"),
        "prob_calibrada_en_tau": r4(float(iso.predict([ptau])[0]))}
    res["cortes_alternativos"] = alt
    # sensibilidad: valoración máxima en las imágenes ambiguas
    for nombre, m0 in (("validacion", val), ("prueba", te)):
        m = m0[m0["birads_rad_max"].between(2, 5)]
        res[nombre]["sensibilidad_valoracion_max"] = {
            "n": int(len(m)), "rho": r4(stats.spearmanr(m["p"], m["birads_rad_max"]).correlation),
            "auc_radiologo": r4(auc(m["y"], m["birads_rad_max"])), "auc_formula": r4(auc(m["y"], m["p"]))}
    # sensibilidad: solo lesiones con patología por biopsia (sin BENIGN_WITHOUT_CALLBACK, cuya etiqueta
    # benigna sale de no haber citado a la paciente, es decir, en parte de la propia valoración del radiólogo)
    vb = v[v["patologia_cbis"].isin(["BENIGN", "MALIGNANT"])]; tb = t[t["patologia_cbis"].isin(["BENIGN", "MALIGNANT"])].copy()
    lrb = LogisticRegression(C=1e6, max_iter=1000).fit(vb[["birads_rad", "u0"]].values, vb["y"].values)
    tb["comb_b"] = lrb.decision_function(tb[["birads_rad", "u0"]].values)
    res["sensibilidad_solo_biopsia"] = {
        "n_prueba": int(len(tb)), "pacientes": int(tb["patient_id"].nunique()),
        "excluidas_sin_biopsia_prueba": int((t["patologia_cbis"] == "BENIGN_WITHOUT_CALLBACK").sum()),
        "distribucion_prueba": describir(tb, "birads_rad"),
        "rho": r4(stats.spearmanr(tb["p"], tb["birads_rad"]).correlation),
        "rho_ic95": [r4(x) for x in boot(tb, lambda q: stats.spearmanr(q["p"], q["birads_rad"]).correlation)],
        "auc_radiologo": r4(auc(tb["y"], tb["birads_rad"])), "auc_formula": r4(auc(tb["y"], tb["p"])),
        "dif": r4(auc(tb["y"], tb["birads_rad"]) - auc(tb["y"], tb["p"])),
        "dif_ic95": [r4(x) for x in boot(tb, lambda q: auc(q["y"], q["birads_rad"]) - auc(q["y"], q["p"]))],
        "auc_combinacion": r4(auc(tb["y"], tb["comb_b"])),
        "comb_menos_radiologo": r4(auc(tb["y"], tb["comb_b"]) - auc(tb["y"], tb["birads_rad"])),
        "comb_menos_radiologo_ic95": [r4(x) for x in boot(tb, lambda q: auc(q["y"], q["comb_b"]) - auc(q["y"], q["birads_rad"]))]}
    out["resultados"] = res
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    te.to_csv(OUT.replace(".json", "_prueba.csv"), index=False)
    print(json.dumps(out, ensure_ascii=False, indent=1)[:200])


if __name__ == "__main__":
    main()
