"""
Asociacion de u0 con los descriptores BI-RADS de CBIS-DDSM con vinculo POR IMAGEN (Cap. 4, Sec. BI-RADS).

Corrige a compute_r3_semantics.py (versión anterior, no incluida en este repositorio), que asigna margen, forma y tipo de calcificacion por paciente
(diccionario indexado por patient_id): todas las imagenes de una paciente recibian el descriptor de su
ultima masa listada, aunque mostraran otra lesion o una calcificacion.

Aqui cada imagen de prueba se vincula con su propia lesion:
  manifest_test.csv (ruta JPEG) -> dicom_info.csv (misma serie y archivo) -> PatientID de CBIS-DDSM
  'Mass-Test_P_00038_LEFT_CC[_k]' -> (tipo, paciente, lado, proyeccion, numero de anomalia k)
  -> fila de mass/calc_case_description_test_set.csv.
Los recortes llevan el numero de anomalia y se vinculan a una sola lesion. Las mamografias completas se
vinculan a todas las lesiones de esa proyeccion; si estas tienen descriptores distintos, la imagen se
marca como ambigua y se excluye de esa prueba.

Pruebas (mismas que compute_r3_semantics.py, misma familia de Holm de 15 pruebas):
  u0 ~ patologia (Mann-Whitney), u0 ~ margen {CIRCUMSCRIBED, ILL_DEFINED, SPICULATED} (Kruskal-Wallis),
  u0 ~ forma {ROUND, OVAL, IRREGULAR} (K-W), u0 en calcificaciones pleomorficas malignas frente a
  puntiformes benignas (M-W), u0 ~ densidad (K-W, grupos con >= 20 imagenes) y 10 canales GAP ~ patologia.
Tamano del efecto de K-W: epsilon^2 = H / (n - 1) (Tomczak y Tomczak, 2014), con n = imagenes comparadas.

Uso (desde la raiz del repositorio):
  python scripts/semantica_u0_por_imagen.py [CSV_DIR] [SALIDA.json]
CSV_DIR por defecto: database/archive/csv (los CSV de la version JPEG de CBIS-DDSM, incluido dicom_info.csv).
Requiere numpy, pandas, scipy.
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "database", "archive", "csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "reports", "auditoria", "semantica_u0_por_imagen.json")
EMB = os.path.join(os.environ.get("TESIS_REP") or os.path.join(REPO, "artifacts"), "output", "embeddings")
EXP = os.path.join(REPO, "reports", "experiments")
MG = ["CIRCUMSCRIBED", "ILL_DEFINED", "SPICULATED"]
SG = ["ROUND", "OVAL", "IRREGULAR"]


def holm(p):
    m = len(p); o = np.argsort(p); adj = np.empty(m)
    for r, i in enumerate(o):
        adj[i] = min(1.0, (m - r) * p[i])
    for r in range(1, m):
        adj[o[r]] = max(adj[o[r]], adj[o[r - 1]])
    return adj


def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    s = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    return float((np.mean(a) - np.mean(b)) / s)


def norm_path(s):
    return s.replace("\\", "/")


def construir_tabla(csv_dir=CSV_DIR):
    """Vincula cada imagen de prueba con su lesion y devuelve (tabla, X, y, top_ch)."""
    nz = np.load(os.path.join(EXP, "kan_robust64_reducer.npz"))
    X = np.load(os.path.join(EMB, "f2robust_test_feats.npy")).astype(np.float64)
    y = np.load(os.path.join(EMB, "f2robust_test_labels.npy")).astype(int)
    gz = (X - nz["gap_mu"]) / nz["gap_sd"]
    z64 = (((gz @ nz["proj_w"].T + nz["proj_b"]) - nz["pca_mean"]) @ nz["pca_components"].T - nz["mu_z"]) / nz["sd_z"]
    u0 = z64[:, 0]
    r0 = ((nz["pca_components"][0] / nz["sd_z"].ravel()[0]) @ nz["proj_w"]) / nz["gap_sd"].ravel()
    top_ch = np.argsort(-np.abs(r0))[:10]

    man = pd.read_csv(os.path.join(REPO, "src", "data", "processed", "manifest_test.csv"))
    assert (man["label"].values == y).all()
    di = pd.read_csv(os.path.join(csv_dir, "dicom_info.csv"))
    dm = pd.read_csv(os.path.join(csv_dir, "mass_case_description_test_set.csv"))
    dc = pd.read_csv(os.path.join(csv_dir, "calc_case_description_test_set.csv"))
    di["key"] = di["image_path"].map(lambda s: "/".join(norm_path(s).split("/")[-2:]))
    man["key"] = man["filepath"].map(lambda s: "/".join(norm_path(s).split("/")[-2:]))
    m = man.merge(di[["key", "PatientID", "SeriesDescription"]], on="key", how="left", validate="one_to_one")
    assert m["PatientID"].notna().all(), "imagenes sin vinculo en dicom_info.csv"

    def parse(pid):
        a = pid.split("_")
        return a[0].split("-")[0], "_".join(a[1:3]), a[3], a[4], (int(a[5]) if len(a) > 5 else None)
    parts = m["PatientID"].map(parse)
    m["tipo"] = [t[0] for t in parts]; m["pac"] = [t[1] for t in parts]
    m["lado"] = [t[2] for t in parts]; m["vista"] = [t[3] for t in parts]; m["anom"] = [t[4] for t in parts]
    assert (m["pac"] == m["patient_id"]).all()

    def rows(r, src):
        s = src[(src["patient_id"] == r.pac) & (src["left or right breast"] == r.lado) & (src["image view"] == r.vista)]
        if r.anom is not None and not pd.isna(r.anom):
            s = s[s["abnormality id"] == int(r.anom)]
        return s

    def one(vals):
        vals = set(vals)
        return vals.pop() if len(vals) == 1 else ("AMBIGUA" if vals else "N/A")

    mar, shp, calc, dens = [], [], [], []
    for r in m.itertuples():
        if r.tipo == "Mass":
            s = rows(r, dm)
            pair = one(zip(s["mass margins"].astype(str).str.strip().str.upper(), s["mass shape"].astype(str).str.strip().str.upper()))
            mar.append(pair[0] if isinstance(pair, tuple) else pair); shp.append(pair[1] if isinstance(pair, tuple) else pair)
            calc.append("N/A"); dens.append(one(s["breast_density"].astype(str)))
        else:
            s = rows(r, dc)
            mar.append("N/A"); shp.append("N/A")
            calc.append(one(s["calc type"].astype(str).str.strip().str.upper())); dens.append(one(s["breast density"].astype(str)))
    m["margen"], m["forma"], m["calc"], m["densidad"], m["u0"] = mar, shp, calc, dens, u0
    return m, X, y, top_ch, dm


def main():
    m, X, y, top_ch, dm = construir_tabla()

    # comparacion con la asignacion por paciente de compute_r3_semantics.py
    mm = {}
    for _, r in dm.iterrows():
        mm[r["patient_id"]] = (str(r["mass margins"]).strip().upper(), str(r["mass shape"]).strip().upper())
    old_m = m["patient_id"].map(lambda p: mm.get(p, ("N/A", "N/A"))[0])
    old_s = m["patient_id"].map(lambda p: mm.get(p, ("N/A", "N/A"))[1])

    tests, raw = {}, {}
    def mw(a, b, key, n_names=("n1", "n0")):
        w = stats.mannwhitneyu(a, b, alternative="two-sided")
        tests[key] = {"U": float(w.statistic), "p_raw": float(w.pvalue), n_names[0]: int(len(a)), n_names[1]: int(len(b)),
                      "d": round(cohens_d(a, b), 3), "r_rb": round(1 - 2 * float(w.statistic) / (len(a) * len(b)), 4),
                      "mean1": round(float(np.mean(a)), 4), "mean0": round(float(np.mean(b)), 4)}
        raw[key] = float(w.pvalue)
    def kw(col, cats, key):
        g = [m.loc[m[col] == c, "u0"].values for c in cats]
        h = stats.kruskal(*g); n = int(sum(len(x) for x in g))
        tests[key] = {"H": round(float(h.statistic), 4), "p_raw": float(h.pvalue), "n": n, "k": len(cats),
                      "n_por_grupo": dict(zip(cats, [int(len(x)) for x in g])),
                      "epsilon2": round(float(h.statistic) / (n - 1), 4)}
        raw[key] = float(h.pvalue)

    mw(m.loc[m["label"] == 1, "u0"].values, m.loc[m["label"] == 0, "u0"].values, "u0_pathology")
    kw("margen", MG, "u0_margins")
    kw("forma", SG, "u0_shapes")
    mw(m.loc[(m["calc"] == "PLEOMORPHIC") & (m["label"] == 1), "u0"].values,
       m.loc[(m["calc"] == "PUNCTATE") & (m["label"] == 0), "u0"].values, "u0_calc")
    dg = [g for g, sub in m.groupby("densidad") if len(sub) >= 20 and g not in ("AMBIGUA", "N/A")]
    kw("densidad", dg, "u0_density")
    for c in top_ch:
        a = X[y == 1, int(c)]; b = X[y == 0, int(c)]
        mw(a, b, f"ch{int(c)}_pathology")
    names = list(raw); adj = holm(np.array([raw[k] for k in names]))
    for k, a in zip(names, adj):
        tests[k]["p_holm"] = float(a)

    out = {"descripcion": "u0 frente a descriptores BI-RADS con vinculo por imagen (ver docstring)",
           "n_imagenes": int(len(m)), "n_pacientes": int(m["patient_id"].nunique()),
           "vinculo": {"mamografias_completas": int((m["SeriesDescription"] == "full mammogram images").sum()),
                       "recortes": int((m["SeriesDescription"] == "cropped images").sum()),
                       "imagenes_completas_ambiguas_margen": int((m["margen"] == "AMBIGUA").sum())},
           "comparacion_con_asignacion_por_paciente": {
               "imagenes_en_prueba_de_margen_antes": int(old_m.isin(MG).sum()),
               "con_descriptor_de_otra_lesion": int((old_m.isin(MG) & (old_m != m["margen"])).sum()),
               "de_ellas_calcificaciones": int((old_m.isin(MG) & (m["tipo"] == "Calc")).sum()),
               "imagenes_en_prueba_de_forma_antes": int(old_s.isin(SG).sum()),
               "forma_de_otra_lesion": int((old_s.isin(SG) & (old_s != m["forma"])).sum())},
           "holm_familia": names, "tests": tests}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps({k: out[k] for k in ("vinculo", "comparacion_con_asignacion_por_paciente")}, indent=1))
    for k in ("u0_pathology", "u0_margins", "u0_shapes", "u0_calc", "u0_density"):
        print(k, {kk: v for kk, v in tests[k].items() if kk != "n_por_grupo"})


if __name__ == "__main__":
    main()
