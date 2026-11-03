# -*- coding: utf-8 -*-
"""
analizar_informes_llm.py -- Métricas, estadística y validación del estudio de informes con modelos de lenguaje.

Lee reports/experiments/estudio_informes_llm.csv (de estudio_informes_llm.py) y calcula, para cada informe:
  Región:     región reportada en la sección de hallazgos (se ignoran las menciones negadas, como
              «CSE: no identificado»; si quedan dos o más regiones distintas, la región es ambigua).
  Categoría:  categoría BI-RADS de la sección de impresión (se ignora el eco de la lista «1, 2, 3, 4 o 5»);
              «coherente» si es 4-5 con clasificación sospechosa o 2-3 con clasificación benigna.
  Tipo:       menciona masa/nódulo o calcificaciones (frases no negadas).
  Sin respaldo: categorías de hallazgos afirmados o negados que no están en la evidencia (léxico fijo, abajo).
Y las tasas por modelo y condición, con IC de Wilson y por remuestreo de pacientes, McNemar exacto entre
condiciones (Holm entre modelos) y, en la condición con evidencia, Q de Cochran y McNemar de cada modelo
frente a qwen2.5:1.5b-instruct (Holm).

Uso:
    python scripts/analizar_informes_llm.py                         # métricas y resumen
    python scripts/analizar_informes_llm.py --planilla              # además, planilla de validación manual (40 informes)
    python scripts/analizar_informes_llm.py --validar RUTA.csv      # acuerdo del detector con la planilla revisada
    python scripts/analizar_informes_llm.py --entrada OTRO.csv      # otro archivo de informes (formato largo)
Salidas: estudio_informes_llm_metricas.csv, estudio_informes_llm_resumen.json (reports/experiments/)
"""
import os, re, sys, json, math, argparse, unicodedata
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--entrada", default=None)
ap.add_argument("--planilla", action="store_true")
ap.add_argument("--validar", default=None)
ap.add_argument("--referencia", default="qwen2.5:1.5b-instruct")
ap.add_argument("--boot", type=int, default=2000)
args = ap.parse_args()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(REPO, "reports", "experiments")
ENT = args.entrada or os.path.join(EXP, "estudio_informes_llm.csv")
BASE = os.path.splitext(ENT)[0]

# ------------------------------------------------------------------ detector
def sin_tildes(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def limpiar(t):
    t = re.sub(r"[*#_`>|]+", " ", str(t))
    return sin_tildes(t.lower())

NEG = re.compile(r"\bno se (observa|identifica|evidencia|aprecia|detecta|visualiza|describe)|\bno (hay|presenta|muestra|identificad|observad|evidencia)|\bsin (evidencia|signos|presencia)|\bausencia de|\bninguna?\b|\bno especificad")
# Versión 2 del estudio: la región se entrega como «región superior externa de la imagen (CSE)», etc.
REG = {"CSE": r"cuadrante superior externo|(region|zona|porcion) superior extern|\bcse\b",
       "CSI": r"cuadrante superior interno|(region|zona|porcion) superior intern|\bcsi\b",
       "CIE": r"cuadrante inferior externo|(region|zona|porcion) inferior extern|\bcie\b",
       "CII": r"cuadrante inferior interno|(region|zona|porcion) inferior intern|\bcii\b",
       "RA": r"retroareolar|region central|zona central"}
REG_TXT = {"Cuadrante Superior Externo (CSE)": "CSE", "Cuadrante Superior Interno (CSI)": "CSI",
           "Cuadrante Inferior Externo (CIE)": "CIE", "Cuadrante Inferior Interno (CII)": "CII",
           "Región Retroareolar / Central": "RA"}
LEX = {  # hallazgos que ninguna de las dos condiciones entrega
    "margen":        r"espiculad|circunscrit|microlobulad|(borde|margen|limite|contorno)(es|s)?\b[^.]{0,30}(oscurecid|indistint|definid|irregular|regular|lis[oa]s?\b|net[oa]s?\b|borros|angulad)",
    "forma":         r"forma (redond|oval|irregular|lobulad|regular|bien definid|definid)|\bredondead|\bovalad|\bovoide",
    "tamano":        r"\d+([.,]\d+)?\s*(mm|cm)\b|milimetr|centimetr|diametro",
    "otras_caracteristicas": r"consistencia|textura|superficie (lis|irregular|rugos)|ecogenic|vasculariz|realce",
    "calcificacion_morfologia": r"pleomorf|amorf|puntiform|ramificad|groser|palomita|distrofic|vasculares|en cascara|agrupad|segmentari|cluster|racimo",
    "distorsion_asimetria": r"distorsion|asimetri",
    "ganglios_piel_pezon": r"adenopat|ganglio|retraccion|engrosamiento|pezon",
    "densidad_mamaria": r"densidad mamaria|mamas? dens|tejido (fibroglandular|denso|graso)|fibroglandular|heterogeneamente dens|predominio graso|\bacr [abcd]\b",
    "tipo_lesion":   r"\bmasa|\bnodul|calcificaci|lesion solida",
    "niega_hallazgo": r"(imagen|mamografia|estudio)( mamografic[oa])? normal|sin hallazgos|no se (identific|observ|evidenci)\w* (hallazgos|lesion|anomal|alteracion)|ausencia de (hallazgos|lesion)",
}
MASA = re.compile(r"\bmasa|\bnodul|lesion solida"); CALC = re.compile(r"calcificaci")

def secciones(t):
    i2 = t.find("hallazgos"); i3 = t.find("impresion"); i4 = t.find("recomendaci", i3 if i3 >= 0 else 0)
    s2 = t[i2:i3] if 0 <= i2 < i3 else t
    s3 = t[i3:i4] if 0 <= i3 < i4 else (t[i3:] if i3 >= 0 else t)
    return s2, s3

def frases(s):
    return [f for f in re.split(r"[.\n;]+", s) if f.strip()]

def region(s2):
    vistas = []
    for f in frases(s2):
        if NEG.search(f): continue
        for k, pat in REG.items():
            if re.search(pat, f) and k not in vistas: vistas.append(k)
    if not vistas: return "ninguna"
    return vistas[0] if len(vistas) == 1 else "varias"

PAT_B = re.compile(r"(?:bi-?\s?rads|categoria)[^.\d\n]{0,45}?(?<![\d.,])([0-6])(?![.,]?\d|\s*%|\s*,\s*\d|\s+(?:y|o|a)\s+\d)")
def birads(s3, t):
    """Primera categoría BI-RADS de la sección de impresión (o, si no hay, del texto): admite «BI-RADS 4»,
    «categoría 4A», «categoría BI-RADS de 4B», «la categoría BI-RADS asignada es 4»; ignora porcentajes
    (45.2 %), decimales y el eco de la lista «1, 2, 3, 4 o 5»."""
    for zona in (s3, t):
        m = PAT_B.search(zona)
        if m: return int(m.group(1))
    return None

def repetitivo(t, veces=5):
    fr = [f.strip() for f in re.split(r"[.\n]+", t) if len(f.strip()) > 20]
    return int(any(fr.count(f) >= veces for f in set(fr)))

def num_birads(txt):
    m = re.search(r"([0-6])", str(txt)); return int(m.group(1)) if m else None

def medir(fila):
    t = limpiar(fila["texto"]); s2, s3 = secciones(t)
    fr = [f for f in frases(t) if not NEG.search(f)]
    reg = region(s2); b = birads(s3, t)
    masa = any(MASA.search(f) for f in fr); calc = any(CALC.search(f) for f in fr)
    lat = str(fila["laterality"]).upper()
    contraria = (r"(mama|lado) derech" if lat == "LEFT" else r"(mama|lado) izquierd") if lat in ("LEFT", "RIGHT") else None
    cats = [k for k, pat in LEX.items() if re.search(pat, t)]
    if contraria and re.search(contraria, t): cats.append("lateralidad_contraria")
    gt_reg = REG_TXT.get(fila["gt_quadrant"]); dada = REG_TXT.get(fila["region_dada"])
    pat = str(fila["pathology_gt"]).upper(); maligna = pat.startswith("MALIGNANT")
    # clasificación entregada en las dos condiciones (versión 2): la de la fórmula, g >= tau
    b_dada = num_birads(fila["birads_dada"]); pred_mal = float(fila["g"]) >= float(fila["tau"])
    tipo = str(fila["abnormality_type"]).lower()
    ev = fila["condicion"] == "evidencia"
    return {
        "region_reportada": reg, "birads_reportada": b, "menciona_masa": int(masa), "menciona_calc": int(calc),
        "region_error": int(reg != gt_reg), "region_omitida": int(reg in ("ninguna", "varias")),
        "copia_region": int(reg == dada) if ev else np.nan,
        "birads_omitida": int(b is None),
        "birads_incompatible": int(b is not None and ((maligna and b <= 3) or (not maligna and b >= 5))),
        "birads_transmitida": (int(b == b_dada) if ev else int(b is not None and (b >= 4) == pred_mal)),
        "birads_coherente": int(b is not None and (b >= 4) == pred_mal),   # 4-5 si es sospechosa, 2-3 si no
        "tipo_erroneo": int((tipo == "mass" and calc and not masa) or (tipo == "calc" and masa and not calc)),
        "sin_respaldo_n": len(cats), "sin_respaldo": int(len(cats) > 0),
        "sin_respaldo_sin_tipo": int(len([c for c in cats if c != "tipo_lesion"]) > 0),
        "categorias_sin_respaldo": ";".join(cats),
        "repetitivo": repetitivo(t),
        "truncado": int(float(fila.get("tokens") or 0) >= float(fila.get("max_tokens") or 1024)),
    }

# ------------------------------------------------------------------ estadística (solo numpy)
def wilson(k, n, z=1.959964):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]

def binom_2col(k, n):
    if n == 0: return 1.0
    pmf = [math.comb(n, i) * 0.5 ** n for i in range(n + 1)]
    pk = pmf[k]; return min(1.0, sum(p for p in pmf if p <= pk * (1 + 1e-9)))

def mcnemar(a, b):  # a, b: 0/1 pareados
    b01 = int(((a == 1) & (b == 0)).sum()); b10 = int(((a == 0) & (b == 1)).sum())
    return {"b01": b01, "b10": b10, "p": binom_2col(min(b01, b10), b01 + b10)}

def chi2_sf(x, gl):
    if gl % 2 == 0:
        return math.exp(-x / 2) * sum((x / 2) ** i / math.factorial(i) for i in range(gl // 2))
    try:
        from scipy.stats import chi2; return float(chi2.sf(x, gl))
    except Exception: return float("nan")

def cochran_q(M):  # M: n x k (0/1)
    M = np.asarray(M, float); k = M.shape[1]; C = M.sum(0); R = M.sum(1); N = M.sum()
    den = k * N - (R ** 2).sum()
    if den == 0: return {"Q": 0.0, "gl": k - 1, "p": 1.0}
    Q = (k - 1) * (k * (C ** 2).sum() - N ** 2) / den
    return {"Q": round(float(Q), 3), "gl": k - 1, "p": chi2_sf(float(Q), k - 1)}

def holm(ps):
    orden = sorted(ps, key=ps.get); m = len(orden); prev = 0.0; out = {}
    for i, k in enumerate(orden):
        adj = min(1.0, max(prev, (m - i) * ps[k])); prev = adj; out[k] = adj
    return out

rng = np.random.default_rng(0)
def ic_paciente(S, col):
    pac = S["patient_id"].values; u = np.unique(pac); grupos = {p: np.where(pac == p)[0] for p in u}
    v = S[col].values.astype(float); res = []
    for _ in range(args.boot):
        idx = np.concatenate([grupos[p] for p in rng.choice(u, len(u))]); res.append(np.nanmean(v[idx]))
    return [round(100 * float(np.percentile(res, 2.5)), 1), round(100 * float(np.percentile(res, 97.5)), 1)]

def tasa(S, col):
    s = S[col].dropna(); k, n = int(s.sum()), int(len(s))
    return {"k": k, "n": n, "pct": round(100 * k / n, 1) if n else None, "ic95_wilson": wilson(k, n),
            "ic95_paciente": ic_paciente(S.dropna(subset=[col]), col) if n else None}

# ------------------------------------------------------------------ cálculo
D = pd.read_csv(ENT)
D = D[D["error"].isna() & D["texto"].notna()].reset_index(drop=True) if "error" in D else D
M = pd.concat([D, pd.DataFrame([medir(r) for _, r in D.iterrows()])], axis=1)
M.to_csv(BASE + "_metricas.csv", index=False, encoding="utf-8")

BIN = ["region_error", "region_omitida", "copia_region", "birads_omitida", "birads_incompatible",
       "birads_transmitida", "birads_coherente", "tipo_erroneo", "sin_respaldo", "sin_respaldo_sin_tipo", "repetitivo", "truncado"]
modelos = list(dict.fromkeys(M["modelo"]))
res = {"archivo": os.path.basename(ENT), "n_informes": int(len(M)), "n_imagenes": int(M["idx"].nunique()),
       "n_pacientes": int(M["patient_id"].nunique()), "modelos": {}, "entre_condiciones": {}, "entre_modelos_evidencia": {}}
for mo in modelos:
    res["modelos"][mo] = {}
    for cond in ("ciego", "evidencia"):
        S = M[(M.modelo == mo) & (M.condicion == cond)]
        r = {c: tasa(S, c) for c in BIN if S[c].notna().any()}
        r["sin_respaldo_media"] = round(float(S["sin_respaldo_n"].mean()), 2)
        r["categorias"] = {k: int(S["categorias_sin_respaldo"].str.contains(k).sum()) for k in list(LEX) + ["lateralidad_contraria"]}
        r["region_reportada"] = S["region_reportada"].value_counts().to_dict()
        r["segundos_por_informe"] = round(float(S["segundos"].mean()), 2) if "segundos" in S else None
        res["modelos"][mo][cond] = r
# ciego frente a evidencia, por modelo (pareado por imagen), Holm entre modelos para cada desenlace
for c in ["region_error", "birads_omitida", "birads_incompatible", "birads_coherente", "tipo_erroneo", "sin_respaldo", "sin_respaldo_sin_tipo"]:
    ps = {}; det = {}
    for mo in modelos:
        P = M[M.modelo == mo].pivot_table(index="idx", columns="condicion", values=c, aggfunc="first").dropna()
        if {"ciego", "evidencia"} <= set(P.columns):
            det[mo] = mcnemar(P["ciego"].values, P["evidencia"].values); ps[mo] = det[mo]["p"]
    for mo, adj in holm(ps).items(): det[mo]["p_holm"] = adj
    res["entre_condiciones"][c] = det
# entre modelos, condición con evidencia
E = M[M.condicion == "evidencia"]
for c in ["birads_transmitida", "copia_region", "tipo_erroneo", "sin_respaldo", "sin_respaldo_sin_tipo", "birads_omitida"]:
    P = E.pivot_table(index="idx", columns="modelo", values=c, aggfunc="first").dropna()
    if P.shape[1] < 2: continue
    out = {"cochran": cochran_q(P.values), "frente_a_referencia": {}}
    if args.referencia in P.columns:
        ps = {mo: mcnemar(P[args.referencia].values, P[mo].values) for mo in P.columns if mo != args.referencia}
        for mo, adj in holm({k: v["p"] for k, v in ps.items()}).items(): ps[mo]["p_holm"] = adj
        out["frente_a_referencia"] = ps
    res["entre_modelos_evidencia"][c] = out
json.dump(res, open(BASE + "_resumen.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)

# resumen en pantalla
cols = ["region_error", "copia_region", "birads_transmitida", "birads_coherente", "birads_omitida", "birads_incompatible", "tipo_erroneo", "sin_respaldo", "sin_respaldo_sin_tipo"]
tab = []
for mo in modelos:
    for cond in ("ciego", "evidencia"):
        r = res["modelos"][mo][cond]; tab.append([mo, cond] + [r[c]["pct"] if c in r else None for c in cols] + [r["sin_respaldo_media"]])
print(pd.DataFrame(tab, columns=["modelo", "cond"] + cols + ["n_sin_resp"]).to_string(index=False))
print("Guardado:", BASE + "_metricas.csv", "y", BASE + "_resumen.json")

# ------------------------------------------------------------------ planilla de validación manual
if args.planilla:
    rs = np.random.default_rng(1); elegidos = []
    for mo in modelos:
        for cond in ("ciego", "evidencia"):
            S = M[(M.modelo == mo) & (M.condicion == cond)]
            elegidos += list(rs.choice(S.index.values, size=min(4, len(S)), replace=False))
    rs.shuffle(elegidos); P = M.loc[elegidos].reset_index(drop=True); P["id"] = range(1, len(P) + 1)
    with open(BASE + "_validacion_informes.md", "w", encoding="utf-8") as f:
        f.write("# Informes para revisar\n\nPara cada informe, llena una fila de la planilla CSV con el mismo número.\n\n")
        for _, r in P.iterrows():
            dada = (f"Datos entregados: región «{r['region_dada']}», categoría «{r['birads_dada']}»." if r["condicion"] == "evidencia"
                    else f"Datos entregados: solo la clasificación del modelo ({'sospechosa de malignidad' if r['g'] >= r['tau'] else 'probablemente benigna'}, puntaje {r['p']:.3f}).")
            f.write(f"\n---\n\n## Informe {r['id']}\n\n{dada} Lateralidad: {r['laterality']}.\n\n```\n{r['texto']}\n```\n")
    plan = P[["id"]].copy()
    for c in ["region_reportada (CSE/CSI/CIE/CII/RA/ninguna/varias)", "birads_reportada (0-6 o vacio)",
              "menciona_masa (0/1)", "menciona_calc (0/1)", "sin_respaldo (0/1)", "cuales_sin_respaldo", "observaciones"]:
        plan[c] = ""
    plan.to_csv(BASE + "_validacion_planilla.csv", index=False, encoding="utf-8-sig")
    P[["id", "modelo", "condicion", "idx", "region_reportada", "birads_reportada", "menciona_masa", "menciona_calc",
       "sin_respaldo", "categorias_sin_respaldo"]].to_csv(BASE + "_validacion_detector.csv", index=False, encoding="utf-8")
    print("Planilla:", BASE + "_validacion_planilla.csv", "| informes:", BASE + "_validacion_informes.md")

# ------------------------------------------------------------------ acuerdo con la revisión manual
if args.validar:
    H = (pd.read_excel(args.validar, sheet_name=0) if args.validar.lower().endswith(".xlsx")
         else pd.read_csv(args.validar, encoding="utf-8-sig", sep=None, engine="python"))
    Dt = pd.read_csv(BASE + "_validacion_detector.csv")
    H.columns = [c.split(" (")[0] for c in H.columns]; J = Dt.merge(H, on="id", suffixes=("_det", "_hum"))
    def kappa(a, b):
        a, b = np.asarray(a, int), np.asarray(b, int); po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
        return round(float((po - pe) / (1 - pe)), 3) if pe < 1 else 1.0
    val = {}
    for c in ["region_reportada", "birads_reportada", "menciona_masa", "menciona_calc", "sin_respaldo"]:
        a = J[f"{c}_det"].fillna("").astype(str).str.replace(".0", "", regex=False).str.strip().replace({"nan": "", "None": ""})
        b = J[f"{c}_hum"].fillna("").astype(str).str.replace(".0", "", regex=False).str.strip().replace({"nan": "", "None": ""})
        val[c] = {"acuerdo_pct": round(100 * float((a == b).mean()), 1), "n": int(len(J))}
        if c in ("menciona_masa", "menciona_calc", "sin_respaldo"):
            val[c]["kappa"] = kappa(a.replace("", "0"), b.replace("", "0"))
    json.dump(val, open(BASE + "_validacion_acuerdo.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(val, ensure_ascii=False, indent=1))
