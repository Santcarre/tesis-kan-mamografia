# -*- coding: utf-8 -*-
"""
explicadores_imagen.py -- LIME y KernelSHAP sobre regiones de la imagen (Cap. 4, fidelidad y localización)

Pregunta. La Sección de fidelidad muestra que la fórmula g(u0) conserva la explicación de la red en el
logit, en la dirección de pesos y en el mapa espacial, pero no canal por canal. Aquí se pregunta, sin
gradientes y caso por caso: ¿la red y la fórmula se apoyan en las mismas regiones de la imagen?, ¿esas
regiones coinciden con Grad-CAM y con el mapa de la fórmula (psi)?, ¿caen sobre la lesión mejor que una
referencia trivial?

Diseño (las 368 mamografías completas con contorno, las mismas del Pointing Game):
  - Regiones: rejilla de celdas de CELDA x CELDA píxeles sobre la imagen de 224 x 224; se usan solo las
    celdas con al menos la mitad de su área dentro del tejido mamario (extract_breast_tissue_mask).
    Una rejilla fija evita añadir dependencias (scikit-image) y hace la partición reproducible.
  - Perturbación: una región "ausente" se sustituye por la misma región de la imagen desenfocada
    (gaussiano, sigma = SIGMA px), para no sacar la imagen de su distribución con parches negros.
  - Dos funciones objetivo evaluadas en la MISMA pasada del extractor: el logit de decisión de la red
    robusta, s = (w1 - w0)·GAP + (b1 - b0), y la fórmula g(u0) = f_mal(u0) - f_ben(u0), con u0 calculada
    desde el mismo GAP con el reductor kan_robust64_reducer.npz.
  - LIME: N_MUESTRAS máscaras binarias al azar (p = 1/2), kernel exponencial sobre la distancia coseno
    a la imagen completa (anchura 0,25) y regresión ridge ponderada (alfa = 1).
  - KernelSHAP estándar: tamaños de coalición proporcionales al kernel de Shapley, pares complementarios
    y restricción de eficiencia sum(phi) = f(todo) - f(nada) (como kernel_shap_std de
    auditoria_explicadores.py). Referencia "nada" = imagen completamente desenfocada.
  - Estabilidad: en los primeros N_ESTAB casos se repiten LIME y KernelSHAP con otra semilla.
  - Mapas de referencia agregados por celda: Grad-CAM de clase maligna (= ReLU(sum w1 A), identidad
    verificada en formula_vs_gradcam_wsss.py), CAM lineal de la dirección de decisión (w1 - w0) y el
    mapa de la fórmula psi (pesos -W de u0).

Medidas por imagen (sobre las celdas de tejido):
  (1) fidelidad red-fórmula: Spearman entre atribuciones de la red y de la fórmula, coincidencia de la
      celda de mayor atribución e IoU del 15 % superior de celdas;
  (2) concordancia: Spearman de LIME/SHAP con los mapas y entre LIME y SHAP;
  (3) localización: acierto de la celda de mayor atribución positiva (toca la lesión) y fracción de la
      atribución positiva dentro de la lesión (EBPG por celdas), frente a: celda del centro del tejido,
      celda de tejido al azar (valor esperado exacto), mapa barajado entre imágenes y nivel de azar por área.
Resúmenes globales y en malignas con IC del 95 % por bootstrap agrupado por paciente (B = 2000, semilla 0)
y diferencias real - referencia con prueba de Wilcoxon.

Añadidos (revisión de octubre de 2026):
  - Convergencia (--convergencia): en 40 imágenes al azar, LIME y KernelSHAP con 1000 a 8000 muestras y dos
    semillas; mide cuánto se reproduce cada explicador consigo mismo y sugiere el presupuesto.
  - Fidelidad red-fórmula con semillas cruzadas: en los casos de estabilidad, además del acuerdo con las
    mismas perturbaciones (que comparten el ruido del explicador), se mide el acuerdo entre la red con una
    semilla y la fórmula con otra, comparable con la estabilidad del explicador consigo mismo.
  - Curvas de borrado e inserción (Petsiuk et al., 2018): se desenfocan (o se restauran) primero las celdas
    que cada explicación marca como más importantes y se mide el puntaje de la red, sigma(s), y el de la
    fórmula, sigma(g); se compara con órdenes al azar. Mide qué explicación es fiel al modelo.

Uso (desde la raíz del repositorio, con el entorno que tiene torch):
    python scripts/explicadores_imagen.py --convergencia            # primero: elegir el número de muestras
    python scripts/explicadores_imagen.py [--n-muestras 1000] [--celda 16] [--sigma 10] [--estab 100] [--smoke]
--smoke procesa 6 imágenes con 200 muestras para comprobar que todo corre (minutos en CPU).
Salidas: reports/auditoria/explicadores_imagen.json, explicadores_imagen_porimagen.csv, explicadores_imagen_mapas.npz
         Figuras/34_explicadores_imagen_ejemplos.png
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--n-muestras", type=int, default=1000)
ap.add_argument("--celda", type=int, default=16)
ap.add_argument("--sigma", type=float, default=10.0)
ap.add_argument("--estab", type=int, default=100)
ap.add_argument("--perm", type=int, default=20, help="repeticiones del mapa barajado")
ap.add_argument("--lote", type=int, default=250)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--convergencia", action="store_true", help="estudio de convergencia y salida")
ap.add_argument("--conv-imagenes", type=int, default=40)
ap.add_argument("--presupuestos", default="1000,2000,4000,8000")
ap.add_argument("--pasos", type=int, default=20, help="pasos de las curvas de borrado e inserción")
ap.add_argument("--azar-ordenes", type=int, default=5)
args = ap.parse_args()
if args.smoke:
    args.n_muestras, args.estab, args.perm = 200, 2, 3

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "src"), os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("TESIS_CODIGOS", REPO)
OUT = os.path.join(REPO, "reports", "auditoria"); os.makedirs(OUT, exist_ok=True)
FIG = os.path.join(REPO, "Figuras"); os.makedirs(FIG, exist_ok=True)
SUF = "_smoke" if args.smoke else ""

import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models
from torchvision.transforms.functional import gaussian_blur
from scipy.stats import wilcoxon
from src.data.roi_loader import get_roi_manifest, CBISROIDataset
from src.explainability.weak_segmentation import extract_breast_tissue_mask, compute_centroid
import ast
FUNCS = {"sin": np.sin, "cos": np.cos, "exp": np.exp, "tanh": np.tanh, "log": np.log, "sqrt": np.sqrt, "abs": np.abs}
BINOPS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}


def compile_expr(src):
    """Evaluador de lista blanca de la fórmula en x_1 (mismo que verificar_formula.py; no usa eval())."""
    tree = ast.parse(src.replace("^", "**"), mode="eval")

    def ev(n, x):
        if isinstance(n, ast.Expression): return ev(n.body, x)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
        if isinstance(n, ast.Name) and n.id == "x_1": return x
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand, x); return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and type(n.op) in BINOPS: return BINOPS[type(n.op)](ev(n.left, x), ev(n.right, x))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCS and len(n.args) == 1 and not n.keywords:
            return FUNCS[n.func.id](ev(n.args[0], x))
        raise ValueError("nodo no permitido en la formula: " + ast.dump(n)[:80])

    return lambda x: ev(tree, np.asarray(x, dtype=np.float64))

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED, T0 = 42, time.time()
CEL = args.celda; assert 224 % CEL == 0; C = 224 // CEL

# ------------------------------------------------------------------ modelo, reductor y fórmula
m = models.resnet18(weights=None); m.fc = nn.Linear(512, 2)
ck = torch.load(os.path.join(REPO, "reports", "models", "resnet18_robust_best.pt"), map_location=DEV, weights_only=False)
m.load_state_dict(ck["model_state_dict"]); m = m.to(DEV).eval()
feat = nn.Sequential(*list(m.children())[:-2]).eval()            # imagen -> A (512 x 7 x 7)
w1 = m.fc.weight[1].detach().double(); w0 = m.fc.weight[0].detach().double()
wd = (w1 - w0); bd = float(m.fc.bias[1] - m.fc.bias[0])

R = np.load(os.path.join(REPO, "reports", "experiments", "kan_robust64_reducer.npz"))
T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), device=DEV)
g_mu, g_sd = T(R["gap_mu"]).ravel(), T(R["gap_sd"]).ravel()
p_w, p_b, p_m = T(R["proj_w"]), T(R["proj_b"]).ravel(), T(R["pca_mean"]).ravel()
pc0 = T(R["pca_components"][0]); mz0, sz0 = float(R["mu_z"].ravel()[0]), float(R["sd_z"].ravel()[0])
v = R["proj_w"].T.astype(np.float64) @ R["pca_components"][0].astype(np.float64)
W_mal = -(v / R["gap_sd"].ravel().astype(np.float64) / sz0)       # dirección de malignidad de u0 (g' < 0)

cj = json.load(open(os.path.join(REPO, "reports", "experiments", "kan_robust64_canonical.json"), encoding="utf-8"))
f_ben, f_mal = compile_expr(cj["fben"]), compile_expr(cj["fmal"])
g_fun = lambda u: np.asarray(f_mal(u), float) - np.asarray(f_ben(u), float)


def u0_de_gap(gap):
    z = ((gap - g_mu) / g_sd) @ p_w.T + p_b
    return ((z - p_m) @ pc0 - mz0) / sz0


@torch.no_grad()
def evaluar(x, xb, Mfull):
    """Mfull [n, C, C] en {0,1} -> (s, g) de la red y de la fórmula para cada imagen perturbada."""
    S, U = [], []
    for k in range(0, Mfull.shape[0], args.lote):
        mk = torch.as_tensor(Mfull[k:k + args.lote], dtype=torch.float32, device=DEV)
        mp = mk.repeat_interleave(CEL, 1).repeat_interleave(CEL, 2).unsqueeze(1)
        gap = feat(xb + (x - xb) * mp).mean(dim=(2, 3)).double()
        S.append((gap @ wd + bd).cpu().numpy()); U.append(u0_de_gap(gap).cpu().numpy())
    s, u = np.concatenate(S), np.concatenate(U)
    return s, g_fun(u)


# ------------------------------------------------------------------ explicadores sobre máscaras
def lime_regiones(M, y, ancho=0.25, alfa=1.0):
    d = M.shape[1]
    cos = M.sum(1) / (np.sqrt(np.maximum(M.sum(1), 1e-12)) * np.sqrt(d))
    w = np.exp(-((1.0 - cos) ** 2) / ancho ** 2)
    A = np.hstack([np.ones((len(M), 1)), M]); sw = np.sqrt(w)
    reg = alfa * np.eye(d + 1); reg[0, 0] = 0.0
    th = np.linalg.solve((A * sw[:, None]).T @ (A * sw[:, None]) + reg, (A * sw[:, None]).T @ (y * sw))
    pred = A @ th; r2 = 1 - np.sum(w * (y - pred) ** 2) / np.sum(w * (y - np.average(y, weights=w)) ** 2)
    return th[1:], float(r2)


def mascaras_shap(d, n, rng):
    tam = np.arange(1, d); pk = (d - 1.0) / (tam * (d - tam)); pk /= pk.sum()
    s = rng.choice(tam, p=pk, size=max(1, n // 2))
    rk = np.argsort(np.argsort(rng.random((len(s), d)), axis=1), axis=1)
    Mh = (rk < s[:, None]).astype(np.float64)
    return np.vstack([Mh, 1.0 - Mh])


def shap_regiones(M, y, f_todo, f_nada):
    delta = f_todo - f_nada
    A = M[:, :-1] - M[:, -1:]; t = y - f_nada - M[:, -1] * delta
    ph, *_ = np.linalg.lstsq(A, t, rcond=None)
    return np.append(ph, delta - ph.sum())


def spear(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3: return np.nan
    ra, rb = pd.Series(a[ok]).rank().values, pd.Series(b[ok]).rank().values
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else np.nan


def iou_top(a, b, q=0.15):
    ok = np.isfinite(a) & np.isfinite(b); k = max(1, int(round(q * ok.sum())))
    ia, ib = set(np.argsort(-np.where(ok, a, -np.inf))[:k]), set(np.argsort(-np.where(ok, b, -np.inf))[:k])
    return len(ia & ib) / len(ia | ib)


def mapa_celdas(mapa224):
    return mapa224.reshape(C, CEL, C, CEL).mean(axis=(1, 3)).ravel()


def up224(cam7, relu):
    t = torch.as_tensor(cam7, dtype=torch.float32)[None, None]
    t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].numpy()
    return np.maximum(t, 0) if relu else t


def localizar(a, lf):
    """a: atribución por celda (nan fuera del tejido); lf: fracción de lesión por celda."""
    ok = np.isfinite(a)
    top = int(np.nanargmax(np.where(ok, a, -np.inf)))
    pos = np.where(ok, np.maximum(a, 0), 0.0)
    return float(lf[top] > 0), float((pos * lf).sum() / pos.sum()) if pos.sum() > 0 else 0.0


# ------------------------------------------------------------------ datos
df = get_roi_manifest(REPO); ds = CBISROIDataset(df, image_size=224)
N = len(ds) if not args.smoke else 6
if not args.smoke: assert N == 368, N
labels_all = np.array([int(l) for l in df["label"].values[:N]])
rng_fig = np.random.default_rng(0)
mal_idx = np.where(labels_all == 1)[0]
casos_fig = sorted(rng_fig.choice(mal_idx, size=min(3, len(mal_idx)), replace=False).tolist())

MET = ("lime_s", "lime_g", "shap_s", "shap_g", "psi", "gradcam", "cam_wdiff")


def preparar(i):
    it = ds[i]; x = it["image_tensor"].unsqueeze(0).to(DEV)
    xb = gaussian_blur(x, kernel_size=[int(6 * args.sigma) | 1] * 2, sigma=[args.sigma] * 2)
    raw = it["image_tensor"].numpy()[0]; disp = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    tissue = extract_breast_tissue_mask(disp); gt = it["mask_np"]
    tf, lf = mapa_celdas(tissue), mapa_celdas(gt)
    cel = np.where(tf >= 0.5)[0]
    if len(cel) < 4:   # imagen sin tejido suficiente (no debería ocurrir)
        cel = np.argsort(-tf)[:4]
    return it, x, xb, disp, tissue, gt, lf, cel


def a_rejilla(Md, cel):
    Mf = np.ones((Md.shape[0], C * C)); Mf[:, cel] = Md
    return Mf.reshape(-1, C, C)


def correr(x, xb, cel, semilla, n):
    d = len(cel); r = np.random.default_rng(semilla)
    Ml = r.integers(0, 2, size=(n, d)).astype(np.float64); Ml[0] = 1.0
    Ms = mascaras_shap(d, n, r)
    Mall = np.vstack([Ml, Ms, np.ones((1, d)), np.zeros((1, d))])
    s, g = evaluar(x, xb, a_rejilla(Mall, cel))
    nl, ns = len(Ml), len(Ms)
    res = {}
    for nom, y in (("s", s), ("g", g)):
        res[f"lime_{nom}"], res[f"r2_{nom}"] = lime_regiones(Ml, y[:nl])
        res[f"shap_{nom}"] = shap_regiones(Ms, y[nl:nl + ns], y[-2], y[-1])
    res["s_x"], res["g_x"] = float(s[0]), float(g[0])
    return res


sig = lambda z: 1.0 / (1.0 + np.exp(-np.asarray(z, float)))
trap = getattr(np, "trapezoid", None) or np.trapz


def curvas(x, xb, cel, ordenes):
    """ordenes: {nombre: permutación de las celdas de tejido}. Devuelve AUC de borrado e inserción
    del puntaje de la red sigma(s) y de la fórmula sigma(g), sobre la fracción de celdas (0..1)."""
    d = len(cel); ks = np.unique(np.round(np.linspace(0, 1, args.pasos + 1) * d).astype(int))
    fr = ks / d; bloques, nombres = [], list(ordenes)
    for nom in nombres:
        o = ordenes[nom]
        for k in ks:
            md = np.ones(d); md[o[:k]] = 0.0            # borrado: se desenfocan las k primeras
            mi = np.zeros(d); mi[o[:k]] = 1.0           # inserción: se restauran las k primeras
            bloques += [md, mi]
    s, g = evaluar(x, xb, a_rejilla(np.vstack(bloques), cel))
    out, nk = {}, len(ks)
    for t, nom in enumerate(nombres):
        base = t * 2 * nk
        for lab, y in (("s", sig(s)), ("g", sig(g))):
            yd, yi = y[base:base + 2 * nk:2], y[base + 1:base + 2 * nk:2]
            out[f"del_{nom}_{lab}"] = float(trap(yd, fr)); out[f"ins_{nom}_{lab}"] = float(trap(yi, fr))
    return out


def orden_desc(a_cel, rng):
    return np.lexsort((rng.random(len(a_cel)), -np.nan_to_num(a_cel, nan=-np.inf)))


# ------------------------------------------------------------------ modo convergencia
if args.convergencia:
    pres = [int(v) for v in args.presupuestos.split(",")]
    sel = np.sort(np.random.default_rng(0).choice(N, size=min(args.conv_imagenes, N), replace=False))
    filas = []
    for t, i in enumerate(sel):
        it, x, xb, disp, tissue, gt, lf, cel = preparar(int(i))
        for n in pres:
            a, b = correr(x, xb, cel, SEED + i, n), correr(x, xb, cel, SEED + 10_000 + i, n)
            f = {"idx": int(i), "n": n, "d": len(cel)}
            for met in ("lime", "shap"):
                f[f"estab_{met}_s"] = spear(a[f"{met}_s"], b[f"{met}_s"])
                f[f"estab_{met}_g"] = spear(a[f"{met}_g"], b[f"{met}_g"])
                f[f"fid_misma_{met}"] = spear(a[f"{met}_s"], a[f"{met}_g"])
                f[f"fid_cruz_{met}"] = 0.5 * (spear(a[f"{met}_s"], b[f"{met}_g"]) + spear(b[f"{met}_s"], a[f"{met}_g"]))
            filas.append(f)
        print(f"[{time.time() - T0:6.0f}s] convergencia {t + 1}/{len(sel)}", flush=True)
    Cv = pd.DataFrame(filas); res = Cv.groupby("n").mean(numeric_only=True).drop(columns=["idx"]).round(3)
    ok = res.index[(res["estab_shap_s"] >= 0.90) & (res["estab_lime_s"] >= 0.90)]
    sug = int(ok.min()) if len(ok) else int(max(pres))
    json.dump({"imagenes": sel.tolist(), "por_presupuesto": res.reset_index().to_dict("records"), "sugerido": sug,
               "criterio": "menor presupuesto con estabilidad >= 0,90 en LIME y KernelSHAP (red)"},
              open(os.path.join(OUT, "explicadores_convergencia.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(res.to_string()); print(f"\nPresupuesto sugerido: {sug} muestras. Corre ahora:\n"
                                  f"    python scripts/explicadores_imagen.py --n-muestras {sug}")
    sys.exit(0)

# ------------------------------------------------------------------ estudio principal
grids = {k: [] for k in MET}; lfs, tis, ctr, rows, fig_data = [], [], [], [], {}
for i in range(N):
    it, x, xb, disp, tissue, gt, lf, cel = preparar(i); d = len(cel)
    cy, cx = compute_centroid(tissue); cc = int(cy // CEL) * C + int(cx // CEL)
    if cc not in set(cel):
        yy, xx = cel // C, cel % C; cc = int(cel[np.argmin((yy - cy / CEL) ** 2 + (xx - cx / CEL) ** 2)])

    r1 = correr(x, xb, cel, SEED + i, args.n_muestras)
    with torch.no_grad():
        A = feat(x)[0].double().cpu().numpy()
    mapas = {"psi": mapa_celdas(up224(np.tensordot(W_mal, A, axes=(0, 0)), relu=False)),
             "gradcam": mapa_celdas(up224(np.tensordot(w1.cpu().numpy(), A, axes=(0, 0)), relu=True)),
             "cam_wdiff": mapa_celdas(up224(np.tensordot(wd.cpu().numpy(), A, axes=(0, 0)), relu=False))}
    full = {}
    for k in MET:
        g_ = np.full(C * C, np.nan)
        g_[cel] = r1[k] if k.startswith(("lime", "shap")) else mapas[k][cel]
        full[k] = g_; grids[k].append(g_)
    lfs.append(lf); tis.append(cel); ctr.append(cc)

    row = {"idx": i, "patient_id": it["patient_id"], "label": int(it["label"]), "n_celdas": d,
           "s_x": r1["s_x"], "g_x": r1["g_x"], "lime_r2_s": r1["r2_s"], "lime_r2_g": r1["r2_g"]}
    for met in ("lime", "shap"):
        a, b = full[f"{met}_s"], full[f"{met}_g"]
        row[f"fid_{met}_spearman"] = spear(a, b)
        row[f"fid_{met}_top1"] = float(np.nanargmax(a) == np.nanargmax(b))
        row[f"fid_{met}_iou15"] = iou_top(a, b)
    for a, b in (("lime_s", "shap_s"), ("lime_g", "shap_g"), ("lime_s", "cam_wdiff"), ("lime_s", "gradcam"),
                 ("lime_g", "psi"), ("shap_s", "cam_wdiff"), ("shap_g", "psi"), ("psi", "cam_wdiff")):
        row[f"conc_{a}__{b}"] = spear(full[a], full[b])
    for k in ("lime_s", "lime_g", "shap_s", "shap_g", "psi", "gradcam"):
        row[f"hit_{k}"], row[f"ebpg_{k}"] = localizar(full[k], lf)
    row["hit_centro_tejido"] = float(lf[cc] > 0)
    row["hit_punto_azar"] = float(np.mean(lf[cel] > 0))
    row["ebpg_azar_area"] = float(np.mean(lf[cel]))
    # curvas de borrado e inserción
    ro = np.random.default_rng(SEED + 20_000 + i)
    ordenes = {k: orden_desc(full[k][cel], ro) for k in MET}
    for t in range(args.azar_ordenes): ordenes[f"azar{t}"] = ro.permutation(d)
    cv = curvas(x, xb, cel, ordenes)
    for lab in ("s", "g"):
        for tipo in ("del", "ins"):
            cv[f"{tipo}_azar_{lab}"] = float(np.mean([cv.pop(f"{tipo}_azar{t}_{lab}") for t in range(args.azar_ordenes)]))
    row.update(cv)
    if i < args.estab:
        r2 = correr(x, xb, cel, SEED + 10_000 + i, args.n_muestras)
        for k in ("lime_s", "lime_g", "shap_s", "shap_g"):
            row[f"estab_{k}"] = spear(r1[k], r2[k])
        for met in ("lime", "shap"):
            row[f"fid_cruz_{met}"] = 0.5 * (spear(r1[f"{met}_s"], r2[f"{met}_g"]) + spear(r2[f"{met}_s"], r1[f"{met}_g"]))
    rows.append(row)
    if i in casos_fig:
        fig_data[i] = {"disp": disp, "gt": gt, **{k: full[k].copy() for k in ("lime_s", "lime_g", "psi")}}
    if (i + 1) % 20 == 0 or i == N - 1:
        print(f"[{time.time() - T0:6.0f}s] {i + 1}/{N}  (celdas de tejido: {d})", flush=True)

D = pd.DataFrame(rows)

# mapa barajado entre imágenes (misma rejilla, celdas de tejido de la imagen evaluada)
rng = np.random.default_rng(0)
for k in ("lime_s", "lime_g", "shap_s", "shap_g", "psi", "gradcam"):
    H, E = np.zeros(N), np.zeros(N); cnt = np.zeros(N)
    for _ in range(args.perm):
        perm = rng.permutation(N)
        while np.any(perm == np.arange(N)): perm = rng.permutation(N)
        for i in range(N):
            a = np.full(C * C, np.nan); src = grids[k][perm[i]]; cel = tis[i]
            a[cel] = src[cel]
            if not np.isfinite(a).any(): continue
            h, e = localizar(a, lfs[i]); H[i] += h; E[i] += e; cnt[i] += 1
    D[f"hit_{k}_barajado"] = H / np.maximum(cnt, 1); D[f"ebpg_{k}_barajado"] = E / np.maximum(cnt, 1)
D.to_csv(os.path.join(OUT, f"explicadores_imagen_porimagen{SUF}.csv"), index=False)
np.savez_compressed(os.path.join(OUT, f"explicadores_imagen_mapas{SUF}.npz"), celda=CEL,
                    **{k: np.vstack(grids[k]) for k in MET}, fraccion_lesion=np.vstack(lfs))

# ------------------------------------------------------------------ resúmenes con bootstrap por paciente
pats = D["patient_id"].values; up_ = np.unique(pats); pidx = {p: np.where(pats == p)[0] for p in up_}
brng = np.random.default_rng(0); BB = 2000 if not args.smoke else 200


def boot(vec, mask):
    sel = [p for p in up_ if mask[pidx[p]].any()]; est = []
    for _ in range(BB):
        ps = brng.integers(0, len(sel), len(sel))
        ii = np.concatenate([pidx[sel[q]][mask[pidx[sel[q]]]] for q in ps]); est.append(np.nanmean(vec[ii]))
    return [float(np.nanpercentile(est, 2.5)), float(np.nanpercentile(est, 97.5))]


def resumen(col, mask):
    v = D[col].values.astype(float)
    return {"media": float(np.nanmean(v[mask])), "ic95": boot(v, mask), "n": int(np.isfinite(v[mask]).sum())}


def contraste(a, b, mask):
    va, vb = D[a].values.astype(float), D[b].values.astype(float); dlt = va - vb
    try: p = float(wilcoxon(va[mask], vb[mask]).pvalue)
    except Exception: p = None
    return {"real": float(np.nanmean(va[mask])), "referencia": float(np.nanmean(vb[mask])),
            "dif": float(np.nanmean(dlt[mask])), "dif_ic95": boot(dlt, mask), "wilcoxon_p": p}


out = {"config": {"n_imagenes": int(N), "n_muestras": args.n_muestras, "celda_px": CEL, "sigma_desenfoque": args.sigma,
                  "n_estabilidad": int(min(args.estab, N)), "perm_barajado": args.perm, "bootstrap_B": BB,
                  "pasos_curvas": args.pasos, "ordenes_azar": args.azar_ordenes,
                  "celdas_tejido_media": float(D["n_celdas"].mean()), "dispositivo": DEV},
       "control_s_vs_g_spearman": spear(D["s_x"].values, D["g_x"].values), "global": {}, "malignas": {}}
cols_res = [c for c in D.columns if c.startswith(("fid_", "conc_", "estab_", "lime_r2", "del_", "ins_"))]
for nom, mask in (("global", np.ones(N, bool)), ("malignas", D["label"].values == 1)):
    o = out[nom]
    o["medidas"] = {c: resumen(c, mask) for c in cols_res}
    o["localizacion"] = {}
    for k in ("lime_s", "lime_g", "shap_s", "shap_g", "psi", "gradcam"):
        o["localizacion"][k] = {
            "hit_top1": resumen(f"hit_{k}", mask), "ebpg": resumen(f"ebpg_{k}", mask),
            "hit_vs_centro_tejido": contraste(f"hit_{k}", "hit_centro_tejido", mask),
            "hit_vs_punto_azar": contraste(f"hit_{k}", "hit_punto_azar", mask),
            "hit_vs_barajado": contraste(f"hit_{k}", f"hit_{k}_barajado", mask),
            "ebpg_vs_azar_area": contraste(f"ebpg_{k}", "ebpg_azar_area", mask),
            "ebpg_vs_barajado": contraste(f"ebpg_{k}", f"ebpg_{k}_barajado", mask)}
    o["referencias"] = {c: resumen(c, mask) for c in ("hit_centro_tejido", "hit_punto_azar", "ebpg_azar_area")}
    o["borrado_insercion_vs_azar"] = {}
    for k in MET:
        for lab in ("s", "g"):
            o["borrado_insercion_vs_azar"][f"{k}_{lab}"] = {
                "borrado": contraste(f"del_{k}_{lab}", f"del_azar_{lab}", mask),
                "insercion": contraste(f"ins_{k}_{lab}", f"ins_azar_{lab}", mask)}
out["tiempo_s"] = round(time.time() - T0, 1)
json.dump(out, open(os.path.join(OUT, f"explicadores_imagen{SUF}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# ------------------------------------------------------------------ figura de ejemplos (casos malignos al azar, semilla 0)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
if fig_data:
    cols = [("disp", "Imagen y contorno"), ("lime_s", "LIME de la red"), ("lime_g", "LIME de la fórmula"), ("psi", r"Mapa de la fórmula $\psi$")]
    fig, ax = plt.subplots(len(fig_data), 4, figsize=(13, 3.4 * len(fig_data)), squeeze=False)
    for r, (i, dd) in enumerate(sorted(fig_data.items())):
        for c, (k, tit) in enumerate(cols):
            a = ax[r, c]; a.imshow(dd["disp"], cmap="gray")
            if k != "disp":
                g_ = dd[k].reshape(C, C); lim = np.nanmax(np.abs(g_)) or 1.0
                a.imshow(np.ma.masked_invalid(np.kron(g_, np.ones((CEL, CEL)))), cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=0.6)
            a.contour(dd["gt"], levels=[0.5], colors="yellow", linewidths=1.2)
            a.set_xticks([]); a.set_yticks([])
            if r == 0: a.set_title(tit, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, f"34_explicadores_imagen_ejemplos{SUF}.png"), dpi=200, bbox_inches="tight")

print("\nResumen (malignas):")
for k, v_ in out["malignas"]["medidas"].items():
    if k.startswith(("fid_", "estab_")): print(f"  {k:28s} {v_['media']:.3f} {v_['ic95']}")
for k, v_ in out["malignas"]["localizacion"].items():
    h = v_["hit_vs_centro_tejido"]
    print(f"  {k:8s} Top-1 {v_['hit_top1']['media']:.3f} | centro tejido {h['referencia']:.3f} dif {h['dif']:+.3f} {h['dif_ic95']}"
          f" | EBPG {v_['ebpg']['media']:.3f} vs área {v_['ebpg_vs_azar_area']['referencia']:.3f}")
print("\nBorrado (menor es mejor) e inserción (mayor es mejor), puntaje de la red | de la fórmula (global):")
for k in MET + ("azar",):
    print(f"  {k:10s} borrado {D[f'del_{k}_s'].mean():.3f} | {D[f'del_{k}_g'].mean():.3f}   inserción {D[f'ins_{k}_s'].mean():.3f} | {D[f'ins_{k}_g'].mean():.3f}")
if "fid_cruz_lime" in D:
    print(f"\nRed-fórmula, mismas perturbaciones: LIME {D['fid_lime_spearman'].mean():.3f}, SHAP {D['fid_shap_spearman'].mean():.3f}")
    print(f"Red-fórmula, semillas cruzadas:    LIME {D['fid_cruz_lime'].mean():.3f}, SHAP {D['fid_cruz_shap'].mean():.3f}")
    print(f"Estabilidad de la red:             LIME {D['estab_lime_s'].mean():.3f}, SHAP {D['estab_shap_s'].mean():.3f}")
print(f"Guardado en {OUT} ({out['tiempo_s']:.0f} s, dispositivo {DEV})")
