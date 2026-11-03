# -*- coding: utf-8 -*-
"""
segmentacion_resolucion.py -- ¿Mejora la localización débilmente supervisada con mapas de mayor resolución?

El mapa de la fórmula, psi = ReLU(sum_c -W_c A_c), sale del último bloque de ResNet-18, que tiene 7 x 7 celdas
con entradas de 224 x 224: no puede delimitar lesiones de unos 13 px de diámetro. Se prueban dos formas de
aumentar la resolución sin entrenar nada, con el MISMO operador focal (>= 60 %, máscara de tejido), las
mismas métricas a 224 x 224 y las mismas referencias que la tesis (control_azar_focal.py):

  (A) Entrada a mayor resolución: la imagen se lleva a 448 o 672 px; ResNet-18 no tiene capas fijas al
      tamaño, así que A tiene 14 x 14 o 21 x 21 celdas y psi y u0 se calculan igual. Se mide también el AUC
      de la fórmula a esa resolución, porque la red se entrenó a 224 px.
  (B) Ventana deslizante con la fórmula: la mamografía a resolución original se recorre con ventanas de
      VENTANAS px (lado), con paso de 1/4 de ventana; cada ventana se lleva a 224 px y se evalúa g(u0)
      (y, como comparación, el logit de la red). El mapa es el promedio de g de las ventanas que cubren cada
      píxel. Se omiten las ventanas con menos de la mitad de tejido. La red se entrenó también con recortes
      centrados en lesiones, así que las ventanas se parecen a lo que vio; pero g mide malignidad, no
      presencia de lesión, y la red nunca vio ventanas de tejido normal como clase.

Métodos: psi_224 (reproduce la tesis), psi_448, psi_672, gradcam_448 (ReLU(sum w1 A), igual a Grad-CAM con
cabeza lineal), vent_formula, vent_red. Referencias: centro del tejido, punto al azar (B veces) y mapa de
otra imagen (B permutaciones sin puntos fijos, por método).
Resúmenes en malignas y global, con IC del 95 % por remuestreo de pacientes y diferencia pareada frente
a psi_224 y frente al centro del tejido.

Uso (desde la raíz del repositorio):
    python scripts/segmentacion_resolucion.py --smoke        # 6 imágenes
    python scripts/segmentacion_resolucion.py [--ventanas 448,896] [--B 20]
Salidas: reports/auditoria/segmentacion_resolucion.json y segmentacion_resolucion_porimagen.csv
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--ventanas", default="448,896", help="lados de ventana en píxeles de la imagen original")
ap.add_argument("--B", type=int, default=20)
ap.add_argument("--lote", type=int, default=128)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
VENT = [int(v) for v in args.ventanas.split(",")]

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "src")):
    if p not in sys.path: sys.path.insert(0, p)
os.environ.setdefault("TESIS_CODIGOS", REPO)
OUT = os.path.join(REPO, "reports", "auditoria"); SUF = "_smoke" if args.smoke else ""

import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models
from PIL import Image
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from src.data.roi_loader import get_roi_manifest, CBISROIDataset
from src.explainability.weak_segmentation import (extract_breast_tissue_mask, segment_wsss_focal_core,
                                                  compute_centroid, compute_segmentation_metrics)
from src.reporting.clinical_evidence_extractor import determine_quadrant
import ast

DEV = "cuda" if torch.cuda.is_available() else "cpu"; T0 = time.time()
Image.MAX_IMAGE_PIXELS = None

# ------------------------------------------------------------------ modelo, reductor y fórmula
m = models.resnet18(weights=None); m.fc = nn.Linear(512, 2)
ck = torch.load(os.path.join(REPO, "reports", "models", "resnet18_robust_best.pt"), map_location=DEV, weights_only=False)
m.load_state_dict(ck["model_state_dict"]); m = m.to(DEV).eval()
feat = nn.Sequential(*list(m.children())[:-2]).eval()
w1 = m.fc.weight[1].detach().double().cpu().numpy(); wd = (m.fc.weight[1] - m.fc.weight[0]).detach().double()
bd = float(m.fc.bias[1] - m.fc.bias[0])
R = np.load(os.path.join(REPO, "reports", "experiments", "kan_robust64_reducer.npz"))
Tn = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), device=DEV)
g_mu, g_sd = Tn(R["gap_mu"]).ravel(), Tn(R["gap_sd"]).ravel()
p_w, p_b, p_m = Tn(R["proj_w"]), Tn(R["proj_b"]).ravel(), Tn(R["pca_mean"]).ravel()
pc0 = Tn(R["pca_components"][0]); mz0, sz0 = float(R["mu_z"].ravel()[0]), float(R["sd_z"].ravel()[0])
v = R["proj_w"].T.astype(np.float64) @ R["pca_components"][0].astype(np.float64)
W_mal = -(v / R["gap_sd"].ravel().astype(np.float64) / sz0)

FUNCS = {"sin": np.sin, "cos": np.cos, "exp": np.exp, "tanh": np.tanh, "log": np.log, "sqrt": np.sqrt, "abs": np.abs}
BINOPS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}
def compile_expr(src):
    tree = ast.parse(src.replace("^", "**"), mode="eval")
    def ev(n, x):
        if isinstance(n, ast.Expression): return ev(n.body, x)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
        if isinstance(n, ast.Name) and n.id == "x_1": return x
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            r = ev(n.operand, x); return -r if isinstance(n.op, ast.USub) else r
        if isinstance(n, ast.BinOp) and type(n.op) in BINOPS: return BINOPS[type(n.op)](ev(n.left, x), ev(n.right, x))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCS and len(n.args) == 1:
            return FUNCS[n.func.id](ev(n.args[0], x))
        raise ValueError("nodo no permitido: " + ast.dump(n)[:80])
    return lambda x: ev(tree, np.asarray(x, dtype=np.float64))
cj = json.load(open(os.path.join(REPO, "reports", "experiments", "kan_robust64_canonical.json"), encoding="utf-8"))
f_ben, f_mal = compile_expr(cj["fben"]), compile_expr(cj["fmal"])
g_fun = lambda u: np.asarray(f_mal(u), float) - np.asarray(f_ben(u), float)

def u0_de_gap(gap):
    z = ((gap - g_mu) / g_sd) @ p_w.T + p_b
    return ((z - p_m) @ pc0 - mz0) / sz0

NORM = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def norm01(a, tissue):
    a = np.where(np.isfinite(a), a, np.nan)
    vals = a[tissue > 0.5] if (tissue > 0.5).any() else a.ravel()
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    if not np.isfinite(lo) or hi <= lo: return np.zeros_like(a, dtype=np.float32)
    return np.nan_to_num((a - lo) / (hi - lo), nan=0.0).clip(0, 1).astype(np.float32)

def up224(arr):
    t = torch.as_tensor(np.asarray(arr, np.float32))[None, None]
    return F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].numpy()

@torch.no_grad()
def mapas_resolucion(pil, lado):
    x = NORM(pil.resize((lado, lado), Image.BILINEAR)).unsqueeze(0).to(DEV)
    A = feat(x)[0].double()
    gap = A.mean(dim=(1, 2))[None]
    u0 = float(u0_de_gap(gap).cpu().numpy()[0]); s = float((gap @ wd + bd).cpu().numpy()[0])
    An = A.cpu().numpy()
    psi = np.maximum(np.tensordot(W_mal, An, axes=(0, 0)), 0)
    gc = np.maximum(np.tensordot(w1, An, axes=(0, 0)), 0)
    return up224(psi), up224(gc), u0, s

@torch.no_grad()
def mapa_ventanas(pil, tissue224, lado):
    W0, H0 = pil.size; f = 224.0 / lado
    w_, h_ = max(224, int(round(W0 * f))), max(224, int(round(H0 * f)))
    x = NORM(pil.resize((w_, h_), Image.BILINEAR)).to(DEV)          # [3, h_, w_]
    st = 56
    ys = list(range(0, h_ - 224 + 1, st)) or [0]; xs = list(range(0, w_ - 224 + 1, st)) or [0]
    if ys[-1] != h_ - 224: ys.append(h_ - 224)
    if xs[-1] != w_ - 224: xs.append(w_ - 224)
    cajas = []
    for y in ys:
        for xx in xs:      # caja en coordenadas 224 x 224 (la imagen se redimensiona sin conservar proporción)
            y0, y1 = int(y * 224 / h_), int(np.ceil((y + 224) * 224 / h_)); x0, x1 = int(xx * 224 / w_), int(np.ceil((xx + 224) * 224 / w_))
            if tissue224[y0:y1, x0:x1].mean() >= 0.5: cajas.append((y, xx, y0, y1, x0, x1))
    sg = np.zeros((224, 224)); ss = np.zeros((224, 224)); cnt = np.zeros((224, 224)); gs, svals = [], []
    for k in range(0, len(cajas), args.lote):
        lote = cajas[k:k + args.lote]
        xb = torch.stack([x[:, c[0]:c[0] + 224, c[1]:c[1] + 224] for c in lote])
        gap = feat(xb).mean(dim=(2, 3)).double()
        g = g_fun(u0_de_gap(gap).cpu().numpy()); s = (gap @ wd + bd).cpu().numpy()
        for c, gv, sv in zip(lote, g, s):
            sg[c[2]:c[3], c[4]:c[5]] += gv; ss[c[2]:c[3], c[4]:c[5]] += sv; cnt[c[2]:c[3], c[4]:c[5]] += 1
        gs += list(g); svals += list(s)
    with np.errstate(invalid="ignore", divide="ignore"):
        mg, ms = sg / cnt, ss / cnt
    mg[cnt == 0] = np.nan; ms[cnt == 0] = np.nan
    return mg, ms, (max(gs) if gs else np.nan), (max(svals) if svals else np.nan), len(cajas)

def quad(c): return determine_quadrant(c[0], c[1], "LEFT", "MLO")      # igual que control_azar_focal.py

def una(cam, tissue, gt, gt_q):
    core, cam_t, peak = segment_wsss_focal_core(cam, tissue, min_tau_ratio=0.60)
    mt = compute_segmentation_metrics(core, gt); c = compute_centroid(core); cb = core > 0.5
    en = float((cam_t * cb).sum())
    return {"centroid_dist": mt["centroid_dist"], "quad_match": float(quad(c) == gt_q), "dice": mt["dice"],
            "recall": mt["recall"], "ebpg_cont": float((cam_t * gt).sum() / (cam_t.sum() + 1e-8)),
            "ebpg_core": float((cam_t * cb * gt).sum() / (en + 1e-8)) if en > 0 else 0.0,
            "hit_peak": float(gt[peak] > 0.5), "core_area": float(cb.sum())}

def punto(cp, gt_c, gt_q):
    return {"centroid_dist": float(np.hypot(cp[0] - gt_c[0], cp[1] - gt_c[1])), "quad_match": float(quad(cp) == gt_q)}

# ------------------------------------------------------------------ datos y mapas
df = get_roi_manifest(REPO); ds = CBISROIDataset(df, image_size=224)
N = len(ds) if not args.smoke else 6
MET = ["psi_224", "psi_448", "psi_672", "gradcam_448"] + [f"vent_formula_{w}" for w in VENT] + [f"vent_red_{w}" for w in VENT]
mapas = {k: [] for k in MET}; tis, gts, gtc, gtq, lab, pac, punt = [], [], [], [], [], [], {}
for i in range(N):
    it = ds[i]; row = df.iloc[i]
    raw = it["image_tensor"].numpy()[0]; disp = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    tissue = extract_breast_tissue_mask(disp); gt = it["mask_np"]
    tis.append(tissue); gts.append(gt); gtc.append(compute_centroid(gt)); gtq.append(quad(compute_centroid(gt)))
    lab.append(int(it["label"])); pac.append(it["patient_id"])
    pil = Image.open(row["image_path"]).convert("RGB")
    p224, _, u224, s224 = mapas_resolucion(pil, 224); p448, gc448, u448, s448 = mapas_resolucion(pil, 448)
    p672, _, u672, s672 = mapas_resolucion(pil, 672)
    for k, a in (("psi_224", p224), ("psi_448", p448), ("psi_672", p672), ("gradcam_448", gc448)):
        mapas[k].append(norm01(a, tissue))
    sc = {"u0_224": u224, "u0_448": u448, "u0_672": u672, "s_224": s224, "s_448": s448, "s_672": s672}
    for w in VENT:
        mg, ms, gmax, smax, nv = mapa_ventanas(pil, tissue, w)
        mapas[f"vent_formula_{w}"].append(norm01(mg, tissue)); mapas[f"vent_red_{w}"].append(norm01(ms, tissue))
        sc[f"gmax_vent_{w}"], sc[f"smax_vent_{w}"], sc[f"n_vent_{w}"] = gmax, smax, nv
    punt[i] = sc
    if (i + 1) % 20 == 0 or i == N - 1:
        print(f"[{time.time()-T0:6.0f}s] {i+1}/{N}", flush=True)
lab = np.array(lab); pac = np.array(pac)

# ------------------------------------------------------------------ métricas, referencias
rng = np.random.default_rng(0); rows = []
for k in MET:
    for i in range(N):
        r = una(mapas[k][i], tis[i], gts[i], gtq[i]); r.update(idx=i, metodo=k, cond="real"); rows.append(r)
    for b in range(args.B):
        perm = rng.permutation(N)
        while N > 1 and np.any(perm == np.arange(N)): perm = rng.permutation(N)
        for i in range(N):
            r = una(mapas[k][perm[i]], tis[i], gts[i], gtq[i]); r.update(idx=i, metodo=k, cond="otra_imagen", rep=b); rows.append(r)
for i in range(N):
    r = punto(compute_centroid(tis[i]), gtc[i], gtq[i]); r.update(idx=i, metodo="-", cond="centro_tejido"); rows.append(r)
    ys, xs = np.where(tis[i] > 0.5)
    for b in range(args.B):
        j = rng.integers(len(ys)); r = punto((float(ys[j]), float(xs[j])), gtc[i], gtq[i]); r.update(idx=i, metodo="-", cond="punto_azar", rep=b); rows.append(r)
Rr = pd.DataFrame(rows); Rr["patient_id"] = pac[Rr["idx"].values]; Rr["label"] = lab[Rr["idx"].values]
Rr.to_csv(os.path.join(OUT, f"segmentacion_resolucion_porimagen{SUF}.csv"), index=False)

def por_imagen(metodo, cond, met):
    sub = Rr[(Rr.metodo == metodo) & (Rr.cond == cond)]
    return sub.groupby("idx")[met].mean().reindex(range(N)).values

upat = np.unique(pac); pidx = {p: np.where(pac == p)[0] for p in upat}; brng = np.random.default_rng(0)
def boot(vv, mask, B=2000 if not args.smoke else 200):
    sel = [p for p in upat if mask[pidx[p]].any()]; est = []
    for _ in range(B):
        ps = brng.integers(0, len(sel), len(sel)); ii = np.concatenate([pidx[sel[q]][mask[pidx[sel[q]]]] for q in ps]); est.append(np.nanmean(vv[ii]))
    return [round(float(np.nanpercentile(est, 2.5)), 4), round(float(np.nanpercentile(est, 97.5)), 4)]
def wil(a, b, mask):
    try: return float(wilcoxon(a[mask], b[mask]).pvalue)
    except Exception: return None

out = {"n": N, "n_malignas": int(lab.sum()), "ventanas": VENT, "B": args.B, "auc": {}, "resumen": {}}
for k, col in (("formula_224", "u0_224"), ("formula_448", "u0_448"), ("formula_672", "u0_672")):
    out["auc"][k] = float(roc_auc_score(lab, -np.array([punt[i][col] for i in range(N)]))) if len(set(lab)) > 1 else None
for k in ("s_224", "s_448", "s_672"):
    out["auc"]["red_" + k[2:]] = float(roc_auc_score(lab, [punt[i][k] for i in range(N)])) if len(set(lab)) > 1 else None
for w in VENT:
    for nom, col in (("formula_ventana_max", f"gmax_vent_{w}"), ("red_ventana_max", f"smax_vent_{w}")):
        out["auc"][f"{nom}_{w}"] = float(roc_auc_score(lab, [punt[i][col] for i in range(N)])) if len(set(lab)) > 1 else None
METR = ["centroid_dist", "quad_match", "dice", "recall", "ebpg_core", "hit_peak"]
for nom, mask in (("malignas", lab == 1), ("global", np.ones(N, bool))):
    o = out["resumen"][nom] = {"referencias": {c: {mt: float(np.nanmean(por_imagen("-", c, mt)[mask])) for mt in ("centroid_dist", "quad_match")}
                                               for c in ("centro_tejido", "punto_azar")}}
    ref224 = {mt: por_imagen("psi_224", "real", mt) for mt in METR}; ctr = {mt: por_imagen("-", "centro_tejido", mt) for mt in ("centroid_dist", "quad_match")}
    for k in MET:
        o[k] = {}
        for mt in METR:
            real, nul = por_imagen(k, "real", mt), por_imagen(k, "otra_imagen", mt)
            e = {"real": float(np.nanmean(real[mask])), "ic95": boot(real, mask), "otra_imagen": float(np.nanmean(nul[mask])),
                 "dif_vs_otra_imagen": float(np.nanmean((real - nul)[mask])), "dif_vs_otra_imagen_ic95": boot(real - nul, mask)}
            if k != "psi_224":
                e["dif_vs_psi_224"] = float(np.nanmean((real - ref224[mt])[mask])); e["dif_vs_psi_224_ic95"] = boot(real - ref224[mt], mask)
                e["wilcoxon_vs_psi_224"] = wil(real, ref224[mt], mask)
            if mt in ctr:
                e["dif_vs_centro_tejido"] = float(np.nanmean((real - ctr[mt])[mask])); e["dif_vs_centro_tejido_ic95"] = boot(real - ctr[mt], mask)
            o[k][mt] = e
out["tiempo_s"] = round(time.time() - T0, 1)
json.dump(out, open(os.path.join(OUT, f"segmentacion_resolucion{SUF}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print("\nAUC en las", N, "mamografías:", {k: (round(v_, 4) if v_ is not None else None) for k, v_ in out["auc"].items()})
for nom in ("malignas", "global"):
    o = out["resumen"][nom]; print(f"\n=== {nom.upper()} === referencias: {o['referencias']}")
    print(f"{'método':22s} {'dist(px)':>9s} {'región':>7s} {'Dice':>6s} {'recall':>7s} {'EBPGnúc':>8s} {'pico':>6s}   dist vs centro tejido")
    for k in MET:
        r = o[k]; dct = r["centroid_dist"].get("dif_vs_centro_tejido"); ic = r["centroid_dist"].get("dif_vs_centro_tejido_ic95")
        print(f"{k:22s} {r['centroid_dist']['real']:9.1f} {100*r['quad_match']['real']:7.1f} {r['dice']['real']:6.3f} {100*r['recall']['real']:7.1f} "
              f"{100*r['ebpg_core']['real']:8.1f} {100*r['hit_peak']['real']:6.1f}   {dct:+.1f} {ic}")
print(f"\nGuardado en {OUT} ({out['tiempo_s']:.0f} s, {DEV})")
