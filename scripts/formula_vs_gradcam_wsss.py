# -*- coding: utf-8 -*-
"""
formula_vs_gradcam_wsss.py  --  ¿Puede la formula simbolica reconstruir el mapa de Grad-CAM?

Fundamento (se demuestra en el texto; aqui se verifica numericamente):
  u0 = sum_c W_c * GAP_c + b  y  GAP_c = (1/49) sum_{x,y} A_c(x,y)  =>  u0 = (1/49) sum_{x,y} psi(x,y) + b,
  psi(x,y) = sum_c W_c A_c(x,y).  Para g(u0) con g' < 0:
        d g / d A_c(x,y) = g'(u0) * W_c / 49   (constante en el espacio)
  =>  Grad-CAM de la formula = ReLU( |g'(u0)| * sum_c (-W_c) A_c ) = |g'| * ReLU(-psi),
  y tras normalizar min-max no depende de g: el mapa de la formula es el CAM lineal de la
  direccion de u0. La parte no lineal de g solo cambia la escala global.
  Analogamente, Grad-CAM (clase 1) de la ResNet con capa lineal es ReLU(sum_c w1_c A_c).
  Por tanto, la fidelidad con que la formula "reconstruye" el Grad-CAM de la red queda fijada
  por la alineacion entre los vectores de pesos  -W (direccion de malignidad de u0)  y  w1 (o w1-w0).

Este script mide:
  (A) cosenos y correlaciones entre -W y los pesos de la capa lineal (w1, w1-w0);
  (B) verificacion numerica de que el Grad-CAM por autograd == ReLU(sum w1 A) (tras normalizar);
  (C) similitud mapa a mapa (Pearson, Spearman, IoU del 15 % superior, distancia entre picos)
      entre psi_KAN y el Grad-CAM robusto, en las 368 imagenes;
  (D) el operador focal WSSS (identico al NB28) aplicado al mapa de la formula, con el control
      de azar 'cam barajado', frente al Grad-CAM robusto.

Uso:  python scripts/formula_vs_gradcam_wsss.py [RAIZ] [SALIDA_DIR] [--B 20] [--smoke]
Salidas: formula_vs_gradcam_wsss.json, formula_vs_gradcam_wsss_porimagen.csv
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("codigos", nargs="?", default=None, help="raiz del repositorio (por defecto, la carpeta padre de scripts/)"); ap.add_argument("salida", nargs="?", default=None)
ap.add_argument("--B", type=int, default=20); ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
CODIGOS = os.path.abspath(args.codigos or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = args.salida or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "auditoria")
os.makedirs(OUT, exist_ok=True)
for p in (CODIGOS, os.path.join(CODIGOS, "src")):
    if p not in sys.path: sys.path.insert(0, p)
os.environ.setdefault("TESIS_CODIGOS", CODIGOS)

import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models
from PIL import Image
from scipy.stats import wilcoxon, spearmanr, pearsonr
from src.data.roi_loader import get_roi_manifest, CBISROIDataset
from src.explainability.weak_segmentation import (extract_breast_tissue_mask, segment_wsss_focal_core,
                                                  compute_centroid, compute_segmentation_metrics)
from src.reporting.clinical_evidence_extractor import determine_quadrant

DEV = "cuda" if torch.cuda.is_available() else "cpu"; T0 = time.time()

# ---------------------------------------------------------------- modelo, reductor, pesos
m = models.resnet18(weights=None); m.fc = nn.Linear(512, 2)
ck = torch.load(os.path.join(CODIGOS, "reports", "models", "resnet18_robust_best.pt"), map_location=DEV)
m.load_state_dict(ck["model_state_dict"]); m = m.to(DEV).eval()
w1 = m.fc.weight[1].detach().cpu().numpy().astype(np.float64)
w0 = m.fc.weight[0].detach().cpu().numpy().astype(np.float64)

R = np.load(os.path.join(CODIGOS, "reports", "experiments", "kan_robust64_reducer.npz"))
v = R["proj_w"].T.astype(np.float64) @ R["pca_components"][0].astype(np.float64)      # 512
W = v / R["gap_sd"][0].astype(np.float64) / float(R["sd_z"][0, 0])                       # u0 = GAP @ W + b
W_mal = -W                                                                              # g' < 0  => direccion de malignidad


def cosine(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


pesos = {
    "cos(-W, w1)": cosine(W_mal, w1), "cos(-W, w1-w0)": cosine(W_mal, w1 - w0),
    "pearson(-W, w1)": float(pearsonr(W_mal, w1)[0]), "pearson(-W, w1-w0)": float(pearsonr(W_mal, w1 - w0)[0]),
    "spearman(-W, w1-w0)": float(spearmanr(W_mal, w1 - w0)[0]),
}
print("(A) alineacion de pesos:", json.dumps(pesos), flush=True)


class Hook:
    def __init__(self, layer):
        layer.register_forward_hook(lambda mod, i, o: setattr(self, "act", o))
        layer.register_full_backward_hook(lambda mod, gi, go: setattr(self, "grad", go[0]))


hk = Hook(m.layer4)


def up(cam7):
    cam7 = F.relu(torch.as_tensor(cam7)).numpy()
    cam7 = (cam7 - cam7.min()) / (cam7.max() - cam7.min() + 1e-8) if cam7.max() > cam7.min() else np.zeros_like(cam7)
    im = Image.fromarray((cam7 * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    return np.array(im, dtype=np.float32) / 255.0


def gradcam_autograd(x, cls=1):
    m.zero_grad(); out = m(x); out[0, cls].backward()
    w = hk.grad[0].mean(dim=(1, 2)); cam = (hk.act[0] * w.view(-1, 1, 1)).sum(0).detach().cpu().numpy()
    return up(cam)


def quad(c): return determine_quadrant(c[0], c[1], "LEFT", "MLO")


def one(cam, tissue, gt, gt_q):
    core, cam_t, peak = segment_wsss_focal_core(cam, tissue, min_tau_ratio=0.60)
    mm = compute_segmentation_metrics(core, gt); c = compute_centroid(core); cb = core > 0.5
    en = float((cam_t * cb).sum())
    return {"centroid_dist": mm["centroid_dist"], "quad_match": float(quad(c) == gt_q), "dice": mm["dice"],
            "recall": mm["recall"], "ebpg_cont": float((cam_t * gt).sum() / (cam_t.sum() + 1e-8)),
            "ebpg_core": float((cam_t * cb * gt).sum() / (en + 1e-8)) if en > 0 else 0.0,
            "hit_peak": float(gt[peak] > 0.5)}


# ---------------------------------------------------------------- datos y mapas
df = get_roi_manifest(CODIGOS); ds = CBISROIDataset(df, image_size=224)
N = len(ds) if not args.smoke else 12
if not args.smoke: assert N == 368, N
M = {k: [] for k in ("gradcam", "cam_w1", "cam_wdiff", "psi")}
tissue, gts, gt_q, labels, pats = [], [], [], [], []
u0_chk, logit_diff, u0_err = [], [], []
b_u0 = float(-(R["gap_mu"][0].astype(np.float64) @ W))  # constante aproximada (se usa solo para verificar monotonia)
for i in range(N):
    it = ds[i]; x = it["image_tensor"].unsqueeze(0).to(DEV)
    raw = it["image_tensor"].numpy()[0]; disp = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    tissue.append(extract_breast_tissue_mask(disp)); g = it["mask_np"]; gts.append(g)
    gt_q.append(quad(compute_centroid(g))); labels.append(int(it["label"])); pats.append(it["patient_id"])
    M["gradcam"].append(gradcam_autograd(x))
    A = hk.act[0].detach().cpu().numpy().astype(np.float64)                      # 512 x 7 x 7
    M["cam_w1"].append(up(np.tensordot(w1, A, axes=(0, 0))))
    M["cam_wdiff"].append(up(np.tensordot(w1 - w0, A, axes=(0, 0))))
    M["psi"].append(up(np.tensordot(W_mal, A, axes=(0, 0))))
    gap = A.mean(axis=(1, 2)); u0_chk.append(float(gap @ W + b_u0)); logit_diff.append(float((w1 - w0) @ gap))
    if (i + 1) % 50 == 0: print(f"[{time.time()-T0:5.0f}s] mapas {i+1}/{N}", flush=True)
labels = np.array(labels); pats = np.array(pats)

# (B) identidad Grad-CAM == CAM lineal
dif_b = [float(np.abs(M["gradcam"][i] - M["cam_w1"][i]).max()) for i in range(N)]
ident = {"max_abs_dif_gradcam_vs_cam_w1": float(np.max(dif_b)), "media": float(np.mean(dif_b))}
print("(B) Grad-CAM autograd vs ReLU(sum w1 A):", ident, flush=True)
corr_u0_logit = float(spearmanr(u0_chk, logit_diff)[0])


# (C) similitud de mapas
def iou_top(a, b, q=85):
    ta, tb = a >= np.percentile(a, q), b >= np.percentile(b, q)
    return float((ta & tb).sum() / max((ta | tb).sum(), 1))


def peak_dist(a, b):
    pa, pb = np.unravel_index(np.argmax(a), a.shape), np.unravel_index(np.argmax(b), b.shape)
    return float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))


sim = []
for i in range(N):
    for ref in ("gradcam", "cam_wdiff"):
        a, b = M["psi"][i].ravel(), M[ref][i].ravel()
        sim.append({"idx": i, "ref": ref, "pearson": float(pearsonr(a, b)[0]) if a.std() > 0 and b.std() > 0 else np.nan,
                    "spearman": float(spearmanr(a, b)[0]), "iou_top15": iou_top(M["psi"][i], M[ref][i]),
                    "dist_picos_px": peak_dist(M["psi"][i], M[ref][i])})
S = pd.DataFrame(sim)

# (D) WSSS con el mapa de la formula + nulo de cam barajado
rng = np.random.default_rng(0); rows = []
for arm, key in (("gradcam_robusto", "gradcam"), ("formula_psi", "psi")):
    for i in range(N):
        r = one(M[key][i], tissue[i], gts[i], gt_q[i]); r.update(idx=i, brazo=arm, cond="real"); rows.append(r)
    for b in range(args.B):
        perm = rng.permutation(N)
        while np.any(perm == np.arange(N)): perm = rng.permutation(N)
        for i in range(N):
            r = one(M[key][perm[i]], tissue[i], gts[i], gt_q[i]); r.update(idx=i, brazo=arm, cond="cam_barajado", rep=b); rows.append(r)
    print(f"[{time.time()-T0:5.0f}s] WSSS {arm} listo", flush=True)
D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, "formula_vs_gradcam_wsss_porimagen.csv"), index=False)

upat = np.unique(pats); pidx = {p: np.where(pats == p)[0] for p in upat}; brng = np.random.default_rng(0)


def boot(vec, mask, B=2000):
    sel = [p for p in upat if mask[pidx[p]].any()]; est = []
    for _ in range(B):
        ps = brng.choice(len(sel), len(sel), replace=True)
        ii = np.concatenate([pidx[sel[q]][mask[pidx[sel[q]]]] for q in ps]); est.append(np.nanmean(vec[ii]))
    return [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))]


def pi(df_, **flt):
    s = df_
    for k, val in flt.items(): s = s[s[k] == val]
    return s.groupby("idx").mean(numeric_only=True)


out = {"n": N, "tiempo_s": round(time.time() - T0, 1), "A_alineacion_pesos": pesos,
       "B_identidad_gradcam_cam_lineal": ident, "spearman_u0_vs_logit_diff": corr_u0_logit,
       "C_similitud_mapas": {}, "D_wsss": {}}
for nm, mask in (("global", np.ones(N, bool)), ("malignas", labels == 1)):
    out["C_similitud_mapas"][nm] = {}
    for ref in ("gradcam", "cam_wdiff"):
        sub = pi(S, ref=ref).reindex(range(N))
        out["C_similitud_mapas"][nm][ref] = {c: {"media": float(np.nanmean(sub[c].values[mask])),
                                                 "ic95": boot(sub[c].values, mask)}
                                             for c in ("pearson", "spearman", "iou_top15", "dist_picos_px")}
    out["D_wsss"][nm] = {}
    for arm in ("gradcam_robusto", "formula_psi"):
        real, nul = pi(D, brazo=arm, cond="real").reindex(range(N)), pi(D, brazo=arm, cond="cam_barajado").reindex(range(N))
        out["D_wsss"][nm][arm] = {}
        for met in ("centroid_dist", "quad_match", "dice", "recall", "ebpg_cont", "ebpg_core", "hit_peak"):
            rv, nv = real[met].values, nul[met].values
            try: p = float(wilcoxon(rv[mask], nv[mask]).pvalue)
            except Exception: p = None
            out["D_wsss"][nm][arm][met] = {"real": float(rv[mask].mean()), "real_ic95": boot(rv, mask),
                                           "nulo_barajado": float(nv[mask].mean()), "dif": float((rv - nv)[mask].mean()),
                                           "dif_ic95": boot(rv - nv, mask), "wilcoxon_p": p}
json.dump(out, open(os.path.join(OUT, "formula_vs_gradcam_wsss.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print("\n(C) similitud psi_KAN vs Grad-CAM robusto (malignas):")
for c, v_ in out["C_similitud_mapas"]["malignas"]["gradcam"].items(): print(f"   {c:14s} {v_['media']:.3f} {v_['ic95']}")
print("(D) WSSS malignas:")
for arm in ("gradcam_robusto", "formula_psi"):
    for met in ("centroid_dist", "quad_match", "dice", "ebpg_core", "hit_peak"):
        v_ = out["D_wsss"]["malignas"][arm][met]
        print(f"   {arm:16s} {met:14s} real {v_['real']:.4f} | nulo {v_['nulo_barajado']:.4f} | dif {v_['dif']:+.4f} {v_['dif_ic95']} p={v_['wilcoxon_p']}")
print(f"Guardado en {OUT} ({time.time()-T0:.0f}s)")
