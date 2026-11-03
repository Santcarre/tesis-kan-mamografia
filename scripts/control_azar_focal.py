# -*- coding: utf-8 -*-
"""
control_azar_focal.py  --  Control de azar y reproduccion del operador focal (Cap. 4, 4.6-4.7)

Motivo. Las cifras del Pointing Game focal de la tesis (EBPG 23-29 %, Top-1 26-33 %)
provienen de un CSV cuyo codigo generador no esta en el repositorio; el notebook 33
no logra reproducirlas (p. ej. P_00202: 0,0037 frente a 0,965). Este script:

  1. Reproduce desde el repositorio el pipeline del notebook 28 (Grad-CAM de layer4,
     mascara tisular, nucleo focal >= 60 %) para ResNet-18 base y robusto sobre las
     368 mamografias con mascara 1:1, y mide: distancia de centroides, acierto de
     cuadrante, Dice, recall, EBPG continuo, EBPG dentro del nucleo, Top-1 del pico.
  2. Calcula controles de azar con el MISMO operador:
       - cam_barajado : se usa el Grad-CAM de OTRA imagen (permutacion sin puntos fijos)
                        con la mascara tisular y la lesion de la imagen evaluada.
       - centro_tejido: el centroide predicho es el centroide de la mascara tisular.
       - punto_azar   : centroide predicho = punto uniforme dentro del tejido.
  3. Resume con IC95 % por bootstrap agrupado por paciente (B=2000, seed 0) y
     diferencias real - nulo.

Uso (desde la raiz del repo, con el entorno que tiene torch):
    python scripts/control_azar_focal.py [RAIZ] [SALIDA_DIR] [--B 20] [--smoke]

--smoke procesa solo 12 imagenes para comprobar que todo corre.
Salidas: control_azar_focal.json y control_azar_focal_porimagen.csv
Comprobacion interna: el Robusto Tisular debe dar ~64,44 px (malignos) y ~38,7 % de
cuadrante; si no, hay un desajuste de pipeline y se reporta en el JSON.
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("codigos", nargs="?", default=None, help="raiz del repositorio (por defecto, la carpeta padre de scripts/)")
ap.add_argument("salida", nargs="?", default=None)
ap.add_argument("--B", type=int, default=20, help="repeticiones de los nulos aleatorios")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

CODIGOS = os.path.abspath(args.codigos or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = args.salida or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "auditoria")
os.makedirs(OUT, exist_ok=True)
for p in (CODIGOS, os.path.join(CODIGOS, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("TESIS_CODIGOS", CODIGOS)

import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models
from PIL import Image
from scipy.stats import wilcoxon

from src.data.roi_loader import get_roi_manifest, CBISROIDataset
from src.explainability.weak_segmentation import (
    extract_breast_tissue_mask, segment_wsss_focal_core, compute_centroid,
    compute_segmentation_metrics)
from src.reporting.clinical_evidence_extractor import determine_quadrant

DEV = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.time()


class GradCAM:
    """Identico al del notebook 28 (layer4, clase maligna)."""
    def __init__(self, model, layer):
        self.model, self.layer = model, layer
        layer.register_forward_hook(lambda m, i, o: setattr(self, "act", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grad", go[0]))

    def __call__(self, x, class_idx=1):
        self.model.zero_grad()
        out = self.model(x)
        out[0, class_idx].backward(retain_graph=False)
        w = self.grad[0].mean(dim=(1, 2))
        cam = F.relu((self.act[0] * w.view(-1, 1, 1)).sum(dim=0)).detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8) if cam.max() > cam.min() else np.zeros_like(cam)
        img = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0


def load_model(ckpt_name):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(512, 2)
    ck = torch.load(os.path.join(CODIGOS, "reports", "models", ckpt_name), map_location=DEV)
    m.load_state_dict(ck["model_state_dict"])
    return m.to(DEV).eval()


def quad(c):
    return determine_quadrant(c[0], c[1], "LEFT", "MLO")   # igual que el notebook 28


def one(cam, tissue, gt, gt_q):
    """Metricas del operador focal para un mapa dado."""
    core, cam_t, peak = segment_wsss_focal_core(cam, tissue, min_tau_ratio=0.60)
    m = compute_segmentation_metrics(core, gt)
    c = compute_centroid(core)
    core_b = core > 0.5
    en_core = float((cam_t * core_b).sum())
    return {
        "centroid_dist": m["centroid_dist"],
        "quad_match": float(quad(c) == gt_q),
        "dice": m["dice"], "recall": m["recall"], "precision": m["precision"],
        "ebpg_cont": float((cam_t * gt).sum() / (cam_t.sum() + 1e-8)),
        "ebpg_core": float((cam_t * core_b * gt).sum() / (en_core + 1e-8)) if en_core > 0 else 0.0,
        "hit_peak": float(gt[peak] > 0.5),
        "core_area": float(core_b.sum()),
    }


def point_metrics(c_pred, gt, gt_c, gt_q):
    return {"centroid_dist": float(np.hypot(c_pred[0] - gt_c[0], c_pred[1] - gt_c[1])),
            "quad_match": float(quad(c_pred) == gt_q)}


# ------------------------------------------------------------------ datos
df = get_roi_manifest(CODIGOS)
ds = CBISROIDataset(df, image_size=224)
N = len(ds) if not args.smoke else 12
if not args.smoke:
    assert N == 368, f"se esperaban 368 mascaras, hay {N}"
print(f"[{time.time()-T0:5.0f}s] {N} mamografias | dispositivo {DEV}", flush=True)

models_ = {"base": load_model("resnet18_baseline_v2.pt"), "robusto": load_model("resnet18_robust_best.pt")}
cams = {k: [] for k in models_}
tissue, gts, gt_c, gt_q, labels, pats = [], [], [], [], [], []
gc = {k: GradCAM(m, m.layer4) for k, m in models_.items()}
for i in range(N):
    it = ds[i]
    x = it["image_tensor"].unsqueeze(0).to(DEV)
    raw = it["image_tensor"].cpu().numpy()[0]
    disp = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    tissue.append(extract_breast_tissue_mask(disp))
    g = it["mask_np"]; gts.append(g); gt_c.append(compute_centroid(g)); gt_q.append(quad(compute_centroid(g)))
    labels.append(int(it["label"])); pats.append(it["patient_id"])
    for k in gc:
        cams[k].append(gc[k](x))
    if (i + 1) % 50 == 0:
        print(f"[{time.time()-T0:5.0f}s] Grad-CAM {i+1}/{N}", flush=True)
labels = np.array(labels); pats = np.array(pats)

# ------------------------------------------------------------------ metricas reales y nulos
rng = np.random.default_rng(0)
rows = []
for k in cams:
    for i in range(N):
        r = one(cams[k][i], tissue[i], gts[i], gt_q[i]); r.update(idx=i, modelo=k, condicion="real"); rows.append(r)
    for b in range(args.B):
        perm = rng.permutation(N)
        while np.any(perm == np.arange(N)):                      # sin puntos fijos
            perm = rng.permutation(N)
        for i in range(N):
            r = one(cams[k][perm[i]], tissue[i], gts[i], gt_q[i]); r.update(idx=i, modelo=k, condicion="cam_barajado", rep=b); rows.append(r)
    print(f"[{time.time()-T0:5.0f}s] {k}: real + {args.B} barajados listos", flush=True)

for i in range(N):                                                # nulos puntuales (independientes del modelo)
    ys, xs = np.where(tissue[i] > 0.5)
    c_t = compute_centroid(tissue[i])
    r = point_metrics(c_t, gts[i], gt_c[i], gt_q[i]); r.update(idx=i, modelo="-", condicion="centro_tejido"); rows.append(r)
    for b in range(args.B):
        j = rng.integers(len(ys)); r = point_metrics((float(ys[j]), float(xs[j])), gts[i], gt_c[i], gt_q[i])
        r.update(idx=i, modelo="-", condicion="punto_azar", rep=b); rows.append(r)

R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "control_azar_focal_porimagen.csv"), index=False)

# ------------------------------------------------------------------ resumen
METR = ["centroid_dist", "quad_match", "dice", "recall", "ebpg_cont", "ebpg_core", "hit_peak"]
upat = np.unique(pats); pidx = {p: np.where(pats == p)[0] for p in upat}
brng = np.random.default_rng(0)


def boot(v, mask, B=2000):
    """IC95 % de la media, remuestreando pacientes. v: vector por imagen."""
    sel = [p for p in upat if mask[pidx[p]].any()]
    est = []
    for _ in range(B):
        ps = brng.choice(len(sel), len(sel), replace=True)
        ii = np.concatenate([pidx[sel[q]][mask[pidx[sel[q]]]] for q in ps])
        est.append(v[ii].mean())
    return [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))]


def per_image(modelo, cond, met):
    sub = R[(R.modelo == modelo) & (R.condicion == cond)]
    return sub.groupby("idx")[met].mean().reindex(range(N)).values      # media sobre repeticiones


out = {"n": N, "n_malignas": int((labels == 1).sum()), "B": args.B, "tiempo_s": round(time.time() - T0, 1), "resumen": {}}
for sub_name, mask in [("malignas", labels == 1), ("global", np.ones(N, bool))]:
    out["resumen"][sub_name] = {}
    for modelo in ("base", "robusto"):
        out["resumen"][sub_name][modelo] = {}
        for met in METR:
            real = per_image(modelo, "real", met)
            nul = per_image(modelo, "cam_barajado", met)
            d = real - nul
            try:
                p = float(wilcoxon(real[mask], nul[mask]).pvalue)
            except Exception:
                p = None
            out["resumen"][sub_name][modelo][met] = {
                "real": float(real[mask].mean()), "real_ic95": boot(real, mask),
                "nulo_cam_barajado": float(nul[mask].mean()), "diferencia": float(d[mask].mean()),
                "diferencia_ic95": boot(d, mask), "wilcoxon_p": p}
    out["resumen"][sub_name]["nulos_puntuales"] = {}
    for cond in ("centro_tejido", "punto_azar"):
        out["resumen"][sub_name]["nulos_puntuales"][cond] = {
            met: float(per_image("-", cond, met)[mask].mean()) for met in ("centroid_dist", "quad_match")}

ref = out["resumen"]["malignas"]["robusto"]
out["comprobacion_nb28"] = {
    "centroid_dist_malignas_robusto": ref["centroid_dist"]["real"], "centroid_dist_esperado_nb28": 64.44,
    "quad_match_malignas_robusto": ref["quad_match"]["real"], "quad_match_esperado_nb28": 0.3871}
json.dump(out, open(os.path.join(OUT, "control_azar_focal.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print("\n=== MALIGNAS (n=%d) ===" % out["n_malignas"])
for modelo in ("base", "robusto"):
    for met in ("centroid_dist", "quad_match", "dice", "ebpg_core", "hit_peak"):
        v = out["resumen"]["malignas"][modelo][met]
        print(f"{modelo:8s} {met:14s} real {v['real']:.4f} {v['real_ic95']} | nulo {v['nulo_cam_barajado']:.4f} | dif {v['diferencia']:+.4f} {v['diferencia_ic95']} p={v['wilcoxon_p']}")
print("nulos puntuales:", out["resumen"]["malignas"]["nulos_puntuales"])
print("comprobacion NB28:", out["comprobacion_nb28"])
print(f"Guardado en {OUT}  ({time.time()-T0:.0f}s)")
