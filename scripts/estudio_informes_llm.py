# -*- coding: utf-8 -*-
"""
estudio_informes_llm.py -- Estudio de informes con modelos de lenguaje locales (Cap. 4, informes).

Amplía el piloto de 32 imágenes a las 368 mamografías de prueba con contorno y a cinco modelos de lenguaje
locales de tres familias (más uno opcional servido con llama.cpp). Dos textos de entrada con las MISMAS
instrucciones (Anexo A), que solo difieren en la información:
  - ciego:     format_control_prompt (la clasificación de la fórmula con su umbral y su puntaje);
  - evidencia: format_evidence_prompt (la misma clasificación, más u0 y g(u0), la categoría BI-RADS
               orientativa de la herramienta y la región de la imagen y el área de la zona marcada por el
               núcleo focal sobre el mapa de la fórmula psi).
Versión 2 (1-nov-2026): el control recibe la misma clasificación que la evidencia (antes, la de la red robusta
con corte 0,5, que discrepaba en 36 de 368 imágenes); la categoría BI-RADS está anclada en la decisión
(src/reporting/birads_herramienta.py); la región se presenta como región de la imagen y se retiró la
compacidad. La corrida anterior no se incluye en este repositorio.

Este script SOLO genera los informes y guarda el texto completo. Las métricas se calculan aparte con
scripts/analizar_informes_llm.py, para poder validar y corregir el detector sin volver a generar.

Desenlaces principales, fijados antes de correr el estudio:
  (1) transmisión de la categoría: en la condición con evidencia, el informe escribe la categoría BI-RADS
      que recibió;
  (2) afirmaciones sin respaldo: el informe afirma o niega hallazgos que no están en la evidencia
      (márgenes, forma, tamaño, morfología de calcificaciones, distorsión, densidad, etc.).
Secundarios: región reportada frente a la del contorno clínico, tipo de lesión erróneo, categoría omitida o
incompatible con la patología.

Ajustes iguales para todos los modelos: temperatura 0,2; hasta 1024 tokens de salida; ventana de 2048 tokens;
semilla = índice de la imagen (la misma en las dos condiciones y en todos los modelos). Una repetición.
En los modelos con razonamiento previo (Qwen 3.x, Gemma 4) se desactiva ese modo, y cualquier bloque
<think>...</think> que aparezca se elimina del texto.

Modelo opcional con llama.cpp (p. ej., Ternary Bonsai, que no corre en Ollama): se nombra con el prefijo
«llamacpp:» y se sirve con llama-server en LLAMACPP_URL (por defecto http://localhost:8080), p. ej.
    python scripts/estudio_informes_llm.py --modelos llamacpp:bonsai-27b

Localizador: la región y el área que recibe el modelo salen del núcleo focal (>= 60 %, máscara
de tejido) del mapa de la fórmula psi calculado con la imagen a RESOLUCION_MAPA px (448 por defecto; ver
segmentacion_resolucion.py: a 448 px el mapa acierta la región más que el centro del tejido). La decisión,
u0, g(u0), el puntaje y la categoría se calculan siempre con la imagen a 224 px, como en el resto de la tesis.

Uso (desde la raíz del repositorio, con Ollama corriendo):
    python scripts/estudio_informes_llm.py --smoke      # 3 imágenes x 5 modelos: comprueba todo y estima el tiempo
    python scripts/estudio_informes_llm.py              # estudio completo (se puede interrumpir y reanudar)
Salidas: reports/experiments/estudio_informes_llm[_smoke].csv  (un renglón por modelo x imagen x condición)
         reports/experiments/estudio_informes_llm_evidencia.json (evidencia por imagen, en caché)
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd, requests

MODELOS = ["qwen2.5:1.5b-instruct", "qwen2.5:3b", "phi3.5", "qwen3.5:4b", "gemma4:12b"]
PIENSAN = ("qwen3", "gemma4", "deepseek-r1", "magistral")      # modelos con razonamiento previo
ap = argparse.ArgumentParser()
ap.add_argument("--modelos", nargs="+", default=MODELOS)
ap.add_argument("--smoke", action="store_true", help="3 imágenes con todos los modelos")
ap.add_argument("--resolucion-mapa", type=int, default=448,
                help="lado de la imagen con que se calcula el mapa psi del localizador (la decisión siempre es a 224)")
args = ap.parse_args()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "src")):
    if p not in sys.path: sys.path.insert(0, p)
os.environ.setdefault("TESIS_CODIGOS", REPO)
EXP = os.path.join(REPO, "reports", "experiments")
SUF = "_smoke" if args.smoke else ""
OUT_CSV = os.path.join(EXP, f"estudio_informes_llm{SUF}.csv")
EV_JSON = os.path.join(EXP, f"estudio_informes_llm_evidencia_{args.resolucion_mapa}.json")
BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
LCPP = os.environ.get("LLAMACPP_URL", "http://localhost:8080")
OPC = {"temperature": 0.2, "num_predict": 1024, "num_ctx": 2048}
T0 = time.time()

# ------------------------------------------------------------------ 1. modelos disponibles en Ollama
olla = [m for m in args.modelos if not m.startswith("llamacpp:")]
falta = []
if olla:
    try:
        tags = requests.get(BASE + "/api/tags", timeout=10).json().get("models", [])
    except Exception as exc:
        sys.exit(f"No se pudo contactar a Ollama en {BASE} ({exc}). Abre Ollama y vuelve a intentar.")
    disp = {t["name"] for t in tags}
    falta = [m for m in olla if m not in disp and m + ":latest" not in disp]
if any(m.startswith("llamacpp:") for m in args.modelos):
    try:
        requests.get(LCPP + "/health", timeout=10).raise_for_status()
    except Exception as exc:
        sys.exit(f"No se pudo contactar a llama-server en {LCPP} ({exc}). Inícialo antes de correr el script.")
if falta:
    print("Faltan modelos en Ollama. Descárgalos con:")
    for m in falta: print(f"    ollama pull {m}")
    sys.exit(1)
print(f"Modelos listos: {', '.join(args.modelos)}", flush=True)

# ------------------------------------------------------------------ 2. evidencia por imagen (en caché)
import torch
from PIL import Image
from src.data.roi_loader import CBISROIDataset, get_roi_manifest
from src.reporting.clinical_evidence_extractor import (ClinicalEvidenceExtractor, determine_quadrant,
                                                       compute_compactness_and_eccentricity)
from src.reporting.llm_engine import format_control_prompt, format_evidence_prompt
from src.explainability.weak_segmentation import extract_breast_tissue_mask, segment_wsss_focal_core, compute_centroid

if os.path.exists(EV_JSON):
    casos = json.load(open(EV_JSON, encoding="utf-8"))
    print(f"Evidencia leída de la caché: {len(casos)} imágenes", flush=True)
else:
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    df = get_roi_manifest(REPO); ds = CBISROIDataset(df, image_size=224)
    ex = ClinicalEvidenceExtractor(device=DEV)
    assert ex.kan64 is not None, "no se pudo cargar la fórmula canónica"
    R = np.load(os.path.join(EXP, "kan_robust64_reducer.npz"))
    v = R["proj_w"].T.astype(np.float64) @ R["pca_components"][0].astype(np.float64)
    W_mal = -(v / R["gap_sd"].ravel().astype(np.float64) / float(R["sd_z"].ravel()[0]))

    import torch.nn.functional as F
    import torchvision.transforms as T
    Image.MAX_IMAGE_PIXELS = None
    NORM = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def mapa_psi(pil, tissue, lado):
        """Igual que segmentacion_resolucion.py: psi = ReLU(sum -W A) con la imagen a lado x lado px,
        interpolado a 224 x 224 y normalizado a [0, 1] dentro del tejido."""
        m = ex.model
        with torch.no_grad():
            x = NORM(pil.resize((lado, lado), Image.BILINEAR)).unsqueeze(0).to(DEV)
            x = m.maxpool(m.relu(m.bn1(m.conv1(x)))); A = m.layer4(m.layer3(m.layer2(m.layer1(x))))[0]
        psi = np.maximum(np.tensordot(W_mal, A.double().cpu().numpy(), axes=(0, 0)), 0)
        t = F.interpolate(torch.as_tensor(psi, dtype=torch.float32)[None, None], size=(224, 224),
                          mode="bilinear", align_corners=False)[0, 0].numpy()
        vals = t[tissue > 0.5] if (tissue > 0.5).any() else t.ravel()
        lo, hi = float(vals.min()), float(vals.max())
        return np.zeros_like(t) if hi <= lo else ((t - lo) / (hi - lo)).clip(0, 1).astype(np.float32)

    casos = []
    for idx in range(len(ds)):
        item = ds[idx]; rel = df.iloc[idx]["rel_img_path"]
        ev = ex.extract_evidence(item, rel)
        raw = item["image_tensor"].numpy()[0]; disp_img = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        tissue = extract_breast_tissue_mask(disp_img)
        pil = Image.open(df.iloc[idx]["image_path"]).convert("RGB")
        core, _, _ = segment_wsss_focal_core(mapa_psi(pil, tissue, args.resolucion_mapa), tissue, min_tau_ratio=0.60)
        cy, cx = compute_centroid(core); area, _, comp, _ = compute_compactness_and_eccentricity(core)
        casos.append({"idx": idx, "patient_id": ev["patient_id"], "view": ev["view"], "laterality": ev["laterality"],
                      "prob_cnn": float(ev["prob_cnn"]), "kan64_z64_0": float(ev["kan64_z64_0"]),
                      "kan64_g": float(ev["kan64_g"]), "kan64_p": float(ev["kan64_p"]), "kan64_th": float(ev["kan64_th"]),
                      "kan64_birads": ev["kan64_birads"],
                      "predicted_quadrant": determine_quadrant(cy, cx, ev["laterality"], ev["view"]),
                      "pred_area_px": int(area), "compactness": float(comp),
                      "gt_quadrant": ev["gt_quadrant"], "abnormality_type": ev["abnormality_type"],
                      "pathology_gt": str(ev["pathology_gt"])})
        if (idx + 1) % 25 == 0: print(f"[{time.time()-T0:5.0f}s] evidencia {idx+1}/{len(ds)}", flush=True)
    json.dump(casos, open(EV_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    err = np.mean([c["predicted_quadrant"] != c["gt_quadrant"] for c in casos])
    err_m = np.mean([c["predicted_quadrant"] != c["gt_quadrant"] for c in casos if str(c["pathology_gt"]).upper().startswith("MALIGNANT")])
    print(f"Evidencia guardada para {len(casos)} imágenes ({len({c['patient_id'] for c in casos})} pacientes). "
          f"Error de región del localizador a {args.resolucion_mapa} px: {err:.3f} (malignas: {err_m:.3f})", flush=True)

if args.smoke: casos = casos[:3]

# Categoría orientativa anclada en la decisión (fuente única: src/reporting/birads_herramienta.py). Se recalcula
# desde g y tau, porque la caché de evidencia se generó con la convención anterior.
from src.reporting import birads_herramienta as BH
for c in casos:
    c["kan64_birads"] = BH.etiqueta(c["kan64_g"], c["kan64_th"])

# ------------------------------------------------------------------ 3. generación (reanudable)
import re
def sin_razonamiento(t):
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    return re.sub(r"^.*?</think>", "", t, flags=re.S).strip()   # bloque sin etiqueta de apertura


def generar(model, prompt, seed):
    piensa = model.lower().startswith(PIENSAN)
    for intento in range(3):
        try:
            if model.startswith("llamacpp:"):
                r = requests.post(LCPP + "/v1/chat/completions", timeout=1800, json={
                    "messages": [{"role": "user", "content": prompt}], "temperature": OPC["temperature"],
                    "max_tokens": OPC["num_predict"], "seed": int(seed),
                    "chat_template_kwargs": {"enable_thinking": False}})
                if r.status_code == 200:
                    j = r.json(); txt = j["choices"][0]["message"].get("content") or ""
                    return sin_razonamiento(txt), (j.get("usage") or {}).get("completion_tokens"), None
            else:
                cuerpo = {"model": model, "prompt": prompt, "stream": False, "options": {**OPC, "seed": int(seed)}}
                if piensa: cuerpo["think"] = False
                r = requests.post(BASE + "/api/generate", json=cuerpo, timeout=1800)
                if r.status_code == 400 and "think" in r.text.lower() and piensa:
                    cuerpo.pop("think"); r = requests.post(BASE + "/api/generate", json=cuerpo, timeout=1800)
                if r.status_code == 200:
                    j = r.json(); return sin_razonamiento(j.get("response", "")), j.get("eval_count"), None
            err = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as exc:
            err = f"EXC {exc}"
        time.sleep(5)
    return "", None, err


def guardar(filas):
    """Escritura atómica: se escribe a un archivo temporal y se reemplaza, para que una interrupción
    en medio de la escritura no deje el CSV dañado."""
    tmp = OUT_CSV + ".tmp"
    pd.DataFrame(filas).to_csv(tmp, index=False, encoding="utf-8"); os.replace(tmp, OUT_CSV)


hechos = set()
filas = []
if os.path.exists(OUT_CSV):
    prev = pd.read_csv(OUT_CSV)
    prev = prev[prev["error"].isna()]
    filas = prev.to_dict("records")
    hechos = {(r["modelo"], int(r["idx"]), r["condicion"]) for r in filas}
    print(f"Reanudando: {len(hechos)} informes ya generados", flush=True)

total = len(args.modelos) * len(casos) * 2
for model in args.modelos:
    t_mod = time.time(); n_mod = 0
    for e in casos:
        for cond, prompt in (("ciego", format_control_prompt(e)), ("evidencia", format_evidence_prompt(e))):
            if (model, e["idx"], cond) in hechos: continue
            t1 = time.time()
            txt, ntok, err = generar(model, prompt, seed=e["idx"])
            filas.append({"modelo": model, "idx": e["idx"], "condicion": cond, "patient_id": e["patient_id"],
                          "view": e["view"], "laterality": e["laterality"], "pathology_gt": e["pathology_gt"],
                          "abnormality_type": e["abnormality_type"], "gt_quadrant": e["gt_quadrant"],
                          "region_dada": e["predicted_quadrant"], "birads_dada": e["kan64_birads"],
                          "prob_cnn": e["prob_cnn"], "u0": e["kan64_z64_0"], "g": e["kan64_g"], "p": e["kan64_p"],
                          "tau": e["kan64_th"], "tokens": ntok, "max_tokens": OPC["num_predict"], "segundos": round(time.time() - t1, 2),
                          "error": err, "texto": txt})
            hechos.add((model, e["idx"], cond)); n_mod += 1
            if len(filas) % 10 == 0: guardar(filas)
        if n_mod and (e["idx"] + 1) % 25 == 0:
            vel = (time.time() - t_mod) / n_mod
            print(f"[{time.time()-T0:6.0f}s] {model}: imagen {e['idx']+1}/{len(casos)}, {vel:.1f} s por informe, "
                  f"{len(hechos)}/{total} informes en total", flush=True)
    guardar(filas)
    if n_mod:
        print(f"[{time.time()-T0:6.0f}s] {model} terminado: {n_mod} informes, {(time.time()-t_mod)/n_mod:.1f} s por informe", flush=True)

D = pd.DataFrame(filas)
n_err = int(D["error"].notna().sum()) if len(D) else 0
print(f"Listo: {len(D)} informes en {OUT_CSV} ({n_err} con error; vuelve a correr el script para reintentarlos). "
      f"Tiempo total: {(time.time()-T0)/60:.1f} min", flush=True)
if args.smoke:
    vel = D.groupby("modelo")["segundos"].median()
    print("Segundos por informe (mediana):", vel.round(1).to_dict())
    print("Horas estimadas por modelo para 368 imágenes x 2 condiciones:", (vel * 736 / 3600).round(1).to_dict())
    print(f"Total estimado: {vel.sum() * 736 / 3600:.1f} h")
    print("Revisa que los textos de", OUT_CSV, "sean informes en español y no contengan razonamiento previo.")
