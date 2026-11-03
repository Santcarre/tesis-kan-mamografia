#!/usr/bin/env python3
"""
generar_datos_ui.py: genera los datos que alimentan la interfaz de Prometeo
(`desktop_app/ui/data/`). Solo necesita numpy y la biblioteca estándar: no usa
PyTorch ni las imágenes de CBIS-DDSM.

Entradas (todas dentro del repositorio):
    reports/experiments/kan_robust64_canonical.json   fórmula g(u0) y umbral de Youden
    reports/experiments/kan_robust64_reducer.npz      reductor S-PCA 512 -> 64
    artifacts/output/embeddings/f2robust_test_*.npy   embeddings GAP del conjunto de prueba (1012 imágenes)
    reports/**.json                                   cifras de la tesis (AUC, EBPG, semillas, ...)
    desktop_app/ui/data/lab_seleccion.json            pacientes del Laboratorio

Salidas (en desktop_app/ui/data/):
    tesis.json          fórmula (términos estructurados), umbral y cifras de la tesis
    lab_global.json     u0 y etiqueta de las 1012 imágenes de prueba (curva ROC y umbral interactivos)
    lab_pacientes.json  recorrido numérico paso a paso de cada paciente del Laboratorio

Autocomprobaciones (el script falla si algo no cuadra con la tesis):
    AUC de la fórmula = 0,7185; sensibilidad 72,49 %; especificidad 59,09 %;
    los términos estructurados reproducen la expresión original;
    u0 y g de cada paciente coinciden con los valores de referencia de lab_seleccion.json.

Uso (desde la raíz del repositorio):
    python scripts/generar_datos_ui.py
"""
import argparse
import ast
import datetime as dt
import json
import math
import pathlib
import re
import sys

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.reporting.formula_eval import compile_expr, sigmoid  # noqa: E402  (biblioteca estándar)

from src.reporting import birads_herramienta as BH  # noqa: E402  fuente única de la categoría BI-RADS


def birads_herramienta(p, p_tau):
    """Categoría orientativa (2-5) anclada en el umbral de decisión; ver src/reporting/birads_herramienta.py."""
    return str(BH.categoria_desde_p(p, p_tau))


def cortes_birads(p_tau):
    return [{"hasta": BH.CORTE_2_3, "cat": "2"}, {"hasta": p_tau, "cat": "3"},
            {"hasta": BH.CORTE_4_5, "cat": "4"}, {"hasta": None, "cat": "5"}]


# ----------------------------------------------------------------------------- utilidades
def leer_json(ruta):
    with open(RAIZ / ruta, "r", encoding="utf-8") as f:
        return json.load(f)


def leer_json_opcional(ruta):
    p = RAIZ / ruta
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def auc_rangos(y, score):
    """AUC de Mann-Whitney con empates por rangos medios."""
    y = np.asarray(y).astype(int)
    s = np.asarray(score, dtype=float)
    orden = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    ss = s[orden]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        r[orden[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def r(x, nd=4):
    return None if x is None else round(float(x), nd)


# ----------------------------------------------------------------------------- fórmula estructurada
def _aplanar(nodo, signo=1.0):
    """Lista de (signo, nodo) con los sumandos de primer nivel."""
    if isinstance(nodo, ast.BinOp) and isinstance(nodo.op, ast.Add):
        return _aplanar(nodo.left, signo) + _aplanar(nodo.right, signo)
    if isinstance(nodo, ast.BinOp) and isinstance(nodo.op, ast.Sub):
        return _aplanar(nodo.left, signo) + _aplanar(nodo.right, -signo)
    if isinstance(nodo, ast.UnaryOp) and isinstance(nodo.op, ast.USub):
        return _aplanar(nodo.operand, -signo)
    return [(signo, nodo)]


def _num(n):
    """Valor numérico de una constante (admite signo unario)."""
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
        return float(n.value)
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        v = _num(n.operand)
        return None if v is None else -v
    return None


def _arg_lineal(n):
    """Descompone `w*x_1 + phi` (o `w*x_1 - phi`) en (w, phi)."""
    fase = 0.0
    if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)):
        c = _num(n.right)
        if c is None:
            return None
        fase = c if isinstance(n.op, ast.Add) else -c
        n = n.left
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
        w = _num(n.left)
        if w is not None and isinstance(n.right, ast.Name) and n.right.id == "x_1":
            return w, fase
    return None


def estructurar(src):
    """Devuelve la fórmula como lista de términos {k: lin|cos|sin|const, a, w, p}."""
    arbol = ast.parse(src.replace("^", "**"), mode="eval").body
    terminos = []
    for signo, n in _aplanar(arbol):
        c = _num(n)
        if c is not None:
            terminos.append({"k": "const", "a": signo * c})
            continue
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
            a = _num(n.left)
            if a is not None and isinstance(n.right, ast.Name) and n.right.id == "x_1":
                terminos.append({"k": "lin", "a": signo * a})
                continue
            if a is not None and isinstance(n.right, ast.Call) and n.right.func.id in ("sin", "cos"):
                wp = _arg_lineal(n.right.args[0])
                if wp is not None:
                    terminos.append({"k": n.right.func.id, "a": signo * a, "w": wp[0], "p": wp[1]})
                    continue
        raise ValueError("Término no reconocido en la fórmula: " + ast.unparse(n)[:90])
    return terminos


def eval_terminos(terminos, x):
    t = 0.0
    for q in terminos:
        if q["k"] == "const":
            t += q["a"]
        elif q["k"] == "lin":
            t += q["a"] * x
        elif q["k"] == "cos":
            t += q["a"] * math.cos(q["w"] * x + q["p"])
        else:
            t += q["a"] * math.sin(q["w"] * x + q["p"])
    return t


def valor_termino(q, x):
    return eval_terminos([q], x)


# ----------------------------------------------------------------------------- pipeline numérico
def cargar_reductor():
    z = np.load(RAIZ / "reports/experiments/kan_robust64_reducer.npz")
    return {k: z[k].astype(np.float64) for k in z.files if z[k].dtype.kind == "f"}


def proyectar(gap, R):
    """GAP-512 -> (gap estandarizado, z64)."""
    gz = (gap - R["gap_mu"]) / R["gap_sd"]
    z = gz @ R["proj_w"].T + R["proj_b"]
    z = ((z - R["pca_mean"]) @ R["pca_components"].T - R["mu_z"]) / R["sd_z"]
    return gz, z


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--salida", default=str(RAIZ / "desktop_app" / "ui" / "data"),
                    help="Carpeta de salida (por defecto desktop_app/ui/data)")
    args = ap.parse_args()
    out = pathlib.Path(args.salida)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ fórmula canónica
    can = leer_json("reports/experiments/kan_robust64_canonical.json")
    f_ben = compile_expr(can["fben"])
    f_mal = compile_expr(can["fmal"])
    t_ben = estructurar(can["fben"])
    t_mal = estructurar(can["fmal"])
    for x in np.linspace(-6, 6, 41):
        assert abs(eval_terminos(t_ben, x) - f_ben(x)) < 1e-9, "términos de f_ben no reproducen la expresión"
        assert abs(eval_terminos(t_mal, x) - f_mal(x)) < 1e-9, "términos de f_mal no reproducen la expresión"

    def g(x):
        return f_mal(x) - f_ben(x)

    tau = float(can["th_youden_val"])
    # Umbral de Youden de la red robusta en validación (con el que la tesis la reporta)
    thr_red = float(leer_json("reports/experiments/youden_val_calibration.json")["Robusto"]["thr_youden_val"])
    mono = leer_json("reports/experiments/r3_monotonicity.json")
    soporte = mono["u0_support"]

    # u0 donde g cruza el umbral (g es estrictamente decreciente en el soporte)
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if g(mid) > tau:
            lo = mid
        else:
            hi = mid
    u0_umbral = (lo + hi) / 2

    # ------------------------------------------------------------------ población de prueba (1012)
    R = cargar_reductor()
    X = np.load(RAIZ / "artifacts/output/embeddings/f2robust_test_feats.npy").astype(np.float64)
    y = np.load(RAIZ / "artifacts/output/embeddings/f2robust_test_labels.npy").astype(int)
    GZ, Z = proyectar(X, R)
    u0 = Z[:, 0]
    gv = np.array([g(float(v)) for v in u0])
    pred = (gv >= tau).astype(int)
    auc = auc_rangos(y, gv)
    tp = int(((y == 1) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    sens, esp = tp / (tp + fn), tn / (tn + fp)
    assert abs(auc - can["AUC_D"]) < 5e-4, f"AUC de la fórmula {auc:.4f} != {can['AUC_D']}"
    assert abs(sens - can["recall_D"]) < 5e-4 and abs(esp - can["spec_D"]) < 5e-4, "sens/esp no coinciden con la tesis"
    print(f"[ok] AUC={auc:.4f}  sens={sens:.4f}  esp={esp:.4f}  ({len(y)} imágenes de prueba)")

    # ------------------------------------------------------------------ cifras de la tesis
    ci = leer_json("reports/experiments/auc_ci_comparison.json")
    rec = leer_json("reports/auditoria/recalculos_auditoria.json")
    point = leer_json("reports/experiments/33_pointing_stats.json")
    nulo = leer_json("reports/experiments/pointing_game_nulo_area.json")
    sems = leer_json("reports/auditoria/semantica_u0_por_imagen.json")   # vínculo por lesión (Cap. 4, BI-RADS)
    llm = leer_json("reports/experiments/estudio_informes_llm_resumen.json")    # estudio de informes (Cap. 4)
    llm_val = leer_json("reports/experiments/estudio_informes_llm_validacion_acuerdo.json")
    llm_ev = leer_json("reports/experiments/estudio_informes_llm_evidencia_448.json")
    segres = leer_json("reports/auditoria/segmentacion_resolucion.json")             # mapas de mayor resolución
    rad = leer_json("reports/auditoria/u0_birads_radiologo.json")                    # u0 frente al BI-RADS del radiólogo
    rob = leer_json("reports/experiments/26_resnet18_robust.json")
    ver = leer_json("reports/auditoria/verificacion_formula.json")
    # Tabla multi-semilla que cita el Capítulo 4 (§4.2): reports/27_spca_robust_comparison.json
    ms27 = leer_json("reports/27_spca_robust_comparison.json")["results"]
    ms_rob = next(q for q in ms27 if "Robusto" in q["experiment"] and "PCA Ciego" not in q["experiment"])
    ms_base = next(q for q in ms27 if "Baseline" in q["experiment"])
    ms_ciego = next(q for q in ms27 if "PCA Ciego" in q["experiment"])
    # Segunda corrida (índice CANONICAL_RESULTS): reports/experiments/r3_spca64_multiseed.json
    ms_r3 = [q for q in leer_json("reports/experiments/r3_spca64_multiseed.json") if q.get("backbone") == "resnet18_robusto"]
    pac = rec["bootstrap_por_paciente"]
    resumen_ms = leer_json_opcional("reports/auditoria/multisemilla/resumen_multisemilla.json")
    corridas = []
    for s in (1, 2, 3):
        c = leer_json_opcional(f"reports/auditoria/multisemilla/run_robusto_s{s}.json")
        if c and not c.get("smoke"):
            corridas.append({"semilla": c["semilla"], "auc_red": r(c["auc_test_red"]),
                             "auc_u0": r(c["spca"]["auc_u0"]["test"]),
                             "sens": r(c["spca"]["u0_youden_val"]["sens"]),
                             "esp": r(c["spca"]["u0_youden_val"]["spec"])})

    tesis = {
        "generado": dt.date.today().isoformat(),
        "generado_con": "scripts/generar_datos_ui.py",
        "formula": {
            "f_ben": t_ben, "f_mal": t_mal,
            "expr_ben": can["fben"], "expr_mal": can["fmal"],
            "caracteres": can["chars"], "variables": can["n_vars"],
            "tau_logit": tau, "tau_prob": r(sigmoid(tau), 4),
            "u0_soporte": soporte, "u0_umbral": r(u0_umbral, 4),
            "sup_gprima": ver["sup_abs_gprima"], "estrictamente_decreciente": ver["estrictamente_decreciente"],
            "gprima_rango_soporte": [mono["gprime_min_support"], mono["gprime_max_support"]],
            "reductor": can["reductor"], "kan": can["kan"],
        },
        "poblacion": {"n": int(len(y)), "n_mal": int(y.sum()), "n_ben": int((1 - y).sum()),
                      "u0_media_mal": r(u0[y == 1].mean()), "u0_media_ben": r(u0[y == 0].mean()),
                      "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}, "sens": r(sens), "esp": r(esp)},
        "auc": {
            "formula": {"auc": r(ci["models"]["R3_formula"]["auc"]),
                        "ic95_imagen": [r(ci["models"]["R3_formula"]["ci95_low"]), r(ci["models"]["R3_formula"]["ci95_high"])],
                        "ic95_paciente": pac["auc_ic95"]["R3_formula"]},
            "red_robusta": {"auc": r(ci["models"]["Robusto"]["auc"]),
                            "ic95_imagen": [r(ci["models"]["Robusto"]["ci95_low"]), r(ci["models"]["Robusto"]["ci95_high"])],
                            "ic95_paciente": pac["auc_ic95"]["Robusto"]},
            "baseline": {"auc": r(ci["models"]["Baseline_FC"]["auc"]),
                         "ic95_imagen": [r(ci["models"]["Baseline_FC"]["ci95_low"]), r(ci["models"]["Baseline_FC"]["ci95_high"])],
                         "ic95_paciente": pac["auc_ic95"]["Baseline_FC"]},
            "logistica_u0": r(rec["logistica"]["logistica_u0"]["auc_test"]),
            "n_pacientes": pac["n_pacientes"],
            "delta_baseline_menos_formula": pac["delta_Baseline_menos_R3"],
        },
        "red_robusta": {"sens_youden_propio": r(rob["metrics"]["recall"]), "esp_youden_propio": r(rob["metrics"]["specificity"])},
        "multisemilla": {
            "fuente": "reports/27_spca_robust_comparison.json (la tabla del Capítulo 4, §4.2)",
            "robusto_spca": {"auc": r(ms_rob["auc_d_mean"]), "sd": r(ms_rob["auc_d_std"]), "auc_red_kan": r(ms_rob["auc_full_mean"]), "sd_red_kan": r(ms_rob["auc_full_std"]),
                             "sens": r(ms_rob["recall_mean"]), "sens_sd": r(ms_rob["recall_std"]), "esp": r(ms_rob["spec_mean"]), "esp_sd": r(ms_rob["spec_std"]),
                             "vars": ms_rob["n_vars_mean"], "chars": ms_rob["char_len_mean"],
                             "semillas": [{"semilla": q["seed"], "auc": r(q["auc_d"]), "vars": q["n_vars"]} for q in ms_rob["seed_records"]]},
            "base_spca": {"auc": r(ms_base["auc_d_mean"]), "sd": r(ms_base["auc_d_std"])},
            "robusto_pca_ciego": {"auc": r(ms_ciego["auc_d_mean"]), "sd": r(ms_ciego["auc_d_std"])},
            "corrida_indice": {"fuente": "reports/experiments/r3_spca64_multiseed.json (la que lista CANONICAL_RESULTS.md)",
                               "auc": r(np.mean([q["AUC_D"] for q in ms_r3])), "sd": r(np.std([q["AUC_D"] for q in ms_r3], ddof=1)),
                               "semillas": [{"semilla": q["seed"], "auc": r(q["AUC_D"])} for q in ms_r3]},
        },
        "extractor_semillas": {"corridas": corridas, "resumen": resumen_ms},
        "ebpg": {
            "gradcam_base": {"v": point["gc_base"]["ebpg_global"], "ic95": point["gc_base"]["ci95"]},
            "gradcam_robusto": {"v": point["gc_rob"]["ebpg_global"], "ic95": point["gc_rob"]["ci95"]},
            "formula": {"v": point["psi"]["ebpg_global"], "ic95": point["psi"]["ci95"]},
            "azar_imagen": {"v": nulo["nulls"]["global"]["null_image_mean"], "ic95": nulo["nulls"]["global"]["null_image_ci95"]},
            "azar_mama": {"v": nulo["nulls"]["global"]["null_breast_mean"], "ic95": nulo["nulls"]["global"]["null_breast_ci95"]},
            "n": point["gc_base"]["n"],
        },
        "semantica": {
            "u0_vs_patologia": {"d": sems["tests"]["u0_pathology"]["d"], "p_holm": sems["tests"]["u0_pathology"]["p_holm"]},
            "u0_vs_margenes": {"H": sems["tests"]["u0_margins"]["H"], "p_holm": sems["tests"]["u0_margins"]["p_holm"]},
            "u0_vs_forma": {"H": sems["tests"]["u0_shapes"]["H"], "p_holm": sems["tests"]["u0_shapes"]["p_holm"]},
            "u0_vs_densidad": {"H": sems["tests"]["u0_density"]["H"], "p_holm": sems["tests"]["u0_density"]["p_holm"]},
            "n_pacientes": sems["n_pacientes"],
        },
        "estudio_llm": {
            "n_informes": llm["n_informes"], "n": llm["n_imagenes"], "pacientes": llm["n_pacientes"],
            "error_localizador": r(float(np.mean([e["predicted_quadrant"] != e["gt_quadrant"] for e in llm_ev])), 4),
            "incompatible_herramienta": r(float(np.mean([
                (str(e["pathology_gt"]).upper().startswith("MALIGNANT") and BH.categoria(e["kan64_g"], e["kan64_th"]) <= 3)
                or (not str(e["pathology_gt"]).upper().startswith("MALIGNANT") and BH.categoria(e["kan64_g"], e["kan64_th"]) >= 5)
                for e in llm_ev])), 4),
            "validacion": {"n": llm_val["sin_respaldo"]["n"], "acuerdo_sin_respaldo": llm_val["sin_respaldo"]["acuerdo_pct"] / 100,
                           "kappa_sin_respaldo": llm_val["sin_respaldo"]["kappa"]},
            "modelos": [
                {"nombre": {"qwen2.5:1.5b-instruct": "Qwen2.5-1.5B", "qwen2.5:3b": "Qwen2.5-3B", "phi3.5": "Phi-3.5-mini",
                            "qwen3.5:4b": "Qwen3.5-4B", "gemma4:12b": "Gemma 4 12B"}.get(m, m),
                 "sin_respaldo_control": r(v["ciego"]["sin_respaldo"]["pct"] / 100, 4),
                 "sin_respaldo_evidencia": r(v["evidencia"]["sin_respaldo"]["pct"] / 100, 4),
                 "coherente_control": r(v["ciego"]["birads_coherente"]["pct"] / 100, 4),
                 "coherente_evidencia": r(v["evidencia"]["birads_coherente"]["pct"] / 100, 4),
                 "error_region_evidencia": r(v["evidencia"]["region_error"]["pct"] / 100, 4),
                 "degenerados_evidencia": r(v["evidencia"]["repetitivo"]["pct"] / 100, 4),
                 "segundos": v["evidencia"]["segundos_por_informe"]}
                for m, v in llm["modelos"].items()],
            "fuente": "reports/experiments/estudio_informes_llm_resumen.json (scripts/estudio_informes_llm.py y analizar_informes_llm.py)",
        },
        "localizador_448": {
            "acierto_malignas_224": r(segres["resumen"]["malignas"]["psi_224"]["quad_match"]["real"], 4),
            "acierto_malignas_448": r(segres["resumen"]["malignas"]["psi_448"]["quad_match"]["real"], 4),
            "centro_tejido_malignas": r(segres["resumen"]["malignas"]["referencias"]["centro_tejido"]["quad_match"], 4),
            "dif_centro_ic95": segres["resumen"]["malignas"]["psi_448"]["quad_match"]["dif_vs_centro_tejido_ic95"],
            "acierto_global_448": r(segres["resumen"]["global"]["psi_448"]["quad_match"]["real"], 4),
            "auc_formula_224": r(segres["auc"]["formula_224"], 4), "auc_formula_448": r(segres["auc"]["formula_448"], 4),
            "fuente": "reports/auditoria/segmentacion_resolucion.json",
        },
        "birads_radiologo": {
            "rho": rad["resultados"]["prueba"]["spearman_p_vs_radiologo"]["rho"],
            "rho_ic95": rad["resultados"]["prueba"]["spearman_p_vs_radiologo"]["ic95"],
            "auc_radiologo": rad["resultados"]["prueba"]["auc_radiologo"]["auc"],
            "auc_formula": rad["resultados"]["prueba"]["auc_formula_mismas_imagenes"]["auc"],
            "auc_combinacion": rad["resultados"]["combinacion_radiologo_u0"]["auc_prueba"],
            "kappa_categoria": rad["resultados"]["cortes_alternativos"]["anclado_en_tau_kappa_max"]["acuerdo_prueba"]["kappa_cuadratico"],
            "n_imagenes": rad["resultados"]["prueba"]["n_imagenes_2a5"],
            "n_pacientes": rad["resultados"]["prueba"]["n_pacientes"],
            "fuente": "reports/auditoria/u0_birads_radiologo.json",
        },
        "birads_cortes": cortes_birads(sigmoid(tau)),
        "birads_nota": BH.NOTA,
    }
    with open(out / "tesis.json", "w", encoding="utf-8") as f:
        json.dump(tesis, f, ensure_ascii=False, indent=1)

    with open(out / "lab_global.json", "w", encoding="utf-8") as f:
        json.dump({"n": int(len(y)), "u0": [round(float(v), 4) for v in u0], "y": [int(v) for v in y]},
                  f, ensure_ascii=False, separators=(",", ":"))

    # ------------------------------------------------------------------ pacientes del Laboratorio
    sel = leer_json("desktop_app/ui/data/lab_seleccion.json")["pacientes"]
    pc0 = R["pca_components"][0]
    sd0 = float(R["sd_z"].ravel()[0])
    dir_gap = (pc0 / sd0) @ R["proj_w"] / R["gap_sd"].ravel()   # d u0 / d gap_c
    pacientes = []
    for p in sel:
        i = p["emb_row"]
        gap = X[i]
        gz = GZ[i]
        z = Z[i]
        u = float(z[0])
        fb, fm = f_ben(u), f_mal(u)
        gg = fm - fb
        assert abs(u - p["u0_ref"]) < 2e-3 and abs(gg - p["g_ref"]) < 2e-3, f"{p['id']}: u0/g no coinciden con la referencia"
        pe = sigmoid(gg)
        eps = 1e-4
        gprima = (g(u + eps) - g(u - eps)) / (2 * eps)
        dg_dgap = dir_gap * gprima
        top = np.argsort(-np.abs(dg_dgap))[:5]
        # contribuciones de cada término a g = f_mal - f_ben en este u0
        contrib = []
        for lado, terms, sgn in (("mal", t_mal, 1.0), ("ben", t_ben, -1.0)):
            for q in terms:
                contrib.append({"lado": lado, "k": q["k"], "a": r(q["a"], 5), "w": r(q.get("w"), 5), "p": r(q.get("p"), 5),
                                "valor": r(sgn * valor_termino(q, u), 5)})
        # vecindad empírica: de las imágenes de prueba con u0 cercano, ¿qué fracción es maligna?
        ancho = 0.3
        m = np.abs(u0 - u) <= ancho
        n_v = int(m.sum()); k_v = int(y[m].sum())
        pct_mal = float((u0[y == 1] <= u).mean()); pct_ben = float((u0[y == 0] <= u).mean())
        pacientes.append({
            **{k: p[k] for k in ("id", "idx", "paciente", "rol", "titulo", "gancho", "leccion", "tipo", "patologia",
                                 "birads_radiologo", "forma", "margenes")},
            "red": {"logit": p["net_logit"], "p": p["net_p"], "pred": "MALIGNANT" if p["net_p"] >= thr_red else "BENIGN", "umbral": r(thr_red, 4)},
            "gap_z": [round(float(v), 2) for v in gz],
            "z64": [round(float(v), 3) for v in z],
            "u0": r(u, 4), "f_ben": r(fb, 4), "f_mal": r(fm, 4), "g": r(gg, 4), "p_sym": r(pe, 4),
            "tau_logit": tau, "tau_prob": r(sigmoid(tau), 4), "margen": r(gg - tau, 4),
            "veredicto": "MALIGNANT" if gg >= tau else "BENIGN",
            "birads_herramienta": birads_herramienta(pe, sigmoid(tau)),
            "gprima": r(gprima, 4),
            "u0_para_cambiar": r(u - u0_umbral, 4),
            "terminos": contrib,
            "canales_top": [{"canal": int(c), "dg_dgap": r(dg_dgap[c], 5), "gap_z": r(gz[c], 3)} for c in top],
            "vecindad": {"ancho": ancho, "n": n_v, "malignas": k_v, "frac_mal": r(k_v / n_v if n_v else None, 4),
                         "ic95": wilson(k_v, n_v)},
            "percentil": {"maligna_u0_menor_o_igual": r(pct_mal, 4), "benigna_u0_menor_o_igual": r(pct_ben, 4)},
            "acierto_formula": bool((gg >= tau) == (p["patologia"] == "MALIGNANT")),
            "acierto_red": bool((p["net_p"] >= thr_red) == (p["patologia"] == "MALIGNANT")),
        })
    with open(out / "lab_pacientes.json", "w", encoding="utf-8") as f:
        json.dump({"generado": dt.date.today().isoformat(), "pacientes": pacientes}, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[ok] {len(pacientes)} pacientes; umbral en u0 = {u0_umbral:.4f}")
    print("Archivos escritos en", out)


if __name__ == "__main__":
    main()
