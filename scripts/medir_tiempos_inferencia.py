#!/usr/bin/env python3
"""
medir_tiempos_inferencia.py: cuánto cuesta cada etapa de la inferencia del modelo canónico RS-KAN.

Pregunta que responde: ¿la fórmula simbólica sirve para equipos de pocos recursos? La fórmula
reemplaza solo la última etapa (la decisión). Este script mide, por separado, lo que cuesta
cada etapa para ver dónde está el tiempo y la memoria:

    imagen -> preproceso -> ResNet-18 (512 números) -> u0 -> fórmula g(u0) -> veredicto

Bloque A (solo biblioteca estándar y NumPy; no necesita PyTorch)
    u0 desde un vector de 512 números, fórmula, veredicto, lote de 1012 casos, tamaño de los
    archivos que hacen falta y comprobación de que el atajo (u0 con un solo producto punto)
    da lo mismo que la proyección completa a 64 variables.
Bloque B (necesita PyTorch y el checkpoint resnet18_robust_best.pt)
    Importar PyTorch, cargar el checkpoint, pasar una imagen de 224x224 por la red (1, 2 y 4 hilos)
    y la capa final de la red (512 -> 2) para compararla con la fórmula.
Bloque C (opcional, necesita Pillow)
    Reducir una matriz sintética de 3000x4000 a 224x224. La lectura del DICOM no se mide.

Uso:
    python scripts/medir_tiempos_inferencia.py                 # todo lo que se pueda
    python scripts/medir_tiempos_inferencia.py --un-nucleo     # simula un equipo de un solo núcleo
    python scripts/medir_tiempos_inferencia.py --sin-torch     # solo el bloque A
    python scripts/medir_tiempos_inferencia.py --salida otro.json

Escribe reports/experiments/tiempos_inferencia_<equipo>.json (o la ruta de --salida).
Los tiempos dependen del equipo: hay que correrlo en cada máquina que se quiera describir.
"""
import os
import sys

# Hay que limitar los hilos ANTES de importar NumPy o PyTorch.
if "--un-nucleo" in sys.argv:
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = "1"

import argparse
import ast
import datetime as dt
import json
import math
import pathlib
import platform
import re
import statistics
import time

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.reporting.formula_eval import compile_expr, sigmoid  # noqa: E402  (biblioteca estándar)


# ----------------------------------------------------------------------------- utilidades
def rss_mb():
    """Memoria residente del proceso en MB, o None si no se puede leer."""
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 2**20, 1)
    except Exception:
        pass
    try:
        with open("/proc/self/status") as f:
            for linea in f:
                if linea.startswith("VmRSS:"):
                    return round(int(linea.split()[1]) / 1024, 1)
    except Exception:
        pass
    return None


def medir(fn, n, calentar=5):
    """Ejecuta fn() n veces y devuelve la mediana y el percentil 95 en microsegundos."""
    for _ in range(calentar):
        fn()
    t = []
    for _ in range(n):
        a = time.perf_counter_ns()
        fn()
        t.append((time.perf_counter_ns() - a) / 1e3)
    t.sort()
    return {"mediana_us": round(statistics.median(t), 2), "p95_us": round(t[int(0.95 * (len(t) - 1))], 2), "n": n}


def info_equipo():
    d = {
        "fecha": dt.datetime.now().isoformat(timespec="seconds"),
        "sistema": platform.platform(),
        "procesador": platform.processor() or platform.machine(),
        "nucleos_logicos": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    try:
        d["nucleos_disponibles_para_el_proceso"] = len(os.sched_getaffinity(0))
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as f:
            for linea in f:
                if linea.startswith("model name"):
                    d["procesador"] = linea.split(":", 1)[1].strip()
                    break
        with open("/proc/meminfo") as f:
            d["ram_total_mb"] = round(int(f.readline().split()[1]) / 1024)
    except Exception:
        try:
            import psutil
            d["ram_total_mb"] = round(psutil.virtual_memory().total / 2**20)
        except Exception:
            pass
    d["hilos_pedidos_al_entorno"] = os.environ.get("OMP_NUM_THREADS")
    return d


def compile_np(src):
    """Igual que compile_expr, pero con NumPy, para evaluar un lote entero de una vez (lista blanca, sin eval)."""
    tree = ast.parse(src.replace("^", "**"), mode="eval")
    fun = {"sin": np.sin, "cos": np.cos, "exp": np.exp, "tanh": np.tanh, "log": np.log, "sqrt": np.sqrt, "abs": np.abs}

    def ev(n, x):
        if isinstance(n, ast.Expression):
            return ev(n.body, x)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name) and n.id == "x_1":
            return x
        if isinstance(n, ast.UnaryOp):
            v = ev(n.operand, x)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left, x), ev(n.right, x)
            return {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
                    ast.Div: lambda: a / b, ast.Pow: lambda: a ** b}[type(n.op)]()
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in fun:
            return fun[n.func.id](ev(n.args[0], x))
        raise ValueError("Nodo no permitido")

    return lambda x: ev(tree, np.asarray(x, dtype=np.float64))


def tam(ruta):
    p = RAIZ / ruta
    return p.stat().st_size if p.exists() else None


# ----------------------------------------------------------------------------- bloque A
def bloque_a(rep):
    can = json.loads((RAIZ / "reports/experiments/kan_robust64_canonical.json").read_text(encoding="utf-8"))
    z = np.load(RAIZ / "reports/experiments/kan_robust64_reducer.npz")
    R = {k: z[k].astype(np.float64) for k in z.files if z[k].dtype.kind == "f"}
    X = np.load(RAIZ / "artifacts/output/embeddings/f2robust_test_feats.npy").astype(np.float64)
    tau = float(can["th_youden_val"])

    f_ben, f_mal = compile_expr(can["fben"]), compile_expr(can["fmal"])
    fb_np, fm_np = compile_np(can["fben"]), compile_np(can["fmal"])

    # Proyección completa (la que usa la tubería general): GAP-512 -> 64 variables
    def proyeccion_completa(gap):
        gz = (gap - R["gap_mu"]) / R["gap_sd"]
        zz = gz @ R["proj_w"].T + R["proj_b"]
        return ((zz - R["pca_mean"]) @ R["pca_components"].T - R["mu_z"]) / R["sd_z"]

    # Atajo: como todo es lineal, u0 = gap . w + b con un solo vector de 512 números
    comp0 = R["pca_components"][0]
    w_u0 = (R["proj_w"].T @ comp0) / (R["gap_sd"].reshape(-1) * R["sd_z"].reshape(-1)[0])
    b_u0 = (-(R["gap_mu"].reshape(-1) @ w_u0)
            + ((R["proj_b"] - R["pca_mean"]) @ comp0 - R["mu_z"].reshape(-1)[0]) / R["sd_z"].reshape(-1)[0])
    u0_completo = proyeccion_completa(X)[:, 0]
    u0_atajo = X @ w_u0 + b_u0
    dif = float(np.max(np.abs(u0_completo - u0_atajo)))
    assert dif < 1e-8, f"el atajo de u0 no coincide con la proyección completa (dif={dif})"

    g_scalar = lambda u: f_mal(u) - f_ben(u)  # noqa: E731
    g_np = lambda u: fm_np(u) - fb_np(u)  # noqa: E731
    veredicto_completo = g_np(u0_completo) >= tau
    veredicto_scalar = np.array([g_scalar(float(u)) >= tau for u in u0_atajo])
    assert (veredicto_completo == veredicto_scalar).all(), "los veredictos no coinciden"

    x0 = X[0]
    w_lista, x_lista = w_u0.tolist(), x0.tolist()

    def decidir_numpy(gap=x0):
        u = float(gap @ w_u0 + b_u0)
        g = g_scalar(u)
        return g >= tau, sigmoid(g)

    def decidir_python_puro():
        u = b_u0
        for a, b in zip(w_lista, x_lista):
            u += a * b
        g = g_scalar(u)
        return g >= tau, sigmoid(g)

    r = {}
    r["u0_un_producto_punto_numpy"] = medir(lambda: x0 @ w_u0 + b_u0, rep)
    r["proyeccion_completa_a_64_variables"] = medir(lambda: proyeccion_completa(x0[None, :]), rep)
    r["formula_g_un_caso"] = medir(lambda: g_scalar(0.3), rep)
    r["decision_completa_desde_512_numpy"] = medir(decidir_numpy, rep)
    r["decision_completa_desde_512_python_puro"] = medir(decidir_python_puro, max(rep // 5, 200))
    r["lote_1012_con_numpy_total"] = medir(lambda: g_np(X @ w_u0 + b_u0) >= tau, max(rep // 50, 50))
    r["lote_1012_con_numpy_por_caso_us"] = round(r["lote_1012_con_numpy_total"]["mediana_us"] / len(X), 3)
    r["archivos_necesarios_bytes"] = {
        "formula_canonica_json": tam("reports/experiments/kan_robust64_canonical.json"),
        "vector_u0_512_coeficientes_float64": int(w_u0.nbytes + 8),
        "reductor_completo_npz": tam("reports/experiments/kan_robust64_reducer.npz"),
        "checkpoint_resnet18_robusto": tam("reports/models/resnet18_robust_best.pt"),
        "caracteres_de_fben_mas_fmal_en_el_json": len(can["fben"]) + len(can["fmal"]),
    }
    r["comprobacion"] = {"diferencia_maxima_u0_atajo_vs_completo": dif, "veredictos_iguales_en_1012": True,
                         "casos": int(len(X))}
    return r


# ----------------------------------------------------------------------------- bloque B
def bloque_b(rep, hilos):
    t0 = time.perf_counter()
    import torch
    from torchvision import models
    t_import = time.perf_counter() - t0
    rss_antes = rss_mb()

    ruta = RAIZ / "reports/models/resnet18_robust_best.pt"
    if not ruta.exists():
        return {"omitido": f"falta {ruta}"}

    t0 = time.perf_counter()
    m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(512, 2)
    ck = torch.load(ruta, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    t_carga = time.perf_counter() - t0
    rss_despues = rss_mb()

    cabeza = m.fc
    extractor = torch.nn.Sequential(*list(m.children())[:-1], torch.nn.Flatten())
    g = torch.Generator().manual_seed(0)
    x1 = torch.randn(1, 3, 224, 224, generator=g)
    x8 = torch.randn(8, 3, 224, 224, generator=g)

    res = {"torch": torch.__version__, "tiempo_importar_torch_s": round(t_import, 2),
           "tiempo_cargar_checkpoint_s": round(t_carga, 2),
           "memoria_mb": {"antes_de_importar_torch_bloque_a": None, "despues_de_importar": rss_antes,
                          "despues_de_cargar_modelo": rss_despues}, "por_hilos": {}}
    with torch.inference_mode():
        feats = extractor(x1)
        for h in hilos:
            torch.set_num_threads(h)
            d = {}
            d["extractor_resnet18_una_imagen"] = medir(lambda: extractor(x1), rep, calentar=3)
            d["extractor_resnet18_lote_de_8"] = medir(lambda: extractor(x8), max(rep // 3, 5), calentar=2)
            d["capa_final_512_a_2_de_la_red"] = medir(lambda: cabeza(feats), max(rep * 20, 500))
            d["red_completa_una_imagen"] = medir(lambda: m(x1), rep, calentar=3)
            res["por_hilos"][str(h)] = d
        res["memoria_mb"]["despues_de_inferir"] = rss_mb()
    return res


# ----------------------------------------------------------------------------- bloque C
def bloque_c(rep):
    try:
        from PIL import Image
    except Exception:
        return {"omitido": "falta Pillow"}
    a = (np.random.default_rng(0).random((3000, 4000)) * 65535).astype(np.uint16)

    def reducir():
        v = a.astype(np.float32)
        v = (v - v.min()) / (v.max() - v.min() + 1e-8)
        Image.fromarray((v * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)

    return {"reducir_3000x4000_a_224x224": medir(reducir, max(rep // 20, 5), calentar=1),
            "nota": "No incluye leer ni decodificar el archivo DICOM."}


# ----------------------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeticiones", type=int, default=2000)
    ap.add_argument("--hilos", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--sin-torch", action="store_true")
    ap.add_argument("--un-nucleo", action="store_true", help="un solo hilo y un solo núcleo (simula un equipo modesto)")
    ap.add_argument("--etiqueta", default=None, help="nombre del equipo para el archivo de salida")
    ap.add_argument("--salida", default=None)
    args = ap.parse_args()

    if args.un_nucleo:
        try:
            os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[0]})
        except Exception:
            try:
                import psutil
                psutil.Process().cpu_affinity([psutil.Process().cpu_affinity()[0]])
            except Exception:
                print("[aviso] no se pudo fijar el proceso a un núcleo; solo se limitaron los hilos")
        args.hilos = [1]

    out = {"equipo": info_equipo(), "modo": "un_nucleo" if args.un_nucleo else "normal"}
    rss0 = rss_mb()
    print("== Bloque A: de 512 números al veredicto (sin PyTorch) ==")
    out["bloque_a"] = bloque_a(args.repeticiones)
    out["bloque_a"]["memoria_mb"] = {"proceso_con_numpy_antes": rss0, "despues": rss_mb()}
    for k, v in out["bloque_a"].items():
        if isinstance(v, dict) and "mediana_us" in v:
            print(f"  {k:48s} mediana {v['mediana_us']:>10.2f} us   p95 {v['p95_us']:>10.2f} us")
    print("  comprobación:", out["bloque_a"]["comprobacion"])

    if not args.sin_torch:
        print("== Bloque B: red convolucional (PyTorch) ==")
        try:
            out["bloque_b"] = bloque_b(max(args.repeticiones // 20, 20), args.hilos)
            out["bloque_b"].get("memoria_mb", {})["antes_de_importar_torch_bloque_a"] = out["bloque_a"]["memoria_mb"]["despues"]
            for h, d in out["bloque_b"].get("por_hilos", {}).items():
                for k, v in d.items():
                    print(f"  [{h} hilo(s)] {k:34s} mediana {v['mediana_us']/1000:>9.2f} ms")
        except ImportError as e:
            out["bloque_b"] = {"omitido": f"PyTorch no está instalado ({e})"}
            print("  omitido:", out["bloque_b"]["omitido"])

    print("== Bloque C: reducir la imagen ==")
    out["bloque_c"] = bloque_c(args.repeticiones)
    print(" ", out["bloque_c"])

    etiqueta = args.etiqueta or re.sub(r"[^A-Za-z0-9]+", "_", platform.node() or "equipo")
    ruta = pathlib.Path(args.salida) if args.salida else RAIZ / "reports/experiments" / f"tiempos_inferencia_{etiqueta}{'_1nucleo' if args.un_nucleo else ''}.json"
    ruta.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Escrito:", ruta)


if __name__ == "__main__":
    main()
