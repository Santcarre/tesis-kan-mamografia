# -*- coding: utf-8 -*-
"""
canales_atribucion.py -- Fidelidad por canales con la misma definición de atribución en la fórmula y en la cabeza.

coherence_kan64.json (scripts/compute_coherence_kan64.py) compara el gradiente de la fórmula respecto de los 512
descriptores, g'(u0) * du0/dz, con la atribución tipo Shapley de la cabeza lineal, a_c * (z_c - mu_c). Son dos
definiciones distintas: el gradiente de la fórmula tiene dirección fija, así que su canal dominante es siempre el
mismo (el 191). Este script repite la comparación con la misma definición en los dos lados (gradiente por
descriptor centrado, que para un modelo afín es su valor de Shapley exacto):

    fórmula:  phi_c = g'(u0) * (du0/dz)_c * (z_c - mu_c)
    cabeza:   phi_c = a_c * (z_c - mu_c),   a = w_1 - w_0

Mismos 200 casos de prueba y mismo fondo (200 casos, semilla 42) que coherence_kan64.json.

Uso (desde la raíz del repo):  python scripts/canales_atribucion.py
Requiere: embeddings (artifacts/), reports/experiments/kan_robust64_{canonical.json,reducer.npz} y el checkpoint
reports/models/resnet18_robust_best.pt. Si PyTorch no está instalado, lee los pesos de la última capa directamente
del archivo .pt (formato zip de PyTorch) con numpy.
Salida: reports/auditoria/canales_atribucion.json
"""
import collections, json, os, pickle, sys, zipfile
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED, N = 42, 200
OUT = os.path.join(RAIZ, "reports", "auditoria", "canales_atribucion.json")


def pesos_fc(ruta):
    """fc.weight (2 x 512) del checkpoint, con PyTorch si está disponible y, si no, leyendo el zip."""
    try:
        import torch
        ck = torch.load(ruta, map_location="cpu", weights_only=False)
        return ck["model_state_dict"]["fc.weight"].double().numpy()
    except ImportError:
        pass
    tipos = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16, "LongStorage": np.int64}
    z = zipfile.ZipFile(ruta)
    pref = [n for n in z.namelist() if n.endswith("data.pkl")][0][:-len("data.pkl")]

    class Vacio:
        def __init__(self, *a, **k): pass
        def __setstate__(self, s): pass

    class Lector(pickle.Unpickler):
        def find_class(self, mod, nombre):
            if nombre in tipos: return nombre
            if mod == "torch._utils" and nombre == "_rebuild_tensor_v2":
                def reconstruir(alm, desp, forma, pasos, *a):
                    if len(forma) == 0: return alm[desp]
                    return np.lib.stride_tricks.as_strided(alm[desp:], shape=forma, strides=[p * alm.itemsize for p in pasos]).copy()
                return reconstruir
            if mod == "collections" and nombre == "OrderedDict": return collections.OrderedDict
            if mod == "builtins": return super().find_class(mod, nombre)
            return Vacio
        def persistent_load(self, pid):
            _, tipo, clave, _, _ = pid
            return np.frombuffer(z.read(f"{pref}data/{clave}"), dtype=tipos.get(tipo, np.float32))
    return Lector(z.open(f"{pref}data.pkl")).load()["model_state_dict"]["fc.weight"].astype(np.float64)


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    W = pesos_fc(os.path.join(RAIZ, "reports", "models", "resnet18_robust_best.pt"))
    a = W[1] - W[0]
    emb = os.environ.get("TESIS_EMBEDDINGS_DIR") or os.path.join(RAIZ, "artifacts", "output", "embeddings")
    Xte = np.load(os.path.join(emb, "f2robust_test_feats.npy")).astype(np.float64)
    cj = json.load(open(os.path.join(RAIZ, "reports", "experiments", "kan_robust64_canonical.json"), encoding="utf-8"))
    nz = np.load(os.path.join(RAIZ, "reports", "experiments", "kan_robust64_reducer.npz"))
    gap_mu, gap_sd, proj_w, proj_b, pc, pm, mu_z, sd_z = [nz[k].astype(np.float64) for k in
        ("gap_mu", "gap_sd", "proj_w", "proj_b", "pca_components", "pca_mean", "mu_z", "sd_z")]
    entorno = {"sin": np.sin, "cos": np.cos, "exp": np.exp, "tanh": np.tanh}
    f_mal = eval("lambda x_1: " + cj["fmal"], entorno); f_ben = eval("lambda x_1: " + cj["fben"], entorno)
    g = lambda u: f_mal(u) - f_ben(u)
    dg = lambda u: (g(u + 1e-6) - g(u - 1e-6)) / 2e-6
    du0_dz = ((pc[0] / sd_z.ravel()[0]) @ proj_w) / gap_sd.ravel()      # u0 es afín en los 512 descriptores
    dir_fija = du0_dz / np.linalg.norm(du0_dz)

    rng = np.random.default_rng(SEED)                                   # mismo sorteo que compute_coherence_kan64.py
    fondo = Xte[rng.choice(Xte.shape[0], size=N, replace=False)]; mu = fondo.mean(axis=0)
    casos = rng.choice(Xte.shape[0], size=N, replace=False)

    res = {"gradiente": {"top1": 0, "rho": []}, "misma_definicion": {"top1": 0, "rho": []}}
    dominantes = []
    for i in casos:
        x = Xte[i]
        phi_cabeza = a * (x - mu)
        z64 = ((((x[None] - gap_mu) / gap_sd) @ proj_w.T + proj_b - pm) @ pc.T - mu_z) / sd_z
        d = dg(float(np.clip(z64.ravel()[0], -10, 10)))
        for clave, phi in (("gradiente", dir_fija * d), ("misma_definicion", d * du0_dz * (x - mu))):
            res[clave]["top1"] += int(np.argmax(np.abs(phi)) == np.argmax(np.abs(phi_cabeza)))
            res[clave]["rho"].append(spearman(phi, phi_cabeza))
        dominantes.append(int(np.argmax(np.abs(d * du0_dz * (x - mu)))))

    salida = {"n_casos": N, "semilla": SEED,
              "nota": "gradiente: definición de coherence_kan64.json; misma_definicion: gradiente por descriptor "
                      "centrado en la fórmula y en la cabeza (valor de Shapley exacto de un modelo afín).",
              "coseno_a_vs_du0dz": float(a @ du0_dz / np.linalg.norm(a) / np.linalg.norm(du0_dz)),
              "canales_dominantes_distintos_formula": len(set(dominantes))}
    for clave, r in res.items():
        salida[clave] = {"top1_coincide": f"{r['top1']}/{N}", "spearman_media": round(float(np.mean(r["rho"])), 4),
                         "spearman_mediana": round(float(np.median(r["rho"])), 4)}
    json.dump(salida, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(salida, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
