"""
Auditoria de LIME y KernelSHAP sobre el logit lineal del extractor robusto.

Uso (desde la raiz del repositorio):  python scripts/auditoria_explicadores.py [RAIZ] [SALIDA.json] [--smoke]

Que hace (todo sobre EMBEDDINGS REALES de prueba, no datos sinteticos):
  A. d=512, presupuestos de muestras [600, 2000, 5000, 20000]:
       - LIME (implementacion del repositorio)
       - KernelSHAP "viejo" (implementacion del repositorio)
       - KernelSHAP estandar (tamanos de coalicion proporcionales al kernel de
         Shapley, pares complementarios, restriccion de eficiencia)
     contra la verdad analitica phi_j = a_j (x_j - media_fondo_j), con a = w1 - w0.
  B. Barrido de dimension d en {12,24,48,128,256,512} (subconjuntos aleatorios de canales,
     logit restringido a esos canales) con presupuesto fijo de 600.
  Metricas por caso: Spearman, concordancia del canal dominante (top-1),
  error relativo L2.  Se reporta media (y mediana) sobre 200 casos AL AZAR.
Salida: reports/auditoria/auditoria_explicadores.json
"""
import json, os, sys, time
import numpy as np

SMOKE = "--smoke" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
CODIGOS = args[0] if args else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = args[1] if len(args) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "reports", "auditoria", "auditoria_explicadores.json")
for p in (CODIGOS, os.path.join(CODIGOS, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

SEED, N_CASES, N_BG = 42, 200, 32
BUDGETS = [600, 2000, 5000, 20000]
DIMS = [12, 24, 32, 48, 64, 128, 256, 512]
if SMOKE:
    N_CASES, BUDGETS, DIMS = 5, [600, 2000], [12, 512]


def rank(a):
    o = np.argsort(a, kind="mergesort"); r = np.empty(len(a)); r[o] = np.arange(len(a))
    for v in np.unique(a):            # empates -> rango medio
        m = a == v
        if m.sum() > 1: r[m] = r[m].mean()
    return r

def spear(a, b):
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0: return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def kernel_shap_std(f, x, bg, n_samples, rng):
    """KernelSHAP estandar: tamanos ~ kernel de Shapley, pares complementarios,
    minimos cuadrados con restriccion sum(phi)=f(x)-E f(bg) (eliminando una variable)."""
    d = x.size
    sizes = np.arange(1, d)
    pk = (d - 1.0) / (sizes * (d - sizes)); pk /= pk.sum()
    n_pairs = max(1, n_samples // 2)
    s = rng.choice(sizes, p=pk, size=n_pairs)
    R = rng.random((n_pairs, d))
    ranks = np.argsort(np.argsort(R, axis=1), axis=1)
    M = (ranks < s[:, None])
    M = np.vstack([M, ~M]).astype(np.float64)                       # [n, d]
    y = np.zeros(M.shape[0])
    for b in bg:                                                    # promedio sobre fondo
        y += np.asarray(f(M * x[None, :] + (1.0 - M) * b[None, :]), float).ravel()
    y /= len(bg)
    f0 = float(np.mean(np.asarray(f(bg), float).ravel()))
    fx = float(np.asarray(f(x[None, :]), float).ravel()[0])
    delta = fx - f0
    A = M[:, :-1] - M[:, -1:]
    t = y - f0 - M[:, -1] * delta
    phi_head, *_ = np.linalg.lstsq(A, t, rcond=None)
    phi = np.append(phi_head, delta - phi_head.sum())
    return phi


def load():
    from src.reporting.clinical_evidence_extractor import _resolve_embeddings_dir
    emb = _resolve_embeddings_dir()
    Xte = np.load(os.path.join(emb, "f2robust_test_feats.npy")).astype(np.float64)
    import torch
    ck = torch.load(os.path.join(CODIGOS, "reports", "models", "resnet18_robust_best.pt"),
                    map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]; W = sd["fc.weight"].numpy().astype(np.float64)
    return Xte, W[1] - W[0]


def metr(est, ref):
    return {"rho": spear(est, ref),
            "top1": float(int(np.argmax(np.abs(est))) == int(np.argmax(np.abs(ref)))),
            "err_rel": float(np.linalg.norm(est - ref) / (np.linalg.norm(ref) + 1e-12))}

def resumen(lst):
    out = {}
    for k in lst[0]:
        v = np.array([m[k] for m in lst], float)
        out[k] = {"media": float(np.nanmean(v)), "mediana": float(np.nanmedian(v))}
    return out


def evalua(Xte, a, d, budget, idx, rng_seed):
    """Devuelve metricas por metodo para dimension d y presupuesto dado."""
    from src.explainability.attribution_baselines import lime_surrogate, kernel_shap
    rs = np.random.default_rng(rng_seed)
    ch = np.sort(rs.choice(Xte.shape[1], size=d, replace=False)) if d < Xte.shape[1] else np.arange(Xte.shape[1])
    X, ad = Xte[:, ch], a[ch]
    f = lambda U: np.asarray(U, float) @ ad
    res = {"lime": [], "ks_viejo": [], "ks_estandar": []}
    for k, i in enumerate(idx):
        x = X[i]
        bg = X[rs.choice(X.shape[0], size=N_BG, replace=False)]
        ref = ad * (x - bg.mean(axis=0))                           # Shapley exacto con ese fondo
        lm = lime_surrogate(f, x, background=bg, n_samples=budget, seed=SEED + k)
        res["lime"].append(metr(lm["coef"] * (x - bg.mean(axis=0)), ref))
        ks = kernel_shap(f, x, background=bg, n_samples=budget, n_background=N_BG, seed=SEED + k)
        res["ks_viejo"].append(metr(ks["shap"], ref))
        res["ks_estandar"].append(metr(kernel_shap_std(f, x, bg, budget, np.random.default_rng(SEED + k)), ref))
    return {m: resumen(v) for m, v in res.items()}


def main():
    t0 = time.time()
    if SMOKE:
        rng = np.random.default_rng(0)
        Xte, a = np.abs(rng.normal(size=(300, 512))), rng.normal(size=512)
    else:
        Xte, a = load()
    idx = np.random.default_rng(SEED).choice(Xte.shape[0], size=min(N_CASES, Xte.shape[0]), replace=False)
    out = {"n_casos": int(len(idx)), "aleatorios": True, "n_fondo": N_BG, "datos": "embeddings reales de prueba" if not SMOKE else "SMOKE",
           "A_presupuesto_d512": {}, "B_dimension_p600": {}}
    for B in BUDGETS:
        out["A_presupuesto_d512"][str(B)] = evalua(Xte, a, 512, B, idx, 1)
        print("A budget", B, {m: round(v["rho"]["media"], 4) for m, v in out["A_presupuesto_d512"][str(B)].items()}, flush=True)
    for d in DIMS:
        out["B_dimension_p600"][str(d)] = evalua(Xte, a, d, 600, idx, 2)
        print("B d", d, {m: round(v["rho"]["media"], 4) for m, v in out["B_dimension_p600"][str(d)].items()}, flush=True)
    out["segundos"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("Guardado:", OUT, out["segundos"], "s")

if __name__ == "__main__":
    main()
