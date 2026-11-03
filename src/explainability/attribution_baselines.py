"""
attribution_baselines.py: Metodos de atribucion post-hoc implementados desde cero
para el estudio de coherencia con la formula simbolica.

Por que no se usan los paquetes `lime` ni `shap`:
  - No estan instalados en el entorno y anadir dependencias contradice la
    politica de reproducibilidad fijada en P2 (versionespineadas).
  - La funcion objetivo de este trabajo, el logit de decision de la CNN
    robusta, es EXACTAMENTE lineal en el embedding GAP de 512 dimensiones:

        s(gap) = (w1 - w0) . gap + (b1 - b0)

    Por tanto sus valores de Shapley tienen forma cerrada y conocida, lo que
    convierte este entorno en un banco de pruebas con verdad de referencia:
    cualquier metodo de atribucion puede validarse contra la verdad analitica
    antes de Comparable con la formula simbolica.

Contenidos:
  linear_shap_exact   valores de Shapley analiticos para un modelo lineal
  kernel_shap         KernelSHAP (Lundberg & Lee) con kernel de Shapley
  lime_surrogate      surrogate lineal local ponderado, al estilo LIME
  coherence_metrics   metricas de acuerdo entre dos vectores de atribucion
"""
import itertools
import math

import numpy as np


# ---------------------------------------------------------------------------
# 1. Verdad de referencia: Shapley analitico para un modelo lineal
# ---------------------------------------------------------------------------
def linear_shap_exact(coef, x, background_mean):
    """Valores de Shapley exactos de s = coef . u en un modelo lineal.

    Para una funcion de valor v(S) = coef . u_S, el valor de Shapley de la
    variable j es

        phi_j = coef_j * (x_j - E[u_j])

    Es decir, la contribucion es el coeficiente por la desviacion de la
    instancia respecto a la media de referencia. Al ser el juego cooperativo
    lineal, esta expresion es exacta y no una aproximacion.

    Args:
        coef: [d] coeficientes de la funcion lineal.
        x: [d] punto a explicar.
        background_mean: [d] media de la distribucion de referencia (u).
                         Si es None se usa el mismo punto (atribucion local).
    Returns:
        [d] valores de Shapley, cuya suma es s(x) - E[s].
    """
    coef = np.asarray(coef, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    if background_mean is None:
        background_mean = x
    bg = np.asarray(background_mean, dtype=np.float64).ravel()
    return coef * (x - bg)


# ---------------------------------------------------------------------------
# 2. KernelSHAP
# ---------------------------------------------------------------------------
def kernel_shap(predict_fn, x, background, n_samples=2000, n_background=64, seed=42):
    """KernelSHAP con regresion lineal ponderada sobre coalitions muestreadas.

    Algoritmo (Lundberg & Lee, 2017):
      1. Se muestrean coalitions z de subconjuntos de caracteristicas,addersiendo
         siempre la coalition vacia y la completa.
      2. Cada coalition se evalua sobre la representacion hibrida: las variables
         encendidas toman el valor de `x` y las apagadas una muestra de
         `background`. El valor v(z) es la media de `predict_fn` sobre esas
         muestras hibridas.
      3. Se ajusta v(z) ≈ phi_0 + sum_j phi_j z_j por minimos cuadrados con los
         pesos del kernel de Shapley, y los coeficientes son los valores de
         Shapley.

    Para la CNN, que es exactamente lineal en `gap`, `linear_shap_exact` da la
    verdad de referencia y este metodo permite contrastarla.

    Args:
        predict_fn: [N, d] -> [N] funcion a explicar.
        x: [d] punto a explicar.
        background: [M, d] muestras de la distribucion de referencia.
        n_samples: numero de coalitions a muestrear.
        n_background: muestras de referencia por coalition.
        seed: semilla del generador.
    Returns:
        dict con 'shap' [d], 'intercept', 'r2', 'n_samples' y 'n_coalitions'.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64).ravel()
    background = np.atleast_2d(np.asarray(background, dtype=np.float64))
    d = x.size
    m = background.shape[0]
    if m == 0:
        raise ValueError("Se requiere al menos una muestra de referencia")

    # --- 1. coalitions: vacia, completa y aleatorias de tamano 1..d-1 ---
    zs = [np.zeros(d), np.ones(d)]
    for _ in range(max(0, int(n_samples) - 2)):
        k = int(rng.integers(1, d))
        z = np.zeros(d)
        z[rng.choice(d, size=k, replace=False)] = 1.0
        zs.append(z)
    Z = np.asarray(zs, dtype=np.float64)                  # [C, d]

    # --- 2. valores v(z) sobre la representacion hibrida ---
    vs = np.empty(Z.shape[0], dtype=np.float64)
    for i, z in enumerate(Z):
        idx = rng.integers(0, m, size=int(n_background))
        h = z[None, :] * x[None, :] + (1.0 - z[None, :]) * background[idx]
        vs[i] = float(np.mean(np.asarray(predict_fn(h), dtype=np.float64).ravel()))

    # --- 3. regresion lineal ponderada con el kernel de Shapley ---
    # El kernel w(s) = (d-1)/(C(d,s) s (d-s)) se define para 1 <= s <= d-1.
    # Para la coalition vacia y la completa el peso es 1 por convencion; su
    # efecto queda absorbido por el intercepto de la regresion.
    sizes = Z.sum(axis=1)
    w = np.zeros(Z.shape[0], dtype=np.float64)
    interior = (sizes > 0) & (sizes < d)
    with np.errstate(divide="ignore", invalid="ignore"):
        comb = np.array([_n_choose(d, int(s)) for s in sizes], dtype=np.float64)
        w[interior] = (d - 1.0) / (comb[interior] * sizes[interior] * (d - sizes[interior]))
    w[~interior] = 1.0
    w[~np.isfinite(w)] = 0.0
    w /= max(1, int(n_background))                        # por el muestreo hibrido

    A = np.hstack([np.ones((Z.shape[0], 1)), Z])
    sw = np.sqrt(w)
    theta, *_ = np.linalg.lstsq(A * sw[:, None], vs * sw, rcond=None)

    pred = A @ theta
    ss_res = float(((vs - pred) ** 2).sum())
    ss_tot = float(((vs - vs.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "shap": theta[1:].copy(),
        "intercept": float(theta[0]),
        "r2": float(r2),
        "n_samples": int(n_samples),
        "n_coalitions": int(Z.shape[0]),
    }


def _n_choose(n, k):
    """Coeficiente binomial entero, para no depender de scipy."""
    if k < 0 or k > n:
        return 0.0
    return float(math.comb(int(n), int(k)))


# ---------------------------------------------------------------------------
# 3. LIME
# ---------------------------------------------------------------------------
def lime_surrogate(
    predict_fn,
    x,
    background=None,
    n_samples=2000,
    kernel_width=0.75,
    seed=42,
    alpha=1.0,
):
    """Surrogate lineal local ponderado, al estilo LIME.

    Genera perturbancias alrededor de `x`, evalua `predict_fn` y ajusta un
    modelo lineal por minimos cuadrados con pesos dados por una funcion kernel
    exponencial sobre la distancia al punto de referencia (L1 en la version
    original; aqui se usa L2 ponderada, que es la formulacion estandar de la
    regresion ridge localised).

    Args:
        predict_fn: [N, d] -> [N] funcion a explicar.
        x: [d] punto a explicar.
        background: [M, d] referencia para la escala de las perturbaciones.
        n_samples: numero de perturbancias.
        kernel_width: anchura del kernel exponencial.
        seed: semilla del generador.
        alpha: peso de la penalizacion ridge.
    Returns:
        dict con 'coef' [d], 'intercept', 'r2', 'intercept', 'n_samples'.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64).ravel()
    d = x.size
    if background is None:
        background = x[None, :]
    background = np.atleast_2d(np.asarray(background, dtype=np.float64))
    scale = background.std(axis=0)
    scale[scale <= 0] = 1.0

    # Perturbaciones binarias alrededor de x (tipo LIME Tabular)
    mask = rng.integers(0, 2, size=(int(n_samples), d)).astype(np.float64)
    mask[0, :] = 1.0                                   # garantiza el punto original
    Xp = x[None, :] * mask + (1.0 - mask) * (
        background[rng.integers(0, background.shape[0], int(n_samples))])

    y = np.asarray(predict_fn(Xp), dtype=np.float64).ravel()

    # Distancia al punto de referencia en escala estandar
    dist = np.sqrt((((Xp - x[None, :]) / scale[None, :]) ** 2).sum(axis=1))
    W = np.exp(-(dist ** 2) / (kernel_width ** 2 * max(d, 1)))

    # Regresion ridge ponderada
    n_obs = Xp.shape[0]
    A = np.hstack([np.ones((n_obs, 1)), Xp])
    sw = np.sqrt(W)
    Aw = A * sw[:, None]
    yw = y * sw
    reg = alpha * np.eye(d + 1)
    reg[0, 0] = 0.0                                   # no penalizar el intercepto
    try:
        theta = np.linalg.solve(Aw.T @ Aw + reg, Aw.T @ yw)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(Aw, yw, rcond=None)[0]

    pred = A @ theta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "coef": theta[1:],
        "intercept": float(theta[0]),
        "r2": r2,
        "n_samples": int(Xp.shape[0]),
        "kernel_width": float(kernel_width),
    }


# ---------------------------------------------------------------------------
# 4. Metricas de coherencia entre dos vectores de atribucion
# ---------------------------------------------------------------------------
def _rankdata(a):
    """Rangos con promedio para empates (equivalente a scipy.stats.rankdata)."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
    # Promediar empates
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    for k in range(counts.size):
        if counts[k] > 1:
            sel = inv == k
            ranks[sel] = ranks[sel].mean()
    return ranks


def spearman(a, b):
    """Correlacion de rangos de Spearman entre dos vectores."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def coherence_metrics(attr_a, attr_b, atol=1e-12):
    """Metricas de acuerdo entre dos vectores de atribucion.

    Args:
        attr_a, attr_b: [d] vectores de atribucion a comparar.
        atol: tolerancia para considerar nulo un valor.
    Returns:
        dict con rho de Spearman, Pearson, tasa de acuerdo de signo,
        acuerdo Top-1 y error relativo L2.
    """
    a = np.asarray(attr_a, dtype=np.float64).ravel()
    b = np.asarray(attr_b, dtype=np.float64).ravel()
    d = a.size

    sa, sb = a.std(), b.std()
    pear = float("nan")
    if sa > atol and sb > atol:
        pear = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))

    signs_match = float(np.mean(np.sign(a) == np.sign(b)))
    top1 = bool(np.argmax(np.abs(a)) == np.argmax(np.abs(b)))
    denom = np.linalg.norm(b)
    rel = float(np.linalg.norm(a - b) / denom) if denom > atol else float("nan")
    return {
        "spearman": spearman(a, b),
        "pearson": pear,
        "sign_agreement": signs_match,
        "top1_agreement": top1,
        "rel_l2_error": rel,
        "n_features": int(d),
    }


def agreement_by_case(attrs_a, attrs_b):
    """Agrega metricas de coherencia caso a caso.

    Args:
        attrs_a: [N, d] atribuciones A por caso.
        attrs_b: [N, d] atribuciones B por caso.
    Returns:
        dict con las metricas medias y las distribuciones.
    """
    A = np.atleast_2d(np.asarray(attrs_a, dtype=np.float64))
    B = np.atleast_2d(np.asarray(attrs_b, dtype=np.float64))
    if A.shape != B.shape:
        raise ValueError("Las matrices de atribucion deben tener igual forma")
    rows = [coherence_metrics(A[i], B[i]) for i in range(A.shape[0])]

    def agg(key):
        vals = [r[key] for r in rows if not np.isnan(r[key])]
        if not vals:
            return {"mean": float("nan"), "median": float("nan"), "min": float("nan"),
                    "max": float("nan"), "n": 0}
        v = np.asarray(vals, dtype=np.float64)
        return {"mean": float(v.mean()), "median": float(np.median(v)),
                "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}

    top1 = [r["top1_agreement"] for r in rows]
    return {
        "n_cases": int(A.shape[0]),
        "spearman": agg("spearman"),
        "pearson": agg("pearson"),
        "sign_agreement": agg("sign_agreement"),
        "rel_l2_error": agg("rel_l2_error"),
        "top1_agreement_rate": float(np.mean(top1)) if top1 else float("nan"),
        "per_case": rows,
    }
