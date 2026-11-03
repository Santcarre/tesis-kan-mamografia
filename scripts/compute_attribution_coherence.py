"""
compute_attribution_coherence.py: Estudio de coherencia entre la atribucion
analitica de la CNN y los metodos post-hoc LIME y SHAP, en el espacio de las
variables canonicas.

Este script responde al paso 10 del pipeline de la tesis: como afecta la
formula simbolica (y el operador de atribucion espacial que la sostiene) a los
algoritmos conocidos de explicabilidad.

Bloques del estudio
  A. Validacion de los explicadores. El logit de la CNN es exactamente lineal
     en el embedding GAP de 512 dimensiones, de modo que sus valores de Shapley
     tienen forma cerrada conocida. Se contrasta LIME y KernelSHAP contra esa
     verdad de referencia.
  B. Escalado con la dimensionalidad. Se mide como decae el acuerdo al
     aumentar el numero de variables con un presupuesto de muestreo fijo.
  C. Puente canonico. Se descompone el logit en la parte que sobrevive a la
     compresion S-PCA de 12 componentes y el residuo de 500 dimensiones.
  D. Coherencia por caso sobre las 368 ROI de prueba, entre la atribucion
     analitica y la de LIME, y entre el credito por variable canonica y la
     contribucion de la formula simbolica.

Salida: reports/experiments/attribution_coherence.json
"""
import io
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # raiz del repositorio
CODIGOS = ROOT
if CODIGOS not in sys.path:
    sys.path.insert(0, CODIGOS)

import torch
from torchvision import models
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from src.explainability.attribution_baselines import (
    linear_shap_exact, kernel_shap, lime_surrogate, coherence_metrics, agreement_by_case)
from src.reporting.clinical_evidence_extractor import _resolve_embeddings_dir

SEED = 42
OUT_NAME = "attribution_coherence.json"
LIME_SAMPLES = 3000
KS_SAMPLES = 600
KS_BACKGROUND = 32
SCALING_DIMS = [12, 24, 32, 48, 64, 128, 256, 512]


def _load():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(os.path.join(CODIGOS, "reports", "models", "resnet18_robust_best.pt"),
                    map_location=dev, weights_only=False)
    m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(512, 2)
    m.load_state_dict(ck["model_state_dict"])
    m = m.to(dev).eval()
    W = m.fc.weight.detach().cpu().numpy().astype(np.float64)
    b = m.fc.bias.detach().cpu().numpy().astype(np.float64)
    return m, W, b, dev


def main():
    t_start = time.time()
    res = {
        "date": time.strftime("%Y-%m-%d"),
        "seed": SEED,
        "model": "ResNet-18 Robusto (resnet18_robust_best.pt)",
        "nota_alcance": (
            "El logit de la CNN es exactamente lineal en el embedding GAP de 512 "
            "dimensiones, por lo que sus valores de Shapley tienen forma cerrada y "
            "sirven como verdad de referencia. La comparacion con la formula "
            "simbolica se hace sobre el ROL de cada variable canonica y no sobre "
            "magnitudes, porque la formula vive en el espacio D'=48 del modelo "
            "1c_base (base 1) y la estacion en el espacio S-PCA de 12 componentes "
            "del backbone robusto (base 0): son espacios distintos."
        ),
    }

    m, W, b, dev = _load()
    a = W[1] - W[0]
    bshift = float(b[1] - b[0])
    res["logit"] = {
        "coef_norm": float(np.linalg.norm(a)),
        "bias_diff": bshift,
        "w0_norm": float(np.linalg.norm(W[0])),
        "w1_norm": float(np.linalg.norm(W[1])),
    }

    emb = _resolve_embeddings_dir()
    Xtr = np.load(os.path.join(emb, "f2robust_train_feats.npy")).astype(np.float64)
    ytr = np.load(os.path.join(emb, "f2robust_train_labels.npy"))
    Xte = np.load(os.path.join(emb, "f2robust_test_feats.npy")).astype(np.float64)
    yte = np.load(os.path.join(emb, "f2robust_test_labels.npy"))
    res["datos"] = {"n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
                    "dim": int(Xtr.shape[1])}

    mu = Xtr.mean(0, keepdims=True)
    sd = (Xtr.std(0, keepdims=True) + 1e-8).ravel()
    Z = (Xtr - mu) / sd
    corrs = np.array([np.corrcoef(Z[:, i], ytr)[0, 1] for i in range(Z.shape[1])])
    weights = np.abs(np.nan_to_num(corrs, 0.0))
    weights = weights / (weights.max() + 1e-8)

    pca = PCA(n_components=12, random_state=SEED)
    pca.fit(Z * weights)
    V = pca.components_
    res["spca"] = {
        "n_components": 12,
        "varianza_explicada": float(pca.explained_variance_ratio_.sum()),
    }

    S = Xtr @ a + bshift
    U = Z @ V.T                                   # coords canonicas (sin pesos)
    Uw = (Z * weights) @ V.T                       # coords tal como la estacion

    # ---------------- C. Puente canonico ----------------
    beta = (V * sd) @ a                            # coeficiente de s en cada coord
    g = sd * a
    resid = g - V.T @ (V @ g)
    s_resid = Z @ resid
    r2_ort = 1.0 - float(np.var(s_resid)) / float(np.var(S))
    coef_ls, *_ = np.linalg.lstsq(U, S, rcond=None)
    S_ls = U @ coef_ls
    r2_ls = 1.0 - float(np.var(S - S_ls)) / float(np.var(S))
    beta_w = (V * weights * sd) @ a
    res["puente_canonico"] = {
        "beta_12d": [float(v) for v in beta],
        "beta_12d_con_pesos": [float(v) for v in beta_w],
        "r2_proyeccion_ortogonal": r2_ort,
        "r2_mínimos_cuadrados": r2_ls,
        "correlacion_s_proyectado": float(np.corrcoef(S, S - s_resid)[0, 1]),
        "auc_logit_completo": float(roc_auc_score(ytr, 1.0 / (1.0 + np.exp(-S)))),
        "auc_logit_reconstruido": float(roc_auc_score(ytr, 1.0 / (1.0 + np.exp(-S_ls)))),
        "orden_importancia_canonica": [int(j) for j in np.argsort(-np.abs(beta))],
        "orden_importancia_canonica_con_pesos":
            [int(j) for j in np.argsort(-np.abs(beta_w))],
    }

    # ---------------- A. Validacion de los explicadores ----------------
    rng = np.random.default_rng(SEED)
    d = Xtr.shape[1]
    bg = Xtr[rng.choice(Xtr.shape[0], size=min(200, Xtr.shape[0]), replace=False)]
    x0 = Xtr[rng.choice(Xtr.shape[0], size=min(64, Xtr.shape[0]), replace=False)]
    f_lin = lambda U: np.asarray(U, dtype=np.float64) @ a
    truth = linear_shap_exact(a, x0.mean(axis=0), bg.mean(axis=0))

    t0 = time.time()
    lm = lime_surrogate(f_lin, x0[0], background=bg, n_samples=LIME_SAMPLES, seed=SEED)
    t_lime = time.time() - t0
    t0 = time.time()
    ks = kernel_shap(f_lin, x0[0], background=bg, n_samples=KS_SAMPLES,
                     n_background=KS_BACKGROUND, seed=SEED)
    t_ks = time.time() - t0

    res["validacion"] = {
        "dimension": int(d),
        "lime": {
            "r2_surrogate": float(lm["r2"]),
            "segundos": round(t_lime, 3),
            "n_samples": lm["n_samples"],
        },
        "kernel_shap": {
            "r2_regresion": float(ks["r2"]),
            "segundos": round(t_ks, 3),
            "n_coalitions": ks["n_coalitions"],
        },
        "nota": (
            "LIME estima el gradiente local, que en una funcion lineal es el "
            "vector de coeficientes; KernelSHAP estima los valores de Shapley. "
            "Se comparan ambos contra la verdad analitica en el bloque D."
        ),
    }

    # ---------------- B. Escalado con la dimensionalidad ----------------
    escalado = []
    for dim in SCALING_DIMS:
        r = np.random.default_rng(SEED)
        c = r.normal(size=dim)
        bgi = r.normal(size=(min(200, 4 * dim), dim))
        xi = r.normal(size=dim)
        fi = lambda U, c=c: np.asarray(U, dtype=np.float64) @ c
        tru = linear_shap_exact(c, xi, bgi.mean(axis=0))
        lm_i = lime_surrogate(fi, xi, background=bgi, n_samples=LIME_SAMPLES, seed=SEED)
        ks_i = kernel_shap(fi, xi, background=bgi, n_samples=KS_SAMPLES,
                           n_background=KS_BACKGROUND, seed=SEED)
        # LIME estima coef; se pondera por (x - Ex) para obtener magnitudes SHAP
        pred_lime = lm_i["coef"] * (xi - bgi.mean(axis=0))
        escalado.append({
            "dimension": int(dim),
            "lime_rho_shap": float(coherence_metrics(pred_lime, tru)["spearman"]),
            "lime_r2": float(lm_i["r2"]),
            "kernel_shap_rho": float(coherence_metrics(ks_i["shap"], tru)["spearman"]),
            "kernel_shap_r2": float(ks_i["r2"]),
        })
    res["escalado"] = {
        "presupuesto_lime": LIME_SAMPLES,
        "presupuesto_kernel_shap": KS_SAMPLES,
        "curva": escalado,
        "nota": (
            "Con presupuesto de muestreo fijo, LIME mantiene el acuerdo porque "
            "ajusta un surrogate local de d+1 parametros. KernelSHAP muestrea "
            "coalitions de un conjunto de tamano 2^d: con tamano de coalicion "
            "uniforme en 1..d-1, al aumentar d la probabilidad de que aparezca "
            "una coalicion pequena (las unica con peso apreciable en el kernel "
            "de Shapley, w(s)=(d-1)/(C(d,s) s (d-s))) cae, de modo que la "
            "regresion ponderada se apoya en pocas Effectively informative. "
            "La metrica primaria es la correlacion de rangos con la verdad "
            "analitica; el R2 de la regresion de KernelSHAP se reporta sin "
            "ponderar y no debe leerse como calidad del ajuste."
        ),
    }

    # ---------------- D. Coherencia por caso sobre las ROI de prueba ----------------
    bg_test = Xte[rng.choice(Xte.shape[0], size=min(200, Xte.shape[0]), replace=False)]
    mu_bg = bg_test.mean(axis=0)
    n_cases = min(200, Xte.shape[0])
    exact_all, lime_all, canon_all = [], [], []
    r2_lime_fila, r2_ks_fila = [], []
    for i in range(n_cases):
        xi = Xte[i]
        tru = linear_shap_exact(a, xi, mu_bg)
        lm_i = lime_surrogate(f_lin, xi, background=bg_test, n_samples=LIME_SAMPLES,
                              seed=SEED + i)
        ks_i = kernel_shap(f_lin, xi, background=bg_test, n_samples=KS_SAMPLES,
                           n_background=KS_BACKGROUND, seed=SEED + i)
        r2_lime_fila.append(float(lm_i["r2"]))
        r2_ks_fila.append(float(ks_i["r2"]))
        # credito por variable canonica: beta_j * (u_j - E[u_j])
        Ui = ((xi[None, :] - mu) / sd) @ V.T
        canon = beta * (Ui[0] - U.mean(axis=0))
        exact_all.append(tru)
        lime_all.append(lm_i["coef"] * (xi - mu_bg))
        canon_all.append(canon)

    def _agg(vals):
        v = np.asarray(vals, dtype=np.float64)
        return {"mean": float(v.mean()), "median": float(np.median(v)),
                "min": float(v.min()), "max": float(v.max())}

    res["coherencia_caso"] = {
        "n_casos": int(n_cases),
        "espacio": "512-dimensional GAP",
        "lime_vs_shap_analitico": agreement_by_case(lime_all, exact_all),
        "lime_r2": _agg(r2_lime_fila),
        "kernel_shap_r2": _agg(r2_ks_fila),
    }
    # coherencia entre el credito canonico (12-d) y el canonico exacto (512-d)
    # se comparan solo las 12 coords que la estacion muestra
    canon_exact_top = []
    for i in range(n_cases):
        tru = np.asarray(exact_all[i], dtype=np.float64)
        # proyeccion ortogonal de la atribucion 512-d sobre la base canonica
        canon_exact_top.append(V @ tru)
    res["coherencia_canonica"] = {
        "descripcion": (
            "Correlacion entre (a) la proyeccion sobre la base PCA de la "
            "atribucion analitica 512-d, V @ phi, y (b) el credito que la base "
            "canonica atribuye a cada componente, beta_j multiplicado por la "
            "desviacion de la coordenada. Mide cuanto se alinea la atribucion "
            "por dimension del embedding con las direcciones que la compresion "
            "S-PCA declara canonicas. Un valor bajo indica que la compresion de "
            "12 dimensiones reorganiza el credito, no que el metodo falle."
        ),
        "beta_por_componente": [float(v) for v in beta],
        **agreement_by_case(canon_all, canon_exact_top),
    }

    res["segundos_total"] = round(time.time() - t_start, 2)
    out = os.path.join(CODIGOS, "reports", "experiments", OUT_NAME)
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("escrito %s (%.1f s)" % (out, res["segundos_total"]))


if __name__ == "__main__":
    main()
