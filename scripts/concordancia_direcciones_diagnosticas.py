"""
Concordancia entre arquitecturas de la direccion diagnostica (Cap. 4, Sec. 4.4).

Pregunta: la variable u0 de la formula (extractor ResNet-18 robusto, S-PCA canonico) reaparece en otras
redes? Para cada extractor (ResNet-18 robusto y base, DenseNet-121, ResNet-50, EfficientNet-B0) se ajusta
en entrenamiento una regresion logistica L2 (C = 0,01) sobre los descriptores GAP estandarizados con la
media y la desviacion de entrenamiento, y se toma su puntaje (logit) en las 1012 imagenes de prueba como
la direccion diagnostica de esa red. En el extractor robusto ese puntaje reproduce -u0 (r ~ 0,98).
Se reportan: AUC de cada puntaje, correlaciones de Pearson y Spearman entre puntajes (y con -u0), IC95 %
por remuestreo de pacientes (B = 2000, semilla 0) y la correlacion dentro de cada clase (benignas, malignas),
que descarta que la concordancia se deba solo a que todos los puntajes predicen la misma etiqueta.

Ademas se reproduce el analisis anterior (S-PCA ponderado por correlacion, 12 componentes, emparejamiento
hungaro; recompute_cross_architecture_correlation.py) y se mide, para el par dominante de cada comparacion,
el AUC de cada eje y su correlacion con u0: el eje compartido con |r| ~ 0,9 no es u0.

Uso (desde la raiz del repositorio):
  python scripts/concordancia_direcciones_diagnosticas.py [SALIDA.json]
Los embeddings se buscan en $TESIS_REP/output/embeddings (por defecto artifacts/output/embeddings);
los de DenseNet-121, ResNet-50, EfficientNet-B0 y ResNet-18 base no se incluyen en el repositorio.
Requiere numpy, pandas, scipy, scikit-learn.
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "reports", "auditoria", "concordancia_direcciones.json")
EMB = os.path.join(os.environ.get("TESIS_REP") or os.path.join(REPO, "artifacts"), "output", "embeddings")
EMB_ROB = os.environ.get("TESIS_EMB_ROBUSTO") or EMB
MODELOS = {"ResNet-18 Robusto": "f2robust", "ResNet-18 Base": "f2base", "DenseNet-121": "f2_densenet121",
           "ResNet-50": "f2_resnet50", "EfficientNet-B0": "f2_efficientnet_b0"}
B, SEED, C = 2000, 0, 0.01


def cargar(pref):
    d = EMB_ROB if pref == "f2robust" else EMB
    f = lambda s, k: np.load(os.path.join(d, f"{pref}_{s}_{k}.npy"))
    return (f("train", "feats").astype(np.float64), f("train", "labels").astype(int),
            f("test", "feats").astype(np.float64), f("test", "labels").astype(int))


def u0_canonico(X_te):
    nz = np.load(os.path.join(REPO, "reports", "experiments", "kan_robust64_reducer.npz"))
    gz = (X_te - nz["gap_mu"]) / nz["gap_sd"]
    z = (((gz @ nz["proj_w"].T + nz["proj_b"]) - nz["pca_mean"]) @ nz["pca_components"].T - nz["mu_z"]) / nz["sd_z"]
    return z[:, 0]


def spca_b(Xtr, ytr, Xte, k=12):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    a, b = (Xtr - mu) / sd, (Xte - mu) / sd
    c = np.array([np.corrcoef(a[:, i], ytr)[0, 1] for i in range(a.shape[1])])
    w = np.abs(np.nan_to_num(c, 0.0)); w = w / (w.max() + 1e-8)
    return PCA(n_components=k, random_state=42).fit(a * w).transform(b * w)


def auc_abs(y, s):
    a = roc_auc_score(y, s)
    return max(a, 1 - a)


def main():
    man = pd.read_csv(os.path.join(REPO, "src", "data", "processed", "manifest_test.csv"))
    up, inv = np.unique(man["patient_id"].values, return_inverse=True)
    idx = [np.where(inv == i)[0] for i in range(len(up))]
    rng = np.random.default_rng(SEED)
    reps = [np.concatenate([idx[j] for j in rng.integers(0, len(up), len(up))]) for _ in range(B)]

    punt, ejes, y = {}, {}, None
    for nombre, pref in MODELOS.items():
        Xtr, ytr, Xte, yte = cargar(pref)
        if y is None:
            y = yte; assert (y == man["label"].values).all()
        assert (yte == y).all()
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        m = LogisticRegression(C=C, max_iter=5000).fit((Xtr - mu) / sd, ytr)
        punt[nombre] = m.decision_function((Xte - mu) / sd)
        ejes[nombre] = spca_b(Xtr, ytr, Xte)
        if pref == "f2robust":
            u0 = u0_canonico(Xte)
    punt_menos_u0 = -u0
    nombres = list(punt)

    def corr_ic(a, b):
        r = [np.corrcoef(a[ii], b[ii])[0, 1] for ii in reps]
        return [round(float(np.percentile(r, 2.5)), 3), round(float(np.percentile(r, 97.5)), 3)]

    res = {"descripcion": "Direccion diagnostica por extractor: logistica L2 (C=0,01) sobre GAP estandarizado; puntaje en prueba",
           "n_test": int(len(y)), "n_pacientes": int(len(up)), "B": B, "semilla_bootstrap": SEED,
           "auc_puntaje": {k: round(float(roc_auc_score(y, v)), 4) for k, v in punt.items()},
           "auc_menos_u0": round(float(roc_auc_score(y, punt_menos_u0)), 4),
           "correlacion_con_menos_u0": {k: {"pearson": round(float(np.corrcoef(v, punt_menos_u0)[0, 1]), 3),
                                            "spearman": round(float(spearmanr(v, punt_menos_u0)[0]), 3),
                                            "ic95_pearson": corr_ic(v, punt_menos_u0),
                                            "pearson_benignas": round(float(np.corrcoef(v[y == 0], punt_menos_u0[y == 0])[0, 1]), 3),
                                            "pearson_malignas": round(float(np.corrcoef(v[y == 1], punt_menos_u0[y == 1])[0, 1]), 3)}
                                        for k, v in punt.items()},
           "matriz_pearson": {a: {b: round(float(np.corrcoef(punt[a], punt[b])[0, 1]), 3) for b in nombres} for a in nombres}}

    # analisis anterior: S-PCA (variante B, 12 comp.) + hungaro, con diagnostico de los ejes emparejados
    ref = ejes["ResNet-18 Robusto"]; hung = {}
    for nombre in nombres[1:]:
        Z = ejes[nombre]
        R = np.array([[np.corrcoef(ref[:, i], Z[:, j])[0, 1] for j in range(12)] for i in range(12)])
        fi, co = linear_sum_assignment(-np.abs(R))
        pares = sorted(zip(fi, co), key=lambda t: -abs(R[t]))
        i, j = pares[0]
        hung[nombre] = {"par_dominante": [int(i), int(j)], "r": round(float(R[i, j]), 4),
                        "media_abs_r_12_pares": round(float(np.mean([abs(R[t]) for t in pares])), 4),
                        "auc_eje_ref": round(float(auc_abs(y, ref[:, i])), 3), "auc_eje_obj": round(float(auc_abs(y, Z[:, j])), 3),
                        "corr_u0_eje_ref": round(float(np.corrcoef(u0, ref[:, i])[0, 1]), 3),
                        "corr_u0_eje_obj": round(float(np.corrcoef(u0, Z[:, j])[0, 1]), 3)}
    res["hungaro_spca_b_12"] = {"nota": "Reproduce cross_architecture_correlation.json; el eje compartido no es u0",
                                "comparaciones": hung,
                                "auc_ejes_robusto": [round(float(auc_abs(y, ref[:, k])), 3) for k in range(12)],
                                "corr_u0_ejes_robusto": [round(float(np.corrcoef(u0, ref[:, k])[0, 1]), 3) for k in range(12)]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
