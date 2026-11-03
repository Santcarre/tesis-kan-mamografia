"""
Bootstrap pareado por paciente para todos los contrastes de AUC de la Tabla 4.x
(tab:contrastes_auc). Mismo método que recalculos_auditoria.py: B=2000, semilla 0,
se remuestrean pacientes completas con reemplazo y en cada réplica se calculan los
AUC de todos los modelos sobre la MISMA muestra (pareado).

Uso (desde la raiz del repo):  python scripts/bootstrap_pares_paciente.py [RAIZ] [SALIDA.json]
RAIZ = raiz del repositorio (por defecto, la carpeta padre de scripts/).
Requiere numpy, pandas, scikit-learn.
"""
import sys, os, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = sys.argv[1] if len(sys.argv) > 1 else REPO   # raiz del repositorio
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "reports", "auditoria", "bootstrap_pares_paciente.json")
COD = ROOT
EMB = os.path.join(ROOT, "artifacts", "output", "embeddings", "f2robust_")
PROB = os.path.join(COD, "reports", "experiments", "youden_val_probs")
SEED, B = 0, 2000

# Puntajes de prueba guardados
m = pd.read_csv(os.path.join(COD, "src", "data", "processed", "manifest_test.csv"))
yt = np.load(os.path.join(PROB, "Baseline_FC_test_labels.npy")).astype(int)
assert (yt == m.label.values).all(), "Las etiquetas no coinciden con el manifiesto"
S = {k: np.load(os.path.join(PROB, f"{k}_test_probs.npy")).ravel()
     for k in ("Baseline_FC", "Robusto", "R3_formula")}

# Control lineal: logística C=0.01 sobre los 512 descriptores estandarizados (igual que recalculos_auditoria.py)
X = {s: np.load(EMB + f"{s}_feats.npy").astype(np.float64) for s in ("train", "test")}
ytr = np.load(EMB + "train_labels.npy").astype(int)
yte_emb = np.load(EMB + "test_labels.npy").astype(int)
assert (yte_emb == yt).all(), "El orden de los embeddings de prueba no coincide con las probabilidades"
mu, sd = X["train"].mean(0), X["train"].std(0) + 1e-8
lr = LogisticRegression(C=0.01, max_iter=5000).fit((X["train"] - mu) / sd, ytr)
S["Logistica_512"] = lr.decision_function((X["test"] - mu) / sd)

PARES = [("Baseline_FC", "R3_formula"), ("Robusto", "R3_formula"),
         ("Logistica_512", "R3_formula"), ("Baseline_FC", "Robusto")]

up, inv = np.unique(m.patient_id.values, return_inverse=True)
idx = [np.where(inv == i)[0] for i in range(len(up))]
rng = np.random.default_rng(SEED)
dif = {f"{a}-{b}": [] for a, b in PARES}
for _ in range(B):
    ii = np.concatenate([idx[j] for j in rng.integers(0, len(up), len(up))])
    yy = yt[ii]
    if yy.min() == yy.max():
        continue
    auc = {k: roc_auc_score(yy, v[ii]) for k, v in S.items()}
    for a, b in PARES:
        dif[f"{a}-{b}"].append(auc[a] - auc[b])

res = {"semilla": SEED, "B": B, "n_imagenes": int(len(yt)), "n_pacientes": int(len(up)),
       "auc_punto": {k: round(float(roc_auc_score(yt, v)), 4) for k, v in S.items()}, "contrastes": {}}
for (a, b) in PARES:
    d = np.array(dif[f"{a}-{b}"])
    res["contrastes"][f"{a} - {b}"] = dict(
        delta_punto=round(float(roc_auc_score(yt, S[a]) - roc_auc_score(yt, S[b])), 4),
        ic95=[round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
        p_bilateral=round(float(2 * min((d <= 0).mean(), (d >= 0).mean())), 4),
        replicas_validas=int(len(d)))
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(res, ensure_ascii=False, indent=2))
