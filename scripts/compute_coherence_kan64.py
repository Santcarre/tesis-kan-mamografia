"""
Bateria de coherencia KAN64 (R3 canonica) vs explicadores.
Parte 1 (CPU): 200 casos test, rho Spearman + Top-k entre
  - verdad lineal phi = a*(x - mu_bg) [a = W1-W0 robusto]
  - LIME por caso (3000 muestras, como attribution_coherence)
  - KAN dg/dgap por regla de la cadena via reductor S-PCA64 canonico + d/du0 analitica
Parte 2 (GPU): psi_KAN en 200 ROIs con mascara, EBPG medio vs psi1 (S-PCA12).
Salida: reports/experiments/coherence_kan64.json
"""
import json, os, sys, time
import numpy as np

CODIGOS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (CODIGOS, os.path.join(CODIGOS, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch
from torchvision import models
from scipy.stats import spearmanr

from src.explainability.attribution_baselines import lime_surrogate, coherence_metrics
from src.reporting.clinical_evidence_extractor import _resolve_embeddings_dir

SEED = 42
N_CASES = 200
LIME_SAMPLES = 3000
OUT = os.path.join(CODIGOS, "reports", "experiments", "coherence_kan64.json")

def load_robust():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(os.path.join(CODIGOS, "reports", "models", "resnet18_robust_best.pt"),
                     map_location=dev, weights_only=False)
    m = models.resnet18(weights=None); m.fc = torch.nn.Linear(512, 2)
    m.load_state_dict(ck["model_state_dict"]); m = m.to(dev).eval()
    W = m.fc.weight.detach().cpu().numpy().astype(np.float64)
    return m, W[1] - W[0], dev

def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    m, a, dev = load_robust()
    emb = _resolve_embeddings_dir()
    Xtr = np.load(os.path.join(emb, "f2robust_train_feats.npy")).astype(np.float64)
    Xte = np.load(os.path.join(emb, "f2robust_test_feats.npy")).astype(np.float64)
    cj = json.load(open(os.path.join(CODIGOS, "reports", "experiments", "kan_robust64_canonical.json"), encoding="utf-8"))
    nz = np.load(os.path.join(CODIGOS, "reports", "experiments", "kan_robust64_reducer.npz"))
    gap_mu = nz["gap_mu"].astype(np.float64); gap_sd = nz["gap_sd"].astype(np.float64)
    proj_w = nz["proj_w"].astype(np.float64); proj_b = nz["proj_b"].astype(np.float64)
    pc, pm = nz["pca_components"].astype(np.float64), nz["pca_mean"].astype(np.float64)
    mu_z, sd_z = nz["mu_z"].astype(np.float64), nz["sd_z"].astype(np.float64)
    import sympy as sp, re
    e_ben, e_mal = sp.sympify(cj["fben"]), sp.sympify(cj["fmal"])
    used = sorted({int(x) for x in re.findall(r"x_(\d+)", cj["fmal"] + cj["fben"])})
    assert used == [1], f"canonica debe ser univariable, hallado {used}"
    u = sp.Symbol("x_1")
    dg_du = sp.lambdify(u, sp.diff(e_mal - e_ben, u), "numpy")
    # fila efectiva dg/dgap (salvo escalar dg/du0 por caso): r0 = (pca_row0/sd_z0) @ proj_w / gap_sd
    r0 = ((pc[0] / sd_z.ravel()[0]) @ proj_w) / gap_sd.ravel()  # [512]
    r0 = r0 / (np.linalg.norm(r0) + 1e-12)

    bg = Xte[rng.choice(Xte.shape[0], size=min(200, Xte.shape[0]), replace=False)]
    mu_bg = bg.mean(axis=0)
    f_lin = lambda U: np.asarray(U, dtype=np.float64) @ a
    idx = rng.choice(Xte.shape[0], size=min(N_CASES, Xte.shape[0]), replace=False)
    rhos_kan_lin, rhos_lime_lin, rhos_kan_lime = [], [], []
    top1_kan_lin, top1_lime_lin, top1_kan_lime = 0, 0, 0
    for k, i in enumerate(idx):
        xi = Xte[i]
        phi_lin = a * (xi - mu_bg)
        # KAN por caso: escalar dg/du0 * direccion fija r0
        gz = (xi.reshape(1, -1) - gap_mu) / gap_sd
        zp = gz @ proj_w.T + proj_b
        zc = (zp - pm) @ pc.T
        z64 = ((zc - mu_z) / sd_z).ravel()
        s = float(np.clip(z64[0], -10, 10))
        phi_kan = r0 * float(dg_du(s))
        lm = lime_surrogate(f_lin, xi, background=bg, n_samples=LIME_SAMPLES, seed=SEED + k)
        phi_lime = lm["coef"] * (xi - mu_bg)
        rhos_kan_lin.append(float(spearmanr(phi_kan, phi_lin).statistic))
        rhos_lime_lin.append(float(spearmanr(phi_lime, phi_lin).statistic))
        rhos_kan_lime.append(float(spearmanr(phi_kan, phi_lime).statistic))
        if int(np.argmax(np.abs(phi_kan))) == int(np.argmax(np.abs(phi_lin))): top1_kan_lin += 1
        if int(np.argmax(np.abs(phi_lime))) == int(np.argmax(np.abs(phi_lin))): top1_lime_lin += 1
        if int(np.argmax(np.abs(phi_kan))) == int(np.argmax(np.abs(phi_lime))): top1_kan_lime += 1
        if (k + 1) % 50 == 0:
            print(f"  caso {k+1}/{len(idx)}", flush=True)

    res = {"n": int(len(idx)), "seed": SEED, "nota": (
        "Formula canonica univariable: dg/dgap tiene direccion fija r0 (S-PCA64) y solo cambia "
        "el escalar dg/du0 por caso. El ranking KAN es por tanto constante salvo signo; "
        "la comparacion honesta es Top-1/dir vs LIME/lineal, no rho por caso."),
        "kan_direccion_fija": True,
        "rho_kan_vs_lineal": {"mean": float(np.mean(rhos_kan_lin)), "median": float(np.median(rhos_kan_lin))},
        "rho_lime_vs_lineal": {"mean": float(np.mean(rhos_lime_lin)), "median": float(np.median(rhos_lime_lin))},
        "rho_kan_vs_lime": {"mean": float(np.mean(rhos_kan_lime)), "median": float(np.median(rhos_kan_lime))},
        "top1_kan_vs_lineal": f"{top1_kan_lin}/{len(idx)}",
        "top1_lime_vs_lineal": f"{top1_lime_lin}/{len(idx)}",
        "top1_kan_vs_lime": f"{top1_kan_lime}/{len(idx)}",
        "canal_top1_kan": int(np.argmax(np.abs(r0))),
        "segundos": round(time.time() - t0, 1)}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"rho KAN/lin {res['rho_kan_vs_lineal']} LIME/lin {res['rho_lime_vs_lineal']} KAN/LIME {res['rho_kan_vs_lime']}")
    print(f"Top1 KAN/lin {res['top1_kan_vs_lineal']} LIME/lin {res['top1_lime_vs_lineal']} KAN/LIME {res['top1_kan_vs_lime']}")
    print(f"Guardado: {OUT} ({res['segundos']}s)")

if __name__ == "__main__":
    main()
