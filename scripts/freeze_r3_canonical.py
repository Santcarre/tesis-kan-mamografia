"""
Congela canonica R3: f2robust + S-PCA64 + KAN[64,32,2] seed 42.
Guarda reductor (gap mu/sd, proj, pca, mu_z/sd_z) + formulas + metricas.
Salidas:
 - reports/experiments/kan_robust64_canonical.json (formulas + metricas + mapping)
 - reports/experiments/kan_robust64_reducer.npz (pesos para inferencia viva)
"""
import json, os, sys, time, re
import numpy as np
import torch

CODIGOS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP_DIR = os.environ.get("TESIS_REP") or os.path.join(CODIGOS, "artifacts")
for p in (CODIGOS, os.path.join(CODIGOS, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models.bottleneck import SupervisedPCAReducer
from src.explainability.f2_symb import SymbHead

EMB_DIR = os.path.join(REP_DIR, "output", "embeddings")
OUT_JSON = os.path.join(CODIGOS, "reports", "experiments", "kan_robust64_canonical.json")
OUT_NPZ = os.path.join(CODIGOS, "reports", "experiments", "kan_robust64_reducer.npz")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
PREFIX = "f2robust"

def main():
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)
    X_tr = np.load(os.path.join(EMB_DIR, f"{PREFIX}_train_feats.npy")).astype(np.float32)
    X_va = np.load(os.path.join(EMB_DIR, f"{PREFIX}_val_feats.npy")).astype(np.float32)
    X_te = np.load(os.path.join(EMB_DIR, f"{PREFIX}_test_feats.npy")).astype(np.float32)
    y_tr = np.load(os.path.join(EMB_DIR, f"{PREFIX}_train_labels.npy")).astype(int)
    y_va = np.load(os.path.join(EMB_DIR, f"{PREFIX}_val_labels.npy")).astype(int)
    y_te = np.load(os.path.join(EMB_DIR, f"{PREFIX}_test_labels.npy")).astype(int)
    mu = X_tr.mean(0, keepdims=True).astype(np.float32)
    sd = (X_tr.std(0, keepdims=True) + 1e-8).astype(np.float32)
    Xtr_z = (X_tr - mu) / sd; Xva_z = (X_va - mu) / sd; Xte_z = (X_te - mu) / sd

    red = SupervisedPCAReducer(in_dim=X_tr.shape[1], bottleneck_dim=64, random_state=SEED, device=DEVICE)
    red.fit(Xtr_z, y_tr, Xva_z, y_va, epochs=30, lr=1e-3, weight_decay=1e-3, batch_size=256)
    z_tr = red.transform(Xtr_z); z_va = red.transform(Xva_z); z_te = red.transform(Xte_z)

    symb = SymbHead(in_dim=64, hidden=32, grid=3, k=3, seed=SEED, device=DEVICE)
    auc_val, auc_full = symb.fit_and_sparsify(
        torch.from_numpy(z_tr).float().to(DEVICE), torch.from_numpy(y_tr).long().to(DEVICE),
        torch.from_numpy(z_va).float().to(DEVICE), torch.from_numpy(y_va).long().to(DEVICE),
        torch.from_numpy(z_te).float().to(DEVICE), torch.from_numpy(y_te).long().to(DEVICE),
        fit_ep=10, sparse_ep=20)
    x_va = torch.from_numpy(z_va).float().to(DEVICE)
    x_te = torch.from_numpy(z_te).float().to(DEVICE)
    with torch.no_grad():
        Lv_ref = symb.kan(x_va).cpu().numpy()
    kp = symb.kan; kp.eval()
    with torch.no_grad():
        kp(x_va)
    kp.symbolic_enabled = True
    symb.fix_layer1(kp)
    symb.fix_layer0(kp, top_k=12, x_val=x_va)
    kp.symbolic_enabled = False
    fben, fmal = symb.make_formula(kp)
    aC_val, aC_test, rC = SymbHead.config_C(kp, x_va, x_te,
        torch.from_numpy(y_va).long().to(DEVICE), torch.from_numpy(y_te).long().to(DEVICE), Lv_ref)
    d_rep = SymbHead.report_D(fben, fmal, z_va, z_te, y_va.astype(np.float64), y_te.astype(np.float64))
    form = f"({fmal}) - ({fben})"
    used = sorted(set(re.findall(r"x_(\d+)", form)))

    # Guardar reductor para inferencia viva
    proj_w = red.proj.weight.detach().cpu().numpy().astype(np.float32)
    proj_b = red.proj.bias.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(OUT_NPZ,
        gap_mu=mu, gap_sd=sd,
        proj_w=proj_w, proj_b=proj_b,
        pca_components=red.pca.components_.astype(np.float32),
        pca_mean=red.pca.mean_.astype(np.float32),
        mu_z=red.mu_z.astype(np.float32), sd_z=red.sd_z.astype(np.float32))
    canon = {"backbone": "resnet18_robusto", "prefix": PREFIX, "seed": SEED,
        "reductor": "S-PCA64 SupervisedPCAReducer D->64 + PCA64 (fit solo train, best_val_auc)",
        "kan": "[64,32,2] grid=3 k=3 top_k=12",
        "AUC_full": round(float(auc_full), 4), "AUC_val": round(float(auc_val), 4),
        "AUC_C": round(float(aC_test), 4), "r_C": round(float(rC), 4),
        "AUC_D": d_rep["auc_D"], "th_youden_val": d_rep["th_youden_val"],
        "J_youden_val": d_rep["J_youden_val"], "recall_D": d_rep["recall_D_youden"],
        "spec_D": d_rep["spec_D_youden"], "n_vars": len(used), "vars": used,
        "chars": len(form), "fben": fben, "fmal": fmal,
        "reducer_npz": "reports/experiments/kan_robust64_reducer.npz",
        "tiempo_s": round(time.time() - t0, 1)}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(canon, f, ensure_ascii=False, indent=2)
    print(f"AUC_full={auc_full:.4f} C={aC_test:.4f} rC={rC:.4f} D={d_rep['auc_D']:.4f} vars={used}")
    print(f"Guardado: {OUT_JSON}\nGuardado: {OUT_NPZ}")

if __name__ == "__main__":
    main()
