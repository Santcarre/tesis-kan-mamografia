"""
R3 completo: S-PCA64 + KAN[64,32,2], 5 semillas x 4 backbones.
Salida: reports/experiments/r3_spca64_multiseed.json
"""
import json, os, sys, time, re
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

CODIGOS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP_DIR = os.environ.get("TESIS_REP") or os.path.join(CODIGOS, "artifacts")
for p in (CODIGOS, os.path.join(CODIGOS, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models.bottleneck import SupervisedPCAReducer
from src.explainability.f2_symb import SymbHead

EMB_DIR = os.path.join(REP_DIR, "output", "embeddings")
OUT = os.path.join(CODIGOS, "reports", "experiments", "r3_spca64_multiseed.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BACKBONES = {
    "resnet18_robusto": "f2robust",
    "densenet121": "f2_densenet121",
    "efficientnet_b0": "f2_efficientnet_b0",
    "resnet50": "f2_resnet50",
}
SEEDS = [42, 123, 7, 2026, 999]

def run_one(name, prefix, seed):
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    X_tr = np.load(os.path.join(EMB_DIR, f"{prefix}_train_feats.npy")).astype(np.float32)
    X_va = np.load(os.path.join(EMB_DIR, f"{prefix}_val_feats.npy")).astype(np.float32)
    X_te = np.load(os.path.join(EMB_DIR, f"{prefix}_test_feats.npy")).astype(np.float32)
    y_tr = np.load(os.path.join(EMB_DIR, f"{prefix}_train_labels.npy")).astype(int)
    y_va = np.load(os.path.join(EMB_DIR, f"{prefix}_val_labels.npy")).astype(int)
    y_te = np.load(os.path.join(EMB_DIR, f"{prefix}_test_labels.npy")).astype(int)
    D_in = int(X_tr.shape[1])
    mu = X_tr.mean(0, keepdims=True); sd = X_tr.std(0, keepdims=True) + 1e-8
    Xtr_z = (X_tr - mu) / sd; Xva_z = (X_va - mu) / sd; Xte_z = (X_te - mu) / sd
    red = SupervisedPCAReducer(in_dim=D_in, bottleneck_dim=64, random_state=seed, device=DEVICE)
    red.fit(Xtr_z, y_tr, Xva_z, y_va, epochs=30, lr=1e-3, weight_decay=1e-3, batch_size=256)
    z_tr = red.transform(Xtr_z); z_va = red.transform(Xva_z); z_te = red.transform(Xte_z)
    symb = SymbHead(in_dim=64, hidden=32, grid=3, k=3, seed=seed, device=DEVICE)
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
    ok, diff = SymbHead.smoke_test(d_rep["auc_D"], aC_test, tol=0.015)
    form = f"({fmal}) - ({fben})"
    used = sorted(set(re.findall(r"x_(\d+)", form)))
    dt = time.time() - t0
    print(f"[{name} s={seed}] full={auc_full:.4f} C={aC_test:.4f} D={d_rep['auc_D']:.4f} vars={len(used)} {dt:.1f}s", flush=True)
    return {"backbone": name, "seed": seed, "D_in": D_in,
        "AUC_full": round(float(auc_full), 4), "AUC_C": round(float(aC_test), 4),
        "r_C": round(float(rC), 4), "AUC_D": d_rep["auc_D"],
        "recall_D": d_rep["recall_D_youden"], "n_vars": len(used),
        "chars": len(form), "smoke_diff": round(float(diff), 4),
        "smoke_ok15": bool(ok), "tiempo_s": round(dt, 1),
        "fben": fben, "fmal": fmal}

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    allr = []
    if os.path.exists(OUT):
        with open(OUT, "r", encoding="utf-8") as f:
            try: allr = json.load(f)
            except Exception: allr = []
    done = {(r.get("backbone"), r.get("seed")) for r in allr if "error" not in r}
    for name, prefix in BACKBONES.items():
        for s in SEEDS:
            if (name, s) in done:
                print(f"skip {name} s={s} ya hecho", flush=True); continue
            print(f"\n=== R3 {name} s={s} ===", flush=True)
            try:
                allr.append(run_one(name, prefix, s))
            except Exception as e:
                import traceback; traceback.print_exc()
                allr.append({"backbone": name, "seed": s, "error": str(e)})
            with open(OUT, "w", encoding="utf-8") as f:
                json.dump(allr, f, ensure_ascii=False, indent=2)
    # resumen
    summ = {}
    for name in BACKBONES:
        vals = [r["AUC_D"] for r in allr if r.get("backbone") == name and "AUC_D" in r]
        if vals:
            summ[name] = {"mean_D": round(float(np.mean(vals)), 4),
                          "std_D": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0,
                          "n": len(vals)}
    print("\nResumen AUC_D:", json.dumps(summ, indent=2))
    print(f"Guardado: {OUT}")

if __name__ == "__main__":
    main()
