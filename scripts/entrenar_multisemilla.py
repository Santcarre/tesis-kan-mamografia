"""
Reentrenamiento multi-semilla del backbone (revision de octubre 2026).

Para cada semilla entrena desde cero (pesos ImageNet) dos ResNet-18 con los protocolos del repo:
  base   : notebook 01  (Adam lr 1e-4, wd 1e-4, CE ponderada, ReduceLROnPlateau(max,0.5,3), 20 ep, paciencia 5, drop_last)
  robusto: notebook 26  (5 ep warmup solo fc lr 1e-3; despues layer3 1e-5 / layer4 5e-5 / fc 2e-4, AdamW wd 1e-2,
                         CosineAnnealing T_max=25, Focal(gamma 2, alpha=[1, n_neg/n_pos]), paciencia 7)
y, sobre sus embeddings GAP de 512-D, repite la etapa S-PCA64 y mide u0, regresion logistica sobre u0 y sobre 512-D.
Con --kan repite ademas la destilacion KAN[64,32,2] (misma receta de freeze_r3_canonical.py).

SUPUESTOS (el codigo exacto del bucle robusto no esta en el repo; se reconstruyó del notebook 26 y su historial):
  * en el warmup el backbone se mantiene en eval() (BN congelada); * el cosine avanza una vez por epoca de fine-tuning;
  * checkpoint = mejor AUC de validacion; * semillas nuevas (no 42) para medir variabilidad, 42 se puede incluir.
Nunca se usa el conjunto de prueba para elegir epoca, umbral ni modelo.

Uso (desde la raiz del repositorio):
  python scripts/entrenar_multisemilla.py --smoke            # prueba rapida (~minutos)
  python scripts/entrenar_multisemilla.py --seeds 1 2 3       # corrida completa (protocolo de la tesis: 3 semillas)
  python scripts/entrenar_multisemilla.py --aggregate         # tabla final
"""
import argparse, json, os, random, sys, time
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DEFAULT = os.path.abspath(os.path.join(HERE, ".."))   # raiz del repositorio
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class Focal(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__(); self.alpha, self.gamma = alpha, gamma
    def forward(self, x, y):
        ce = F.cross_entropy(x, y, reduction="none", weight=self.alpha)
        return (((1 - torch.exp(-ce)) ** self.gamma) * ce).mean()


def youden(y, s):
    f, t, th = roc_curve(y, s); return float(th[(t - f).argmax()])


def op(y, s, th):
    p = s >= th
    return float((p & (y == 1)).sum() / (y == 1).sum()), float(((~p) & (y == 0)).sum() / (y == 0).sum())


@torch.no_grad()
def predict(model, loader, embed=False):
    model.eval(); P, L, E = [], [], []
    fc = model.fc
    if embed: model.fc = nn.Identity()
    for x, y in loader:
        o = model(x.to(DEVICE))
        if embed: E.append(o.cpu().numpy())
        else: P.append(torch.softmax(o, 1)[:, 1].cpu().numpy())
        L.append(y.numpy())
    model.fc = fc
    return (np.concatenate(E) if embed else np.concatenate(P)), np.concatenate(L)


def loaders(cod, seed, smoke, drop_last=False):
    sys.path.insert(0, cod); sys.path.insert(0, os.path.join(cod, "src"))
    from src.data import MammographyDataset
    from src.data.preprocessing import get_train_transforms, get_val_transforms
    d = os.path.join(cod, "src", "data", "processed")
    df = {s: pd.read_csv(os.path.join(d, f"manifest_{s}.csv")) for s in ("train", "val", "test")}
    if smoke:
        df = {k: v.sample(160, random_state=0).reset_index(drop=True) for k, v in df.items()}
    g = torch.Generator(); g.manual_seed(seed)
    mk = lambda s, tf, sh, dl=False: DataLoader(MammographyDataset(df[s], transform=tf), batch_size=32, shuffle=sh,
                                                drop_last=dl, num_workers=2, pin_memory=(DEVICE == "cuda"), generator=g if sh else None)
    tr_aug = get_train_transforms(); va = get_val_transforms()
    return dict(train=mk("train", tr_aug, True, drop_last), train_eval=mk("train", va, False), val=mk("val", va, False), test=mk("test", va, False)), df["train"]


def new_resnet():
    m = models.resnet18(weights=ResNet18_Weights.DEFAULT); m.fc = nn.Linear(512, 2); return m.to(DEVICE)


def train_epoch(model, loader, opt, crit, frozen_bn=False):
    model.train()
    if frozen_bn: model.eval()
    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        tot += loss.item() * len(y); n += len(y)
    return tot / n


def run_base(L, dtr, seed, ep_max):
    model = new_resnet(); n0, n1 = (dtr.label == 0).sum(), (dtr.label == 1).sum()
    w = torch.tensor([1.0 / n0, 1.0 / n1], dtype=torch.float32); w = (w / w.sum()).to(DEVICE)  # igual a MammographyDataset.get_class_weights()
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    best, best_ep, bad, best_sd, hist = -1, 0, 0, None, []
    for ep in range(1, ep_max + 1):
        l = train_epoch(model, L["train"], opt, crit)
        p, y = predict(model, L["val"]); a = roc_auc_score(y, p); sch.step(a); hist.append((ep, l, a))
        print(f"  [base s{seed}] ep {ep:02d} loss {l:.4f} val_auc {a:.4f}", flush=True)
        if a > best: best, best_ep, bad, best_sd = a, ep, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 5: break
    model.load_state_dict(best_sd); return model, best, best_ep, hist


def run_robust(L, dtr, seed, ep_warm, ep_ft):
    model = new_resnet(); n0, n1 = int((dtr.label == 0).sum()), int((dtr.label == 1).sum())
    crit = Focal(torch.tensor([1.0, n0 / n1], device=DEVICE), 2.0)
    for p in model.parameters(): p.requires_grad = False
    for p in model.fc.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-2)
    best, best_ep, bad, best_sd, hist = -1, 0, 0, None, []
    for ep in range(1, ep_warm + 1):
        l = train_epoch(model, L["train"], opt, crit, frozen_bn=True)
        p, y = predict(model, L["val"]); a = roc_auc_score(y, p); hist.append((ep, l, a))
        print(f"  [robusto s{seed}] warmup {ep} loss {l:.4f} val_auc {a:.4f}", flush=True)
        if a > best: best, best_ep, best_sd = a, ep, {k: v.clone() for k, v in model.state_dict().items()}
    for p in model.layer3.parameters(): p.requires_grad = True
    for p in model.layer4.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW([{"params": model.fc.parameters(), "lr": 2e-4}, {"params": model.layer3.parameters(), "lr": 1e-5},
                             {"params": model.layer4.parameters(), "lr": 5e-5}], weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(ep_ft, 1)); bad = 0
    for k in range(1, ep_ft + 1):
        ep = ep_warm + k; l = train_epoch(model, L["train"], opt, crit); sch.step()
        p, y = predict(model, L["val"]); a = roc_auc_score(y, p); hist.append((ep, l, a))
        print(f"  [robusto s{seed}] ft {ep} loss {l:.4f} val_auc {a:.4f}", flush=True)
        if a > best: best, best_ep, bad, best_sd = a, ep, 0, {k2: v.clone() for k2, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 7: break
    model.load_state_dict(best_sd); return model, best, best_ep, hist


def spca_study(cod, E, Y, seed, kan):
    sys.path.insert(0, cod); sys.path.insert(0, os.path.join(cod, "src"))
    from src.models.bottleneck import SupervisedPCAReducer
    mu, sd = E["train"].mean(0, keepdims=True), E["train"].std(0, keepdims=True) + 1e-8
    Zs = {k: ((E[k] - mu) / sd).astype(np.float32) for k in E}
    red = SupervisedPCAReducer(in_dim=512, bottleneck_dim=64, random_state=seed, device=DEVICE)
    red.fit(Zs["train"], Y["train"], Zs["val"], Y["val"], epochs=30, lr=1e-3, weight_decay=1e-3, batch_size=256)
    Z = {k: red.transform(Zs[k]) for k in Zs}; u = {k: Z[k][:, 0] for k in Z}
    sgn = -1.0 if roc_auc_score(Y["val"], u["val"]) < 0.5 else 1.0   # orientacion fijada en VALIDACION
    out = {"signo_u0": sgn, "auc_u0": {k: round(float(roc_auc_score(Y[k], sgn * u[k])), 4) for k in u}}
    th = youden(Y["val"], sgn * u["val"]); se, sp = op(Y["test"], sgn * u["test"], th)
    out["u0_youden_val"] = {"sens": round(se, 4), "spec": round(sp, 4)}
    def logit(F_, C=1.0):
        m = LogisticRegression(C=C, max_iter=5000).fit(F_["train"], Y["train"]); sv, st = m.decision_function(F_["val"]), m.decision_function(F_["test"])
        t = youden(Y["val"], sv); a, b = op(Y["test"], st, t)
        return {"auc_val": round(float(roc_auc_score(Y["val"], sv)), 4), "auc_test": round(float(roc_auc_score(Y["test"], st)), 4), "sens": round(a, 4), "spec": round(b, 4)}
    out["logistica_u0"] = logit({k: u[k][:, None] for k in u})
    out["logistica_512_C0.01"] = logit(Zs, 0.01)
    if kan:
        from src.explainability.f2_symb import SymbHead
        import re
        zt = lambda k: torch.from_numpy(Z[k]).float().to(DEVICE); yt = lambda k: torch.from_numpy(Y[k]).long().to(DEVICE)
        sym = SymbHead(in_dim=64, hidden=32, grid=3, k=3, seed=seed, device=DEVICE)
        av, af = sym.fit_and_sparsify(zt("train"), yt("train"), zt("val"), yt("val"), zt("test"), yt("test"), fit_ep=10, sparse_ep=20)
        with torch.no_grad(): Lv = sym.kan(zt("val")).cpu().numpy()
        kp = sym.kan; kp.eval()
        with torch.no_grad(): kp(zt("val"))
        kp.symbolic_enabled = True; sym.fix_layer1(kp); sym.fix_layer0(kp, top_k=12, x_val=zt("val")); kp.symbolic_enabled = False
        fb, fm = sym.make_formula(kp)
        aC_v, aC_t, rC = SymbHead.config_C(kp, zt("val"), zt("test"), yt("val"), yt("test"), Lv)
        d = SymbHead.report_D(fb, fm, Z["val"], Z["test"], Y["val"].astype(np.float64), Y["test"].astype(np.float64))
        form = f"({fm}) - ({fb})"
        out["kan"] = {"auc_red_completa_test": round(float(af), 4), "auc_C_test": round(float(aC_t), 4), "r_C": round(float(rC), 4),
                      "auc_D_formula": d["auc_D"], "recall_D": d["recall_D_youden"], "spec_D": d["spec_D_youden"],
                      "vars": sorted(set(re.findall(r"x_(\d+)", form))), "chars": len(form)}
    return out


def aggregate(res_dir):
    rows = [json.load(open(os.path.join(res_dir, f), encoding="utf-8")) for f in sorted(os.listdir(res_dir)) if f.endswith(".json") and f.startswith("run_")]
    if not rows: sys.exit("No hay corridas en " + res_dir)
    get = {"AUC prueba (red)": lambda r: r["auc_test_red"], "AUC u0 (prueba)": lambda r: r["spca"]["auc_u0"]["test"],
           "AUC u0 (entren.)": lambda r: r["spca"]["auc_u0"]["train"], "AUC logistica u0": lambda r: r["spca"]["logistica_u0"]["auc_test"],
           "AUC logistica 512 (C=0,01)": lambda r: r["spca"]["logistica_512_C0.01"]["auc_test"],
           "Sens. u0 (Youden val)": lambda r: r["spca"]["u0_youden_val"]["sens"], "Espec. u0 (Youden val)": lambda r: r["spca"]["u0_youden_val"]["spec"]}
    if all("kan" in r["spca"] for r in rows): get["AUC formula KAN"] = lambda r: r["spca"]["kan"]["auc_D_formula"]
    out = {}
    for m in ("base", "robusto"):
        rs = [r for r in rows if r["modelo"] == m]
        if not rs: continue
        out[m] = {"semillas": [r["semilla"] for r in rs], **{k: {"media": round(float(np.mean([f(r) for r in rs])), 4), "de": round(float(np.std([f(r) for r in rs], ddof=1)) if len(rs) > 1 else 0, 4),
                  "min": round(float(min(f(r) for r in rs)), 4), "max": round(float(max(f(r) for r in rs)), 4)} for k, f in get.items()}}
    json.dump(out, open(os.path.join(res_dir, "resumen_multisemilla.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT); ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--models", nargs="+", default=["base", "robusto"], choices=["base", "robusto"])
    ap.add_argument("--kan", action="store_true"); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    cod = a.root; res = os.path.join(a.root, "reports", "auditoria" + ("_smoke" if a.smoke else ""), "multisemilla")
    os.makedirs(res, exist_ok=True)
    if a.aggregate: return aggregate(res)
    print(f"Dispositivo: {DEVICE} | torch {torch.__version__}" + (f" | {torch.cuda.get_device_name(0)}" if DEVICE == "cuda" else ""))
    for seed in a.seeds:
        for mname in a.models:
            tag = f"run_{mname}_s{seed}"; fj = os.path.join(res, tag + ".json")
            if os.path.exists(fj): print("ya existe", tag); continue
            t0 = time.time(); set_seed(seed); L, dtr = loaders(cod, seed, a.smoke, drop_last=(mname == "base"))
            if mname == "base": model, bv, be, hist = run_base(L, dtr, seed, 2 if a.smoke else 20)
            else: model, bv, be, hist = run_robust(L, dtr, seed, 1 if a.smoke else 5, 1 if a.smoke else 25)
            pv, yv = predict(model, L["val"]); pt, yt_ = predict(model, L["test"])
            th = youden(yv, pv); se, sp = op(yt_, pt, th)
            E, Y = {}, {}
            for k, key in (("train", "train_eval"), ("val", "val"), ("test", "test")): E[k], Y[k] = predict(model, L[key], embed=True)
            np.save(os.path.join(res, tag + "_test_probs.npy"), pt); np.save(os.path.join(res, tag + "_val_probs.npy"), pv)
            np.save(os.path.join(res, tag + "_test_labels.npy"), yt_)
            for k in E: np.save(os.path.join(res, f"{tag}_{k}_feats.npy"), E[k].astype(np.float32))
            torch.save({"model_state_dict": model.state_dict(), "seed": seed, "best_epoch": be}, os.path.join(res, tag + ".pt"))
            set_seed(seed); sp_res = spca_study(cod, E, Y, seed, a.kan)
            out = {"modelo": mname, "semilla": seed, "mejor_epoca": be, "mejor_auc_val": round(float(bv), 4), "auc_test_red": round(float(roc_auc_score(yt_, pt)), 4),
                   "umbral_youden_val": round(th, 4), "sens_test": round(se, 4), "spec_test": round(sp, 4), "spca": sp_res, "segundos": round(time.time() - t0, 1), "smoke": a.smoke}
            json.dump(out, open(fj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print(f"== {tag}: AUC red {out['auc_test_red']} | u0 {sp_res['auc_u0']['test']} | logistica u0 {sp_res['logistica_u0']['auc_test']} | {out['segundos']} s", flush=True)
    aggregate(res)


if __name__ == "__main__":
    main()
