"""
f2_symb: harness reutilizable del Frente 2 (Etapa B - cabeza simbolica).

Entrena una cabeza KAN[64,32,2] sobre features 64-D (de cualquier generador:
linear, KAN0, PCA, autoencoder, MLP supervisado...), poda por fidelidad, fija
funciones simbolicas y evalua la formula aislada con el mapping CORREGIDO de
pykan (`x_m -> col m-1` de X[:, kept]).

Factoriza la logica que N19/N20 tenian embebida, para que N21+ importe y use
funciones sin duplicar codigo en cada notebook.
"""
import io
import re
import json
import contextlib
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, roc_curve
import sympy as sp  # noqa: F401  (necesario para el evaluador)


# ---------------------------------------------------------------------------
# Normalizacion y metricas
# ---------------------------------------------------------------------------
def softmax2(L):
    e = np.exp(L - np.max(L, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def load_zscore(prefix, emb_dir):
    """Carga features '{prefix}_{split}_feats.npy' + labels y z-score con stats de train."""
    feats, labs = {}, {}
    for split in ("train", "val", "test"):
        f = os.path.join(emb_dir, f"{prefix}_{split}_feats.npy")
        l = os.path.join(emb_dir, f"{prefix}_{split}_labels.npy")
        feats[split] = np.load(f).astype(np.float32)
        labs[split] = np.load(l).astype(int)
        assert feats[split].shape[0] == labs[split].shape[0], f"{prefix}/{split} no alineado"
        assert np.isfinite(feats[split]).all(), f"{prefix}/{split} NaN/inf"
    mu = feats["train"].mean(0, keepdims=True)
    sd = feats["train"].std(0, keepdims=True) + 1e-8
    for split in ("train", "val", "test"):
        feats[split] = (feats[split] - mu) / sd
    return feats, labs


def auc_from_prob(y, p):
    return float(roc_auc_score(y, p))


def youden_threshold(scores, labels):
    fpr, tpr, thr = roc_curve(labels, scores)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thr[idx]), float(tpr[idx]), float(fpr[idx]), float(j[idx])


# ---------------------------------------------------------------------------
# Mascaras / sparsificacion / importancia
# ---------------------------------------------------------------------------
def count_masked(kan_model):
    n = 0
    for l in range(len(kan_model.act_fun)):
        n += int((kan_model.act_fun[l].mask.data > 0).sum().item())
    return n


def compute_importance(kan_model, x):
    kan_model.eval()
    with torch.no_grad():
        kan_model(x)
    imp = []
    for l in range(len(kan_model.spline_postacts)):
        pa = kan_model.spline_postacts[l]
        if pa is not None:
            imp.append(pa.abs().mean(dim=0).cpu().numpy())
    return imp


def apply_mask(kan_model, x, keep_frac):
    imp = compute_importance(kan_model, x)
    for l in range(len(kan_model.act_fun)):
        imp_l = imp[l]
        flat = imp_l.flatten()
        kk = max(1, int(keep_frac * flat.size))
        thr = np.sort(flat)[::-1][kk - 1]
        mask = (imp_l >= thr).T
        kan_model.act_fun[l].mask.data = torch.from_numpy(
            mask.astype(np.float32)).to(kan_model.act_fun[l].mask.device)


# ---------------------------------------------------------------------------
# Cabeza simbolica
# ---------------------------------------------------------------------------
class SymbHead:
    """Cabeza KAN entrenada con fit + sparsificacion por mascara.

    Attr:
        kan: modulo pykan (envuelto por KANHead).
    """

    DEF_LIB = ["x", "x^2", "sin", "cos", "tanh", "exp", "log"]

    def __init__(self, in_dim, hidden=32, grid=3, k=3, seed=42, device="cpu", lib=None):
        from src.models import KANHead
        self.in_dim = in_dim
        self.hidden = hidden
        self.grid = grid
        self.k = k
        self.seed = seed
        self.device = device
        self.lib = list(lib if lib is not None else self.DEF_LIB)
        head = KANHead(width=[in_dim, hidden, 2], grid=grid, k=k,
                       seed=seed, device=device).to(device)
        self.kan = head.kan  # pykan MultKAN directo (act_fun, prune_input, symbolic, ...)

    # ---- forward helpers --------------------------------------------------
    def prob(self, x):
        self.kan.eval()
        with torch.no_grad():
            L = self.kan(x).cpu().numpy()
        return softmax2(L)[:, 1]

    def auc(self, x, y):
        y = y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
        return float(roc_auc_score(y, self.prob(x)))

    # ---- fit + sparsify ---------------------------------------------------
    def fit_and_sparsify(self, x_tr, y_tr, x_val, y_val, x_te, y_te,
                         fit_ep=10, sparse_ep=30, batch=512,
                         lr=1e-3, wd=1e-4, class_weights=None):
        """Dos fases: fit (sin mascara) + sparsificacion iterativa por mascara.

        Devuelve (auc_val, auc_test) finales.
        """
        tr_len = len(x_tr)
        crit = nn.CrossEntropyLoss(weight=class_weights)
        opt = torch.optim.Adam(self.kan.parameters(), lr=lr, weight_decay=wd)

        for phase, epochs, do_sparse in (("fit", fit_ep, False),
                                         ("sparsify", sparse_ep, True)):
            for ep in range(1, epochs + 1):
                self.kan.train()
                perm = torch.randperm(tr_len, device=self.device)
                for i in range(0, tr_len, batch):
                    idx = perm[i:i + batch]
                    opt.zero_grad()
                    loss = crit(self.kan(x_tr[idx]), y_tr[idx])
                    if torch.isfinite(loss):
                        loss.backward()
                        opt.step()
                if do_sparse and ep % 5 == 0:
                    keep = max(0.05, 1.0 - ep * 0.03)
                    apply_mask(self.kan, x_val, keep)
                if ep % 5 == 0:
                    print(f"  [{phase}] ep {ep:02d}: AUC test={self.auc(x_te, y_te):.4f} "
                          f"vivas={count_masked(self.kan)}")

        auc_val = self.auc(x_val, y_val)
        auc_test = self.auc(x_te, y_te)
        return auc_val, auc_test

    # ---- poda por fidelidad ----------------------------------------------
    def prune_sweep(self, x_val, x_te, y_val, y_te, auc_ref,
                    dims=(48, 32, 24, 16), r_min=0.95, dA_max=0.01):
        """Barrido de D' con criterio de fidelidad (r>=r_min y dAUC<=dA_max).

        Devuelve (D_sweet, kept_dims, tabla).
        """
        self.kan.eval()
        with torch.no_grad():
            self.kan(x_val)
        post = self.kan.spline_postacts[0].abs().mean(dim=0)
        scores = post.sum(dim=0).cpu().numpy()
        order = np.argsort(scores)[::-1]

        with torch.no_grad():
            Lv_full = self.kan(x_val).cpu().numpy()[:, 1]
        yv = y_val.cpu().numpy() if torch.is_tensor(y_val) else np.asarray(y_val)
        yt = y_te.cpu().numpy() if torch.is_tensor(y_te) else np.asarray(y_te)

        D_sweet = None
        tabla = []
        for Dp in dims:
            kept = sorted(order[:Dp].tolist())
            kp = self.kan.prune_input(active_inputs=kept, log_history=False)
            kp.eval()
            with torch.no_grad():
                Lv_kp = kp(x_val).cpu().numpy()[:, 1]
                Lte_kp = kp(x_te).cpu().numpy()[:, 1]
            r = float(pearsonr(Lv_full, Lv_kp)[0])
            a_val = float(roc_auc_score(yv, Lv_kp))
            a_test = float(roc_auc_score(yt, Lte_kp))
            dA = auc_ref - a_val
            tabla.append({"D": Dp, "r": round(r, 4), "dAUC_val": round(dA, 4),
                          "auc_val": round(a_val, 4), "auc_test": round(a_test, 4)})
            if r >= r_min and dA <= dA_max and D_sweet is None:
                D_sweet = Dp
        if D_sweet is None:
            D_sweet = dims[-1]
        kept_dims = sorted(order[:D_sweet].tolist())
        return D_sweet, [int(d) for d in kept_dims], tabla

    # ---- fix simbolico ----------------------------------------------------
    @staticmethod
    def _quei(fn):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn()

    def fix_layer1(self, kp):
        n_hidden = kp.width[1][0] if isinstance(kp.width[1], list) else kp.width[1]
        n_out = kp.width[-1][0] if isinstance(kp.width[-1], list) else kp.width[-1]
        fixed = 0
        for jj in range(n_hidden):
            for o in range(n_out):
                try:
                    best = self._quei(lambda: kp.suggest_symbolic(1, jj, o, lib=self.lib, verbose=False))
                    fn = best[0] if isinstance(best, tuple) else best
                    self._quei(lambda: kp.fix_symbolic(1, jj, o, fn, verbose=False, log_history=False))
                    fixed += 1
                except Exception:
                    pass
        return fixed

    def fix_layer0(self, kp, top_k=None, x_val=None):
        if x_val is not None:
            kp.eval()
            with torch.no_grad():
                kp(x_val)
        post_s = kp.spline_postacts[0].abs().mean(dim=0)
        imp_s = post_s.cpu().numpy()
        edges = sorted([(float(imp_s[j, i]), i, j)
                        for i in range(imp_s.shape[1]) for j in range(imp_s.shape[0])], reverse=True)
        if top_k is not None:
            edges = edges[:top_k]
        fixed = 0
        for a, i, j in edges:
            try:
                best = self._quei(lambda: kp.suggest_symbolic(0, i, j, lib=self.lib, verbose=False))
                fn = best[0] if isinstance(best, tuple) else best
                self._quei(lambda: kp.fix_symbolic(0, i, j, fn, verbose=False, log_history=False))
                fixed += 1
            except Exception:
                pass
        return fixed

    def build_pruned(self, kept_dims):
        kp = self.kan.prune_input(active_inputs=kept_dims, log_history=False)
        kp.eval()
        return kp

    # ---- formulas ---------------------------------------------------------
    @staticmethod
    def make_formula(kp):
        kp.symbolic_enabled = True
        outputs, _ = kp.symbolic_formula()
        kp.symbolic_enabled = False
        return str(outputs[0]), str(outputs[1])

    @staticmethod
    def leg(s):
        return {"kb": round(len(s) / 1024, 1), "sin": s.count("sin("), "cos": s.count("cos("),
                "n_vars": len(set(re.findall(r"x_(\d+)", s)))}

    # ---- evaluacion C y D -------------------------------------------------
    @staticmethod
    def config_C(kp, x_val, x_te, y_val, y_te, Lv_ref):
        kp.symbolic_enabled = True
        with torch.no_grad():
            Lv_C = kp(x_val).cpu().numpy()[:, 1]
            Lte_C = kp(x_te).cpu().numpy()[:, 1]
        kp.symbolic_enabled = False
        yv = y_val.cpu().numpy() if torch.is_tensor(y_val) else np.asarray(y_val)
        yt = y_te.cpu().numpy() if torch.is_tensor(y_te) else np.asarray(y_te)
        a_val = float(roc_auc_score(yv, Lv_C))
        a_test = float(roc_auc_score(yt, Lte_C))
        r = float(pearsonr(Lv_ref[:, 1], Lv_C)[0])
        return a_val, a_test, r

    @staticmethod
    def eval_formula_vec(formula, X_kept, clip_val=10.0):
        """Evalua la formula en coordenadas podadas CORREGIDAS.

        pykan numera variables 1-based en coordenadas podadas: x_m -> columna m-1.
        """
        expr = sp.sympify(formula)
        used = sorted({int(m) for m in re.findall(r"x_(\d+)", formula)})
        syms = [sp.Symbol("x_%d" % i) for i in used]
        fn = sp.lambdify(syms, expr, "numpy")
        if clip_val is not None:
            vals = [np.clip(X_kept[:, i - 1], -clip_val, clip_val) for i in used]
        else:
            vals = [X_kept[:, i - 1] for i in used]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            out = fn(*vals) if vals else np.full(X_kept.shape[0], float(expr))
        return np.asarray(out, dtype=np.float64).reshape(-1)

    @staticmethod
    def formula_to_latex(formula):
        """Convierte la formula algebraica de pykan a expresion LaTeX limpia."""
        try:
            expr = sp.sympify(formula)
            return sp.latex(expr)
        except Exception:
            return formula

    @staticmethod
    def report_D(fben, fmal, Xv_kept, Xt_kept, yv64, yb64, clip_val=10.0):
        sv = SymbHead.eval_formula_vec(fmal, Xv_kept, clip_val=clip_val) - SymbHead.eval_formula_vec(fben, Xv_kept, clip_val=clip_val)
        st = SymbHead.eval_formula_vec(fmal, Xt_kept, clip_val=clip_val) - SymbHead.eval_formula_vec(fben, Xt_kept, clip_val=clip_val)
        auc = float(roc_auc_score(yb64, st))
        th, _, _, j = youden_threshold(sv, yv64)
        rec = float((st >= th)[yb64 == 1].mean())
        spec = float((st < th)[yb64 == 0].mean())
        return {"auc_D": round(auc, 4), "th_youden_val": round(th, 4),
                "J_youden_val": round(j, 4), "recall_D_youden": round(rec, 4),
                "spec_D_youden": round(spec, 4)}

    @staticmethod
    def smoke_test(auc_D, auc_C, tol=0.01):
        diff = abs(auc_D - auc_C)
        ok = diff <= tol
        print(f"  |AUC_D({auc_D:.4f}) - AUC_C({auc_C:.4f})| = {diff:.4f} -> {'OK' if ok else 'FALLO'}")
        return ok, diff