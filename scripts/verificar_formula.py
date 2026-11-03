# -*- coding: utf-8 -*-
"""
verificar_formula.py -- Verificacion independiente de la formula canonica g(u0) (Cap. 4; Anexo A)

Recalcula, a partir de los artefactos del repositorio y SIN checkpoint, imagenes ni sympy:
  1. u0 de prueba = primera coordenada canonica de los embeddings GAP (reductor kan_robust64_reducer.npz).
  2. g(u0) = f_mal - f_ben, evaluando la expresion de kan_robust64_canonical.json con un evaluador
     aritmetico de lista blanca (no se usa eval()).
  3. AUC de g y de u0 en prueba (Mann-Whitney) y su relacion: AUC(g) = 1 - AUC(u0) si g es decreciente.
  4. g'(u0) por diferencias centrales en una malla fina del soporte observado y en los casos de prueba:
     signo, minimo y maximo, y supremo de |g'|.
  5. Numero de caracteres de la expresion y de variables.
Compara con reports/experiments/r3_monotonicity.json y escribe reports/auditoria/verificacion_formula.json.

Uso (desde la raiz del repositorio):  python scripts/verificar_formula.py [RAIZ] [SALIDA.json]
Requiere solo numpy.
"""
import ast, json, math, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = sys.argv[1] if len(sys.argv) > 1 else REPO
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "reports", "auditoria", "verificacion_formula.json")
EXP = os.path.join(ROOT, "reports", "experiments")
EMB = os.path.join(ROOT, "artifacts", "output", "embeddings", "f2robust_")

FUNCS = {"sin": np.sin, "cos": np.cos, "exp": np.exp, "tanh": np.tanh, "log": np.log, "sqrt": np.sqrt, "abs": np.abs}
BINOPS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}


def compile_expr(src):
    """Devuelve f(x) para una expresion en la variable x_1, aceptando solo la gramatica aritmetica de pykan."""
    tree = ast.parse(src.replace("^", "**"), mode="eval")

    def ev(n, x):
        if isinstance(n, ast.Expression):
            return ev(n.body, x)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.Name) and n.id == "x_1":
            return x
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand, x)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and type(n.op) in BINOPS:
            return BINOPS[type(n.op)](ev(n.left, x), ev(n.right, x))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCS and len(n.args) == 1 and not n.keywords:
            return FUNCS[n.func.id](ev(n.args[0], x))
        raise ValueError("nodo no permitido en la formula: " + ast.dump(n)[:80])

    return lambda x: ev(tree, np.asarray(x, dtype=np.float64))


def auc(y, s):
    """AUC de Mann-Whitney con rangos medios (puntuacion alta = clase 1)."""
    s = np.asarray(s, dtype=np.float64)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    ss = s[order]; i = 0
    while i < len(ss):                      # rangos medios para empates
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n1 = int((y == 1).sum()); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


cj = json.load(open(os.path.join(EXP, "kan_robust64_canonical.json"), encoding="utf-8"))
ref = json.load(open(os.path.join(EXP, "r3_monotonicity.json"), encoding="utf-8"))
R = np.load(os.path.join(EXP, "kan_robust64_reducer.npz"))
Xte = np.load(EMB + "test_feats.npy").astype(np.float64)
yte = np.load(EMB + "test_labels.npy").astype(int)


def u0_of(X):
    g = (X - R["gap_mu"]) / R["gap_sd"]
    z = g @ R["proj_w"].T + R["proj_b"]
    z = ((z - R["pca_mean"]) @ R["pca_components"].T - R["mu_z"]) / R["sd_z"]
    return z[:, 0]


u0 = u0_of(Xte)
f_mal, f_ben = compile_expr(cj["fmal"]), compile_expr(cj["fben"])
g = lambda x: f_mal(x) - f_ben(x)


def dg(x, h=1e-5):                           # diferencias centrales (g es suave: sumas de sen/cos/lineal)
    x = np.asarray(x, dtype=np.float64)
    return (g(x + h) - g(x - h)) / (2 * h)


lo, hi = float(u0.min()), float(u0.max())
grid = np.linspace(lo, hi, 10000)
d_grid, d_test = dg(grid), dg(u0)
auc_g, auc_u0 = auc(yte, g(u0)), auc(yte, u0)

res = {
    "n_test": int(len(yte)), "n_mal": int((yte == 1).sum()), "n_ben": int((yte == 0).sum()),
    "u0_soporte": [round(lo, 4), round(hi, 4)],
    "auc_g": round(auc_g, 4), "auc_u0": round(auc_u0, 4), "auc_complemento": round(1 - auc_u0, 4),
    "gprima_min_malla": round(float(d_grid.min()), 4), "gprima_max_malla": round(float(d_grid.max()), 4),
    "gprima_min_prueba": round(float(d_test.min()), 4), "gprima_max_prueba": round(float(d_test.max()), 4),
    "estrictamente_decreciente": bool(d_grid.max() < 0),
    "sup_abs_gprima": round(float(np.abs(d_grid).max()), 4),
    "caracteres_reportados_en_json": cj.get("chars"),
    "variables_reportadas": cj.get("n_vars"),
}
tol = 5e-4
checks = {
    "u0_soporte": (ref["u0_support"], res["u0_soporte"], max(abs(a - b) for a, b in zip(ref["u0_support"], res["u0_soporte"])) < tol),
    "auc_g": (ref["auc_g_formula"], res["auc_g"], abs(ref["auc_g_formula"] - res["auc_g"]) < tol),
    "gprima_min_soporte": (ref["gprime_min_support"], res["gprima_min_malla"], abs(ref["gprime_min_support"] - res["gprima_min_malla"]) < tol),
    "gprima_max_soporte": (ref["gprime_max_support"], res["gprima_max_malla"], abs(ref["gprime_max_support"] - res["gprima_max_malla"]) < tol),
    "estrictamente_decreciente": (ref["is_strictly_decreasing"], res["estrictamente_decreciente"], ref["is_strictly_decreasing"] == res["estrictamente_decreciente"]),
}
res["comparacion_con_r3_monotonicity"] = {k: {"artefacto": a, "recalculado": b, "coincide": bool(c)} for k, (a, b, c) in checks.items()}
res["coincide_con_artefacto"] = bool(all(c for _, _, c in checks.values()))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(res, ensure_ascii=False, indent=2))
print("\nCOINCIDE CON r3_monotonicity.json:", res["coincide_con_artefacto"])
