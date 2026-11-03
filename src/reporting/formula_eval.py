"""
formula_eval.py: lectura y evaluación de la fórmula canónica g(u0) = f_mal(u0) - f_ben(u0).

Módulo puro (solo biblioteca estándar). Lee `reports/experiments/kan_robust64_canonical.json`
y evalúa la expresión con un intérprete aritmético de lista blanca: no usa eval() ni sympy.
Lo comparten el backend de la estación y el generador de PDF.
"""
import ast
import json
import math
import os
from typing import Callable, Optional, Tuple

_FUNCS = {"sin": math.sin, "cos": math.cos, "exp": math.exp, "tanh": math.tanh, "log": math.log,
          "sqrt": math.sqrt, "abs": abs}


def compile_expr(src: str) -> Callable[[float], float]:
    """Compila una expresión en la variable x_1 aceptando solo la aritmética de pykan."""
    tree = ast.parse(src.replace("^", "**"), mode="eval")

    def ev(n, x: float) -> float:
        if isinstance(n, ast.Expression):
            return ev(n.body, x)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name) and n.id == "x_1":
            return x
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand, x)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left, x), ev(n.right, x)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.Div):
                return a / b
            if isinstance(n.op, ast.Pow):
                return a ** b
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS
                and len(n.args) == 1 and not n.keywords):
            return _FUNCS[n.func.id](ev(n.args[0], x))
        raise ValueError("Nodo no permitido en la fórmula: " + ast.dump(n)[:80])

    return lambda x: ev(tree, float(x))


def sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def load_canonical(codigos_dir: Optional[str] = None) -> Optional[dict]:
    """Devuelve el JSON canónico (fben, fmal, th_youden_val, ...) o None si no existe."""
    base = codigos_dir or os.environ.get("TESIS_CODIGOS") or os.getcwd()
    path = os.path.join(base, "reports", "experiments", "kan_robust64_canonical.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def g_function(codigos_dir: Optional[str] = None) -> Optional[Tuple[Callable[[float], float], float]]:
    """(g, tau_logit) con g(u0) = f_mal(u0) - f_ben(u0); None si falta el artefacto."""
    data = load_canonical(codigos_dir)
    if not data or "fben" not in data or "fmal" not in data:
        return None
    f_ben, f_mal = compile_expr(data["fben"]), compile_expr(data["fmal"])
    return (lambda u: f_mal(u) - f_ben(u)), float(data.get("th_youden_val", 0.0))
