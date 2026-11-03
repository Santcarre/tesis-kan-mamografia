"""
Pruebas que no necesitan PyTorch, FastAPI ni los datos de CBIS-DDSM.

Comprueban tres cosas:
  1. la regla de decisión de la estación (decision.py);
  2. la fórmula canónica y su evaluador de lista blanca (src/reporting/formula_eval.py);
  3. que la interfaz web y sus datos (`desktop_app/ui/data/*.json`) son coherentes con la fórmula.

Ejecutar desde la raíz del repositorio:  python -m pytest desktop_app/tests/test_sin_torch.py -v
"""
import json
import math
import os
import re
from pathlib import Path

import pytest

from desktop_app.backend.services.decision import (
    apply_formula_decision,
    birads_from_p,
    birads_int,
)
from src.reporting.formula_eval import compile_expr, g_function, sigmoid

RAIZ = Path(__file__).resolve().parents[2]
UI = RAIZ / "desktop_app" / "ui"
DATOS = UI / "data"


def _raiz_codigos() -> Path:
    """Carpeta con src/ y reports/, con el mismo orden que backend/config.py: la variable
    TESIS_CODIGOS, la raíz del repositorio y, en el árbol de trabajo original, la carpeta Códigos."""
    env = os.environ.get("TESIS_CODIGOS")
    for c in (Path(env) if env else None, RAIZ, RAIZ / "Códigos"):
        if c is not None and (c / "reports" / "experiments").is_dir():
            return c
    return RAIZ


CODIGOS = _raiz_codigos()
CANON = json.loads((CODIGOS / "reports" / "experiments" / "kan_robust64_canonical.json").read_text(encoding="utf-8"))
TAU = float(CANON["th_youden_val"])


def _cargar(nombre):
    return json.loads((DATOS / nombre).read_text(encoding="utf-8"))


# ───────────────────────── 1. Decisión ─────────────────────────

# Cortes de src/reporting/birads_herramienta.py: 2 | 0,3572 | 3 | sigma(tau*) = 0,5324 | 4 | 0,6761 | 5
@pytest.mark.parametrize("p,esperado", [(0.10, 2), (0.357, 2), (0.358, 3), (0.53, 3), (0.533, 4), (0.60, 4), (0.676, 4), (0.677, 5), (0.95, 5)])
def test_birads_cortes(p, esperado):
    assert birads_int(birads_from_p(p)) == esperado


def test_birads_sin_subcategorias():
    # CBIS-DDSM no subdivide la categoría 4, así que la herramienta tampoco
    for p in (0.45, 0.55, 0.70):
        assert re.search(r"4[ABC]", birads_from_p(p)) is None


@pytest.mark.parametrize("g", [x / 100 for x in range(-300, 301, 7)] + [TAU, TAU - 1e-9])
def test_birads_coherente_con_el_veredicto(g):
    ev = apply_formula_decision({"kan64_g": g, "kan64_th": TAU, "prob_cnn": 0.5})
    cat = birads_int(ev["birads_suggested"])
    assert (cat >= 4) == (ev["decision"] == "MALIGNANT")


def test_decide_la_formula_no_la_red():
    ev = {"kan64_g": 1.0, "kan64_th": TAU, "kan64_z64_0": -1.0, "prob_cnn": 0.01}
    apply_formula_decision(ev)
    assert ev["decision_source"] == "formula"
    assert ev["decision"] == "MALIGNANT"          # la red diría benigna (1%), pero g >= tau*
    assert ev["decision_u0"] == -1.0
    assert ev["prob_net_robust"] == 0.01


def test_umbral_es_inclusivo():
    ev = {"kan64_g": TAU, "kan64_th": TAU, "prob_cnn": 0.0}
    assert apply_formula_decision(ev)["decision"] == "MALIGNANT"
    ev = {"kan64_g": TAU - 1e-6, "kan64_th": TAU, "prob_cnn": 1.0}
    assert apply_formula_decision(ev)["decision"] == "BENIGN"


def test_justo_bajo_el_umbral_es_categoria_3():
    # Antes la convención daba 4B con veredicto benigno; ahora la categoría sigue al veredicto.
    ev = apply_formula_decision({"kan64_g": 0.12, "kan64_th": TAU, "prob_cnn": 0.3})
    assert ev["decision"] == "BENIGN"
    assert birads_int(ev["birads_suggested"]) == 3


def test_respaldo_declarado_si_faltan_artefactos():
    ev = apply_formula_decision({"prob_cnn": 0.8})
    assert ev["decision_source"] == "cnn_fallback"
    assert ev["decision"] == "MALIGNANT" and ev["decision_u0"] is None


def test_decision_idempotente():
    ev = {"kan64_g": -0.3, "kan64_th": TAU, "kan64_z64_0": 0.5, "prob_cnn": 0.4}
    a = dict(apply_formula_decision(ev))
    b = dict(apply_formula_decision(ev))
    assert a == b


# ───────────────────────── 2. Fórmula ─────────────────────────

def _g():
    g, tau = g_function(str(CODIGOS))
    return g, tau


def test_formula_carga_y_umbral():
    g, tau = _g()
    assert tau == pytest.approx(TAU)
    assert sigmoid(tau) == pytest.approx(0.5324, abs=5e-4)


def test_formula_estrictamente_decreciente_y_pendiente():
    g, _ = _g()
    xs = [-6 + 12 * i / 2400 for i in range(2401)]
    ys = [g(x) for x in xs]
    assert all(b < a for a, b in zip(ys, ys[1:])), "g debería decrecer en [-6, 6]"
    h = xs[1] - xs[0]
    pend = max(abs((b - a) / h) for a, b in zip(ys, ys[1:]))
    assert pend == pytest.approx(0.8798, abs=0.01)


def test_u0_umbral_equivalente():
    g, tau = _g()
    lo, hi = -2.0, 2.0
    for _ in range(80):                           # g decrece: g(u) >= tau  <=>  u <= u*
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if g(mid) >= tau else (lo, mid)
    assert lo == pytest.approx(-0.0331, abs=2e-3)


@pytest.mark.parametrize("mal", [
    "__import__('os').system('x')", "x_1.real", "open('f')", "[1][0]", "x_1 if 1 else 2",
    "lambda: 1", "(1).__class__", "foo(x_1)", "sin(x_1, 2)", "sin(x=1)",
])
def test_evaluador_rechaza_lo_no_permitido(mal):
    with pytest.raises((ValueError, SyntaxError)):
        compile_expr(mal)(0.0)


def test_evaluador_aritmetica_basica():
    f = compile_expr("2*cos(0.5*x_1 + 1) - x_1^2 + 3")
    assert f(0.7) == pytest.approx(2 * math.cos(0.5 * 0.7 + 1) - 0.49 + 3)


# ───────────────────────── 3. Interfaz y datos ─────────────────────────

def test_index_referencias_existen():
    html = (UI / "index.html").read_text(encoding="utf-8")
    refs = re.findall(r'(?:href|src)="([^"#]+)"', html)
    locales = [r for r in refs if not re.match(r"^(https?:)?//", r)]
    assert locales, "el index no referencia nada"
    for r in locales:
        assert (UI / r).exists(), f"index.html referencia {r}, que no existe"
    assert "Prometeo" in html


def test_modulos_js_resuelven():
    for js in (UI / "js").glob("*.js"):
        for ruta in re.findall(r"""(?:from\s+|import\()\s*['"](\.[^'"]+)['"]""", js.read_text(encoding="utf-8")):
            assert (js.parent / ruta).resolve().exists(), f"{js.name} importa {ruta}, que no existe"


def test_rutas_de_la_interfaz_son_relativas():
    # La interfaz se sirve en /ui/: una ruta absoluta ("/data/...") se rompería.
    for f in list(UI.glob("*.html")) + list((UI / "js").glob("*.js")) + list((UI / "css").glob("*.css")):
        txt = f.read_text(encoding="utf-8")
        assert not re.search(r"""fetch\(\s*['"`]/(?!api)""", txt), f"{f.name}: fetch con ruta absoluta"
        assert not re.search(r"""(?:href|src)=["']/(?!/)""", txt), f"{f.name}: href/src absoluto"


def test_terminos_de_tesis_json_igualan_la_formula():
    g, _ = _g()
    t = _cargar("tesis.json")["formula"]

    def lado(terminos, x):
        s = 0.0
        for k in terminos:
            if k["k"] == "lin":
                s += k["a"] * x
            elif k["k"] == "const":
                s += k["a"]
            elif k["k"] == "sin":
                s += k["a"] * math.sin(k["w"] * x + k["p"])
            elif k["k"] == "cos":
                s += k["a"] * math.cos(k["w"] * x + k["p"])
            else:
                raise AssertionError(k)
        return s

    for x in (-3.0, -1.2, -0.0331, 0.0, 0.55, 2.4):
        assert lado(t["f_mal"], x) - lado(t["f_ben"], x) == pytest.approx(g(x), abs=1e-9)


def test_pacientes_del_laboratorio_son_coherentes():
    g, tau = _g()
    sel = {p["id"]: p for p in _cargar("lab_seleccion.json")["pacientes"]}
    pac = _cargar("lab_pacientes.json")["pacientes"]
    assert len(pac) == len(sel) == 9
    for p in pac:
        assert p["id"] in sel and p["idx"] == sel[p["id"]]["idx"] and p["paciente"] == sel[p["id"]]["paciente"]
        assert len(p["gap_z"]) == 512 and len(p["z64"]) == 64
        assert p["u0"] == pytest.approx(p["z64"][0], abs=1e-3)
        assert p["g"] == pytest.approx(g(p["u0"]), abs=2e-3)
        assert p["p_sym"] == pytest.approx(sigmoide(p["g"]), abs=2e-3)
        assert p["veredicto"] == ("MALIGNANT" if p["g"] >= tau else "BENIGN")
        assert p["tau_logit"] == pytest.approx(tau, abs=1e-4)
        assert p["birads_herramienta"] == str(birads_from_p(p["p_sym"])).split()[1]
        etiqueta_real = sel[p["id"]]["patologia"]
        assert p["acierto_formula"] == (p["veredicto"] == etiqueta_real)


def sigmoide(x):
    return sigmoid(x)


def test_poblacion_y_matriz_de_confusion():
    pob = _cargar("tesis.json")["poblacion"]
    c = pob["confusion"]
    assert c["tp"] + c["fn"] == pob["n_mal"] == 418
    assert c["tn"] + c["fp"] == pob["n_ben"] == 594
    assert pob["n"] == pob["n_mal"] + pob["n_ben"] == 1012
    assert pob["sens"] == pytest.approx(c["tp"] / (c["tp"] + c["fn"]), abs=1e-4)
    assert pob["esp"] == pytest.approx(c["tn"] / (c["tn"] + c["fp"]), abs=1e-4)


def test_lab_global_alineado():
    gl = _cargar("lab_global.json")
    assert gl["n"] == len(gl["u0"]) == len(gl["y"]) == 1012
    assert sum(gl["y"]) == 418
    g, tau = _g()
    # La matriz de confusión de la tesis es la de aplicar g >= tau* a estas u0.
    tp = sum(1 for u, y in zip(gl["u0"], gl["y"]) if y == 1 and g(u) >= tau)
    fp = sum(1 for u, y in zip(gl["u0"], gl["y"]) if y == 0 and g(u) >= tau)
    c = _cargar("tesis.json")["poblacion"]["confusion"]
    assert (tp, fp) == (c["tp"], c["fp"])
