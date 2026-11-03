"""
Monotonia de la formula canonica en el rango de u0 de las tres particiones (Cap. 4, Sec. 4.2).

r3_monotonicity.json demuestra g'(u0) < 0 en el rango de prueba. Este script repite el argumento
en el rango que cubren entrenamiento, validacion y prueba juntos, que incluye los datos con que se
fijo el umbral de Youden. Pasos:
  1. Calcula u0 en las tres particiones con el reductor congelado (kan_robust64_reducer.npz).
  2. Lee g = f_mal - f_ben de kan_robust64_canonical.json y obtiene la cota global de la segunda
     derivada, |g''| <= sum_m |c_m| w_m^2, sumando sobre los terminos c*sin(w*x + b) y c*cos(w*x + b)
     (los terminos lineales no aportan).
  3. Evalua g' con paso complejo (exacto a precision de maquina) en una malla de 10 000 puntos.
     Entre dos nodos a distancia h, g' no puede superar el maximo de la malla en mas de cota*h/2.

Uso (desde la raiz del repositorio):
  python scripts/monotonia_rango_completo.py [SALIDA.json]
Los embeddings se buscan en $TESIS_EMB_ROBUSTO (por defecto artifacts/output/embeddings).
Requiere numpy.
"""
import json, os, re, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "reports", "auditoria", "monotonia_rango_completo.json")
EMB = os.environ.get("TESIS_EMB_ROBUSTO") or os.path.join(REPO, "artifacts", "output", "embeddings")
N = 10000

can = json.load(open(os.path.join(REPO, "reports", "experiments", "kan_robust64_canonical.json"), encoding="utf-8"))
nz = np.load(os.path.join(REPO, "reports", "experiments", "kan_robust64_reducer.npz"))


def u0(X):
    gz = (X - nz["gap_mu"]) / nz["gap_sd"]
    z = (((gz @ nz["proj_w"].T + nz["proj_b"]) - nz["pca_mean"]) @ nz["pca_components"].T - nz["mu_z"]) / nz["sd_z"]
    return z[:, 0]


def g(x):
    env = {"sin": np.sin, "cos": np.cos, "x_1": x}
    return eval(can["fmal"], env) - eval(can["fben"], env)


def gprima(x):
    return g(x.astype(complex) + 1e-20j).imag / 1e-20


num = r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?"
patron = re.compile(rf"({num})\*(?:sin|cos)\(({num})\*x_1")
cota_g2 = 0.0
for expr in (can["fmal"], can["fben"]):
    for c, w in patron.findall(expr.replace(" ", "")):
        cota_g2 += abs(float(c)) * float(w) ** 2

rangos = {}
for s in ("train", "val", "test"):
    u = u0(np.load(os.path.join(EMB, f"f2robust_{s}_feats.npy")).astype(np.float64))
    rangos[s] = [round(float(u.min()), 4), round(float(u.max()), 4)]

res = {"cota_segunda_derivada": round(cota_g2, 4), "rangos_u0": rangos, "malla_puntos": N, "intervalos": {}}
a_all = min(r[0] for r in rangos.values()); b_all = max(r[1] for r in rangos.values())
for nombre, (a, b) in {"prueba": rangos["test"], "tres_particiones": [a_all, b_all]}.items():
    t = np.linspace(a, b, N); v = gprima(t); h = t[1] - t[0]
    res["intervalos"][nombre] = {"a": a, "b": b, "h": h, "gprima_min_malla": round(float(v.min()), 5),
                                 "gprima_max_malla": round(float(v.max()), 5),
                                 "cota_superior_gprima": round(float(v.max() + cota_g2 * h / 2), 5),
                                 "estrictamente_decreciente": bool(v.max() + cota_g2 * h / 2 < 0)}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(res, indent=2))
