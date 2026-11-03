#!/usr/bin/env python3
"""
generar_laboratorio.py: crea las imágenes del Laboratorio de Prometeo para los nueve
pacientes de `desktop_app/ui/data/lab_seleccion.json`.

Necesita lo mismo que la app local: PyTorch, los puntos de control del modelo
(`model/` o los activos de la versión del repositorio) y las mamografías de CBIS-DDSM.
Usa los mismos servicios que la app (CasesService y RenderService), así que lo que ves
en el Laboratorio es exactamente lo que calcula la estación.

Escribe, para cada paciente, en `desktop_app/ui/data/lab/<id>/`:
    original.png   mamografía a 224x224 (lo que ve la red)
    cam.png        mapa de la red (Grad-CAM)
    psikan.png     mapa de la fórmula (psi_KAN)
    lesion.png     la lesión real (máscara del conjunto) sobre la mamografía
    wsss.png, zoom.png   foco y lupa (opcionales; la interfaz no los usa por ahora)

Comprueba además que u0 y g de la estación coinciden con los de lab_pacientes.json
(calculados sin PyTorch por scripts/generar_datos_ui.py); si no coinciden, avisa.

Nota sobre las imágenes: CBIS-DDSM se distribuye con licencia CC BY 3.0 (TCIA). Si
publicas estos recortes, conserva la atribución (ver README). Si prefieres no
publicarlos, déjalos fuera del repositorio: la interfaz muestra un aviso en su lugar.

Uso (desde la raíz del repositorio):
    python scripts/generar_laboratorio.py [--casos mal-masa,ben-masa] [--salida RUTA]
"""
import argparse
import base64
import json
import pathlib
import sys

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

CAPAS = {"clean": "original", "cam": "cam", "psikan": "psikan", "wsss": "wsss", "zoom": "zoom"}


def _guardar_b64(data_uri: str, ruta: pathlib.Path) -> None:
    ruta.write_bytes(base64.b64decode(data_uri.split(",", 1)[1]))


def _capa_lesion(item, ruta: pathlib.Path) -> None:
    """Mamografía con el contorno de la lesión real (máscara de CBIS-DDSM)."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    img = item["image_tensor"].cpu().numpy()
    if img.ndim == 3:
        img = img[0] if img.shape[0] in (1, 3) else img[:, :, 0]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    mask = np.asarray(item["mask_np"], dtype=np.float32)
    fig = Figure(figsize=(4.2, 4.2), dpi=160)
    ax = fig.subplots()
    ax.imshow(img, cmap="gray")
    if mask.sum() > 0 and mask.shape == img.shape:
        ax.contour(mask, levels=[0.5], colors=["#4fd1e0"], linewidths=2.0)
    ax.text(0.04, 0.93, "Lesión real", transform=ax.transAxes, fontsize=8.2, color="#FFFFFF", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="#0c545c", ec="#4fd1e0", alpha=0.92))
    ax.axis("off")
    fig.savefig(ruta, format="png", bbox_inches="tight", pad_inches=0.04, dpi=160)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--salida", default=str(RAIZ / "desktop_app" / "ui" / "data" / "lab"))
    ap.add_argument("--casos", default="", help="ids separados por coma (por defecto, los nueve)")
    args = ap.parse_args()

    from desktop_app.backend.services.cases_service import CasesService
    from desktop_app.backend.services.render_service import RenderService

    datos = RAIZ / "desktop_app" / "ui" / "data"
    sel = json.loads((datos / "lab_seleccion.json").read_text(encoding="utf-8"))["pacientes"]
    ref = {p["id"]: p for p in json.loads((datos / "lab_pacientes.json").read_text(encoding="utf-8"))["pacientes"]}
    quiero = {c.strip() for c in args.casos.split(",") if c.strip()}

    svc = CasesService()
    rs = RenderService(svc)
    salida = pathlib.Path(args.salida)
    problemas = 0
    for p in sel:
        if quiero and p["id"] not in quiero:
            continue
        idx = p["idx"]
        if idx >= len(svc.df_roi) or str(svc.df_roi.iloc[idx]["patient_id"]) != p["paciente"]:
            print(f"[error] {p['id']}: el índice {idx} no corresponde a {p['paciente']} en tu manifiesto; no se genera.")
            problemas += 1
            continue
        ev = svc.extract_evidence(idx)
        u0, g = ev.get("decision_u0"), ev.get("decision_g")
        r = ref[p["id"]]
        if u0 is None or abs(u0 - r["u0"]) > 5e-3 or abs(g - r["g"]) > 5e-3:
            print(f"[aviso] {p['id']}: la estación da u0={u0}, g={g}; el recorrido numérico tiene u0={r['u0']}, g={r['g']}.")
            problemas += 1
        carpeta = salida / p["id"]
        carpeta.mkdir(parents=True, exist_ok=True)
        for capa, nombre in CAPAS.items():
            _guardar_b64(rs.render_layer(idx, capa), carpeta / f"{nombre}.png")
        _capa_lesion(svc.get_raw_item(idx), carpeta / "lesion.png")
        print(f"[ok] {p['id']} ({p['paciente']}): u0={u0:+.3f} g={g:+.3f} -> {carpeta}")
    print("Listo." if not problemas else f"Terminado con {problemas} aviso(s): revisa los mensajes de arriba.")
    return 1 if problemas else 0


if __name__ == "__main__":
    sys.exit(main())
