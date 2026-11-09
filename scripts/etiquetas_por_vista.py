# -*- coding: utf-8 -*-
"""
etiquetas_por_vista.py -- Cuántas etiquetas de los manifiestos difieren de la patología de su propia vista y cuánto
cambia el AUC de prueba con la etiqueta por vista.

scripts/build_manifests.py asigna a TODAS las imágenes de una paciente la patología de la primera lesión que aparece
en las tablas de CBIS-DDSM (calc_train, calc_test, mass_train, mass_test). En pacientes con lesiones de distinta
patología, algunas imágenes quedan con una etiqueta que no es la de sus lesiones. Este script toma como referencia:
  - recorte: la patología de la lesión del recorte;
  - mamografía completa: maligna si alguna lesión registrada en esa vista (paciente, lado, proyección) es maligna.
y recalcula el AUC de prueba de la fórmula y de las dos redes con las probabilidades guardadas.

Uso (desde la raíz del repo):  python scripts/etiquetas_por_vista.py [CARPETA_CSV]
CARPETA_CSV = carpeta con los CSV de CBIS-DDSM (dicom_info.csv y las cuatro tablas de casos);
por defecto, database/archive/csv. Solo usa pandas y numpy.
Salida: reports/auditoria/etiquetas_por_vista.json
"""
import json, os, re, sys
import numpy as np, pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RAIZ, "database", "archive", "csv")
PROB = os.path.join(RAIZ, "reports", "experiments", "youden_val_probs")
OUT = os.path.join(RAIZ, "reports", "auditoria", "etiquetas_por_vista.json")


def uid(ruta):
    partes = str(ruta).strip().replace("\\", "/").split("/")
    return partes[-2] if len(partes) >= 3 else None


def auc(y, s):
    y = np.asarray(y); r = pd.Series(np.asarray(s, float)).rank().values
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


casos = pd.concat([pd.read_csv(os.path.join(CSV, f + ".csv")) for f in
                   ("calc_case_description_train_set", "calc_case_description_test_set",
                    "mass_case_description_train_set", "mass_case_description_test_set")])
casos["pat"] = casos["pathology"].str.strip().str.upper()
casos["vista"] = (casos["patient_id"].str.strip() + "_" + casos["left or right breast"].str.strip() + "_"
                  + casos["image view"].str.strip())
maligna_vista = casos.groupby("vista")["pat"].apply(lambda s: int((s == "MALIGNANT").any())).to_dict()
recortes = {}
for _, r in casos.iterrows():
    recortes.setdefault(uid(r["cropped image file path"]), []).append(r["pat"])
di = pd.read_csv(os.path.join(CSV, "dicom_info.csv"))
info = {str(r.image_path).replace("CBIS-DDSM/jpeg/", "").replace("/", "\\"):
        (str(r.SeriesInstanceUID).strip(), str(r.PatientID), str(r.SeriesDescription)) for _, r in di.iterrows()}

salida = {"referencia": "recorte: patología de su lesión; mamografía completa: maligna si alguna lesión de la vista lo es"}
for particion in ("train", "val", "test"):
    m = pd.read_csv(os.path.join(RAIZ, "src", "data", "processed", f"manifest_{particion}.csv"))
    ref = []
    for ruta in m["filepath"]:
        serie, pid, desc = info[ruta.replace("database\\archive\\jpeg\\", "").replace("database/archive/jpeg/", "").replace("/", "\\")]
        v = re.match(r".*?_(P_\d+)_(LEFT|RIGHT)_(CC|MLO)", pid)
        if "cropped" in desc and serie in recortes:
            ref.append(int(any(p == "MALIGNANT" for p in recortes[serie])))
        elif v:
            ref.append(maligna_vista.get(f"{v.group(1)}_{v.group(2)}_{v.group(3)}", -1))
        else:
            ref.append(-1)
    m["ref"] = ref
    dif = m[(m.ref >= 0) & (m.ref != m.label)]
    salida[particion] = {"n": int(len(m)), "sin_referencia": int((m.ref < 0).sum()), "distintas": int(len(dif)),
                         "porcentaje": round(100 * len(dif) / int((m.ref >= 0).sum()), 2),
                         "pacientes": int(dif["patient_id"].nunique()),
                         "maligna_en_manifiesto_benigna_en_vista": int((dif.label == 1).sum()),
                         "benigna_en_manifiesto_maligna_en_vista": int((dif.label == 0).sum())}
    if particion == "test":
        salida["test"]["pacientes_afectados"] = sorted(dif["patient_id"].unique().tolist())
        y = np.load(os.path.join(PROB, "Baseline_FC_test_labels.npy")).astype(int)
        assert (y == m["label"].values).all(), "el orden de las probabilidades no coincide con el manifiesto"
        ok = (m.ref >= 0).values
        salida["auc_prueba"] = {}
        for k in ("R3_formula", "Robusto", "Baseline_FC"):
            s = np.load(os.path.join(PROB, f"{k}_test_probs.npy")).ravel()
            salida["auc_prueba"][k] = {"etiqueta_manifiesto": round(auc(m["label"], s), 4),
                                       "etiqueta_por_vista": round(auc(m["ref"][ok], s[ok]), 4)}
json.dump(salida, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(salida, ensure_ascii=False, indent=2))
