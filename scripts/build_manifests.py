"""
Construye los manifiestos CSV (train/val/test) desde los datos crudos del CBIS-DDSM.

VERSION 2: Usa TODAS las vistas disponibles por paciente y elimina data leakage.

NO REQUIERE DEPENDENCIAS EXTERNAS (solo libreria estandar de Python).

Estrategia:
1. Cargar dicom_info.csv para obtener rutas validas a JPEG
2. Cargar CSVs de casos para obtener labels (pathology)
3. Cruzar por patient_id para construir manifiestos
4. NUEVO: Incluir todas las vistas (full mammogram + cropped) por paciente
5. NUEVO: Eliminar pacientes duplicados entre train y test (sin data leakage)

Uso:
    python scripts/build_manifests.py
"""
import os
import csv
import random
from pathlib import Path

# Configuracion
SEED = 42
random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV_DIR = PROJECT_ROOT / "database" / "archive" / "csv"
RAW_JPEG_DIR = PROJECT_ROOT / "database" / "archive" / "jpeg"
PROCESSED_DIR = PROJECT_ROOT / "src" / "data" / "processed"

LABEL_MAP = {
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
    "MALIGNANT": 1,
}


def read_csv(filepath):
    """Lee un CSV y retorna lista de diccionarios."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def extract_base_patient_id(dicom_patient_id):
    """Extrae el patient_id base desde el PatientID de dicom_info.

    Formatos de entrada:
        Calc-Training_P_00005_RIGHT_CC -> P_00005
        Mass-Test_P_00882_RIGHT_CC_1 -> P_00882
        Calc-Test_P_00562_LEFT_CC_2 -> P_00562

    Retorna: patient_id base (P_XXXXX)
    """
    parts = dicom_patient_id.split("_")
    if len(parts) >= 3:
        return f"{parts[1]}_{parts[2]}"
    return None


def build_manifests():
    """Construye los manifiestos train/val/test."""

    print("=" * 60)
    print("Construccion de Manifiestos CBIS-DDSM")
    print("=" * 60)

    # Paso 1: Cargar mapeo de dicom_info.csv
    print("\n[1/3] Cargando mapeo desde dicom_info.csv...")
    dicom_rows = read_csv(RAW_CSV_DIR / "dicom_info.csv")
    print(f"  Total filas: {len(dicom_rows)}")

    # Mapeo: patient_id -> lista de {filepath, series_desc}
    patient_to_images = {}

    for row in dicom_rows:
        dicom_pid = row.get("PatientID", "").strip()
        image_path = row.get("image_path", "").strip()
        series_desc = row.get("SeriesDescription", "").strip().lower()

        if not dicom_pid or not image_path:
            continue

        base_pid = extract_base_patient_id(dicom_pid)
        if not base_pid:
            continue

        # Convertir ruta CBIS-DDSM/jpeg/UID/file.jpg -> database/archive/jpeg/UID/file.jpg
        jpeg_rel = image_path.replace("CBIS-DDSM/jpeg/", "")
        jpeg_full = RAW_JPEG_DIR / jpeg_rel

        if not jpeg_full.exists():
            continue

        if base_pid not in patient_to_images:
            patient_to_images[base_pid] = []

        # Incluir full mammogram y cropped images (excluir ROI masks)
        if "roi" in series_desc or "mask" in series_desc:
            continue
        
        entry = {
            "filepath": str(jpeg_full.relative_to(PROJECT_ROOT)),
            "series": series_desc,
        }

        # Insertar full mammogram primero, luego cropped
        if "full" in series_desc:
            patient_to_images[base_pid].insert(0, entry)
        else:
            patient_to_images[base_pid].append(entry)

    print(f"  Pacientes con imagenes validas: {len(patient_to_images)}")

    # Paso 2: Cargar CSVs de casos para obtener labels
    print("\n[2/3] Cargando CSVs de casos...")
    case_files = {
        "calc_train": "calc_case_description_train_set.csv",
        "calc_test": "calc_case_description_test_set.csv",
        "mass_train": "mass_case_description_train_set.csv",
        "mass_test": "mass_case_description_test_set.csv",
    }

    cases = {}
    for key, filename in case_files.items():
        cases[key] = read_csv(RAW_CSV_DIR / filename)
        print(f"  {filename}: {len(cases[key])} filas")

    # Paso 3: Construir manifiestos cruzando datos
    print("\n[3/3] Construyendo manifiestos...")

    def build_records(case_keys, split_name):
        """Construye registros para un split dado.
        
        VERSION 2: Incluye TODAS las vistas disponibles por paciente.
        """
        records = []
        missing = 0
        seen_patients = set()

        for key in case_keys:
            for row in cases[key]:
                pathology = row.get("pathology", "").strip().upper()
                label = LABEL_MAP.get(pathology)
                if label is None:
                    continue

                patient_id = row.get("patient_id", "").strip()
                if not patient_id or patient_id in seen_patients:
                    continue

                seen_patients.add(patient_id)

                images = patient_to_images.get(patient_id, [])
                if images:
                    # NUEVO: Usar TODAS las vistas disponibles para este paciente
                    for img_entry in images:
                        records.append({
                            "filepath": img_entry["filepath"],
                            "label": label,
                            "patient_id": patient_id,
                        })
                else:
                    missing += 1

        print(f"  {split_name}: {len(records)} registros, {missing} sin imagen")
        return records, missing

    # Construir train
    train_full, train_missing = build_records(["calc_train", "mass_train"], "Train (full)")
    print(f"    -> {len(train_full)} registros, {train_missing} faltantes")

    # Split train/val por patient_id
    patients = list(set(r["patient_id"] for r in train_full))
    random.shuffle(patients)
    n_val = int(len(patients) * 0.20)
    val_patients = set(patients[:n_val])

    records_train = [r for r in train_full if r["patient_id"] not in val_patients]
    records_val = [r for r in train_full if r["patient_id"] in val_patients]

    # Verificar que no hay fuga
    train_p = set(r["patient_id"] for r in records_train)
    val_p = set(r["patient_id"] for r in records_val)
    if train_p & val_p:
        print("    ERROR: Fuga de datos!")
        return
    print(f"    -> Train: {len(records_train)}, Val: {len(records_val)}")

    # Construir test
    test_full, test_missing = build_records(["calc_test", "mass_test"], "Test")
    print(f"    -> {len(test_full)} registros, {test_missing} faltantes")
    
    # NUEVO: Eliminar data leakage train-test
    print("\n[4/4] Eliminando data leakage train-test...")
    train_full_patients = set(r["patient_id"] for r in train_full)
    test_full_patients = set(r["patient_id"] for r in test_full)
    overlap = train_full_patients & test_full_patients
    
    if overlap:
        print(f"  Encontrados {len(overlap)} pacientes duplicados entre train y test")
        print(f"  Removiendo pacientes duplicados del test set...")
        records_test = [r for r in test_full if r["patient_id"] not in overlap]
        print(f"  Test antes: {len(test_full)} registros, despues: {len(records_test)} registros")
    else:
        print(f"  No se encontro data leakage (pacientes completamente separados)")
        records_test = test_full

    # Guardar manifiestos
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    def save_csv(records, filepath):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filepath", "label", "patient_id"])
            writer.writeheader()
            writer.writerows(records)

    save_csv(records_train, PROCESSED_DIR / "manifest_train.csv")
    save_csv(records_val, PROCESSED_DIR / "manifest_val.csv")
    save_csv(records_test, PROCESSED_DIR / "manifest_test.csv")

    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DE MANIFESTOS GENERADOS")
    print("=" * 60)

    def count_labels(records):
        n = len(records)
        benign = sum(1 for r in records if r["label"] == 0)
        malign = sum(1 for r in records if r["label"] == 1)
        patients = len(set(r["patient_id"] for r in records))
        return n, benign, malign, patients

    print(f"\n{'Conjunto':<10} {'Imagenes':>10} {'Benignas':>10} {'Malignas':>10} {'Pacientes':>10}")
    print("-" * 52)
    for name, recs in [("Train", records_train), ("Val", records_val), ("Test", records_test)]:
        n, b, m, p = count_labels(recs)
        print(f"{name:<10} {n:>10} {b:>10} {m:>10} {p:>10}")

    print(f"\nArchivos guardados en: {PROCESSED_DIR}")
    for f in ["manifest_train.csv", "manifest_val.csv", "manifest_test.csv"]:
        print(f"  {f}")

    print("\nConstruccion de manifiestos completada")


if __name__ == "__main__":
    build_manifests()
