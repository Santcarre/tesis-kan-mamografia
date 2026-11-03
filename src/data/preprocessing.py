"""
Transformaciones y split por patient_id para CBIS-DDSM.

Configuración de normalización ImageNet:
    mu = [0.485, 0.456, 0.406]
    sigma = [0.229, 0.224, 0.225]
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms

# Constantes de normalización ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
INPUT_SIZE    = 224
SEED          = 42


def get_train_transforms() -> transforms.Compose:
    """Transformaciones de aumentación para entrenamiento.

    Aplica: volteo horizontal (p=0.5), rotación ±8°, afín leve,
    ColorJitter, resize a 224x224, ToTensor, normalización ImageNet.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms() -> transforms.Compose:
    """Transformaciones deterministas para validación y test.

    Solo resize a 224x224, ToTensor y normalización ImageNet.
    Sin aleatoriedad.
    """
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def split_by_patient(
    manifest_df: pd.DataFrame,
    val_size: float = 0.20,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide el manifiesto en train/val a nivel de patient_id.

    Garantiza que imágenes del mismo paciente no aparezcan en
    ambos conjuntos (evita data leakage).

    Args:
        manifest_df: DataFrame con columna 'patient_id'.
        val_size: Proporción de pacientes para validación (default 0.20).
        seed: Semilla aleatoria (default SEED=42).

    Returns:
        tuple (train_df, val_df)
    """
    patient_ids = manifest_df['patient_id'].unique()
    train_ids, val_ids = train_test_split(
        patient_ids, test_size=val_size, random_state=seed
    )
    train_df = manifest_df[manifest_df['patient_id'].isin(train_ids)]
    val_df   = manifest_df[manifest_df['patient_id'].isin(val_ids)]
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def zscore_normalize_features(
    train_feats,
    val_feats=None,
    test_feats=None,
):
    """Normalización Z-score de embeddings calculada solo sobre train.

    Args:
        train_feats (np.ndarray): Embeddings de entrenamiento [N, D].
        val_feats: Embeddings de validación (opcional).
        test_feats: Embeddings de test (opcional).

    Returns:
        Tuple con arrays normalizados (solo los no-None).
    """
    import numpy as np
    mu  = train_feats.mean(axis=0)
    std = train_feats.std(axis=0) + 1e-8
    results = [(train_feats - mu) / std]
    if val_feats is not None:
        results.append((val_feats - mu) / std)
    if test_feats is not None:
        results.append((test_feats - mu) / std)
    return tuple(results)
