"""
Métricas de evaluación para clasificación binaria de mamografías.

Métricas reportadas:
    Accuracy, Precision, Recall, F1-Score, ROC-AUC
    + Calibración de umbral mediante Índice de Youden.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)


def compute_all_metrics(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    y_prob: list | np.ndarray = None,
    pos_label: int = 1,
) -> dict:
    """Calcula el conjunto completo de métricas de clasificación.

    Args:
        y_true: Etiquetas reales (0=benigno, 1=maligno).
        y_pred: Predicciones binarias.
        y_prob: Probabilidades de la clase positiva (necesario para AUC).
        pos_label: Clase positiva. Default 1 (maligno).

    Returns:
        Dict con accuracy, precision, recall, f1, roc_auc, confusion_matrix.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        'recall':    recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        'f1':        f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, np.array(y_prob))

    return metrics


def compute_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple:
    """Encuentra el umbral óptimo usando el Índice de Youden.

    J = Sensibilidad + Especificidad - 1 = TVP - TFP

    En el contexto clínico (diagnóstico de cáncer), se prioriza
    minimizar falsos negativos (maximizar sensibilidad).

    Args:
        y_true: Etiquetas reales [N].
        y_prob: Probabilidades de clase maligna [N].

    Returns:
        tuple (umbral_óptimo, J_max, fpr_at_threshold, tpr_at_threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    youden_j = tpr - fpr
    best_idx  = np.argmax(youden_j)
    return (
        thresholds[best_idx],
        youden_j[best_idx],
        fpr[best_idx],
        tpr[best_idx],
    )


def apply_threshold(y_prob: np.ndarray, threshold: float) -> np.ndarray:
    """Aplica un umbral de decisión sobre las probabilidades.

    Args:
        y_prob: Probabilidades de clase maligna [N].
        threshold: Umbral de decisión.

    Returns:
        Predicciones binarias [N].
    """
    return (np.array(y_prob) >= threshold).astype(int)


def print_metrics_table(metrics: dict, dataset_name: str = 'Conjunto'):
    """Imprime las métricas en formato tabla."""
    print(f"\n{'='*50}")
    print(f"  Métricas — {dataset_name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k == 'confusion_matrix':
            print(f"  {'Matriz de confusión':25s}:")
            for row in v:
                print(f"    {row}")
        else:
            print(f"  {k:25s}: {v:.4f}")
    print(f"{'='*50}")
