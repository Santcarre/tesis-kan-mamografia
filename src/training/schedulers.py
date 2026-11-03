"""
Schedulers de learning rate y early stopping.
"""
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


def get_scheduler(optimizer: torch.optim.Optimizer, patience: int = 2, factor: float = 0.5):
    """ReduceLROnPlateau en modo max (maximiza métrica de validación).

    Args:
        optimizer: Optimizador de PyTorch.
        patience: Épocas de paciencia antes de reducir lr. Default 2.
        factor: Factor de reducción del lr. Default 0.5.

    Returns:
        ReduceLROnPlateau scheduler.
    """
    return ReduceLROnPlateau(
        optimizer, mode='max', factor=factor, patience=patience
    )


class EarlyStopping:
    """Parada temprana basada en una métrica de validación (modo max).

    Args:
        patience (int): Épocas a esperar sin mejora antes de detener. Default 4.
        delta (float): Mejora mínima para considerar que hubo progreso. Default 0.
    """

    def __init__(self, patience: int = 4, delta: float = 0.0):
        self.patience = patience
        self.delta    = delta
        self.best     = None
        self.counter  = 0
        self.stop     = False

    def __call__(self, metric: float) -> bool:
        """Evalúa si se debe detener el entrenamiento.

        Args:
            metric: Valor de la métrica de validación (mayor = mejor).

        Returns:
            True si se debe detener el entrenamiento.
        """
        if self.best is None or metric > self.best + self.delta:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop
