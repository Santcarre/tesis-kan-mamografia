"""
Funciones de pérdida con ponderación por clase para datos desbalanceados.
"""
import torch
import torch.nn as nn


def get_weighted_loss(class_weights: torch.Tensor = None, device: str = 'cpu') -> nn.CrossEntropyLoss:
    """CrossEntropyLoss con pesos de clase opcionales.

    Args:
        class_weights: Tensor [num_classes] con pesos inversamente proporcionales
                       a la frecuencia de cada clase. Si None, pérdida sin pesos.
        device: Dispositivo donde mover los pesos.

    Returns:
        nn.CrossEntropyLoss configurada.
    """
    if class_weights is not None:
        class_weights = class_weights.to(device)
    return nn.CrossEntropyLoss(weight=class_weights)
