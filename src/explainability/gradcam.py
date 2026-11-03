"""
Grad-CAM sobre el último bloque convolucional de ResNet18 (layer4).

También soporta Grad-CAM focalizado en una dimensión latente específica
(e.g., x_27 del vector GAP, la más crítica según el análisis simbólico).

Referencia:
    Selvaraju et al. (2017). Grad-CAM: Visual explanations from deep networks.
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class GradCAM:
    """Grad-CAM sobre una capa objetivo de una red convolucional.

    Args:
        model: Modelo PyTorch (e.g., HybridResNetKAN).
        target_layer: Capa convolucional objetivo (e.g., model.backbone.backbone[-1]).
        device: Dispositivo ('cpu' o 'cuda').
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module, device: str = 'cpu'):
        self.model        = model
        self.target_layer = target_layer
        self.device       = device
        self._gradients   = None
        self._activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def compute(
        self,
        image_tensor: torch.Tensor,
        class_idx: int = 1,
        latent_dim: int = None,
    ) -> np.ndarray:
        """Calcula el mapa de calor Grad-CAM.

        Args:
            image_tensor: Tensor [1, 3, H, W] normalizado.
            class_idx: Clase objetivo (0=benigno, 1=maligno). Default 1.
            latent_dim: Si se especifica, calcula el gradiente respecto a
                        esta dimensión del vector GAP (e.g., 27 para x_27)
                        en lugar de la clase de salida.

        Returns:
            cam: Mapa de calor np.ndarray [H, W] en rango [0, 1].
        """
        self.model.eval()
        image_tensor = image_tensor.to(self.device)
        image_tensor.requires_grad_(True)

        output = self.model(image_tensor)  # [1, 2]

        self.model.zero_grad()
        if latent_dim is not None:
            # Grad-CAM focalizado en una dimensión del vector GAP
            target = self.model.extract_embeddings(image_tensor)[:, latent_dim]
            target.backward(torch.ones_like(target))
        else:
            target = output[0, class_idx]
            target.backward()

        # Pooling global de los gradientes
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam     = (weights * self._activations).sum(dim=1).squeeze(0)  # [H, W]
        cam     = torch.relu(cam).cpu().numpy()

        # Normalizar a [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    def overlay_on_image(
        self,
        original_image: Image.Image,
        cam: np.ndarray,
        alpha: float = 0.5,
    ) -> Image.Image:
        """Superpone el mapa de calor sobre la imagen original.

        Args:
            original_image: Imagen PIL sin normalizar.
            cam: Mapa de calor [H, W] en [0, 1].
            alpha: Transparencia del mapa. Default 0.5.

        Returns:
            Imagen PIL con Grad-CAM superpuesto.
        """
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        import matplotlib

        h, w = original_image.size
        cam_resized = np.array(
            Image.fromarray(np.uint8(cam * 255)).resize((w, h), Image.BILINEAR)
        ) / 255.0

        colormap   = cm.get_cmap('jet')
        heatmap    = colormap(cam_resized)[:, :, :3]  # RGB
        orig_array = np.array(original_image) / 255.0

        overlay = alpha * heatmap + (1 - alpha) * orig_array
        overlay = np.clip(overlay, 0, 1)
        return Image.fromarray(np.uint8(overlay * 255))
