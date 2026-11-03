"""
pca_attribution.py: Mapeo interpretable multiescala del Frente 2.

Conecta la formula simbolica extraida con las caracteristicas convolucionales de
ResNet18 y genera mapas de calor espaciales tipo Grad-CAM sobre mamografias.

Propiedad matematica clave:
Dado que GAP es lineal (z_c = mean(A_c)) y PCA es lineal (u_k = sum_c W_{k,c} * z_c),
la contribucion espacial de cada canal convolucional esta exactamente ponderada
por las cargas del componente principal W_{k,c}.

Para la formula simbolica s(u_1), la atribucion espacial combina la derivada
analitica s'(u_1) con las cargas W_{0,c}, produciendo un mapa de calor
directamente explicativo de la formula algebraica.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm


def get_top_pca_channels(pca_components, comp_idx=0, top_n=10):
    """Retorna los top_n canales de ResNet18 con mayor peso absoluto en el componente comp_idx.

    Args:
        pca_components: Matriz [n_comps, 512] de pca.components_.
        comp_idx: Indice del componente (0 = PC_0 = x_1).
        top_n: Numero de canales a reportar.

    Returns:
        lista de dicts: [{'channel': c, 'weight': w, 'abs_weight': |w|}, ...]
    """
    weights = pca_components[comp_idx]
    order = np.argsort(np.abs(weights))[::-1][:top_n]
    return [
        {"channel": int(c), "weight": float(weights[c]), "abs_weight": float(np.abs(weights[c]))}
        for c in order
    ]


class PCAGradCAM:
    """Calcula mapas de calor de atribucion espacial para componentes PCA y formulas simbolicas."""

    def __init__(self, backbone_model: nn.Module, target_layer: nn.Module = None, device: str = "cpu"):
        self.model = backbone_model
        self.device = device
        if target_layer is None:
            # Por defecto layer4 de ResNet18
            if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "layer4"):
                self.target_layer = self.model.backbone.layer4
            elif hasattr(self.model, "layer4"):
                self.target_layer = self.model.layer4
            else:
                raise ValueError("No se pudo inferir target_layer (layer4) del modelo.")
        else:
            self.target_layer = target_layer

        self._activations = None
        self._hook = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, outp):
        self._activations = outp.detach()

    def remove_hook(self):
        if self._hook is not None:
            self._hook.remove()

    def compute_component_cam(self, image_tensor: torch.Tensor, pca_components: np.ndarray, comp_idx: int = 0) -> np.ndarray:
        """Calcula el mapa de calor espacial para el componente PCA comp_idx (e.g. comp_idx=0 para x_1).

        Args:
            image_tensor: [1, 3, H, W] normalizado.
            pca_components: [n_comps, 512] (pca.components_).
            comp_idx: Indice del componente.

        Returns:
            cam: np.ndarray [H, W] en rango [0, 1].
        """
        self.model.eval()
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            _ = self.model(image_tensor)

        # Activaciones de layer4: [1, 512, 7, 7]
        acts = self._activations[0]  # [512, 7, 7]
        W = torch.from_numpy(pca_components[comp_idx].astype(np.float32)).to(self.device)  # [512]
        
        # Ponderacion directa de canales
        cam = (acts * W.view(-1, 1, 1)).sum(dim=0)  # [7, 7]
        cam = F.relu(cam)

        cam_np = cam.cpu().numpy()
        if cam_np.max() > cam_np.min():
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        else:
            cam_np = np.zeros_like(cam_np)

        # Upsample a tamano original de entrada
        H, W_dim = image_tensor.shape[2], image_tensor.shape[3]
        cam_resized = np.array(Image.fromarray(cam_np).resize((W_dim, H), Image.BILINEAR))
        return cam_resized

    def compute_psi_field(
        self,
        image_tensor: torch.Tensor,
        weight_vector: np.ndarray,
        signed: bool = True,
    ) -> np.ndarray:
        """Calcula el campo continuo de densidad simbólica psi_j(u, v) de la tesis.

        Implementa la ecuación (eq:psi-derivation):

            psi_j(u, v) = sum_c (v_{j,c} / sigma_c) * A_c(u, v)

        donde ``weight_vector`` ya contiene v_{j,c} / sigma_c (las cargas efectivas
        del proyector S-PCA, ver ``w1_eff`` / ``w2_eff``) y ``A_c(u, v)`` son las
        activaciones del canal ``c`` en layer4.

        A diferencia de ``compute_component_cam``, este método NO aplica ReLU
        (la definición de la tesis no es positiva) y normaliza de forma simétrica
        alrededor de cero, de modo que el signo del campo se conserva y es
        interpretable con un mapa divergente.

        Args:
            image_tensor: [1, 3, H, W] normalizado.
            weight_vector: [512] cargas v_{j,c} / sigma_c.
            signed: Si es True, conserva el signo (psi analítico). Si es False,
                aplica ReLU (equivalente al CAM usado por la segmentación WSSS).

        Returns:
            np.ndarray [H, W]. Con ``signed=True`` el rango es aproximadamente
            [-1, 1]; con ``signed=False`` es [0, 1].
        """
        self.model.eval()
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            _ = self.model(image_tensor)

        acts = self._activations[0]  # [512, h, w]
        wv = torch.from_numpy(
            np.asarray(weight_vector, dtype=np.float32).reshape(-1)
        ).to(self.device)

        field = (acts * wv.view(-1, 1, 1)).sum(dim=0)  # [h, w]
        if not signed:
            field = F.relu(field)

        f_np = field.cpu().numpy()

        if signed:
            # Normalización simétrica: preserva el signo y la magnitud relativa.
            m = float(np.abs(f_np).max())
            f_np = f_np / m if m > 1e-12 else np.zeros_like(f_np)
        else:
            lo, hi = float(f_np.min()), float(f_np.max())
            f_np = (f_np - lo) / (hi - lo + 1e-8) if hi > lo else np.zeros_like(f_np)

        H, W_dim = image_tensor.shape[2], image_tensor.shape[3]
        # fromarray necesita tipos enteros o float controlados; usamos float32->uint8
        # solo para el reescalado bilinear, y devolvemos el float reescalado.
        idx = np.array(
            Image.fromarray((np.clip(f_np, -1.0, 1.0) * 127.5 + 127.5).astype(np.uint8))
            .resize((W_dim, H), Image.BILINEAR),
            dtype=np.float32,
        )
        out = (idx - 127.5) / 127.5
        return np.clip(out, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def overlay_on_image(original_image: Image.Image, cam: np.ndarray, alpha: float = 0.4, colormap: str = "inferno") -> Image.Image:
        """Superpone el mapa CAM normalizado sobre una imagen PIL."""
        cmap = cm.get_cmap(colormap)
        cam_colored = cmap(cam)[:, :, :3]  # [H, W, 3] en [0, 1]
        cam_img = Image.fromarray((cam_colored * 255).astype(np.uint8))

        base_img = original_image.convert("RGB").resize(cam_img.size)
        blended = Image.blend(base_img, cam_img, alpha=alpha)
        return blended


def compute_gradcam_baseline(model: nn.Module, image_tensor: torch.Tensor, target_class: int = 1, device: str = "cpu") -> np.ndarray:
    """Calcula Grad-CAM para ResNet-18 baseline con capa lineal.

    Args:
        model: ResNet18 con fc=Linear(512, 2).
        image_tensor: [1, 3, 224, 224] normalizado.
        target_class: Clase objetivo (1 = maligno).
        device: Dispositivo de cómputo.

    Returns:
        cam_resized: np.ndarray [224, 224] en rango [0, 1].
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    activations, gradients = None, None

    # layer4 de ResNet18
    target_layer = model.layer4 if hasattr(model, "layer4") else model.backbone.layer4

    def fhook(module, inp, outp):
        nonlocal activations
        activations = outp

    def bhook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    h1 = target_layer.register_forward_hook(fhook)
    h2 = target_layer.register_full_backward_hook(bhook)

    out = model(image_tensor)
    model.zero_grad()
    score = out[0, target_class]
    score.backward()

    h1.remove()
    h2.remove()

    w = gradients.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * activations).sum(dim=1)).squeeze().detach().cpu().numpy()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    else:
        cam = np.zeros_like(cam)

    H, W_dim = image_tensor.shape[2], image_tensor.shape[3]
    cam_resized = np.array(Image.fromarray(cam).resize((W_dim, H), Image.BILINEAR))
    return cam_resized


def compute_gradcam_kan(model: nn.Module, image_tensor: torch.Tensor, target_class: int = 1, device: str = "cpu") -> np.ndarray:
    """Calcula Grad-CAM para HybridResNetKAN retropropagando a través de la cabeza KAN.

    Args:
        model: HybridResNetKAN.
        image_tensor: [1, 3, 224, 224] normalizado.
        target_class: Clase objetivo (1 = maligno).
        device: Dispositivo de cómputo.

    Returns:
        cam_resized: np.ndarray [224, 224] en rango [0, 1].
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    activations = None

    # layer4 es el bloque 7 del FeatureExtractor
    layer4 = model.backbone.backbone[7]
    for p in layer4.parameters():
        p.requires_grad = True

    def fhook(module, inp, outp):
        nonlocal activations
        activations = outp
        activations.retain_grad()

    h1 = layer4.register_forward_hook(fhook)

    out = model(image_tensor)
    model.zero_grad()
    score = out[0, target_class]
    score.backward()

    h1.remove()

    gradients = activations.grad
    w = gradients.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * activations).sum(dim=1)).squeeze().detach().cpu().numpy()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    else:
        cam = np.zeros_like(cam)

    H, W_dim = image_tensor.shape[2], image_tensor.shape[3]
    cam_resized = np.array(Image.fromarray(cam).resize((W_dim, H), Image.BILINEAR))
    return cam_resized


def evaluate_pointing_game(cam: np.ndarray, roi_mask: np.ndarray, percentile: float = 80.0) -> dict:
    """Evalúa métricas cuantitativas de fidelidad espacial sobre máscara ROI.

    Args:
        cam: np.ndarray [H, W] mapa de saliencia en [0, 1].
        roi_mask: np.ndarray [H, W] binario (1 = lesión, 0 = fondo).
        percentile: Umbral percentil para calcular IoU.

    Returns:
        dict con:
            'ebpg': float Fracción de masa de atención dentro del tumor [0, 1].
            'hit': float 1.0 si argmax(cam) está dentro de la ROI, 0.0 si no.
            'iou': float Coeficiente de Jaccard a umbral percentil.
    """
    total_energy = cam.sum() + 1e-8
    hit_energy = (cam * roi_mask).sum()
    ebpg = float(hit_energy / total_energy)

    max_idx = np.unravel_index(np.argmax(cam), cam.shape)
    hit_top1 = float(roi_mask[max_idx] > 0.5)

    thresh = np.percentile(cam, percentile)
    cam_bin = (cam >= thresh).astype(np.float32)
    intersection = (cam_bin * roi_mask).sum()
    union = ((cam_bin + roi_mask) > 0).sum() + 1e-8
    iou = float(intersection / union)

    return {
        "ebpg": ebpg,
        "hit": hit_top1,
        "iou": iou
    }

