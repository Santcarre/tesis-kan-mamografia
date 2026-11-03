"""
HybridResNetKAN: Modelo híbrido ResNet18 + KAN.

Arquitectura:
    ResNet18 (backbone, extracción hasta layer4)
    → Global Average Pooling (vector 512-D)
    → KAN(width=[512, 32, 2], grid=3, k=3)  (clasificador)

Protocolo de entrenamiento en 2 fases:
    Fase 1: Solo KAN, backbone congelado (8 épocas, lr=1e-3)
    Fase 2: KAN + layer4 descongelado (10 épocas, lr_kan=5e-4, lr_backbone=1e-5)
"""
import torch
import torch.nn as nn
from .backbones import FeatureExtractor, get_backbone
from .kan_head import KANHead
from .bottleneck import Bottleneck

# Mapeo de keys del state_dict de ResNet18 (torchvision) a los índices del
# Sequential del FeatureExtractor: children[:-2] de resnet18 =
#   [0]=conv1, [1]=bn1, [2]=relu, [3]=maxpool, [4]=layer1, [5]=layer2,
#   [6]=layer3, [7]=layer4
_RESNET18_TO_SEQ = {
    "conv1": "0", "bn1": "1",
    "layer1": "4", "layer2": "5", "layer3": "6", "layer4": "7",
}


def load_backbone_from_baseline(baseline_path, model, device="cpu"):
    """Copia los pesos del backbone de un ResNet18 baseline (entrenado en la
    base de datos, con head FC) al backbone del modelo híbrido.

    El baseline se guardó como `models.resnet18` completo (conv1, bn1,
    layer1..4, fc). El FeatureExtractor del híbrido usa
    `nn.Sequential(*children[:-2])`, así que mapeamos:
        conv1.*   -> backbone.backbone.0.*
        bn1.*     -> backbone.backbone.1.*
        layer1-4.* -> backbone.backbone.{4..7}.*
    Los keys del head FC (`fc.*`) se ignoran (se reemplaza por la KAN).

    Args:
        baseline_path (str): Ruta al checkpoint `resnet18_baseline_v2.pt`.
        model (HybridResNetKAN): Modelo híbrido recién construido.
        device (str): Dispositivo.
    """
    ckpt = torch.load(baseline_path, map_location=device, weights_only=False)
    src = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    target = model.backbone.backbone.state_dict()
    mapping = {}
    for key in src:
        if key.startswith("fc."):
            continue
        parts = key.split(".")
        if parts[0] in _RESNET18_TO_SEQ:
            mapped = f"{_RESNET18_TO_SEQ[parts[0]]}." + ".".join(parts[1:])
            if mapped in target:
                mapping[mapped] = src[key]

    missing, unexpected = model.backbone.backbone.load_state_dict(mapping, strict=False)
    print(f"[backbone] {len(mapping)} tensores copiados desde {baseline_path}")
    print(f"[backbone] no mapeados: {len(unexpected)} | faltantes: {len(missing)}")
    if missing:
        print(f"[backbone] faltantes: {list(missing)[:5]}")
    if unexpected:
        print(f"[backbone] no usados (head FC): {list(unexpected)[:5]}")
    return model


class HybridResNetKAN(nn.Module):
    """Clasificador híbrido ResNet18 + KAN para mamografías.

    Args:
        pretrained (bool): Usar pesos ImageNet en ResNet18. Default True.
        kan_width (list[int]): Arquitectura KAN. Default [512, 32, 2].
        kan_grid (int): Rejilla B-spline. Default 3.
        kan_k (int): Orden spline. Default 3.
        seed (int): Semilla global. Default 42.
        device (str): Dispositivo de cómputo.
    """

    def __init__(
        self,
        pretrained: bool = True,
        kan_width: list = None,
        kan_grid: int = 3,
        kan_k: int = 3,
        seed: int = 42,
        device: str = 'cpu',
    ):
        super().__init__()
        if kan_width is None:
            kan_width = [512, 32, 2]

        self.device = device
        self.seed   = seed

        # Backbone: ResNet18 congelado con GAP
        self.backbone: FeatureExtractor = get_backbone(
            name='resnet18', pretrained=pretrained, freeze=True
        )

        # Cabezal KAN
        self.kan_head = KANHead(
            width=kan_width, grid=kan_grid, k=kan_k,
            seed=seed, device=device,
        )

    # ------------------------------------------------------------------
    # Métodos de control de fases
    # ------------------------------------------------------------------

    def set_phase1(self):
        """Fase 1: Congela el backbone completo, solo entrena KAN."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.kan_head.parameters():
            p.requires_grad = True
        print("[Fase 1] Backbone congelado. Solo KAN se actualiza.")

    def set_phase2(self):
        """Fase 2: Descongela layer4 del backbone + KAN para fine-tuning."""
        self.backbone.unfreeze_layer4()
        for p in self.kan_head.parameters():
            p.requires_grad = True
        print("[Fase 2] layer4 + KAN descongelados para fine-tuning.")

    def get_phase2_param_groups(self, lr_kan: float = 5e-4, lr_backbone: float = 1e-5):
        """Grupos de parámetros con tasas de aprendizaje diferenciadas para Fase 2.

        Args:
            lr_kan: Learning rate para la KAN.
            lr_backbone: Learning rate para layer4 del backbone.

        Returns:
            list de dicts para torch.optim.Adam.
        """
        kan_params      = list(self.kan_head.parameters())
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        return [
            {'params': kan_params,      'lr': lr_kan},
            {'params': backbone_params, 'lr': lr_backbone},
        ]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de imágenes [B, 3, 224, 224]

        Returns:
            logits: [B, 2]
        """
        embeddings = self.backbone(x)        # [B, 512]
        logits     = self.kan_head(embeddings)  # [B, 2]
        return logits

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extrae el vector GAP de 512-D sin pasar por la KAN."""
        with torch.no_grad():
            return self.backbone(x)


def load_backbone_from_hybrid_checkpoint(ckpt_path, model, device="cpu"):
    """Copia el backbone de un checkpoint híbrido (HybridResNetKAN) a otro modelo.

    Útil para el Frente 2: trasladar el backbone entrenado 1C-Base a un
    HybridResNetKAN_Bottleneck sin reentrenarlo.

    Args:
        ckpt_path (str): Ruta al checkpoint híbrido (state_dict con 'backbone.*').
        model (nn.Module): Modelo destino con atributo '.backbone' (FeatureExtractor).
        device (str): Dispositivo.

    Returns:
        model con el backbone cargado.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    src = ckpt["model_state"] if "model_state" in ckpt else ckpt

    # Extrae solo las claves del backbone origen (prefijo 'backbone.')
    src_b = {k[len("backbone."):]: v for k, v in src.items() if k.startswith("backbone.")}
    missing, unexpected = model.backbone.load_state_dict(src_b, strict=False)
    print(f"[backbone] {len(src_b)} tensores copiados desde {ckpt_path}")
    print(f"[backbone] faltantes: {len(missing)} | no usados: {len(unexpected)}")
    return model


class HybridResNetKAN_Bottleneck(nn.Module):
    """Híbrido ResNet18 + bottleneck + KAN (Frente 2).

    Arquitectura:
        ResNet18 (backbone, GAP 512-D, típicamente congelado)
        → Bottleneck (Linear 512→64 entrenable)
        → KAN (width por defecto [64, 16, 2])

    El Z-score de las features se aplica ANTES de alimentar la KAN, usando
    estadísticas calculadas solo sobre train. Por eso el forward espera que
    `x` ya esté normalizado (las stats se aplican externamente para que la
    fórmula cerrada las pueda reproducir).

    Args:
        backbone_name (str): 'resnet18' (default).
        pretrained (bool): Usar pesos ImageNet en el backbone. Default False
            (se cargan desde checkpoint baseline/1C-Base).
        freeze_backbone (bool): Congelar el backbone. Default True (frente 2
            usa features ya entrenadas; no se reentrena el backbone).
        bottleneck_dim (int): Salida del bottleneck. Default 64.
        kan_width (list[int]): Anchura KAN. Default [64, 16, 2].
        kan_grid (int): Rejilla B-spline. Default 3.
        kan_k (int): Orden del spline. Default 3.
        seed (int): Semilla. Default 42.
        device (str): Dispositivo.
    """

    def __init__(
        self,
        pretrained: bool = False,
        bottleneck_dim: int = 64,
        kan_width: list = None,
        kan_grid: int = 3,
        kan_k: int = 3,
        freeze_backbone: bool = True,
        seed: int = 42,
        device: str = 'cpu',
    ):
        super().__init__()
        if kan_width is None:
            kan_width = [bottleneck_dim, 16, 2]

        self.device = device
        self.seed = seed
        self.bottleneck_dim = bottleneck_dim

        self.backbone: FeatureExtractor = get_backbone(
            name='resnet18', pretrained=pretrained, freeze=freeze_backbone
        )
        self.bottleneck = Bottleneck(in_dim=self.backbone.feat_dim, out_dim=bottleneck_dim)
        self.kan_head = KANHead(
            width=kan_width, grid=kan_grid, k=kan_k,
            seed=seed, device=device,
        )

    def set_phase1(self):
        """Fase 1: backbone + bottleneck congelados, solo KAN se actualiza."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.bottleneck.parameters():
            p.requires_grad = False
        for p in self.kan_head.parameters():
            p.requires_grad = True
        print("[F2] Fase 1: backbone y bottleneck congelados. Solo KAN se actualiza.")

    def set_phase2(self):
        """Fase 2: descongela bottleneck + KAN para fine-tuning (backbone sigue congelado)."""
        for p in self.bottleneck.parameters():
            p.requires_grad = True
        for p in self.kan_head.parameters():
            p.requires_grad = True
        print("[F2] Fase 2: bottleneck + KAN descongelados (backbone congelado).")

    def get_phase2_param_groups(self, lr_kan: float = 5e-4, lr_bottleneck: float = 1e-3):
        kan_params = list(self.kan_head.parameters())
        bot_params = [p for p in self.bottleneck.parameters() if p.requires_grad]
        return [
            {'params': kan_params, 'lr': lr_kan},
            {'params': bot_params, 'lr': lr_bottleneck},
        ]

    def forward_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """GAP 512-D (con gradiente, para entrenamiento del bottleneck)."""
        return self.backbone(x)

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """GAP 512-D sin gradiente."""
        with torch.no_grad():
            return self.backbone(x)

    def bottleneck_features(self, emb: torch.Tensor) -> torch.Tensor:
        """Aplica el bottleneck: 512 -> bottleneck_dim."""
        return self.bottleneck(emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.extract_embeddings(x)          # [B, 512]
        bot = self.bottleneck(emb)                # [B, bottleneck_dim]
        return self.kan_head(bot)                 # [B, 2]
