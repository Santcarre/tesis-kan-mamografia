"""
Backbones CNN preentrenados en ImageNet para extracción de características.

Soporta: ResNet18 (512-D), VGG16 (512-D), InceptionV3 (2048-D), EfficientNet-B0 (1280-D).
Todos devuelven el vector de características tras Global Average Pooling (GAP).
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    VGG16_Weights,
    Inception_V3_Weights,
    EfficientNet_B0_Weights,
    DenseNet121_Weights,
)

# Dimensión del vector GAP para cada backbone
BACKBONE_FEAT_DIM = {
    'resnet18':       512,
    'resnet50':      2048,
    'vgg16':          512,
    'inceptionv3':   2048,
    'efficientnet_b0': 1280,
    'densenet121':   1024,
}


class FeatureExtractor(nn.Module):
    """Wrapper que devuelve embeddings GAP de un backbone CNN.

    Args:
        backbone_name (str): Nombre del backbone.
        pretrained (bool): Usar pesos ImageNet (default True).
        freeze (bool): Congelar backbone salvo las últimas capas (default True).
    """

    def __init__(self, backbone_name: str, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.feat_dim = BACKBONE_FEAT_DIM[self.backbone_name]
        self.backbone, self.gap = self._build_backbone(pretrained)
        if freeze:
            self._freeze_backbone()

    def _build_backbone(self, pretrained: bool):
        name = self.backbone_name
        weights = None

        if name == 'resnet18':
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            # Extraer hasta layer4, sin FC ni avgpool original
            backbone = nn.Sequential(*list(base.children())[:-2])
            gap = nn.AdaptiveAvgPool2d(1)

        elif name == 'resnet50':
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            # Extraer hasta layer4, sin FC ni avgpool original
            backbone = nn.Sequential(*list(base.children())[:-2])
            gap = nn.AdaptiveAvgPool2d(1)

        elif name == 'vgg16':
            weights = VGG16_Weights.DEFAULT if pretrained else None
            base = models.vgg16(weights=weights)
            backbone = base.features
            gap = nn.AdaptiveAvgPool2d(1)

        elif name == 'inceptionv3':
            weights = Inception_V3_Weights.DEFAULT if pretrained else None
            base = models.inception_v3(weights=weights)
            # En torchvision reciente el builder fuerza aux_logits=True con weights;
            # se desactiva manualmente para extraer características (sin clasificador).
            base.aux_logits = False
            if base.AuxLogits is not None:
                base.AuxLogits = None
            base.fc = nn.Identity()
            backbone = base
            gap = nn.Identity()  # InceptionV3 ya hace pooling interno

        elif name == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            backbone = base.features
            gap = nn.AdaptiveAvgPool2d(1)

        elif name == 'densenet121':
            weights = DenseNet121_Weights.DEFAULT if pretrained else None
            base = models.densenet121(weights=weights)
            # features (excepto clasificador lineal) + global average pool
            backbone = base.features
            gap = nn.AdaptiveAvgPool2d(1)

        else:
            raise ValueError(f"Backbone '{name}' no soportado. Opciones: {list(BACKBONE_FEAT_DIM.keys())}")

        return backbone, gap

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_stage(self):
        """Descongela la última etapa convolucional para fine-tuning en Fase 2."""
        if self.backbone_name in ('resnet18', 'resnet50'):
            for param in self.backbone[-1].parameters():  # último bloque = layer4
                param.requires_grad = True
        elif self.backbone_name == 'densenet121':
            for name, child in self.backbone.named_children():
                if name in ('denseblock4', 'norm5'):
                    for param in child.parameters():
                        param.requires_grad = True
        elif self.backbone_name == 'efficientnet_b0':
            for block in self.backbone[7:]:
                for param in block.parameters():
                    param.requires_grad = True
        else:
            for param in list(self.backbone.children())[-1].parameters():
                param.requires_grad = True

    def unfreeze_layer4(self):
        """Compatibilidad con código legacy."""
        self.unfreeze_last_stage()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if isinstance(features, tuple):  # InceptionV3 aux output
            features = features[0]
        if self.gap is not nn.Identity():
            features = self.gap(features)
        return features.flatten(1)  # [B, feat_dim]


def get_backbone(name: str, pretrained: bool = True, freeze: bool = True) -> FeatureExtractor:
    """Factory para obtener un extractor de características.

    Args:
        name: 'resnet18' | 'vgg16' | 'inceptionv3' | 'efficientnet_b0'
        pretrained: Usar pesos ImageNet.
        freeze: Congelar el backbone.

    Returns:
        FeatureExtractor listo para usar.
    """
    return FeatureExtractor(backbone_name=name, pretrained=pretrained, freeze=freeze)
