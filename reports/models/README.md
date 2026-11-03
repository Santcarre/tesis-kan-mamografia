# reports/models/

Esta carpeta está vacía a propósito: los checkpoints (`*.pt`) no se versionan en git.

Para los scripts marcados con **C** descargue `resnet18_robust_best.pt` y `resnet18_baseline_v2.pt` (43 MB cada uno) de los activos de la versión (*Release*) del repositorio y colóquelos aquí. Verifique su integridad:

```
SHA-256  c64a0ca03b1e5f6adca9f98de5354dbb8f1338026e822429d49fa84366e178e1  resnet18_robust_best.pt
SHA-256  224e1631c90801373e7bfb9ea4eabfc6e70d3e0ae59f58e5c9677ddf8f1e99da  resnet18_baseline_v2.pt
```

Ambos checkpoints contienen el `state_dict` de una ResNet-18 de `torchvision` con cabeza lineal de 2 clases (`fc: 512 → 2`), bajo la clave `model_state_dict`.
