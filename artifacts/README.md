# artifacts/

`output/embeddings/f2robust_{train,val,test}_{feats,labels}.npy`: vectores GAP de 512 dimensiones que produce el extractor ResNet-18 robusto (checkpoint `reports/models/resnet18_robust_best.pt`) sobre las particiones de `src/data/processed/manifest_*.csv`, con sus etiquetas (0 = benigno, 1 = maligno). Con estos archivos se reproducen sin GPU ni imágenes la proyección S-PCA, la fórmula y los análisis de `scripts/` marcados con el requisito **E**.

| Archivo | Filas |
|---|---|
| `f2robust_train_*` | 4278 |
| `f2robust_val_*` | 1134 |
| `f2robust_test_*` | 1012 |

Los embeddings no contienen píxeles ni metadatos de pacientes.
