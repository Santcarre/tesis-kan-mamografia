"""
MammographyDataset: Dataset de PyTorch para mamografías CBIS-DDSM.
Las imágenes se cargan desde rutas en un DataFrame de manifiesto CSV.
"""
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# Directorio base del proyecto (Códigos/)
# __file__ = .../Codigos/src/data/dataset.py -> subir 3 niveles
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MammographyDataset(Dataset):
    """Dataset de mamografías cargado desde un manifiesto CSV.

    El CSV debe tener al menos las columnas:
        - 'filepath': ruta absoluta o relativa a la imagen JPEG
        - 'label':    0 = benigno, 1 = maligno
        - 'patient_id': identificador de paciente (para split sin fuga de datos)

    Args:
        manifest_df (pd.DataFrame): DataFrame ya filtrado (train/val/test).
        root_dir (str): Directorio raíz para resolver rutas relativas.
            Si es vacio, usa PROJECT_ROOT (directorio del proyecto).
        transform: Transformaciones torchvision a aplicar.
    """

    LABEL_MAP = {'benign': 0, 'malignant': 1, 0: 0, 1: 1}

    def __init__(self, manifest_df: pd.DataFrame, root_dir: str = '', transform=None):
        self.df = manifest_df.reset_index(drop=True)
        self.root_dir = root_dir if root_dir else PROJECT_ROOT
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['filepath'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.LABEL_MAP.get(row['label'], int(row['label']))
        return image, label

    def get_class_weights(self):
        """Calcula pesos de clase inversamente proporcionales a su frecuencia.

        Returns:
            torch.Tensor: Tensor [weight_class0, weight_class1]
        """
        import torch
        labels = self.df['label'].map(self.LABEL_MAP).values
        counts = pd.Series(labels).value_counts().sort_index()
        weights = 1.0 / counts
        weights = weights / weights.sum()  # normalizar
        return torch.tensor(weights.values, dtype=torch.float32)
