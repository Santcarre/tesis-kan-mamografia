"""
Bottleneck para el Frente 2 (fórmula simbólica cerrada).

Reduce el embedding GAP de 512-D a una representación de menor dimensión
(por defecto 64-D) sobre la que se entrena la KAN. Dos opciones:

    - 'linear':  proyección entrenable Linear(512, 64). El bottleneck se
                 entrena junto con la KAN (frente 2).
    - 'pca':     proyección PCA fija calculada SOLO sobre el split de train
                 (no usa validación/test). No es entrenable.
"""
import numpy as np
import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    """Capa de reducción entrenable Linear(512 -> bottleneck_dim).

    Args:
        in_dim (int): Dimensión de entrada. Default 512.
        out_dim (int): Dimensión del bottleneck. Default 64.
    """

    def __init__(self, in_dim: int = 512, out_dim: int = 64):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def pca_bottleneck(feats_train: np.ndarray, out_dim: int = 64):
    """Encaja PCA sobre las features de train y devuelve una matriz de proyección.

    Args:
        feats_train (np.ndarray): Embeddings de entrenamiento [N, 512].
        out_dim (int): Componentes a conservar.

    Returns:
        W (np.ndarray): Matriz [out_dim, 512] (rows = componentes), para
        aplicar como x @ W.T.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=out_dim)
    pca.fit(feats_train)
    return pca.components_  # [out_dim, 512]


def apply_pca(feats: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Aplica la proyección PCA: out = feats @ W.T."""
    return feats @ W.T


class SupervisedPCAReducer:
    """
    Reducción en dos etapas para alta dimensionalidad (D -> 64 -> PCA):
    1. Etapa Supervisada: Proyección lineal W: R^D -> R^64 entrenada para clasificar
       (maximiza la informacion mutua practica con la etiqueta y = benigno/maligno).
    2. Etapa Ortogonal (S-PCA): Descomposición en valores singulares (PCA) sobre el
       espacio latente 64-D supervisado, concentrando la varianza discriminante en
       las primeras componentes (PC0, PC1).
    """
    def __init__(self, in_dim: int, bottleneck_dim: int = 64, random_state: int = 42, device: str = "cuda"):
        self.in_dim = in_dim
        self.bottleneck_dim = bottleneck_dim
        self.random_state = random_state
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        self.proj = nn.Linear(in_dim, bottleneck_dim).to(self.device)
        self.head = nn.Linear(bottleneck_dim, 2).to(self.device)
        self.pca = None
        self.mu_z = None
        self.sd_z = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = 30, lr: float = 1e-3, weight_decay: float = 1e-3, batch_size: int = 256):
        """Entrena el bottleneck supervisado sobre las características GAP y ajusta PCA."""
        from sklearn.decomposition import PCA
        from sklearn.metrics import roc_auc_score
        
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        weights = torch.tensor([1.0, float(n_neg) / float(n_pos)], device=self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        
        optimizer = torch.optim.Adam(list(self.proj.parameters()) + list(self.head.parameters()),
                                     lr=lr, weight_decay=weight_decay)
        
        t_X_tr = torch.from_numpy(X_train).float().to(self.device)
        t_y_tr = torch.from_numpy(y_train).long().to(self.device)
        n_samples = len(X_train)
        
        best_val_auc = -1.0
        best_state = None
        
        for ep in range(epochs):
            self.proj.train()
            self.head.train()
            perm = torch.randperm(n_samples, device=self.device)
            for i in range(0, n_samples, batch_size):
                idx = perm[i:i + batch_size]
                optimizer.zero_grad()
                z = self.proj(t_X_tr[idx])
                loss = criterion(self.head(z), t_y_tr[idx])
                loss.backward()
                optimizer.step()
                
            if X_val is not None and y_val is not None:
                self.proj.eval()
                self.head.eval()
                with torch.no_grad():
                    t_X_va = torch.from_numpy(X_val).float().to(self.device)
                    probs_va = torch.softmax(self.head(self.proj(t_X_va)), dim=1)[:, 1].cpu().numpy()
                    val_auc = float(roc_auc_score(y_val, probs_va))
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = {
                        "proj": {k: v.cpu().clone() for k, v in self.proj.state_dict().items()},
                        "head": {k: v.cpu().clone() for k, v in self.head.state_dict().items()}
                    }

        if best_state is not None:
            self.proj.load_state_dict({k: v.to(self.device) for k, v in best_state["proj"].items()})
            self.head.load_state_dict({k: v.to(self.device) for k, v in best_state["head"].items()})
            
        self.proj.eval()
        with torch.no_grad():
            z_train = self.proj(t_X_tr).cpu().numpy()
            
        self.pca = PCA(n_components=self.bottleneck_dim, random_state=self.random_state)
        z_train_pca = self.pca.fit_transform(z_train).astype(np.float32)
        
        self.mu_z = z_train_pca.mean(0, keepdims=True)
        self.sd_z = z_train_pca.std(0, keepdims=True) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Proyecta X a través del bottleneck supervisado y rota con PCA (z-score)."""
        self.proj.eval()
        with torch.no_grad():
            t_X = torch.from_numpy(X).float().to(self.device)
            z = self.proj(t_X).cpu().numpy()
        z_pca = self.pca.transform(z).astype(np.float32)
        return (z_pca - self.mu_z) / self.sd_z