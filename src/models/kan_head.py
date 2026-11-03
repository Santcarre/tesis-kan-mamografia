"""
KANHead: Cabezal clasificador basado en Kolmogorov-Arnold Networks (KAN).

Usa la biblioteca pykan (pip install pykan).
Arquitectura principal: KAN(width=[512, 32, 2], grid=3, k=3)

Referencia:
    Liu et al. (2025). KAN: Kolmogorov-Arnold Networks.
"""
import torch
import torch.nn as nn


class KANHead(nn.Module):
    """Cabezal KAN para clasificación binaria benigno/maligno.

    Reemplaza la capa lineal final de una CNN por una KAN que
    parametriza transformaciones univariadas mediante B-splines.

    Args:
        width (list[int]): Dimensiones de cada capa. Default [512, 32, 2].
        grid (int): Número de intervalos de la rejilla B-spline. Default 3.
        k (int): Orden del spline. Default 3.
        seed (int): Semilla para reproducibilidad. Default 42.
        device (str): Dispositivo ('cpu' o 'cuda').
    """

    def __init__(
        self,
        width: list = None,
        grid: int = 3,
        k: int = 3,
        seed: int = 42,
        device: str = 'cpu',
    ):
        super().__init__()
        if width is None:
            width = [512, 32, 2]
        self.width  = width
        self.grid   = grid
        self.k      = k
        self.seed   = seed
        self.device = device
        self.kan = self._build_kan()

    def _build_kan(self):
        try:
            from kan import KAN  # pykan
        except ImportError:
            raise ImportError(
                "Instala pykan con: pip install pykan\n"
                "Referencia: https://github.com/KindXiaoming/pykan"
            )
        torch.manual_seed(self.seed)
        kan = KAN(
            width=self.width,
            grid=self.grid,
            k=self.k,
            seed=self.seed,
            device=self.device,
        )
        # El branch simbólico de pykan hace el forward ~100x más lento y solo
        # se necesita tras fix_symbolic(). Durante el entrenamiento (B-splines
        # puras) su aporte es cero, así que lo desactivamos por defecto.
        # Resultados idénticos verificados. reactivar: self.kan.symbolic_enabled = True
        kan.symbolic_enabled = False
        return kan

    def update_grid(self, x: torch.Tensor):
        """Actualiza la rejilla B-spline usando una muestra de activaciones.

        Debe llamarse durante el entrenamiento para adaptar la rejilla
        al rango real de los embeddings de entrada.

        Args:
            x (torch.Tensor): Batch de embeddings [B, D_in].
        """
        with torch.no_grad():
            self.kan.update_grid_from_samples(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kan(x)
