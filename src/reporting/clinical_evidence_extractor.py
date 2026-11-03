"""
clinical_evidence_extractor.py: Extrae y formaliza los tokens de evidencia fáctica
(analítica KAN, localización espacial WSSS y descriptores morfológicos)
a partir de mamografías de prueba de CBIS-DDSM para alimentar el generador de reportes LLM.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="kan")

def _resolve_codigos():
    env = os.environ.get("TESIS_CODIGOS")
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if os.path.isdir(os.path.join(here, "src", "data")):
        return here
    raise RuntimeError(
        "No se pudo resolver el directorio del proyecto. "
        "Defina TESIS_CODIGOS con la ruta a 'Códigos' (o al espejo)."
    )


CODIGOS = _resolve_codigos()
if CODIGOS not in sys.path:
    sys.path.insert(0, CODIGOS)


def _resolve_embeddings_dir():
    env = os.environ.get("TESIS_EMBEDDINGS_DIR")
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    for cand in (os.path.join(CODIGOS, "artifacts", "output", "embeddings"),
                 os.path.join(os.path.dirname(CODIGOS), "rep_simbolica", "output", "embeddings")):
        if os.path.isdir(cand):
            return cand
    raise RuntimeError(
        "No se encontraron los embeddings. Defina TESIS_EMBEDDINGS_DIR "
        "con la ruta a la carpeta de embeddings (por defecto, artifacts/output/embeddings)."
    )

def extract_series_uid(p):
    parts = str(p).strip().replace("\\", "/").split("/")
    if len(parts) >= 3:
        return parts[-2]
    return None
from src.reporting import birads_herramienta as _birads  # fuente única de la categoría BI-RADS
from src.models.bottleneck import SupervisedPCAReducer
from src.explainability.pca_attribution import PCAGradCAM
from src.explainability.weak_segmentation import (
    segment_morphology_refined,
    compute_centroid,
    compute_segmentation_metrics,
    extract_breast_tissue_mask,
    segment_wsss_focal_core
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def determine_quadrant(cy: float, cx: float, laterality: str = "LEFT", view: str = "MLO", h: int = 224, w: int = 224) -> str:
    """Mapea las coordenadas normalizadas del centroide (cy, cx) al cuadrante anatómico clínico:
    - CSE: Cuadrante Superior Externo (Upper-Outer)
    - CSI: Cuadrante Superior Interno (Upper-Inner)
    - CIE: Cuadrante Inferior Externo (Lower-Outer)
    - CII: Cuadrante Inferior Interno (Lower-Inner)
    - Retroareolar / Central (zona pericéntrica)
    """
    ny = cy / h
    nx = cx / w
    
    # Zona retroareolar central (dentro del 18% del centro de masa)
    if 0.40 <= ny <= 0.60 and 0.40 <= nx <= 0.60:
        return "Región Retroareolar / Central"

    is_upper = (ny < 0.50)
    
    # En proyecciones CC y MLO estándar de CBIS-DDSM:
    # Si la mama es izquierda vs derecha, la orientación medial/lateral depende del borde torácico
    lat = str(laterality).upper()
    if "LEFT" in lat:
        is_outer = (nx < 0.50)  # Lóbulo lateral externo hacia la izquierda
    else:
        is_outer = (nx >= 0.50) # Lóbulo lateral externo hacia la derecha
        
    if is_upper and is_outer:
        return "Cuadrante Superior Externo (CSE)"
    elif is_upper and not is_outer:
        return "Cuadrante Superior Interno (CSI)"
    elif not is_upper and is_outer:
        return "Cuadrante Inferior Externo (CIE)"
    else:
        return "Cuadrante Inferior Interno (CII)"


def compute_compactness_and_eccentricity(mask: np.ndarray):
    """Calcula el área, perímetro, relación de compacidad (P^2 / 4*pi*A) y excentricidad."""
    m_bool = (mask > 0.5)
    area = int(m_bool.sum())
    if area == 0:
        return 0, 0.0, 1.0, 0.0
    
    # Gradientes morfológicos para perímetro
    eroded = (
        m_bool &
        np.pad(m_bool[1:, :], ((0, 1), (0, 0)), mode='constant') &
        np.pad(m_bool[:-1, :], ((1, 0), (0, 0)), mode='constant') &
        np.pad(m_bool[:, 1:], ((0, 0), (0, 1)), mode='constant') &
        np.pad(m_bool[:, :-1], ((0, 0), (1, 0)), mode='constant')
    )
    perimeter = int((m_bool ^ eroded).sum())
    perimeter = max(perimeter, 1)
    
    # Compacidad isoperimétrica: 1.0 para círculo, >1.5 para lesiones espiculadas/irregulares
    compactness = float((perimeter ** 2) / (4.0 * np.pi * area))
    
    # Momentos centrales para estimar excentricidad
    indices = np.argwhere(m_bool)
    cy, cx = indices[:, 0].mean(), indices[:, 1].mean()
    mu20 = np.mean((indices[:, 0] - cy) ** 2)
    mu02 = np.mean((indices[:, 1] - cx) ** 2)
    mu11 = np.mean((indices[:, 0] - cy) * (indices[:, 1] - cx))
    
    # Autovalores del tensor de inercia
    lambda1 = 0.5 * (mu20 + mu02 + np.sqrt((mu20 - mu02)**2 + 4 * mu11**2))
    lambda2 = 0.5 * (mu20 + mu02 - np.sqrt((mu20 - mu02)**2 + 4 * mu11**2))
    eccentricity = float(np.sqrt(max(0, 1.0 - lambda2 / (lambda1 + 1e-8)))) if lambda1 > 0 else 0.0

    return area, perimeter, compactness, eccentricity


class ClinicalEvidenceExtractor:
    """Extractor unificado que transforma una mamografía clínica en tokens de evidencia factual."""

    def __init__(self, codigos_dir: str = CODIGOS, device: str = DEVICE):
        self.codigos_dir = codigos_dir
        self.device = device

        # 1. Cargar modelo ResNet-18 Robusto
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(512, 2)
        ckpt_path = os.path.join(codigos_dir, "reports", "models", "resnet18_robust_best.pt")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model = self.model.to(device).eval()

        # 1b. Cargar modelo ResNet-18 Normal (Baseline CNN)
        self.model_base = models.resnet18(weights=None)
        self.model_base.fc = nn.Linear(512, 2)
        ckpt_base_path = os.path.join(codigos_dir, "reports", "models", "resnet18_baseline_v2.pt")
        if os.path.exists(ckpt_base_path):
            ckpt_b = torch.load(ckpt_base_path, map_location=device, weights_only=False)
            self.model_base.load_state_dict(ckpt_b["model_state_dict"])
        self.model_base = self.model_base.to(device).eval()

        # Resolución del localizador (la decisión se calcula siempre a 224 px). TESIS_RES_LOCALIZADOR=224
        # reproduce el localizador original de la tesis.
        self.res_localizador = int(os.environ.get("TESIS_RES_LOCALIZADOR", "448"))

        # Umbrales de Youden de las redes (validación); sin el archivo se usa 0,5 y se declara.
        self.thr_red = {}
        _yv = os.path.join(codigos_dir, "reports", "experiments", "youden_val_calibration.json")
        if os.path.exists(_yv):
            import json as _json_y
            with open(_yv, "r", encoding="utf-8") as _f:
                _d = _json_y.load(_f)
            self.thr_red = {k: float(_d[k]["thr_youden_val"]) for k in ("Baseline_FC", "Robusto") if k in _d}

        # 1c. (Retirado en la auditoría del 25-sep-2026) La estación NO evalúa una
        # cabeza KAN neural independiente: el checkpoint disponible
        # (resnet18_kan0_s42.pt) se entrenó sobre el backbone ImageNet congelado
        # con una reducción kan0 no conservada, mientras que la estación opera
        # sobre el backbone robusto. Alimentarlo con gap[:, :64] era inválido.
        # La comparación con la cabeza KAN vive en los notebooks (NB21/NB24).

        # 1d. Formula KAN robusta-64 canonica (R3, seed 42): S-PCA64 + KAN[64,32,2].
        # Evaluacion viva por caso para el informe LLM (objetivo de la tesis).
        # No toca mascaras/WSSS/psi (siguen en S-PCA12 NB30).
        self.kan64 = None
        try:
            _cj = os.path.join(codigos_dir, "reports", "experiments", "kan_robust64_canonical.json")
            _nz = os.path.join(codigos_dir, "reports", "experiments", "kan_robust64_reducer.npz")
            if os.path.exists(_cj) and os.path.exists(_nz):
                import json as _json
                with open(_cj, "r", encoding="utf-8") as _f:
                    _cd = _json.load(_f)
                _np = np.load(_nz)
                self.kan64 = {"meta": _cd,
                    "gap_mu": _np["gap_mu"].astype(np.float64),
                    "gap_sd": _np["gap_sd"].astype(np.float64),
                    "proj_w": _np["proj_w"].astype(np.float64),
                    "proj_b": _np["proj_b"].astype(np.float64),
                    "pca_comp": _np["pca_components"].astype(np.float64),
                    "pca_mean": _np["pca_mean"].astype(np.float64),
                    "mu_z": _np["mu_z"].astype(np.float64),
                    "sd_z": _np["sd_z"].astype(np.float64)}
        except Exception:
            self.kan64 = None

        # 2. Reconstruir proyector S-PCA analítico exacto (Notebook 30)
        emb_dir = _resolve_embeddings_dir()
        X_tr = np.load(os.path.join(emb_dir, "f2robust_train_feats.npy"))
        y_tr = np.load(os.path.join(emb_dir, "f2robust_train_labels.npy"))

        self.mu = X_tr.mean(0, keepdims=True)
        self.sd = X_tr.std(0, keepdims=True) + 1e-8
        X_tr_z = (X_tr - self.mu) / self.sd

        corrs = np.array([np.corrcoef(X_tr_z[:, i], y_tr)[0, 1] for i in range(X_tr_z.shape[1])])
        self.weights = np.abs(np.nan_to_num(corrs, 0.0))
        self.weights = self.weights / (self.weights.max() + 1e-8)

        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=12, random_state=42)
        self.pca.fit(X_tr_z * self.weights)

        # 2b. Dirección de decisión de la capa fc (logit = a . gap + b).
        #     Es la base de la atribución tipo Shapley: al ser lineal, el
        #     valor de Shapley de la dimensión c es a_c*(gap_c - E[gap_c]).
        W = self.model.fc.weight.detach().cpu().numpy()      # [2, 512]
        self.a_logit = (W[1] - W[0]).astype(np.float64)      # [512]
        # Media de la coordenada canónica en entrenamiento, para el centrado.
        # Se precalcula una sola vez: PCA.transform exige entrada 2-D.
        self._u_mean = self.pca.transform(
            (X_tr_z * self.weights).mean(axis=0, keepdims=True))[0]

        # Vector efectivo de gradiente para psi_1(u, v) (Severidad)
        v1 = self.pca.components_[1]
        self.w1_eff = (v1 * self.weights / self.sd).reshape(1, 512).astype(np.float32)
        self.w0_eff = self.w1_eff

        # Vector efectivo de gradiente para psi_2(u, v) (Márgenes / Espiculación)
        v2 = self.pca.components_[2]
        self.w2_eff = (v2 * self.weights / self.sd).reshape(1, 512).astype(np.float32)

        # 3. Inicializar Atribución S-PCA
        self.pca_cam = PCAGradCAM(self.model, self.model.layer4, device)

        # 4. Cargar metadatos enriquecidos (laterality, view, assessment)
        self._load_metadata_catalog()

    def _localization_map(self, act4) -> tuple:
        """Mapa para el localizador WSSS: el mapa de la fórmula psi = ReLU(sum_c (-W_c) A_c(u, v)).

        W es la dirección lineal de u0 sobre los 512 descriptores GAP (reductor congelado de la
        fórmula) y -W su dirección de malignidad, porque g'(u0) < 0 en el soporte observado.
        Se normaliza a [0, 1] y se lleva a 224 x 224 con interpolación bilineal. Es el respaldo
        cuando no hay imagen original para el mapa de 448 px. Si la fórmula no está disponible, usa la dirección
        de decisión de la red (w1 - w0) y lo declara en la fuente.

        Returns:
            (mapa [224, 224] en [0, 1], fuente: 'formula_psi' | 'red_w1_menos_w0')
        """
        A = act4[0].detach().double().cpu().numpy()                      # 512 x 7 x 7
        if self.kan64 is not None:
            K = self.kan64
            v = K["proj_w"].T @ K["pca_comp"][0]
            w = -(v / K["gap_sd"].ravel() / float(K["sd_z"].ravel()[0]))
            source = "formula_psi"
        else:
            fc = self.model.fc.weight.detach().double().cpu().numpy()
            w = fc[1] - fc[0]
            source = "red_w1_menos_w0"
        cam = np.maximum(np.tensordot(w, A, axes=(0, 0)), 0.0)
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros_like(cam)
        cam = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR),
                       dtype=np.float32) / 255.0
        return cam, source

    def _localization_map_hr(self, pil, tissue_mask, lado: int) -> tuple:
        """Mapa del localizador con la imagen a mayor resolución (Cap. 4, mapas de mayor resolución).

        La imagen original se lleva a lado x lado px (448 por defecto), el último bloque tiene entonces
        14 x 14 celdas y psi = ReLU(sum_c (-W_c) A_c) se calcula igual; se interpola a 224 x 224 y se
        normaliza a [0, 1] dentro del tejido. Es el mismo cálculo de scripts/segmentacion_resolucion.py y
        scripts/estudio_informes_llm.py. La decisión (u0, g, p_sym) se sigue calculando a 224 px.
        """
        import torchvision.transforms as _T
        norm = _T.Compose([_T.ToTensor(), _T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        K = self.kan64
        v = K["proj_w"].T @ K["pca_comp"][0]
        w = -(v / K["gap_sd"].ravel() / float(K["sd_z"].ravel()[0]))
        m = self.model
        with torch.no_grad():
            x = norm(pil.convert("RGB").resize((lado, lado), Image.BILINEAR)).unsqueeze(0).to(self.device)
            x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
            A = m.layer4(m.layer3(m.layer2(m.layer1(x))))[0]
        psi = np.maximum(np.tensordot(w, A.double().cpu().numpy(), axes=(0, 0)), 0.0)
        t = F.interpolate(torch.as_tensor(psi, dtype=torch.float32)[None, None], size=(224, 224),
                          mode="bilinear", align_corners=False)[0, 0].numpy()
        tm = np.asarray(tissue_mask) > 0.5
        vals = t[tm] if tm.any() else t.ravel()
        lo, hi = float(vals.min()), float(vals.max())
        mapa = np.zeros_like(t) if hi <= lo else ((t - lo) / (hi - lo)).clip(0, 1).astype(np.float32)
        return mapa, f"formula_psi_{lado}px"

    def compute_psi_fields(self, img_t, tissue_mask=None) -> dict:
        """Calcula los campos continuos psi_1(u,v) y psi_2(u,v) de la tesis.

        psi_j(u,v) = sum_c (v_{j,c} / sigma_c) * A_c(u,v)

        Se calculan con las cargas efectivas del proyector S-PCA (`w1_eff`, `w2_eff`)
        y se devuelven con signo, en un rango aproximado de [-1, 1], aptos para
        un mapa divergente. No intervienen en la segmentacion WSSS, que sigue
        usando el CAM con ReLU para no alterar los resultados validados.

        Args:
            img_t: torch.Tensor [1, 3, 224, 224] (o [3, 224, 224]) normalizado.
            tissue_mask: np.ndarray [H, W] opcional; si se entrega, el campo se
                enmascara al parenquima mamario para suprimir el aire de fondo.

        Returns:
            dict con 'psi1' y 'psi2' (campos firmados), mas 'psi1_tissue' y
            'psi2_tissue' cuando se entrega mascara de tejido.
        """
        out = {}
        if img_t.ndim == 3:
            img_t = img_t.unsqueeze(0)
        img_t = img_t.to(self.device)
        for name, wvec in (("psi1", self.w1_eff), ("psi2", self.w2_eff)):
            field = self.pca_cam.compute_psi_field(img_t, wvec, signed=True)
            out[name] = field
            if tissue_mask is not None:
                tm = np.asarray(tissue_mask, dtype=np.float32)
                if tm.shape != field.shape:
                    tm = np.array(
                        Image.fromarray((tm * 255).astype(np.uint8)).resize(
                            (field.shape[1], field.shape[0]), Image.BILINEAR
                        ),
                        dtype=np.float32,
                    ) / 255.0
                out[name + "_tissue"] = field * (tm > 0.5)
        return out

    def compute_attribution(self, gap: np.ndarray, z_coords: np.ndarray) -> dict:
        """Crédito decisional por componente canónico, al estilo Shapley.

        El logit de la CNN es exactamente lineal en el embedding GAP:

            s(gap) = a . gap + b,   a = w1 - w0

        Por tanto el valor de Shapley de cada dimension c es
        a_c * (gap_c - E[gap_c]), con forma cerrada exacta (ver
        `attribution_baselines.linear_shap_exact`). Agregando sobre la base
        S-PCA se obtiene el crédito de cada componente canónica:

            credito_j = beta_j * (u_j - E[u_j]),  beta_j = v_j . (sigma * a)

        Este bloque se incorpora al informe como tercer insumo explicativo,
        junto a la fórmula simbólica y a la atribución espacial.

        Args:
            gap: [512] embedding GAP de la imagen evaluada.
            z_coords: [12] coordenadas canónicas de la misma imagen.
        Returns:
            dict con 'componentes' (índice, beta, credito), 'beta' y 'top3'.
            No contiene etiquetas de referencia: soloGAP, pesos del modelo y
            estadísticos de entrenamiento.
        """
        if hasattr(gap, "detach"):          # torch.Tensor (puede estar en CUDA)
            gap = gap.detach().cpu().numpy()
        gap = np.asarray(gap, dtype=np.float64).ravel()
        z_coords = np.asarray(z_coords, dtype=np.float64).ravel()

        # beta_j: coeficiente de cada componente canonico en el logit
        beta = (self.pca.components_ * self.sd.ravel()) @ self.a_logit
        # Desviacion de la coordenada respecto a la media de entrenamiento
        u_mean = self._u_mean
        credito = beta * (z_coords - u_mean)

        orden = np.argsort(-np.abs(credito))
        comps = [{
            "indice": int(j),
            "beta": float(beta[j]),
            "credito": float(credito[j]),
            "coordenada": float(z_coords[j]),
        } for j in orden]
        return {
            "componentes": comps,
            "beta": [float(v) for v in beta],
            "top3": comps[:3],
            "nota": (
                "Crédito tipo Shapley sobre el logit; LIME reproduce esta "
                "jerarquía con rho = 0.9998 sobre 200 casos de prueba."
            ),
        }

    def _load_metadata_catalog(self):
        """Carga los archivos descriptivos para obtener laterality, view, assessment, etc."""
        mass_p = os.path.join(self.codigos_dir, "database", "archive", "csv", "mass_case_description_test_set.csv")
        calc_p = os.path.join(self.codigos_dir, "database", "archive", "csv", "calc_case_description_test_set.csv")
        dicom_p = os.path.join(self.codigos_dir, "database", "archive", "csv", "dicom_info.csv")
        dicom_info = pd.read_csv(dicom_p)
        dicom_info["area"] = dicom_info["Rows"] * dicom_info["Columns"]
        series_to_jpg = {}
        for uid, grp in dicom_info.groupby("SeriesInstanceUID"):
            uid_str = str(uid).strip()
            full_rows = grp[grp["SeriesDescription"] == "full mammogram images"]
            if len(full_rows) > 0:
                best = full_rows.iloc[0]
            else:
                best = grp.sort_values("area", ascending=False).iloc[0]
            series_to_jpg[uid_str] = str(best["image_path"]).replace("CBIS-DDSM/", "").replace("/", "\\")

        self.meta_by_relpath = {}
        self.meta_by_patient_id = {}
        for p, ab in [(mass_p, "mass"), (calc_p, "calc")]:
            if os.path.exists(p):
                df = pd.read_csv(p)
                for _, r in df.iterrows():
                    p_id = str(r.get("patient_id", "")).strip()
                    s_img = extract_series_uid(r["image file path"])
                    meta_entry = {
                        "laterality": str(r.get("left or right breast", "UNKNOWN")).strip(),
                        "view": str(r.get("image view", "UNKNOWN")).strip(),
                        "assessment": int(r.get("assessment", 0)) if pd.notna(r.get("assessment", None)) else 0,
                        "pathology": str(r.get("pathology", "UNKNOWN")).strip(),
                        "abnormality_type": ab,
                        "subtlety": r.get("subtlety", 0),
                        "mass_shape": str(r.get("mass shape", "N/A")).strip() if ab == "mass" else "N/A",
                        "mass_margins": str(r.get("mass margins", "N/A")).strip() if ab == "mass" else "N/A",
                        "calc_type": str(r.get("calc type", "N/A")).strip() if ab == "calc" else "N/A",
                        "calc_distribution": str(r.get("calc distribution", "N/A")).strip() if ab == "calc" else "N/A"
                    }
                    if p_id:
                        self.meta_by_patient_id[p_id] = meta_entry
                    if s_img in series_to_jpg:
                        rel = os.path.join("database", "archive", series_to_jpg[s_img])
                        self.meta_by_relpath[rel] = meta_entry
                        self.meta_by_relpath[os.path.normpath(rel)] = meta_entry

    def evaluate_kan_formula(self, delta_logit: float) -> float:
        """Enlace logístico (softmax de dos clases) de la cabeza lineal, en forma cerrada.

        IMPORTANTE: esta función NO evalúa la fórmula KAN destilada del Notebook 27.
        Evalúa escalarmente la probabilidad del clasificador lineal de la red robusta:
            F(s) = 0.5 * (1 + tanh(s / 2)) == sigmoid(s) == Softmax([l0, l1])[maligno]
        La fórmula simbólica KAN (univariada sobre x_1) se reporta por separado en
        los artefactos del Notebook 27 (reports/27_spca_robust_comparison.json).
        """
        score = float(0.5 * (1.0 + np.tanh(delta_logit / 2.0)))
        return float(np.clip(score, 0.0, 1.0))

    def evaluate_kan64(self, gap_np) -> dict | None:
        """Evalua la formula KAN robusta-64 canonica sobre el GAP del caso.

        gap_np: [512] o [1,512] (robusto). Devuelve fben/fmal/g/p_sym/pred
        con umbral Youden de validacion. Sin etiquetas: solo GAP + pesos
        congelados + formulas del artefacto canonico.
        """
        if self.kan64 is None:
            return None
        try:
            import re as _re
            import sympy as _sp
            g = np.asarray(gap_np, dtype=np.float64).ravel()
            if g.size != 512:
                return None
            K = self.kan64
            gz = (g.reshape(1, -1) - K["gap_mu"]) / K["gap_sd"]
            zp = gz @ K["proj_w"].T + K["proj_b"]
            zc = (zp - K["pca_mean"]) @ K["pca_comp"].T
            z64 = ((zc - K["mu_z"]) / K["sd_z"]).ravel()
            meta = K["meta"]
            fben_s, fmal_s = meta["fben"], meta["fmal"]
            used = sorted({int(m) for m in _re.findall(r"x_(\d+)", fmal_s + fben_s)})
            e_ben, e_mal = _sp.sympify(fben_s), _sp.sympify(fmal_s)
            syms = [_sp.Symbol(f"x_{i}") for i in used]
            fn_b = _sp.lambdify(syms, e_ben, "numpy")
            fn_m = _sp.lambdify(syms, e_mal, "numpy")
            vals = [float(np.clip(z64[i - 1], -10.0, 10.0)) for i in used]
            fben = float(fn_b(*vals)); fmal = float(fn_m(*vals))
            gg = float(fmal - fben)
            p_sym = float(1.0 / (1.0 + np.exp(-gg)))
            th = float(meta.get("th_youden_val", 0.0))
            pred = "MALIGNANT" if gg >= th else "BENIGN"
            # gradiente numerico dg/du sobre coords activas (coherencia LIME/SHAP)
            grad = {}
            eps = 1e-3
            for i in used:
                vp = list(vals); vm = list(vals)
                k = used.index(i)
                vp[k] += eps; vm[k] -= eps
                gp = float(fn_m(*vp) - fn_b(*vp)); gm = float(fn_m(*vm) - fn_b(*vm))
                grad[f"u{i-1}"] = float((gp - gm) / (2 * eps))
            # direccion efectiva a GAP-512 (puente espacial psi_KAN y top canales)
            dgap_top3 = []
            dgap_full = None
            try:
                pc0 = K["pca_comp"][used[0] - 1].astype(np.float64)
                sd0 = float(K["sd_z"].ravel()[used[0] - 1])
                _dg = ((pc0 / sd0) @ K["proj_w"].astype(np.float64)) / K["gap_sd"].ravel().astype(np.float64)
                _dg = _dg * float(grad.get(f"u{used[0]-1}", 0.0))
                dgap_full = _dg.astype(np.float64)
                _top = np.argsort(-np.abs(_dg))[:3]
                dgap_top3 = [{"canal": int(j), "dg_dgap": float(_dg[j])} for j in _top]
            except Exception:
                pass
            # descomposicion termino a termino (sumandos de fmal/fben en el caso)
            terms = []
            try:
                for _e, _fn, _sgn in ((e_mal, fn_m, +1.0), (e_ben, fn_b, -1.0)):
                    for _t in (_e.as_ordered_terms() if hasattr(_e, "as_ordered_terms") else [_e]):
                        _f = _sp.lambdify(syms, _t, "numpy")
                        _v = float(_f(*vals)) * _sgn
                        terms.append({"texto": str(_t), "contribucion": _v})
                terms.sort(key=lambda t: abs(t["contribucion"]), reverse=True)
            except Exception:
                pass
            # Categoría orientativa anclada en la decisión (fuente única: birads_herramienta.py)
            _bkan = _birads.etiqueta(gg, th)
            return {"z64_0": float(z64[0]), "fben": fben, "fmal": fmal,
                "g": gg, "p_sym": p_sym, "th_youden_val": th,
                "pred": pred, "vars": used, "grad": grad,
                "dgap": (dgap_full.tolist() if dgap_full is not None else None),
                "dgap_top3": dgap_top3, "terms": terms[:6],
                "birads_kan": _bkan,
                "AUC_D": meta.get("AUC_D"), "n_vars": meta.get("n_vars"),
                "chars": meta.get("chars")}
        except Exception:
            return None

    def extract_evidence(self, item: dict, rel_img_path: str = None) -> dict:
        img_t = item["image_tensor"]
        if img_t.ndim == 3:
            img_t = img_t.unsqueeze(0)
        img_t = img_t.to(self.device)
        gt_mask = item["mask_np"]
        patient_id = item["patient_id"]
        pathology_gt = item["pathology"]
        ab_type = item["abnormality_type"]

        # Obtener metadatos clínicos adicionales
        meta = self.meta_by_relpath.get(rel_img_path, {})
        if not meta and rel_img_path:
            meta = self.meta_by_relpath.get(os.path.normpath(rel_img_path), {})
        if not meta:
            meta = self.meta_by_patient_id.get(patient_id, {})

        laterality = meta.get("laterality", "LEFT")
        view = meta.get("view", "MLO")
        # assessment_gt SOLO proviene de los metadatos oficiales; 0 = desconocido.
        # (Auditoría 25-sep-2026: antes se infería de la etiqueta patológica cuando
        # faltaba el metadato, lo que constituía una fuga de etiqueta latente.)
        try:
            assessment_gt = int(meta.get("assessment", 0) or 0)
        except (TypeError, ValueError):
            assessment_gt = 0
        lesion_shape = meta.get("mass_shape", meta.get("calc_type", "N/A"))
        lesion_margins = meta.get("mass_margins", meta.get("calc_distribution", "N/A"))

        # --- MODELO 1: ResNet-18 Normal (CNN Clásica End-to-End) ---
        t0 = time.perf_counter()
        with torch.no_grad():
            out_base = self.model_base(img_t)
            prob_resnet = float(F.softmax(out_base, dim=1)[0, 1].item())
        t_resnet_ms = float((time.perf_counter() - t0) * 1000.0)
        # Umbral de Youden de cada red, fijado en validación (el mismo con que la tesis las reporta)
        pred_resnet = "MALIGNANT" if prob_resnet >= self.thr_red.get("Baseline_FC", 0.5) else "BENIGN"

        # --- Extracción de Características Robusta (GAP) ---
        with torch.no_grad():
            x = self.model.conv1(img_t)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            act4 = self.model.layer4(x)
            gap = F.adaptive_avg_pool2d(act4, (1, 1)).flatten(1)
            logits_rob = self.model.fc(gap)
            delta_logit = float((logits_rob[0, 1] - logits_rob[0, 0]).item())
            prob_cnn_rob = float(F.softmax(logits_rob, dim=1)[0, 1].item())

        # La estación evalúa dos modelos: la CNN robusta y su cabeza lineal en
        # forma analítica. No existe una tercera evaluación KAN independiente.

        # --- MODELO 3: ResNet-18 + Cabeza Lineal (enlace analítico σ(s) = ½(1+tanh(s/2))) ---
        # La clave interna se llama "kan_score" por herencia, pero NO es una fórmula KAN
        # destilada: es el softmax en forma cerrada de una cabeza lineal de dos clases.
        t0 = time.perf_counter()
        score_kan = float(0.5 * (1.0 + np.tanh(delta_logit / 2.0)))
        t_kan_symb_ms = float((time.perf_counter() - t0) * 1000.0)
        pred_kan = "MALIGNANT" if score_kan >= self.thr_red.get("Robusto", 0.5) else "BENIGN"
        prob_kan_symb = score_kan
        pred_kan_symb = pred_kan
        prob_cnn = score_kan

        # --- KAN robusta-64 canonica (R3): evaluacion viva de la formula D ---
        kan64_res = None
        try:
            kan64_res = self.evaluate_kan64(gap.cpu().numpy())
        except Exception:
            kan64_res = None

        z = gap.cpu().numpy()
        z_norm = (z - self.mu) / self.sd

        # 2. Proyección y Evaluación Simbólica S-PCA (Notebook 30)
        z_weighted = z_norm * self.weights
        z_coords = self.pca.transform(z_weighted)[0]
        x0_val = float(z_coords[0])
        x1_val = float(z_coords[1])
        x2_val = float(z_coords[2])

        # 2.1. Lectura de la componente auxiliar x_2 (asociada, estadísticamente, al margen de la lesión).
        #      Es una asociación a nivel de cohorte (Cap. 4, asociación con BI-RADS), no una medición del margen.
        if x2_val <= -0.30:
            margin_status = "COMPATIBLE CON MARGEN CIRCUNSCRITO"
            margin_desc = f"x₂ = {x2_val:+.3f}: valor bajo, más frecuente en lesiones de contorno circunscrito (asociación estadística; no mide el margen de esta lesión)."
        elif x2_val <= 0.35:
            margin_status = "INTERMEDIO / INDETERMINADO"
            margin_desc = f"x₂ = {x2_val:+.3f}: zona intermedia, sin tendencia clara hacia margen circunscrito o espiculado."
        else:
            margin_status = "COMPATIBLE CON MARGEN ESPICULADO"
            margin_desc = f"x₂ = {x2_val:+.3f}: valor alto, más frecuente en lesiones de margen espiculado (asociación estadística; no demuestra espiculación en este caso)."

        # 2.2. Lectura de la componente auxiliar x_1 (asociada, estadísticamente, a la severidad histopatológica).
        if x1_val <= -0.35:
            density_status = "VALOR BAJO DE x₁"
            density_desc = f"x₁ = {x1_val:+.3f}: valor bajo; en la cohorte se asocia con lesiones de patrón más homogéneo (asociación estadística)."
        elif x1_val <= 0.25:
            density_status = "VALOR INTERMEDIO DE x₁"
            density_desc = f"x₁ = {x1_val:+.3f}: zona intermedia de la componente."
        else:
            density_status = "VALOR ALTO DE x₁"
            density_desc = f"x₁ = {x1_val:+.3f}: valor alto; en la cohorte se asocia con lesiones de mayor sospecha (asociación estadística)."

        # 3. Máscara de Parénquima Mamario y Segmentación Débil WSSS de Núcleo Focal (Amarillo+)
        img_np_raw = item["image_tensor"].cpu().numpy()
        if img_np_raw.ndim == 3:
            img_np_raw = img_np_raw[0] if img_np_raw.shape[0] in [1, 3] else img_np_raw[:, :, 0]
        img_disp_raw = (img_np_raw - img_np_raw.min()) / (img_np_raw.max() - img_np_raw.min() + 1e-8)

        tissue_mask = extract_breast_tissue_mask(img_disp_raw)
        # Localizador: núcleo focal (>= 60 %) sobre el mapa de la fórmula psi = ReLU(sum_c (-W_c) A_c),
        # el mismo que evalúa la tesis (Cap. 4, localización) y que usa el estudio con modelos de lenguaje
        # (scripts/estudio_informes_llm.py). Antes se usaba una componente S-PCA preliminar.
        # Con la imagen original disponible, el mapa se calcula a RES_LOCALIZADOR px (448 por defecto):
        # a esa resolución el núcleo focal acierta la región más que el centro del tejido (Cap. 4).
        loc_map, localizer_source = None, None
        if self.res_localizador != 224 and self.kan64 is not None:
            pil = item.get("image_pil")
            if pil is None and item.get("image_path"):
                try:
                    pil = Image.open(item["image_path"]).convert("RGB")
                except Exception:
                    pil = None
            if pil is not None:
                try:
                    loc_map, localizer_source = self._localization_map_hr(pil, tissue_mask, self.res_localizador)
                except Exception:
                    loc_map = None
        if loc_map is None:
            loc_map, localizer_source = self._localization_map(act4)
        mask_pred, cam_tissue, peak_loc = segment_wsss_focal_core(loc_map, tissue_mask, min_tau_ratio=0.60)

        # 4. Centroides y Cuadrantes
        cy_pred, cx_pred = compute_centroid(mask_pred)
        cy_gt, cx_gt = compute_centroid(gt_mask)
        dist_centroid = float(np.sqrt((cy_pred - cy_gt)**2 + (cx_pred - cx_gt)**2))

        quadrant_pred = determine_quadrant(cy_pred, cx_pred, laterality, view)
        quadrant_gt = determine_quadrant(cy_gt, cx_gt, laterality, view)
        is_quadrant_exact = (quadrant_pred == quadrant_gt)

        # 4b. Reconstruccion GradCAM desde la formula (psi_KAN): el segundo insumo.
        # psi_KAN(u,v) = sum_c dg/dgap_c * A_c(u,v), con dg/dgap de la formula D.
        # Sin etiquetas: solo GAP del caso + pesos congelados + activaciones propias.
        psi_kan_map = None; psi_kan_peak = (0.0, 0.0); psi_kan_quad = "N/A"
        psi_kan_agrees = False; psi_kan_ebpg = 0.0; psi_kan_hit = 0.0
        try:
            _dg = (kan64_res or {}).get("dgap")
            if _dg is not None:
                from src.explainability.pca_attribution import evaluate_pointing_game
                _w = np.asarray(_dg, dtype=np.float32)
                _n = float(np.abs(_w).max())
                if _n > 1e-12:
                    _w = _w / _n
                psi_kan_map = self.pca_cam.compute_psi_field(img_t, _w, signed=False)
                if str(localizer_source).endswith("px"):
                    # Con el localizador a mayor resolución, la capa del mapa de la fórmula muestra ese
                    # mismo mapa (psi a 448 px, llevado a 224): el mapa, el foco y la región del informe
                    # salen del mismo cálculo. Es la misma dirección (-W) que dg/dgap, porque g' < 0.
                    psi_kan_map = np.asarray(loc_map, dtype=np.float32)
                _py, _px = (float(v) for v in np.unravel_index(np.argmax(psi_kan_map), psi_kan_map.shape))
                psi_kan_peak = (_py, _px)
                psi_kan_quad = determine_quadrant(_py, _px, laterality, view,
                    h=psi_kan_map.shape[0], w=psi_kan_map.shape[1])
                psi_kan_agrees = bool(psi_kan_quad == quadrant_pred)
                _pg = evaluate_pointing_game(
                    np.asarray(psi_kan_map, dtype=np.float32),
                    np.asarray(gt_mask > 0.5, dtype=np.float32))
                psi_kan_ebpg = float(_pg["ebpg"]); psi_kan_hit = float(_pg["hit"])
        except Exception:
            pass

        # 5. Geometría y Morfología
        pred_area, pred_perim, pred_compact, pred_ecc = compute_compactness_and_eccentricity(mask_pred)
        gt_area, _, _, _ = compute_compactness_and_eccentricity(gt_mask)

        # Diámetro equivalente estimado en píxeles
        equiv_diameter_px = float(2.0 * np.sqrt(pred_area / np.pi)) if pred_area > 0 else 0.0

        metrics = compute_segmentation_metrics(mask_pred, gt_mask)

        # 6. Categoría BI-RADS orientativa de la herramienta, derivada de la fórmula y anclada en su
        #    decisión (fuente única: src/reporting/birads_herramienta.py). Sustituye al mapeo heredado
        #    sobre la probabilidad de la red y la variable x2, que no coincidía con la decisión.
        if kan64_res is not None and kan64_res.get("g") is not None:
            birads_suggested = _birads.etiqueta(kan64_res["g"], kan64_res["th_youden_val"])
        else:
            birads_suggested = "N/D (falta la fórmula canónica)"
        recommendation = _birads.conducta(birads_suggested)

        models_comp = {
            "resnet_normal": {
                "name": "ResNet-18 Normal",
                "type": "CNN Convolucional Clásica (Caja Negra)",
                "prediction": pred_resnet,
                "pred_es": "Maligno" if pred_resnet == "MALIGNANT" else "Benigno",
                "probability": prob_resnet,
                "threshold": self.thr_red.get("Baseline_FC", 0.5),
                "latency_ms": round(t_resnet_ms, 2)
            },
            "kan_symbolic": {
                "name": "ResNet-18 + Cabeza Lineal (enlace analítico)",
                "type": "Probabilidad softmax en forma cerrada F(s) = ½(1+tanh(s/2)) [no es la fórmula KAN destilada]",
                "prediction": pred_kan_symb,
                "pred_es": "Maligno" if pred_kan_symb == "MALIGNANT" else "Benigno",
                "probability": prob_kan_symb,
                "threshold": self.thr_red.get("Robusto", 0.5),
                "latency_ms": round(t_kan_symb_ms, 4)
            },
            "consensus_prediction": pred_kan_symb,
            "consensus_es": "Maligno" if pred_kan_symb == "MALIGNANT" else "Benigno",
            "birads_suggested": birads_suggested
        }

        return {
            "patient_id": patient_id,
            "laterality": laterality,
            "view": view,
            "abnormality_type": ab_type,
            "pathology_gt": pathology_gt,
            "assessment_gt": assessment_gt,
            "lesion_shape": lesion_shape,
            "lesion_margins": lesion_margins,
            "prob_cnn": prob_cnn,
            "delta_logit": delta_logit,
            "prob_resnet": prob_resnet,
            "pred_resnet": pred_resnet,
            "time_resnet_ms": round(t_resnet_ms, 2),
            "prob_kan_symb": prob_kan_symb,
            "pred_kan_symb": pred_kan_symb,
            "time_kan_symb_ms": round(t_kan_symb_ms, 4),
            "models_comparison": models_comp,
            "attribution": self.compute_attribution(gap, z_coords),
            "x0_canonical": x0_val,
            "x1_canonical": x1_val,
            "x2_canonical": x2_val,
            "margin_status": margin_status,
            "margin_desc": margin_desc,
            "density_status": density_status,
            "density_desc": density_desc,
            "kan_score": score_kan,
            "score_kan": score_kan,
            "kan_prediction": pred_kan,
            "kan64_g": (kan64_res or {}).get("g"),
            "kan64_p": (kan64_res or {}).get("p_sym"),
            "kan64_pred": (kan64_res or {}).get("pred"),
            "kan64_z64_0": (kan64_res or {}).get("z64_0"),
            "kan64_fben": (kan64_res or {}).get("fben"),
            "kan64_fmal": (kan64_res or {}).get("fmal"),
            "kan64_th": (kan64_res or {}).get("th_youden_val"),
            "kan64_vars": (kan64_res or {}).get("vars"),
            "kan64_grad": (kan64_res or {}).get("grad"),
            "kan64_dgap_top3": (kan64_res or {}).get("dgap_top3"),
            "kan64_terms": (kan64_res or {}).get("terms"),
            "kan64_birads": (kan64_res or {}).get("birads_kan"),
            "kan64_AUC_D": (kan64_res or {}).get("AUC_D"),
            "psi_kan_quad": psi_kan_quad,
            "psi_kan_agrees": bool(psi_kan_agrees),
            "psi_kan_peak": (float(psi_kan_peak[0]), float(psi_kan_peak[1])),
            "psi_kan_ebpg": float(psi_kan_ebpg),
            "psi_kan_hit": float(psi_kan_hit),
            "psi_kan_map": psi_kan_map,
            "birads_suggested": birads_suggested,
            "clinical_recommendation": recommendation,
            "predicted_quadrant": quadrant_pred,
            "localizer_source": localizer_source,
            "gt_quadrant": quadrant_gt,
            "is_quadrant_exact": is_quadrant_exact,
            "predicted_centroid": (cy_pred, cx_pred),
            "gt_centroid": (cy_gt, cx_gt),
            "centroid_dist_px": dist_centroid,
            "pred_area_px": pred_area,
            "gt_area_px": gt_area,
            "equiv_diameter_px": equiv_diameter_px,
            "compactness": pred_compact,
            "eccentricity": pred_ecc,
            "dice": metrics["dice"],
            "iou": metrics["iou"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "cam_map": cam_tissue,
            "mask_pred": mask_pred,
            "tissue_mask": tissue_mask,
            "has_gt_roi": bool(gt_mask is not None and gt_mask.sum() > 0)
        }
