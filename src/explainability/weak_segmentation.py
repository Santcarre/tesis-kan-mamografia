"""
weak_segmentation.py: Módulo de segmentación débilmente supervisada (WSSS)
a partir de mapas de atribución continua (Grad-CAM y S-PCA).

Permite derivar máscaras binarias del contorno tumoral sin requerir
anotaciones a nivel de píxel durante el entrenamiento, evaluando
múltiples estrategias de umbralización y refinamiento morfológico.
"""
import numpy as np
import scipy.ndimage as ndi


def compute_otsu_threshold(heatmap: np.ndarray, min_val: float = 0.05) -> float:
    """Calcula el umbral óptimo de Otsu sobre un mapa continuo en [0, 1].

    Considera solo píxeles con activación mayor a min_val para evitar
    que el fondo negro de la mamografía sesgue el umbral.
    """
    flat = (heatmap * 255.0).astype(np.uint8).flatten()
    valid = flat[flat > int(min_val * 255)]
    if len(valid) < 50:
        return 0.5

    hist, _ = np.histogram(valid, bins=256, range=(0, 256))
    total = valid.size
    sum_total = np.dot(np.arange(256), hist)

    current_max = 0.0
    threshold = 128
    sum_b = 0.0
    w_b = 0.0

    for i in range(256):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += i * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        between_var = w_b * w_f * ((m_b - m_f) ** 2)
        if between_var > current_max:
            current_max = between_var
            threshold = i

    return float(threshold / 255.0)


def compute_energy_quantile_threshold(heatmap: np.ndarray, q: float = 0.30) -> float:
    """Determina el umbral tau tal que los píxeles seleccionados acumulan el
    top q (e.g. 30%) de la energía de activación total.
    """
    total_energy = heatmap.sum()
    if total_energy <= 1e-8:
        return 0.5
    
    # Ordenar píxeles de mayor a menor
    sorted_vals = np.sort(heatmap.flatten())[::-1]
    cum_energy = np.cumsum(sorted_vals)
    target = total_energy * q
    idx = np.searchsorted(cum_energy, target)
    idx = min(idx, len(sorted_vals) - 1)
    return float(sorted_vals[idx])


def segment_fixed_threshold(heatmap: np.ndarray, tau: float = 0.5) -> np.ndarray:
    """Binarización por umbral fijo."""
    return (heatmap >= tau).astype(np.float32)


def segment_otsu(heatmap: np.ndarray, min_val: float = 0.05) -> np.ndarray:
    """Binarización por umbralización adaptativa de Otsu."""
    tau = compute_otsu_threshold(heatmap, min_val=min_val)
    return (heatmap >= tau).astype(np.float32)


def segment_energy_quantile(heatmap: np.ndarray, q: float = 0.30) -> np.ndarray:
    """Binarización por cuantil de energía acumulada."""
    tau = compute_energy_quantile_threshold(heatmap, q=q)
    return (heatmap >= tau).astype(np.float32)


def segment_morphology_refined(
    heatmap: np.ndarray,
    tau: float = None,
    use_otsu: bool = True,
    peak_coord: tuple = None,
    open_size: int = 3,
    close_size: int = 7
) -> np.ndarray:
    """Binarización adaptativa seguida de refinamiento morfológico:
    1. Apertura (elimina ruido aislado)
    2. Filtrado de componente conexo (selecciona el componente del pico o el mayor)
    3. Cierre y relleno de huecos (unifica la lesión)
    """
    if tau is None:
        tau = compute_otsu_threshold(heatmap) if use_otsu else 0.5

    raw_bin = (heatmap >= tau).astype(bool)
    if raw_bin.sum() == 0:
        return np.zeros_like(heatmap, dtype=np.float32)

    # 1. Apertura morfológica
    struct_open = ndi.generate_binary_structure(2, 1)
    if open_size > 3:
        struct_open = np.ones((open_size, open_size), dtype=bool)
    opened = ndi.binary_opening(raw_bin, structure=struct_open)
    if opened.sum() == 0:
        opened = raw_bin  # fallback si apertura borra todo

    # 2. Etiquetado de componentes conexos
    labeled, n_features = ndi.label(opened)
    if n_features == 0:
        return opened.astype(np.float32)

    # Si se provee la coordenada del pico y cae en un componente, seleccionarlo
    if peak_coord is not None and labeled[peak_coord[0], peak_coord[1]] > 0:
        selected_label = labeled[peak_coord[0], peak_coord[1]]
        best_mask = (labeled == selected_label)
    else:
        # Si no, seleccionar el componente que maximiza la suma de calor de activación
        comp_energies = [heatmap[labeled == l].sum() for l in range(1, n_features + 1)]
        best_label = int(np.argmax(comp_energies)) + 1
        best_mask = (labeled == best_label)

    # 3. Cierre morfológico y relleno de huecos
    struct_close = np.ones((close_size, close_size), dtype=bool)
    closed = ndi.binary_closing(best_mask, structure=struct_close)
    filled = ndi.binary_fill_holes(closed)

    return filled.astype(np.float32)


def compute_centroid(mask: np.ndarray):
    """Calcula las coordenadas (y, x) del centroide de una máscara binaria."""
    if mask.sum() == 0:
        return (mask.shape[0] / 2.0, mask.shape[1] / 2.0)
    indices = np.argwhere(mask > 0.5)
    return (float(indices[:, 0].mean()), float(indices[:, 1].mean()))


def compute_segmentation_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """Calcula las métricas estándar de segmentación clínica:
    - Dice Similarity Coefficient (DSC)
    - Intersection over Union (IoU / Jaccard)
    - Precision (Valor predictivo positivo)
    - Recall (Sensibilidad de detección del área tumoral)
    - Centroid Distance (Distancia euclidiana entre centroides en píxeles)
    """
    pred = (pred_mask > 0.5).astype(bool)
    gt = (gt_mask > 0.5).astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    sum_areas = pred.sum() + gt.sum()

    iou = float(intersection / union) if union > 0 else 0.0
    dice = float(2.0 * intersection / sum_areas) if sum_areas > 0 else 0.0
    precision = float(intersection / pred.sum()) if pred.sum() > 0 else 0.0
    recall = float(intersection / gt.sum()) if gt.sum() > 0 else 0.0

    # Distancia de centroides
    cy_pred, cx_pred = compute_centroid(pred)
    cy_gt, cx_gt = compute_centroid(gt)
    centroid_dist = float(np.sqrt((cy_pred - cy_gt) ** 2 + (cx_pred - cx_gt) ** 2))

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "centroid_dist": centroid_dist,
        "pred_area": int(pred.sum()),
        "gt_area": int(gt.sum())
    }


def extract_breast_tissue_mask(img_disp: np.ndarray, min_threshold: float = 0.06) -> np.ndarray:
    """Extrae la silueta anatómica sólida del parénquima mamario, eliminando el fondo de aire.

    1. Binariza por encima del umbral de aire (I > min_threshold).
    2. Aplica cierre morfológico y relleno de huecos para consolidar el tejido.
    3. Conserva únicamente la componente conexa más grande (la mama), descartando chasis o etiquetas.
    """
    raw_tissue = (img_disp > min_threshold).astype(bool)
    struct = np.ones((7, 7), dtype=bool)
    closed = ndi.binary_closing(raw_tissue, structure=struct)
    filled = ndi.binary_fill_holes(closed)

    labeled, n = ndi.label(filled)
    if n > 0:
        sizes = [np.sum(labeled == l) for l in range(1, n + 1)]
        largest_label = np.argmax(sizes) + 1
        tissue_mask = (labeled == largest_label).astype(np.float32)
    else:
        tissue_mask = np.ones_like(img_disp, dtype=np.float32)

    return tissue_mask


def segment_wsss_focal_core(
    cam_map: np.ndarray,
    tissue_mask: np.ndarray = None,
    min_tau_ratio: float = 0.60
) -> tuple:
    """Delimita el núcleo focal de la lesión a partir del umbral en tonalidad amarilla (≥ 60%).

    1. Enmascara la saliencia espectral con el parénquima mamario (fondo negro a cero absoluto).
    2. Fija el umbral adaptativo en la tonalidad amarilla-roja (evitando el halo difuso verde).
    3. Realiza apertura y cierre morfológico centrado en el pico de activación de tejido.

    Retorna: (mask_core, cam_tissue, peak_coord)
    """
    if tissue_mask is not None:
        cam_tissue = cam_map * tissue_mask
    else:
        cam_tissue = cam_map.copy()

    max_act = float(cam_tissue.max())
    if max_act < 1e-4:
        return np.zeros_like(cam_map, dtype=np.float32), cam_tissue, (0, 0)

    # Identificar píxeles activos en tejido
    if tissue_mask is not None and tissue_mask.sum() > 0:
        tissue_vals = cam_tissue[tissue_mask > 0.5]
    else:
        tissue_vals = cam_tissue[cam_tissue > 0.05]

    if len(tissue_vals) > 0:
        tau_pct = float(np.percentile(tissue_vals, 85.0))
        tau = max(min_tau_ratio * max_act, tau_pct)
    else:
        tau = min_tau_ratio * max_act

    peak_coord = np.unravel_index(np.argmax(cam_tissue), cam_tissue.shape)

    mask_core = segment_morphology_refined(
        cam_tissue,
        tau=tau,
        use_otsu=False,
        peak_coord=peak_coord,
        open_size=3,
        close_size=5
    )

    return mask_core, cam_tissue, peak_coord

