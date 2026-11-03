"""
Análisis simbólico de las aristas KAN:
    - Criterio Sproud para identificar aristas más relevantes
    - Ajuste simbólico automático (suggest_symbolic / fix_symbolic)
    - Visualización de funciones spline univariadas

Resultados de referencia (cuaderno 08, backbone ResNet-18):
    Las dimensiones latentes {9, 13, 57} concentran mayor variación relevante.
    Funciones predominantes: sin, cos, identidad (todas con R² > 0.99).
"""
import numpy as np
import torch


def compute_sproud_importance(kan_model, n_top: int = 12) -> list[tuple]:
    """Calcula importancia Sproud de las aristas de la capa 0 de la KAN.

    Sproud = media del valor absoluto de las activaciones post-spline.
    Ordena descendentemente y devuelve los n_top pares (i, j) más importantes.

    Args:
        kan_model: Modelo KAN (pykan) con activaciones cacheadas tras un forward.
        n_top: Número de aristas top a retornar. Default 12.

    Returns:
        Lista de tuplas (importancia, i, j) ordenada descendentemente.
    """
    # Acceder a las activaciones de la capa 0
    try:
        acts = kan_model.acts[0]  # [B, in_dim, out_dim] o similar
    except (AttributeError, IndexError):
        raise RuntimeError(
            "Ejecuta un forward pass con cache_activations=True antes de llamar esta función."
        )

    # Importancia = media absoluta sobre el batch
    importance = acts.abs().mean(dim=0)  # [in_dim, out_dim]
    results = []
    for i in range(importance.shape[0]):
        for j in range(importance.shape[1]):
            results.append((importance[i, j].item(), i, j))

    results.sort(reverse=True)
    return results[:n_top]


def run_symbolic_fit(
    kan_model,
    top_edges: list[tuple],
    lib: list[str] = None,
) -> list[dict]:
    """Ajuste simbólico automático sobre las aristas prioritarias.

    Para cada arista (i, j), sugiere la función simbólica del catálogo
    que mejor ajusta el spline aprendido (según R²).

    Args:
        kan_model: Modelo KAN (pykan).
        top_edges: Lista de (importancia, i, j) de compute_sproud_importance.
        lib: Catálogo de funciones. Default: ['x', 'sin', 'cos', 'exp', 'log'].

    Returns:
        Lista de dicts con {i, j, importance, func_name, r2}.
    """
    if lib is None:
        lib = ['x', 'sin', 'cos', 'exp', 'log']

    results = []
    for importance, i, j in top_edges:
        try:
            kan_model.suggest_symbolic(0, i, j, lib=lib, verbose=False)
            func_name, r2 = kan_model.fix_symbolic(0, i, j, lib=lib)
            results.append({'i': i, 'j': j, 'importance': importance,
                            'func_name': func_name, 'r2': r2})
        except Exception as e:
            results.append({'i': i, 'j': j, 'importance': importance,
                            'func_name': 'ERROR', 'r2': 0.0, 'error': str(e)})
    return results


def prune_to_sparse(kan_model, threshold: float = 1e-3, frac: float = 0.99):
    """Poda aristas de la KAN conservando la contribución dominante.

    Combina dos criterios:
        - Umbral relativo (threshold * max_abs): elimina aristas casi nulas.
        - Fracción de energía (frac): conserva las aristas de mayor magnitud
          que acumulan `frac` de la activación post-spline media total. Esto
          garantiza una fórmula corta aunque la L1 no haya colapsado todo.

    Args:
        kan_model: KAN (pykan) con activaciones cacheadas (hacer forward antes).
        threshold: Umbral relativo al máximo para considerar una arista nula.
        frac: Fracción de energía a conservar (default 0.99).

    Returns:
        dict con 'n_edges_total', 'n_edges_kept' y 'n_edges_zero'.
    """
    if not hasattr(kan_model, 'spline_postacts'):
        raise RuntimeError("Necesita forward cacheado antes de podar (kan_model(x))")
    total_edges = 0
    kept = 0
    for l in range(len(kan_model.spline_postacts)):
        acts = kan_model.spline_postacts[l]          # [B, out, in]
        if acts is None:
            continue
        mean_abs = acts.abs().mean(dim=0)            # [out, in]
        max_a = float(mean_abs.max().item())
        if max_a <= 0:
            total_edges += int(torch.numel(mean_abs))
            continue
        thr = threshold * max_a

        # Umbral absoluto: aristas bajo thr se consideran nulas
        keep_thr = (mean_abs >= thr)

        # Fracción de energía: conservar las top aristas que suman `frac`
        flat = mean_abs.flatten().cpu().numpy()
        order = np.argsort(flat)[::-1]
        cum = np.cumsum(flat[order])
        total = flat.sum()
        if total > 0 and frac < 1.0:
            k = int(np.searchsorted(cum, frac * total) + 1)
            keep_energy = torch.zeros_like(mean_abs, dtype=torch.bool)
            for idx in order[:k]:
                o, i = divmod(int(idx), mean_abs.shape[1])
                keep_energy[o, i] = True
            keep = keep_energy.to(mean_abs.device) & keep_thr
        else:
            keep = keep_thr

        # materializar la poda en la máscara de la capa
        try:
            kan_model.act_fun[l].mask.data = keep.T.float().to(
                kan_model.act_fun[l].mask.device)
        except Exception:
            pass
        total_edges += int(torch.numel(keep))
        kept += int(keep.sum().item())

    n_zero = total_edges - kept
    return {'n_edges_total': total_edges, 'n_edges_kept': kept, 'n_edges_zero': n_zero}


def symbolic_all_survivors(kan_model, lib: list = None, r2_min: float = 0.97):
    """Ajusta simbólicamente TODAS las aristas vivas (máscara > 0) de la KAN.

    A diferencia de run_symbolic_fit (que usa top-N por Sproud), aquí se
    recorren las aristas que permanecen activas tras la poda a cero y se fija
    la mejor función simbólica. Es el paso que permite construir la 'fórmula
    cerrada' del frente 2.

    Args:
        kan_model: KAN (pykan) ya podada (máscaras puestas a cero).
        lib (list[str]): funciones candidatas.
        r2_min (float): umbral de R²; las aristas vivas que no lo alcanzan
            se registran igual para reportar calidad.

    Returns:
        list[dict]: con i, j, func_name, r2, layer.
    """
    if lib is None:
        lib = ['x', 'sin', 'cos', 'exp', 'log']
    results = []
    kan_model.symbolic_enabled = True
    try:
        for l in range(len(kan_model.spline_postacts)):
            acts = kan_model.spline_postacts[l]
            if acts is None:
                continue
            mask = kan_model.act_fun[l].mask.data       # copy del mask
            id_ = torch.nonzero(mask > 0, as_tuple=False)
            for ij in id_:
                i, j = int(ij[0]), int(ij[1])
                try:
                    best = kan_model.suggest_symbolic(l, i, j, lib=lib, verbose=False)
                    if isinstance(best, tuple):
                        fn = best[0]
                        r2 = best[2] if len(best) > 2 else None
                    else:
                        fn, r2 = best, None
                    kan_model.fix_symbolic(l, i, j, fn, verbose=False, log_history=False)
                    r2_val = None
                    if r2 is not None:
                        try:
                            r2_f = float(r2)
                            if np.isfinite(r2_f):
                                r2_val = round(r2_f, 4)
                        except Exception:
                            r2_val = None
                    results.append({'layer': l, 'i': i, 'j': j,
                                    'func_name': str(fn), 'r2': r2_val})
                except Exception as e:
                    results.append({'layer': l, 'i': i, 'j': j,
                                    'func_name': 'ERROR', 'r2': None, 'error': str(e)[:80]})
    finally:
        kan_model.symbolic_enabled = False
    return results


def plot_top_splines(
    kan_model,
    top_edges: list[tuple],
    save_path: str = 'Figuras/kan_sproud_splines.png',
    n_cols: int = 4,
):
    """Grafica las funciones univariadas de las aristas con mayor puntaje Sproud.

    Args:
        kan_model: Modelo KAN (pykan) con activaciones cacheadas.
        top_edges: Lista de (importancia, i, j).
        save_path: Ruta de salida para la figura.
        n_cols: Columnas en el grid de subplots.
    """
    import matplotlib.pyplot as plt
    import os

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    n = len(top_edges)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for idx, (imp, i, j) in enumerate(top_edges):
        ax = axes[idx]
        try:
            # pykan: plot_node(layer, node) o acceso directo a las activaciones
            xs = kan_model.spline_preacts[0][:, j, i].cpu().numpy()
            ys = kan_model.spline_postacts[0][:, j, i].cpu().numpy()
            sort_idx = xs.argsort()
            ax.plot(xs[sort_idx], ys[sort_idx], linewidth=2, color='royalblue')
        except Exception:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'$\\phi_{{({i},{j})}}$ | Sproud={imp:.3f}', fontsize=9)
        ax.set_xlabel(f'$z_{{{i}}}$', fontsize=8)
        ax.set_ylabel(f'$h_{{{j}}}$', fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Funciones Univariadas KAN — Aristas de Mayor Importancia Sproud', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figura guardada en {save_path}')
