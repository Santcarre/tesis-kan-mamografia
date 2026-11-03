# Índice de resultados: cifra → archivo → script

Toda cifra citada en la tesis debe poder trazarse a uno de los archivos de la sección «Vigentes».

Requisitos de los scripts: **E** = embeddings incluidos en `artifacts/`, **C** = checkpoints `resnet18_robust_best.pt` y `resnet18_baseline_v2.pt` (publicados como activos de la versión del repositorio), **D** = imágenes CBIS-DDSM, **P** = probabilidades guardadas en `reports/experiments/youden_val_probs/`. Los scripts se ejecutan desde la raíz del repositorio.

## Vigentes

| Tema | Archivo | Script | Req. |
|---|---|---|---|
| Partición por paciente (manifiestos de entrenamiento, validación y prueba; SHA-256 en el Anexo A) | `src/data/processed/manifest_{train,val,test}.csv` | `scripts/build_manifests.py` | CSV de CBIS-DDSM |
| Fórmula canónica ($g(u_0)$, 749 caracteres) | `reports/experiments/kan_robust64_canonical.json` | `scripts/freeze_r3_canonical.py` | E |
| Reductor S-PCA 512→64 de la fórmula | `reports/experiments/kan_robust64_reducer.npz` | `scripts/freeze_r3_canonical.py` | E |
| Monotonía ($g'<0$, $\sup\lvert g'\rvert=0{,}8798$), soporte de $u_0$, AUC de $g$ | `reports/experiments/r3_monotonicity.json` | verificado de forma independiente por `scripts/verificar_formula.py` → `reports/auditoria/verificacion_formula.json` | E |
| AUC con IC95 % por paciente (Tabla 4.1), bootstrap pareado por paciente, logísticas de control sobre $u_0$, $(u_0,u_1)$, 64 coordenadas y 512 descriptores ($C=1$ y $C=0{,}01$), sensibilidad a especificidad fija | `reports/auditoria/recalculos_auditoria.json` | `scripts/recalculos_auditoria.py` | E, P |
| *Bootstrap* pareado por paciente de todos los contrastes de AUC (tabla de contrastes del Cap. 4) | `reports/auditoria/bootstrap_pares_paciente.json` | `scripts/bootstrap_pares_paciente.py` | E, P |
| Control logístico de la Tabla 4.1 (512 descriptores del extractor robusto, $C=0{,}01$): AUC, IC por paciente, Youden, DeLong y bootstrap pareado frente a la fórmula, calibración (Brier, ECE) | `reports/auditoria/logistica_robusta_control.json` | `scripts/logistica_robusta_control.py` | E, P |
| DeLong entre modelos convolucionales (componentes de covarianza) e IC por imagen (versión anterior, ya no citada); su logística sobre GAP-512 usa el extractor **base** y no se cita | `reports/experiments/auc_ci_comparison.json` | `scripts/recompute_auc_ci_baselines.py` | E, P |
| Umbrales de Youden fijados en validación | `reports/experiments/youden_val_calibration.json`, `youden_val_probs/` | (calculado en el repositorio de trabajo con los checkpoints de cada modelo; no incluido) | — |
| Estabilidad entre extractores (5 semillas × 4 *backbones*) | `reports/experiments/r3_spca64_multiseed.json` | `scripts/run_r3_multiseed.py` | E |
| Estabilidad de la KAN sobre S-PCA en el extractor robusto (5 semillas; AUC 0,7170 ± 0,0021; tabla del Cap. 4, §4.2). La semilla 42 de esta tabla (0,7165) es una corrida aparte de la fórmula canónica (0,7185) | `reports/27_spca_robust_comparison.json` | `pipeline/27_spca_robust_evaluation.ipynb` | E |
| Estabilidad de coeficientes del reductor | `reports/experiments/r3_coef_stability.json` | `scripts/compute_r3_coef_stability.py` | E |
| Asociación de $u_0$ con BI-RADS citada en la tesis (vínculo por imagen; margen, forma, calcificaciones y densidad; $\varepsilon^2 = H/(n-1)$; Holm sobre 15 pruebas) | `reports/auditoria/semantica_u0_por_imagen.json` | `scripts/semantica_u0_por_imagen.py` (requiere los CSV de CBIS-DDSM, incluido `dicom_info.csv`) | E, D |
| Concordancia entre arquitecturas de la dirección diagnóstica (logística $C=0{,}01$ por extractor; correlación con $-u_0$, IC por paciente, dentro de clase) y diagnóstico del emparejamiento húngaro (el eje compartido no es $u_0$; Tabla 4.9 y Fig. 4.x) | `reports/auditoria/concordancia_direcciones.json` | `scripts/concordancia_direcciones_diagnosticas.py` (requiere embeddings de otros *backbones*, no incluidos) | — |
| Emparejamiento húngaro sobre S-PCA ponderada por correlación (12 componentes; matrices de la figura de correlación cruzada) | `reports/experiments/cross_architecture_correlation.json` | `scripts/recompute_cross_architecture_correlation.py` (requiere embeddings de otros *backbones*, no incluidos) | — |
| Benchmark de reducción de dimensión (PCA / lineal / S-PCA) | `reports/experiments/benchmark_reducciones_multibackbone_val.json`, `tab_benchmark_reducciones_val.tex`, `tab_pca_ciego_multibackbone_val.tex` | (repositorio de trabajo; no incluido) | — |
| Poda y punto dulce de compresión | `reports/experiments/poda_val_calibration.json`, `punto_dulce_val.json` | (repositorio de trabajo; no incluido) | — |
| Extractor robusto: curvas y métricas | `reports/experiments/26_resnet18_robust.json` | cuaderno `pipeline/26_resnet18_robust_training.ipynb` | D, GPU |
| Segmentación débilmente supervisada (368 máscaras) | `reports/experiments/28_weak_segmentation_summary.json`, `28_weak_segmentation_comparison.csv` | cuaderno `pipeline/28_weakly_supervised_segmentation.ipynb` | C, D |
| LIME frente al Shapley analítico del logit lineal (200 casos de prueba) | `reports/experiments/attribution_coherence.json` (su tabla de $\beta$ se sustituye por `reports/auditoria/beta_canonico.json`) | `scripts/compute_attribution_coherence.py` | E, C |
| Auditoría de LIME / KernelSHAP sobre el logit lineal | `reports/auditoria/auditoria_explicadores.json` | `scripts/auditoria_explicadores.py` | E, C |
| Descomposición del logit en la base canónica ($\beta$) | `reports/auditoria/beta_canonico.json` | `scripts/beta_canonico.py` | E, C |
| Crédito por caso (fidelidad de canales) | `reports/auditoria/credito_por_caso.json` | `scripts/credito_por_caso.py` | E, C |
| Recálculos de auditoría (AUC, umbrales, sensibilidad) | `reports/auditoria/recalculos_auditoria.json` | `scripts/recalculos_auditoria.py` | E, P |
| Pointing Game continuo (EBPG) bajo protocolo único: Grad-CAM del baseline, Grad-CAM robusto y fórmula | `reports/experiments/33_pointing_stats.json`, `33_pointing_unified.csv`, `tab_pointing_unified.tex`, `33_roi_audit.json` | cuaderno `pipeline/33_reconstruccion_pointing_game_roi.ipynb` | C, D |
| Nivel de azar del Pointing Game por área de la imagen y del tejido | `reports/experiments/pointing_game_nulo_area.json` | `scripts/nulo_area_pointing.py` | D |
| Control de azar y operador focal del Pointing Game | `reports/auditoria/control_azar_focal.json`, `control_azar_focal_porimagen.csv` | `scripts/control_azar_focal.py` | C, D |
| Fórmula frente a Grad-CAM (mapas, WSSS, Pearson) | `reports/auditoria/formula_vs_gradcam_wsss.json`, `formula_vs_gradcam_wsss_porimagen.csv` | `scripts/formula_vs_gradcam_wsss.py` | C, D |
| Estabilidad del extractor robusto (semillas 1, 2, 3) | `reports/auditoria/multisemilla/run_robusto_s*.json` | `scripts/entrenar_multisemilla.py` (`--aggregate` genera `resumen_multisemilla.json`) | D, GPU |
| Coherencia KAN frente a atribuciones post-hoc | `reports/experiments/coherence_kan64.json`, `coherence_kan_self.json`, `coherence_kan_self_active.json` | `scripts/compute_coherence_kan64.py`, `scripts/compute_coherence_kan_self.py`, `scripts/check_lime_active_magnitude.py` | C |
| Asociación de $u_0$ con la categoría BI-RADS del radiólogo ($\rho$, AUC, combinación, sesgo de verificación) y cortes de la categoría orientativa de la herramienta ($\kappa_w$; Sección «Asociación de $u_0$ con la valoración del radiólogo») | `reports/auditoria/u0_birads_radiologo.json`, `u0_birads_radiologo_prueba.csv` | `scripts/u0_birads_radiologo.py`; figura: `scripts/u0_birads_radiologo_figura.py` | E, D |
| Asociación de $u_0$ con margen y forma dentro de cada clase (Holm sobre seis pruebas; una observación por paciente) | `reports/auditoria/semantica_u0_dentro_de_clase.json` | `scripts/semantica_u0_dentro_de_clase.py` | E, D |
| Explicaciones por regiones de la imagen: LIME y KernelSHAP de la red y de la fórmula (8000 muestras), estabilidad, curvas de borrado e inserción y localización | `reports/auditoria/explicadores_imagen.json`, `explicadores_imagen_porimagen.csv`, `explicadores_imagen_mapas.npz` | `scripts/explicadores_imagen.py --n-muestras 8000`; figura 4.8 del mismo script | C, D |
| Convergencia de los explicadores (1000 a 8000 muestras, 40 imágenes) | `reports/auditoria/explicadores_convergencia.json` | `scripts/explicadores_imagen.py --convergencia` | C, D |
| Mapas de mayor resolución (448 y 672 px; ventana deslizante) y localizador de la estación y del estudio de informes | `reports/auditoria/segmentacion_resolucion.json`, `segmentacion_resolucion_porimagen.csv` | `scripts/segmentacion_resolucion.py` | C, D |
| Monotonía de $g$ en el rango de las tres particiones | `reports/auditoria/monotonia_rango_completo.json` | `scripts/monotonia_rango_completo.py` | E |
| Estudio de informes con cinco modelos de lenguaje (368 mamografías, 3680 informes, dos condiciones) | `reports/experiments/estudio_informes_llm.csv`, `estudio_informes_llm_metricas.csv`, `estudio_informes_llm_resumen.json`, `estudio_informes_llm_evidencia_448.json` | `scripts/estudio_informes_llm.py` (generación, Ollama) y `scripts/analizar_informes_llm.py` (detector y pruebas) | C, D, modelo local |
| Validación del detector del estudio de informes (40 informes revisados a mano) | `reports/experiments/estudio_informes_llm_validacion_planilla_revisada.xlsx`, `estudio_informes_llm_validacion_acuerdo.json`, `estudio_informes_llm_validacion_informes.md` | `scripts/analizar_informes_llm.py --planilla` y `--validar` | — |
| Figuras 4.1, 4.3 y 4.4 de la tesis (rótulos en español) | `reports/figures/26_resnet18_robust_training_curves.png`, `33_r3_g_calibration.png`, `27_spca_robust_comparison.png` | `scripts/figura_curvas_extractor.py`, `scripts/figura_calibracion_formula.py`, `scripts/figura_spca_robusto.py` | E |
| Datos de la interfaz de Prometeo (cifras de Resultados y Laboratorio) | `desktop_app/ui/data/tesis.json`, `lab_*.json`, `lab/` | `scripts/generar_datos_ui.py`, `scripts/generar_laboratorio.py` | E (+C, D para las imágenes) |
| Costo de cómputo por etapa y memoria (equipo de sobremesa de 12 hilos, más medición limitada a un núcleo) | `reports/experiments/tiempos_inferencia_mi_pc.json`, `tiempos_inferencia_mi_pc_1nucleo.json` | `scripts/medir_tiempos_inferencia.py` | — |

## Fuera de este índice

El repositorio contiene solo lo que usa la versión final de la tesis. Los experimentos descartados durante el trabajo (versiones anteriores del estudio de informes y de algunos análisis, corridas inválidas, pilotos de una semilla y evaluaciones de localización cuyo código generador no está disponible) quedaron en el repositorio de trabajo y no se publican. Lo que se cita en la tesis está en la tabla anterior.

## Entorno

Windows, Python 3.10, PyTorch 2.14 (*nightly*, CUDA 13.0), GPU NVIDIA RTX 5060; semilla canónica 42 (véase `requirements-lock.txt`). Las semillas del estudio del extractor son 1, 2 y 3.
