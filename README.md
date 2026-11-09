# Modelo Canónico RS-KAN: explicación de una sola variable para la clasificación de mamografías

Código, artefactos y resultados del trabajo de grado **«Clasificación explicable de mamografías mediante una arquitectura híbrida ResNet18-KAN y representación simbólica compacta»**.

| | |
|---|---|
| **Autor** | Daniel Santiago Carreño Briceño |
| **Director** | Andrés Felipe Rios Moreno |
| **Programa** | Matemáticas, Facultad de Ciencias, Universidad El Bosque (Bogotá, Colombia) |
| **Año** | 2026 |
| **Datos** | CBIS-DDSM (público y desidentificado), partición estricta por paciente |

> **Aviso.** Este repositorio es material académico. No es un dispositivo médico, no ha sido validado clínicamente y sus salidas no deben usarse para decisiones diagnósticas.

---

## 1. Qué contiene el trabajo

Una ResNet-18 ajustada a mamografías produce un vector de 512 descriptores (GAP). Una proyección supervisada (S-PCA, 512 → 64) alimenta una red de Kolmogorov-Arnold KAN[64, 32, 2] que se poda y se destila en una **fórmula cerrada que depende de una sola coordenada** $u_0$, una proyección lineal de los 512 descriptores:

```
Mamografía 224×224 ──ResNet-18──▶ GAP ∈ ℝ⁵¹² ──S-PCA──▶ u ∈ ℝ⁶⁴ ──KAN podada──▶ g(u₀) = f_mal − f_ben
                       (robusto)                (u₀ = 1.ª coord.)               (749 caracteres, monótona)
```

La contribución es una **explicación de fidelidad**: la fórmula es una versión comprimida y auditable de la red que conserva su logit, su dirección de decisión y su mapa espacial, y solo en parte sus canales individuales. **No es una mejora del AUC.**

### Resultados principales (partición de prueba: 1012 imágenes, 318 pacientes)

| Resultado | Valor |
|---|---|
| ROC-AUC de la fórmula $g(u_0)$ | **0,7185** (IC 95 % por paciente: 0,674 a 0,759) |
| ROC-AUC de la red robusta de la que se destila / del clasificador denso de referencia | 0,7219 / 0,7325 |
| Diferencia de AUC (DeLong) frente a la red robusta / frente al clasificador denso | $p = 0{,}58$ / $p = 0{,}37$ (sin diferencia significativa) |
| Control lineal: regresión logística sobre los 512 descriptores del mismo extractor ($C = 0{,}01$) | AUC 0,7161 (IC 95 % por paciente: 0,673 a 0,755); sin diferencia con la fórmula (DeLong $p = 0{,}60$) |
| Monotonía y regularidad | $g$ es estrictamente decreciente en el rango de prueba: $-0{,}88 \le g'(u_0) \le -0{,}44$ (demostrado con una cota de la segunda derivada y una malla fina) |
| Relación con $u_0$ | como $g$ es decreciente, ordena los casos al revés que $u_0$: discrimina exactamente lo mismo que $u_0$ y que una regresión logística sobre $u_0$ |
| Punto de operación | sensibilidad 72,49 % y especificidad 59,09 % con el umbral de Youden fijado en validación (es un punto de operación, no una mejor discriminación) |
| Fidelidad al logit de la red | $R^2 = 0{,}91$ en prueba; $u_0$ domina el crédito en el 89,1 % de los casos |
| Fidelidad de dirección / de mapa espacial | coseno 0,70 con la dirección de decisión; correlación 0,85–0,87 con el mapa de esa dirección |
| Fidelidad de canales | parcial: con la misma definición de atribución en los dos lados, $\rho = 0{,}79$ entre canales, pero el canal dominante coincide solo en 26 de 200 casos |
| Explicaciones por regiones de la imagen | LIME y KernelSHAP (8000 muestras) le atribuyen a la fórmula casi lo mismo que a la red ($\rho = 0{,}90$) |
| Relación con el radiólogo | $\rho = 0{,}45$ con su categoría BI-RADS, también dentro de cada patología; AUC del radiólogo 0,779 frente a 0,728 de la fórmula, y 0,815 combinados; categoría orientativa de la herramienta con $\kappa_w = 0{,}44$ |
| Localización (EBPG continuo, malignas) | 1,51 %: por encima del azar por área de la imagen (0,71 %) y por debajo del azar por área de tejido (1,87 %) |
| Localización (operador focal) | a 224 px, ni la fórmula ni Grad-CAM superan al predictor trivial «centro del tejido mamario»; con la imagen a 448 px, el foco de la fórmula acierta la región en el 47,8 % de las malignas (centro del tejido: 34,0 %) |
| Informes con cinco modelos de lenguaje locales (368 mamografías, 3680 informes) | con la evidencia del modelo, las afirmaciones sin respaldo bajan en los cuatro modelos que las hacían (Phi-3.5: 87,5 % → 1,4 %), pero el informe copia los errores de la evidencia (región equivocada en el 56,5 %) |
| Variabilidad al reentrenar el extractor (3 semillas) | AUC de $u_0$ entre 0,694 y 0,714 (0,706 ± 0,011): el 0,7185 canónico es una realización algo favorable |
| Costo de cómputo (un equipo de sobremesa, CPU) | fórmula y veredicto en unas decenas de µs y ~30 MB sin PyTorch; la red (29–67 ms) y reducir la imagen (67 ms) concentran el costo, así que la fórmula **no acelera** el sistema |

Los valores proceden de los archivos de `reports/` (véase la sección 4) y se citan en los capítulos 4 a 6 de la tesis.

### Alcance y límites

- La fórmula **no discrimina mejor** que $u_0$ ni que la red: su valor es la trazabilidad (una variable, constantes explícitas, monotonía verificable).
- La sensibilidad del 72,49 % depende del umbral elegido en validación.
- La localización es limitada: a la resolución de la red (mapa de 7×7) ningún mapa supera al «centro del tejido mamario»; con la imagen a 448 px el foco de la fórmula sí lo supera en acierto de región, pero su centroide queda más lejos de la lesión, y esa resolución se eligió mirando la prueba, así que el resultado es exploratorio (controles en `reports/auditoria/`).
- LIME es exacto sobre un logit afín; KernelSHAP estándar converge cuando el presupuesto de muestras supera aproximadamente la dimensión (`reports/auditoria/auditoria_explicadores.json`).
- El extractor robusto muestra sobreajuste (AUC de entrenamiento cercano a 0,95 frente a 0,72 en prueba); la variabilidad entre semillas del extractor se estudia con `scripts/entrenar_multisemilla.py` (semillas 1, 2 y 3).
- El estudio de informes mide invenciones con un detector automático validado en 40 informes, que las subestima; la referencia de esa validación la propuso un asistente de IA y la revisó el autor, sin un segundo anotador humano, y ningún radiólogo evaluó la calidad clínica de los textos.
- Las etiquetas de los manifiestos se asignaron por paciente (la patología de su primera lesión en las tablas de CBIS-DDSM): el 1,6 % de las imágenes de prueba y el 3,3 % de las de entrenamiento no coinciden con la patología de su vista. Con la etiqueta por vista, el AUC de prueba de la fórmula es 0,7141, el de la red robusta 0,7178 y el de la red base 0,7270 (`scripts/etiquetas_por_vista.py`).

---

## 2. Estructura del repositorio

```
.
├── README.md  LICENSE  requirements.txt  requirements-lock.txt  .env.example
├── src/                    # código modular (datos, modelos, explicabilidad, evaluación, informes)
│   └── data/processed/     # manifest_{train,val,test}.csv: partición por paciente (SHA-256 en el Anexo A)
├── scripts/                # scripts de cómputo de los resultados de la tesis (véase la sección 4)
├── Figuras/                # figuras nuevas del capítulo 4 (BI-RADS del radiólogo, explicadores por regiones)
├── pipeline/               # cuadernos de los experimentos (véase la sección 5)
├── reports/
│   ├── CANONICAL_RESULTS.md  # índice: cifra → archivo → script
│   ├── experiments/        # artefactos canónicos (fórmula, reductor S-PCA, AUC con IC, semántica, LLM…)
│   ├── auditoria/          # resultados de la revisión final (explicadores, fidelidad, controles de azar, semillas)
│   ├── figures/  tables/   # figuras y tablas LaTeX generadas
│   └── models/             # (vacío) checkpoints; no se versionan, véase la sección 3
├── artifacts/output/embeddings/   # embeddings GAP (512-D) del extractor robusto: train/val/test
└── desktop_app/            # Prometeo: laboratorio interactivo de la tesis (web estática + app local)
```

---

## 3. Instalación y datos

**Requisitos.** Python 3.10 o superior. Los resultados se produjeron en Windows con una GPU NVIDIA (PyTorch con CUDA); los scripts que solo usan embeddings corren en CPU.

```bash
git clone https://github.com/Santcarre/tesis-kan-mamografia.git
cd tesis-kan-mamografia
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt   # instale antes torch/torchvision que correspondan a su GPU o CPU
```

`requirements-lock.txt` registra el entorno exacto de los experimentos (con versiones *nightly* de PyTorch que ya no se pueden descargar); es solo de consulta.

**Qué viene incluido y qué no**

| Elemento | Estado |
|---|---|
| Manifiestos de partición (`src/data/processed/`) | incluidos |
| Embeddings GAP del extractor robusto (`artifacts/output/embeddings/f2robust_*.npy`, 13 MB) | incluidos |
| Artefactos canónicos y resultados (`reports/experiments/`, `reports/auditoria/`) | incluidos |
| Checkpoints `resnet18_robust_best.pt` (extractor robusto, 44,8 MB) y `resnet18_baseline_v2.pt` (referencia, 44,8 MB; solo para el brazo «Grad-CAM del baseline») | **no incluidos**; se publican como activos de la versión (*Release*) del repositorio. SHA-256: `c64a0ca03b1e5f6adca9f98de5354dbb8f1338026e822429d49fa84366e178e1` (robusto) y `224e1631c90801373e7bfb9ea4eabfc6e70d3e0ae59f58e5c9677ddf8f1e99da` (baseline). Colóquelos en `reports/models/` |
| Imágenes CBIS-DDSM | **no incluidas**. El trabajo usa la versión JPEG publicada en Kaggle ([CBIS-DDSM: Breast Cancer Image Dataset](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset), versión 1); al descomprimirla en `database/` queda la estructura que esperan los scripts, `database/archive/csv/` y `database/archive/jpeg/`, en la raíz del repo (o defina `TESIS_CODIGOS`). Los DICOM originales están en TCIA (ver «Datos y créditos») |

Variables de entorno opcionales (véase `.env.example`): `TESIS_CODIGOS` (raíz del repo), `TESIS_REP` (por defecto `artifacts/`), `TESIS_EMBEDDINGS_DIR`, `TESIS_PDF_DIR`, `OPENROUTER_API_KEY` (solo si se usa el modelo remoto del generador de informes).

---

## 4. Reproducción de los resultados

Ejecute los scripts **desde la raíz del repositorio**. Requisitos: **E** = embeddings incluidos, **C** = checkpoint (Release), **D** = imágenes CBIS-DDSM.

| Resultado en la tesis | Script | Salida | Req. |
|---|---|---|---|
| Partición por paciente (manifiestos; SHA-256 en el Anexo A) | `scripts/build_manifests.py` | `src/data/processed/manifest_{train,val,test}.csv` | CSV de CBIS-DDSM |
| Fórmula canónica y reductor S-PCA | `scripts/freeze_r3_canonical.py` | `reports/experiments/kan_robust64_{canonical.json,reducer.npz}` | E |
| Verificación independiente de la fórmula (AUC, $g'<0$, soporte de $u_0$; solo `numpy`) | `scripts/verificar_formula.py` | `reports/auditoria/verificacion_formula.json` (compara con `reports/experiments/r3_monotonicity.json`) | E |
| DeLong entre modelos (componentes de covarianza) | `scripts/recompute_auc_ci_baselines.py` | `reports/experiments/auc_ci_comparison.json` | E |
| Control lineal de la Tabla 4.1 (logística sobre 512 descriptores del extractor robusto): AUC, IC por paciente, DeLong, bootstrap pareado y calibración | `scripts/logistica_robusta_control.py` | `reports/auditoria/logistica_robusta_control.json` | E |
| Tiempos de inferencia y memoria por etapa | `scripts/medir_tiempos_inferencia.py` | `reports/experiments/tiempos_inferencia_*.json` | E (+C para la red) |
| Estabilidad entre extractores (5 semillas × 4 *backbones*) | `scripts/run_r3_multiseed.py` | `reports/experiments/r3_spca64_multiseed.json` | E |
| Estabilidad de coeficientes del reductor | `scripts/compute_r3_coef_stability.py` | `reports/experiments/r3_coef_stability.json` | E |
| Invarianza entre arquitecturas | `scripts/recompute_cross_architecture_correlation.py` | `reports/experiments/cross_architecture_correlation.json` | embeddings de otros *backbones* (no incluidos) |
| Auditoría de LIME / KernelSHAP | `scripts/auditoria_explicadores.py` | `reports/auditoria/auditoria_explicadores.json` | E, C |
| Descomposición del logit en la base canónica ($\beta$) | `scripts/beta_canonico.py` | `reports/auditoria/beta_canonico.json` | E, C |
| Crédito por caso (fidelidad de canales) | `scripts/credito_por_caso.py` | `reports/auditoria/credito_por_caso.json` | E, C |
| AUC con IC95 % por paciente, bootstrap pareado y logísticas de control (§4.1.1) | `scripts/recalculos_auditoria.py` | `reports/auditoria/recalculos_auditoria.json` | E |
| *Bootstrap* pareado por paciente de todos los contrastes de AUC | `scripts/bootstrap_pares_paciente.py` | `reports/auditoria/bootstrap_pares_paciente.json` | E |
| Pointing Game continuo (EBPG) bajo protocolo único, 3 brazos × 368 máscaras | `pipeline/33_reconstruccion_pointing_game_roi.ipynb` | `reports/experiments/33_pointing_stats.json`, `33_pointing_unified.csv`, `tab_pointing_unified.tex` | C, D |
| Nivel de azar del Pointing Game por área de la imagen y del tejido | `scripts/nulo_area_pointing.py` | `reports/experiments/pointing_game_nulo_area.json` | D |
| Control de azar del Pointing Game y operador focal | `scripts/control_azar_focal.py` | `reports/auditoria/control_azar_focal*.{json,csv}` | C, D |
| Fórmula frente a Grad-CAM (mapas y WSSS) | `scripts/formula_vs_gradcam_wsss.py` | `reports/auditoria/formula_vs_gradcam_wsss*.{json,csv}` | C, D |
| Estabilidad del extractor robusto (semillas 1, 2, 3) | `scripts/entrenar_multisemilla.py` | `reports/auditoria/multisemilla/` | D, GPU |
| Coherencia KAN frente a atribuciones post-hoc | `scripts/compute_coherence_kan64.py`, `scripts/compute_coherence_kan_self.py`, `scripts/check_lime_active_magnitude.py` | `reports/experiments/coherence_*.json` | C |
| Fidelidad por canales con la misma definición de atribución en la fórmula y en la cabeza | `scripts/canales_atribucion.py` | `reports/auditoria/canales_atribucion.json` | E, C |
| Etiquetas de los manifiestos frente a la patología de cada vista, y AUC de prueba con la etiqueta por vista | `scripts/etiquetas_por_vista.py` | `reports/auditoria/etiquetas_por_vista.json` | CSV de CBIS-DDSM |
| Asociación de $u_0$ con la categoría BI-RADS del radiólogo y cortes de la categoría de la herramienta | `scripts/u0_birads_radiologo.py` (figura: `scripts/u0_birads_radiologo_figura.py`) | `reports/auditoria/u0_birads_radiologo.json`, `u0_birads_radiologo_prueba.csv` | E, D |
| Asociación de $u_0$ con BI-RADS por imagen y dentro de cada clase | `scripts/semantica_u0_por_imagen.py`, `scripts/semantica_u0_dentro_de_clase.py` (figura: `scripts/semantica_u0_figura.py`) | `reports/auditoria/semantica_u0_*.json` | E, D |
| Explicaciones por regiones de la imagen (LIME y KernelSHAP de la red y de la fórmula, borrado e inserción) y su convergencia | `scripts/explicadores_imagen.py --n-muestras 8000` (antes, `--convergencia`) | `reports/auditoria/explicadores_imagen*.{json,csv,npz}`, `explicadores_convergencia.json` | C, D |
| Mapas a mayor resolución (448 y 672 px, ventana deslizante) | `scripts/segmentacion_resolucion.py` | `reports/auditoria/segmentacion_resolucion*.{json,csv}` | C, D |
| Monotonía en el rango de las tres particiones | `scripts/monotonia_rango_completo.py` | `reports/auditoria/monotonia_rango_completo.json` | E |
| Estudio de informes con cinco modelos de lenguaje y su validación | `scripts/estudio_informes_llm.py`, `scripts/analizar_informes_llm.py` (`--planilla`, `--validar`) | `reports/experiments/estudio_informes_llm*.{csv,json,xlsx,md}` | C, D, Ollama |
| Figuras 4.1, 4.3 y 4.4 de la tesis | `scripts/figura_curvas_extractor.py`, `scripts/figura_calibracion_formula.py`, `scripts/figura_spca_robusto.py` | `reports/figures/` | E |
| Datos de la interfaz de Prometeo | `scripts/generar_datos_ui.py`, `scripts/generar_laboratorio.py` | `desktop_app/ui/data/` | E (+C, D para las imágenes) |

Ejemplos:

```bash
python scripts/verificar_formula.py                       # recalcula AUC y monotonía de g(u0); solo numpy, segundos
python scripts/recalculos_auditoria.py                    # rápido, solo embeddings y probabilidades guardadas
python scripts/auditoria_explicadores.py --smoke           # prueba de humo de la auditoría (requiere el checkpoint)
python scripts/control_azar_focal.py --smoke               # prueba de humo (12 imágenes)
python scripts/entrenar_multisemilla.py --seeds 1 2 3      # entrena el extractor robusto con 3 semillas (horas, GPU)
```

El índice completo «cifra → archivo → script» está en [`reports/CANONICAL_RESULTS.md`](reports/CANONICAL_RESULTS.md). Los SHA-256 de los manifiestos y de los artefactos citados en el Anexo A de la tesis están consignados en ese anexo.

---

## 5. Cuadernos (`pipeline/`)

Los cuadernos registran los experimentos tal como se ejecutaron; **el camino reproducible de los resultados finales son los scripts de la sección 4**. Ejecútelos desde la carpeta `pipeline/` (resuelven la raíz del repo con `..` o con `TESIS_CODIGOS`).

| Cuadernos | Contenido |
|---|---|
| `01`–`05` | Línea base ResNet-18, arquitecturas KAN, poda, reducción de dimensión comparativa, PCA |
| `23`, `24` | *Benchmark* de reducción de dimensión en cuatro *backbones* |
| `26`, `27`, `28` | Extractor robusto, evaluación de S-PCA, segmentación débilmente supervisada |
| `30` | Variables semánticas e invarianza entre arquitecturas |
| `33` | Pointing Game continuo bajo protocolo único y auditoría geométrica del cargador ROI |

---

## 6. Prometeo: el laboratorio de la tesis

`desktop_app/` es la parte interactiva de la tesis. Nueve casos de prueba (ocho pacientes) muestran, paso a paso y con explicaciones informales, cómo se llega al informe: la red mira la mamografía, 512 números se resumen en uno (u₀), la fórmula cerrada lo evalúa, un umbral decide y la biopsia dice la verdad. Además permite mover u₀ y el umbral sobre la fórmula, ver las cifras de la tesis con lo que significan (y lo que no) y, con la app local, recorrer los 368 casos con máscara de lesión o subir una imagen propia. Es una herramienta de investigación: los informes que genera son **borradores sin validez clínica** y la categoría BI-RADS es orientativa: está anclada en el umbral de decisión y no es un riesgo de malignidad. El mapa de la fórmula y su foco se calculan con la imagen a 448 px (la decisión, a 224 px), y el informe redactado usa por defecto Phi-3.5-mini servido con Ollama (`ollama pull phi3.5`).

```bash
# Solo navegador (sin PyTorch): Inicio, Laboratorio, Fórmula y Resultados
cd desktop_app/ui && python -m http.server 8080

# App local completa (checkpoints en reports/models/, imágenes CBIS-DDSM, Node.js 18+)
python desktop_app/run_prometeo.py
```

Los datos de la interfaz (`desktop_app/ui/data/`) se regeneran con `python scripts/generar_datos_ui.py` y las imágenes del Laboratorio con `python scripts/generar_laboratorio.py`. Detalles en [`desktop_app/README.md`](desktop_app/README.md).

---

## 7. Datos y créditos

- **CBIS-DDSM (cita del conjunto):** Sawyer-Lee, R., Gimenez, F., Hoogi, A. y Rubin, D. (2016). *Curated Breast Imaging Subset of Digital Database for Screening Mammography (CBIS-DDSM)* [Conjunto de datos]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY. Licencia CC BY 3.0; uso sujeto a la [política de uso de datos de TCIA](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/). Las imágenes que muestra Prometeo son recortes de 224×224 de este conjunto, con mapas superpuestos (adaptación); no se usan para identificar a ninguna paciente.
- **CBIS-DDSM (artículo):** Lee, R. S., Gimenez, F., Hoogi, A., Miyake, K. K., Gorovoy, M. y Rubin, D. L. (2017). *A curated mammography data set for use in computer-aided detection and diagnosis research.* Scientific Data, 4, 170177. Distribuido por The Cancer Imaging Archive (TCIA) bajo licencia CC BY 3.0. Las imágenes de `desktop_app/demo_samples/` proceden de este conjunto.
- **Versión JPEG usada:** Awsaf (2021). *CBIS-DDSM: Breast Cancer Image Dataset* [Conjunto de datos], versión 1. Kaggle. https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset. Licencia CC BY-SA 3.0.
- **TCIA:** Clark, K. et al. (2013). *The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository.* Journal of Digital Imaging, 26(6), 1045–1057.
- La pseudonimización de pacientes (`P_xxxxx`) es la del conjunto público; este repositorio no contiene datos clínicos adicionales.

## 8. Licencia y cita

Código bajo licencia MIT (véase `LICENSE`). Los datos de CBIS-DDSM mantienen su propia licencia.

```bibtex
@thesis{carreno2026rskan,
  author = {Carreño Briceño, Daniel Santiago},
  title  = {Clasificación explicable de mamografías mediante una arquitectura híbrida ResNet18-KAN y representación simbólica compacta},
  school = {Universidad El Bosque},
  type   = {Trabajo de grado (Matemáticas)},
  year   = {2026},
  url    = {https://github.com/Santcarre/tesis-kan-mamografia}
}
```
