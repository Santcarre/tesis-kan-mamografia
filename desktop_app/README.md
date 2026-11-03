# Prometeo — el laboratorio de la tesis

Prometeo es la parte interactiva de la tesis *Modelo canónico RS-KAN*. Sirve para **mirar y probar** lo que se descubrió: cómo una mamografía pasa por la red, se resume en un solo número (u₀), entra a una fórmula cerrada y termina en un informe. Es una herramienta de investigación hecha para una tesis de pregrado. **No es un dispositivo médico, no está validada clínicamente y no debe usarse para decidir sobre la salud de nadie.**

Hay dos formas de usarla:

| Modo | Qué necesita | Qué incluye |
|---|---|---|
| **Estático** (abrir la web) | Solo un navegador | Inicio, Laboratorio, Fórmula y Resultados. Todo se calcula en el navegador con los datos de `ui/data/`. |
| **Con la app local** | Python + PyTorch + CBIS-DDSM + Node (Electron) | Lo anterior, más el Explorador (368 casos con máscara de lesión), el Estudio externo (subir una imagen) y el PDF del informe. |

La interfaz detecta sola si hay servidor (`/api/health`) y marca con una etiqueta «app» lo que lo necesita.

---

## 1. Qué hay en cada sección

- **Inicio.** Qué es esto y por dónde empezar: la idea de que un solo número (u₀) decide el veredicto.
- **Laboratorio.** Nueve casos de prueba (ocho pacientes), cada uno con su recorrido en diez pasos y explicaciones informales: se conoce el caso (y se apuesta cómo saldrá), la red lo mira, se resumen 512 números en uno, u₀ cae en la regla, la fórmula lo evalúa, dónde pesó cada zona, el umbral decide, el informe y la verdad (la biopsia). Los casos incluyen aciertos, un falso negativo, una falsa alarma y uno donde la fórmula acierta y la red no. Cada uno trae su cautela: un caso suelto no demuestra nada.
- **Fórmula.** La fórmula `g(u₀) = f_mal(u₀) − f_ben(u₀)` completa, con dos controles: u₀ (cómo cambian las dos ramas, `g`, `p_sym` y el veredicto) y el umbral (cómo cambian aciertos y errores en las 1012 imágenes de prueba, con la curva ROC).
- **Resultados.** Las cifras de la tesis, cada una con «qué significa», «qué no significa» y su fuente: separación entre benignas y malignas (AUC), aciertos y errores con el umbral, estabilidad entre semillas, si los mapas miran donde deben (frente al azar), qué mide u₀ (incluida su relación con la categoría BI-RADS del radiólogo) y el estudio de informes con cinco modelos de lenguaje. Cierra con «Lo que no sabemos».
- **Explorador** (app local). Los 368 casos con máscara, con la mamografía, el mapa de la red (Grad-CAM) y el de la fórmula (ψ_KAN).
- **Estudio externo** (app local). Una imagen propia (JPEG/PNG/DICOM) pasa por la misma tubería. Los archivos se procesan en el equipo; sin máscara de lesión no hay comparación espacial.

## 2. Cómo se decide (y qué no significa)

```
mamografía ─▶ ResNet-18 robusta ─▶ GAP (512) ─▶ S-PCA (64) ─▶ u₀ ─▶ g(u₀) ─▶ p_sym = σ(g)
```

- **Veredicto:** *maligna* si `g(u₀) ≥ τ* = 0,1296` (probabilidad 0,5324; equivale a `u₀ ≤ −0,0331`). τ* es el umbral de Youden calculado en validación. u₀ bajo = sospechoso.
- **p_sym no es una probabilidad calibrada.** Es la sigmoide de `g`. Por ejemplo, entre las imágenes de prueba con u₀ ≈ 0,55 solo el 27 % es maligna, mientras que `p_sym` dice 41,6 %.
- **BI-RADS de la herramienta** es una categoría orientativa (2, 3, 4 o 5) anclada en la decisión de la fórmula: 2–3 si es benigna y 4–5 si es sospechosa, con cortes en `p_sym` de 0,3572 y 0,6761 ajustados en validación a la valoración del radiólogo de CBIS-DDSM (κ ponderado 0,44 en prueba; `src/reporting/birads_herramienta.py`). No es un riesgo de malignidad ni una recomendación médica.
- **Cifras de referencia (conjunto de prueba, 1012 imágenes, 318 pacientes):** AUC de la fórmula 0,7185 (IC 95 % por paciente 0,674–0,759; por imagen, más estrecho, 0,687–0,749); red robusta 0,7219; línea base 0,7325. Sensibilidad 72,5 %, especificidad 59,1 %. La fórmula **no supera** a la red; su valor es que es una expresión de una variable que se puede leer y auditar.
- **Mapas.** La energía del mapa de la fórmula dentro de la lesión supera al azar *de la imagen* (EBPG 0,0112 frente a 0,0057), pero ningún mapa supera al azar *dentro de la mama* (0,0153). No se presenta como localización.
- **Localizador.** La región y el foco se calculan con el mapa de la fórmula a partir de la imagen a 448 px (la decisión sigue a 224 px): en las lesiones malignas acierta la región en el 47,8 % de los casos, frente al 34,0 % del centro del tejido. `TESIS_RES_LOCALIZADOR=224` vuelve al localizador original.

El informe de texto y el PDF son **borradores sin validez clínica**. Pueden generarse de forma determinista (plantilla anclada a los números) o, si hay Ollama en la máquina, con un modelo de lenguaje local: por defecto Phi-3.5-mini (`ollama pull phi3.5`), el de mejor equilibrio en el estudio de informes; otro modelo se elige con la variable `PROMETEO_LLM`.

## 3. Cómo abrirla

### Modo estático (sin instalar nada)

```bash
cd desktop_app/ui
python -m http.server 8080        # y abrir http://127.0.0.1:8080/
```

Las rutas son relativas, así que también funciona publicada en cualquier hosting estático (por ejemplo GitHub Pages).

### Con la app local

Requisitos: entorno de la raíz (`pip install -r requirements.txt` con torch), el checkpoint del modelo (véase el README principal), las imágenes CBIS-DDSM y Node.js 18+.

```bat
desktop_app\run_prometeo.bat                         :: Windows, un clic (Electron + servidor)
python desktop_app/run_prometeo.py                   :: multiplataforma
python desktop_app/run_prometeo.py --backend-only    :: solo el servidor
```

Con el servidor activo, la interfaz queda en <http://127.0.0.1:8000/> y la documentación de la API en `/docs`.

## 4. Regenerar los datos y las imágenes del Laboratorio

`ui/data/*.json` se calcula **sin PyTorch** (NumPy y la biblioteca estándar) desde los embeddings, el reductor S-PCA y la fórmula canónica del repositorio:

```bash
python scripts/generar_datos_ui.py        # tesis.json, lab_global.json, lab_pacientes.json
```

El script se autoverifica (la fórmula del navegador contra la de `kan_robust64_canonical.json`, la matriz de confusión, los AUC). Las imágenes del Laboratorio (original, mapa de la red, mapa de la fórmula, lesión real) las genera la misma app, con PyTorch y CBIS-DDSM:

```bash
python scripts/generar_laboratorio.py [--casos mal-masa,ben-masa]
```

Escribe en `ui/data/lab/<id>/*.png` y avisa si `u0` o `g` de la estación no coinciden con los del recorrido numérico. Si faltan las imágenes, el Laboratorio muestra un aviso en su lugar y el resto del recorrido funciona. Los casos se eligen en `ui/data/lab_seleccion.json`.

> Las imágenes provienen de CBIS-DDSM (TCIA, CC BY 3.0): si se publican, hay que conservar la atribución (README principal, §7).

## 5. Estructura

```
desktop_app/
├── backend/          # FastAPI: cohorte, casos, capas, fórmula, inferencia externa, informes
│   └── services/decision.py   # regla de decisión y BI-RADS de la herramienta (sin torch)
├── ui/               # interfaz web (HTML + módulos ES + CSS, sin build)
│   ├── js/  css/
│   └── data/         # JSON generados por scripts/generar_datos_ui.py (+ lab/ con imágenes)
├── electron/         # ventana de escritorio (lanza el servidor y espera /api/health)
├── tests/            # pruebas
├── demo_samples/     # cuatro ejemplos de CBIS-DDSM para el estudio externo
└── run_prometeo.{py,bat}
```

## 6. Pruebas

```bash
python -m pytest desktop_app/tests/test_sin_torch.py -v   # no necesita PyTorch ni FastAPI
python -m pytest desktop_app/tests/ -v                    # todas (necesitan torch, fastapi y los datos)
```

`test_sin_torch.py` comprueba la regla de decisión, el evaluador de la fórmula (lista blanca, sin `eval`), que la interfaz referencia archivos que existen y que `ui/data/` es coherente con la fórmula canónica (cada paciente del Laboratorio, la matriz de confusión, los términos de la fórmula). El resto de los módulos prueban la API: `test_api_cases`, `test_api_render`, `test_api_formula`, `test_api_inference`, `test_api_reports`, `test_api_health`, `test_api_ui`, `test_electron_config` y el caso de punta a punta `test_e2e_clinical_p00141`.
