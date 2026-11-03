// Resultados: las cifras de la tesis, cada una con lo que significa y lo que NO significa.
import { S } from './data.js';
import { puntosIC } from './charts.js';
import { f, pc, esc } from './util.js';

const pval = (p) => p < 0.001 ? 'p < 0,001' : 'p = ' + f(p, 3);

function bloque({ titulo, visual, significa, no, fuente }) {
  return `<article class="resultado">
    <div class="res-cuerpo"><h2>${titulo}</h2>${visual}</div>
    <div class="res-notas">
      <div class="sig"><h3>Qué significa</h3>${significa}</div>
      <div class="nosig"><h3>Qué no significa</h3>${no}</div>
      <p class="fuente">Fuente: ${fuente}</p>
    </div></article>`;
}

export async function montar(el) {
  const T = S.tesis, A = T.auc, P = T.poblacion, E = T.ebpg, M = T.multisemilla, SM = T.semantica, L = T.estudio_llm, LH = T.localizador_448, RB = T.birads_radiologo;
  const ic = (m) => [{ lo: m.ic95_imagen[0], hi: m.ic95_imagen[1] }, { lo: m.ic95_paciente[0], hi: m.ic95_paciente[1] }];

  const auc = bloque({
    titulo: '¿Qué tan bien separa benignas de malignas?',
    visual: puntosIC({
      dominio: [0.62, 0.8], paso: 0.04,
      filas: [
        { label: 'Red base (la de referencia)', v: A.baseline.auc, ic: ic(A.baseline), color: 'var(--tinta-3)' },
        { label: 'Red robusta (la reentrenada)', v: A.red_robusta.auc, ic: ic(A.red_robusta), color: 'var(--tinta-2)' },
        { label: 'La fórmula', v: A.formula.auc, ic: ic(A.formula), color: 'var(--sospechoso)', fuerte: true },
      ],
    }) + `<p class="nota">AUC en el conjunto de prueba (${P.n} imágenes, ${A.n_pacientes} pacientes). Barra gruesa: intervalo de confianza al 95 % por imagen. Barra fina: el mismo intervalo contando por paciente, que es más prudente porque varias imágenes son de la misma persona. 0,5 sería lanzar una moneda.</p>`,
    significa: `<p>Las tres quedan en torno a 0,72–0,73 y sus intervalos se solapan casi por completo. La fórmula conserva el ${pc(A.formula.auc / A.red_robusta.auc, 1)} del AUC de la red robusta de la que sale, y lo hace con <b>una sola variable</b> y 749 caracteres.</p>
      <p>La diferencia entre la red base y la fórmula (${f(A.delta_baseline_menos_formula.media * 100, 1)} puntos) no es distinguible de cero: ${pval(A.delta_baseline_menos_formula.p_bilateral)}.</p>`,
    no: `<p>Un AUC de 0,72 es una discriminación <b>modesta</b>: no sirve como prueba diagnóstica por sí sola. Tampoco significa que la fórmula sea mejor que la red: es equivalente, y una regresión logística sobre u₀ (AUC ${f(A.logistica_u0, 3)}) ordena los casos igual.</p><p>Es un solo conjunto de datos, digitalizado de película, sin validación en otro hospital.</p>`,
    fuente: '<code>reports/experiments/auc_ci_comparison.json</code>, <code>reports/auditoria/recalculos_auditoria.json</code>',
  });

  const c = P.confusion;
  const oper = bloque({
    titulo: 'Aciertos y errores con el umbral de la tesis',
    visual: `<div class="cifras-fila">
        <div><span class="cifra">${pc(P.sens)}</span><small>sensibilidad: detecta ${c.tp} de las ${P.n_mal} malignas</small></div>
        <div><span class="cifra">${pc(P.esp)}</span><small>especificidad: descarta ${c.tn} de las ${P.n_ben} benignas</small></div>
      </div>
      <p class="nota">${c.fp} benignas quedan marcadas como sospechosas y ${c.fn} malignas se le escapan. Puedes mover el umbral en la pestaña Fórmula.</p>`,
    significa: `<p>El umbral se eligió en validación (criterio de Youden) y se aplicó tal cual en prueba. Es un punto de equilibrio entre dos errores, no el único posible.</p>`,
    no: `<p>La sensibilidad de ${pc(P.sens, 0)} <b>no es mayor capacidad de detección</b>: es solo el punto de la curva donde quedó el umbral. Con ${pc(1 - P.esp, 0)} de falsas alarmas, no podría usarse sola para decidir a quién biopsiar.</p>`,
    fuente: '<code>reports/experiments/kan_robust64_canonical.json</code>, recalculado por <code>scripts/generar_datos_ui.py</code>',
  });

  const sd = M.robusto_spca;
  const estab = bloque({
    titulo: '¿Es un resultado estable o un golpe de suerte?',
    visual: puntosIC({
      dominio: [0.70, 0.735], paso: 0.005, fmt: (v) => f(v, 3),
      filas: sd.semillas.map(s => ({ label: `Semilla ${s.semilla}`, v: s.auc, ic: [], color: 'var(--sospechoso)' })).concat([{ label: 'Promedio', v: sd.auc, ic: [{ lo: sd.auc - sd.sd, hi: sd.auc + sd.sd }], color: 'var(--tinta)', fuerte: true }]),
    }) + `<p class="nota">AUC de la fórmula con cinco semillas de la etapa S-PCA + KAN: ${f(sd.auc, 4)} ± ${f(sd.sd, 4)} (${pc(sd.sens)} ± ${pc(sd.sens_sd)} de sensibilidad y ${pc(sd.esp)} ± ${pc(sd.esp_sd)} de especificidad). La fórmula canónica (semilla 42, una corrida aparte) da ${f(A.formula.auc, 4)}.</p>`,
    significa: `<p>Repetir la reducción y la destilación con otras semillas da casi lo mismo: la variación es de unas pocas milésimas.</p>`,
    no: (() => {
      const ES = T.extractor_semillas, R = ES.resumen && ES.resumen.robusto;
      const lista = (v) => v.length > 1 ? v.slice(0, -1).join(', ') + ' y ' + v[v.length - 1] : (v[0] || '');
      const auc = ES.corridas.map(q => f(q.auc_u0, 3));
      const resumen = R ? ` (promedio ${f(R['AUC u0 (prueba)'].media, 3)} ± ${f(R['AUC u0 (prueba)'].de, 3)}; AUC de la propia red ${f(R['AUC prueba (red)'].media, 3)} ± ${f(R['AUC prueba (red)'].de, 3)})` : '';
      const techo = R && A.formula.auc > R['AUC u0 (prueba)'].max && A.red_robusta.auc > R['AUC prueba (red)'].max
        ? ` Además, el extractor de la tesis (${f(A.formula.auc, 4)} con la fórmula, ${f(A.red_robusta.auc, 4)} la red) queda por encima de las ${ES.corridas.length} corridas. Con tan pocas semillas no se puede saber cuán típico es, pero conviene esperar que otro entrenamiento dé algo menos.` : '';
      return `<p>Esas semillas solo cambian la etapa S-PCA + KAN; la red que extrae los 512 números es <b>siempre la misma</b>. Al reentrenar la propia red desde cero con ${ES.corridas.length} semillas, el AUC de u₀ en prueba fue ${lista(auc)}${resumen}: la variación real es mayor que la de la tabla.${techo}</p>
      <p class="nota">Como g es decreciente en u₀, el AUC de la fórmula es el mismo que el de u₀. El reentrenamiento se reconstruyó a partir del cuaderno 26 (el código exacto del bucle robusto no está en el repositorio), con semillas nuevas y sin usar el conjunto de prueba para elegir época, umbral ni modelo.</p>`;
    })(),
    fuente: `<code>reports/27_spca_robust_comparison.json</code> (Capítulo 4, §4.2), <code>reports/auditoria/multisemilla/</code>`,
  });

  const loc = bloque({
    titulo: '¿Mira donde debe mirar?',
    visual: puntosIC({
      dominio: [0, 0.02], paso: 0.005, fmt: (v) => pc(v, 1),
      filas: [
        { label: 'Azar sobre toda la imagen', v: E.azar_imagen.v, ic: [{ lo: E.azar_imagen.ic95[0], hi: E.azar_imagen.ic95[1] }], color: 'var(--umbral)' },
        { label: 'Azar dentro de la mama', v: E.azar_mama.v, ic: [{ lo: E.azar_mama.ic95[0], hi: E.azar_mama.ic95[1] }], color: 'var(--umbral)' },
        { label: 'Grad-CAM, red base', v: E.gradcam_base.v, ic: [{ lo: E.gradcam_base.ic95[0], hi: E.gradcam_base.ic95[1] }], color: 'var(--tinta-3)' },
        { label: 'Grad-CAM, red robusta', v: E.gradcam_robusto.v, ic: [{ lo: E.gradcam_robusto.ic95[0], hi: E.gradcam_robusto.ic95[1] }], color: 'var(--tinta-2)' },
        { label: 'Mapa de la fórmula', v: E.formula.v, ic: [{ lo: E.formula.ic95[0], hi: E.formula.ic95[1] }], color: 'var(--sospechoso)', fuerte: true },
      ],
    }) + `<p class="nota">Fracción de la energía del mapa que cae dentro de la lesión real (EBPG), en ${E.n} casos con máscara. Los puntos ámbar son lo que daría un mapa repartido al azar.</p>`,
    significa: `<p>Todos los mapas superan al azar repartido sobre la imagen completa, pero solo porque se concentran en la mama. Ninguno supera al azar repartido dentro de la mama: señalan zonas amplias, no la lesión.</p><p>Con la imagen a 448 píxeles, el foco del mapa de la fórmula acierta la región de la lesión en el ${pc(LH.acierto_malignas_448)} de las malignas, frente al ${pc(LH.centro_tejido_malignas)} del centro del tejido y al ${pc(LH.acierto_malignas_224)} a 224 píxeles. Es el localizador que usa la estación; la decisión se sigue tomando a 224 píxeles, porque a 448 la red ya no discrimina (AUC ${f(LH.auc_formula_448, 2)}).</p>`,
    no: `<p>No implica que la fórmula “vea” el tumor. Por eso en el Laboratorio los mapas se presentan como pista y no como explicación. Tampoco prueba que la red decida mal: solo que estos mapas no son una prueba de que decida bien.</p>`,
    fuente: '<code>reports/experiments/33_pointing_stats.json</code>, <code>pointing_game_nulo_area.json</code>, <code>reports/auditoria/segmentacion_resolucion.json</code>',
  });

  const sem = bloque({
    titulo: '¿Qué mide u₀?',
    visual: `<table class="tabla"><thead><tr><th>u₀ frente a…</th><th class="num">Resultado</th></tr></thead><tbody>
        <tr><td>Biopsia (maligna / benigna)</td><td class="num">tamaño del efecto d = ${f(SM.u0_vs_patologia.d, 2)}, ${pval(SM.u0_vs_patologia.p_holm)}</td></tr>
        <tr><td>Bordes de la masa</td><td class="num">Kruskal-Wallis H = ${f(SM.u0_vs_margenes.H, 1)}, ${pval(SM.u0_vs_margenes.p_holm)}</td></tr>
        <tr><td>Forma de la masa</td><td class="num">H = ${f(SM.u0_vs_forma.H, 1)}, ${pval(SM.u0_vs_forma.p_holm)}</td></tr>
        <tr><td>Densidad del tejido de fondo</td><td class="num">H = ${f(SM.u0_vs_densidad.H, 1)}, ${pval(SM.u0_vs_densidad.p_holm)}</td></tr>
        <tr><td>Categoría BI-RADS del radiólogo</td><td class="num">ρ = ${f(RB.rho, 2)} (IC ${f(RB.rho_ic95[0], 2)} a ${f(RB.rho_ic95[1], 2)})</td></tr>
      </tbody></table><p class="nota">${SM.n_pacientes} pacientes. Valores p corregidos con Holm. La fila del radiólogo usa las ${RB.n_imagenes} imágenes de prueba con categoría 2 a 5 (${RB.n_pacientes} pacientes): él discrimina mejor (AUC ${f(RB.auc_radiologo, 3)} frente a ${f(RB.auc_formula, 3)} de la fórmula en esas imágenes), pero juntos llegan a ${f(RB.auc_combinacion, 3)}. La categoría orientativa de la herramienta coincide con la suya con κ ponderado = ${f(RB.kappa_categoria, 2)}.</p>`,
    significa: `<p>u₀ se asocia con la biopsia, con el tipo de borde y con la forma de la lesión, y no con la densidad del fondo. Es coherente con que u₀ responda a la lesión y no al tejido que la rodea. Además sube con la categoría BI-RADS del radiólogo, también dentro de las benignas y dentro de las malignas: no solo repite la biopsia.</p>`,
    no: `<p>Es una asociación estadística, no una medición: u₀ no “mide espiculación”. Y que la densidad no salga significativa no demuestra que no haya relación, solo que no la detectamos.</p>`,
    fuente: '<code>reports/auditoria/semantica_u0_por_imagen.json</code>, <code>reports/auditoria/u0_birads_radiologo.json</code>',
  });

  const llm = bloque({
    titulo: 'Informes redactados por modelos de lenguaje',
    visual: `<table class="tabla"><thead><tr><th>Modelo</th><th class="num">Inventa rasgos: sin → con evidencia</th><th class="num">Categoría coherente: sin → con evidencia</th><th class="num">s / informe</th></tr></thead><tbody>
        ${L.modelos.map(m => `<tr><td>${m.nombre}</td><td class="num">${pc(m.sin_respaldo_control)} → ${pc(m.sin_respaldo_evidencia)}</td><td class="num">${pc(m.coherente_control)} → ${pc(m.coherente_evidencia)}</td><td class="num">${f(m.segundos, 1)}</td></tr>`).join('')}
      </tbody></table><p class="nota">${L.n_informes} informes de ${L.n} imágenes (${L.pacientes} pacientes). «Inventa rasgos»: afirma o niega hallazgos que no estaban en los datos. El detector automático coincide con la revisión manual en el ${pc(L.validacion.acuerdo_sin_respaldo)} de ${L.validacion.n} informes (κ = ${f(L.validacion.kappa_sin_respaldo, 2)}).</p>`,
    significa: `<p>Con la evidencia del modelo, los modelos de lenguaje dejan casi de inventar rasgos de la lesión y escriben una categoría coherente con la clasificación. Pero copian la evidencia tal como llega: con ella, la región del informe se equivoca en el ${pc(L.error_localizador)} de las imágenes, lo mismo que el localizador, y la categoría es incompatible con la biopsia en el ${pc(L.incompatible_herramienta)}, la tasa de la propia herramienta. Phi-3.5-mini, el que usa la estación, es el de mejor equilibrio.</p>`,
    no: `<p>No muestra que los informes sean útiles en la práctica: nadie con formación radiológica los evaluó. Las tasas salen de un detector automático al que, según la revisión manual, se le escapan algunas invenciones, así que son un mínimo. Cada informe se generó una sola vez.</p>`,
    fuente: '<code>reports/experiments/estudio_informes_llm_resumen.json</code> (Capítulo 4, informes; textos de entrada en el Anexo A)',
  });

  el.innerHTML = `
    <h1 class="vista-titulo">Resultados</h1>
    <p class="vista-intro">Las cifras de la tesis, tal cual salen de los archivos del repositorio. Cada una viene con lo que permite decir y, sobre todo, con lo que no.</p>
    ${auc}${oper}${estab}${loc}${sem}${llm}
    <article class="resultado limites"><div class="res-cuerpo"><h2>Lo que no sabemos</h2>
      <ul><li>Todo se midió en un solo conjunto público (CBIS-DDSM, mamografías digitalizadas de película). No hay validación en otro hospital ni con equipos digitales actuales.</li>
      <li>La comparación con el radiólogo es retrospectiva: usa la categoría BI-RADS que trae el conjunto, no una lectura nueva. No se evaluó si la herramienta ayuda a una persona a decidir mejor.</li>
      <li>La categoría BI-RADS que muestra la herramienta es una regla sobre p_sym, no una calibración clínica. Y p_sym no es una probabilidad calibrada: en las imágenes con un u₀ parecido, la fracción real de malignas suele ser distinta.</li>
      <li>Los casos del Laboratorio se escogieron para enseñar; no son una muestra representativa.</li></ul></div></article>`;
}
