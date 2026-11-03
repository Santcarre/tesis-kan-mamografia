// Laboratorio: recorrido paso a paso de nueve casos de prueba (ocho pacientes), desde la imagen hasta el informe.
import { S, g, sigmoide, tau, uUmbral, fBen, fMal, veredicto, biradsHerramienta, soporte } from './data.js';
import { regla, moverMarcador, curvas, escalaG, moverEscalaG } from './charts.js';
import { $, $$, f, pc, esc, guardar, leer, TIPO, descriptor, esMaligna, claseVeredicto } from './util.js';
import { abrirModal } from './app.js';
import * as api from './api.js';

// Orden fijo (mezclado a propósito para que el orden no delate la respuesta)
const ORDEN = ['p00141-a', 'mal-calc', 'ben-masa', 'fn-masa', 'formula-gana', 'ben-calc', 'p00141-b', 'mal-masa', 'red-gana'];
const PASOS = ['Conoce el caso', 'La red la mira', '512 números', 'Se queda con uno', 'u₀ en la regla', 'La fórmula', 'Dónde pesó cada zona', 'El umbral decide', 'El informe', 'La verdad'];
const estado = { i: 0, paso: 0, apuestas: leer('apuestas', {}) };
let raiz = null;

const lista = () => ORDEN.map(id => S.pac.find(p => p.id === id)).filter(Boolean);
const actual = () => lista()[estado.i];

function imagen(p, capa, alt, clase = '') {
  return `<img class="lab-img ${clase}" src="data/lab/${p.id}/${capa}.png" alt="${esc(alt)}" data-capa="${capa}" width="448" height="448">`;
}
function sinImagen(capa) {
  return `<div class="sin-imagen"><b>Falta esta imagen</b><p>Las mamografías no vienen en el repositorio. Si tienes el conjunto CBIS-DDSM, genéralas con <code>python scripts/generar_laboratorio.py</code>.</p><p class="nota">Capa: ${esc(capa)}. Todo lo demás del recorrido funciona igual.</p></div>`;
}

export async function montar(el) {
  raiz = el;
  const [, id, paso] = (location.hash || '').slice(1).split('/');
  if (id) { const k = lista().findIndex(p => p.id === id); if (k >= 0) estado.i = k; }
  if (paso && !Number.isNaN(+paso)) estado.paso = Math.max(0, Math.min(PASOS.length - 1, +paso - 1));
  el.innerHTML = `
    <h1 class="vista-titulo">Laboratorio</h1>
    <p class="vista-intro">Elige una paciente y recorre, paso a paso, lo que pasa entre la mamografía y el veredicto. Antes de ver nada, puedes apostar. Las nueve están en el conjunto de prueba, que el modelo nunca vio al entrenar, pero las escogimos a mano para enseñar: <b>no representan el desempeño real</b> (eso está en Resultados).</p>
    <div class="lab-sel" id="lab-sel" role="tablist" aria-label="Pacientes"></div>
    <div class="lab-cuerpo">
      <article class="lab-texto hoja" id="lab-texto" aria-live="polite"></article>
      <div class="lab-visual" id="lab-visual"></div>
    </div>
    <nav class="lab-pasos" id="lab-pasos" aria-label="Pasos del recorrido"></nav>`;
  el.addEventListener('error', (e) => {
    if (e.target.classList?.contains('lab-img')) { const c = e.target.dataset.capa; e.target.outerHTML = sinImagen(c); }
  }, true);
  dibujarSelector(); dibujarPasos(); mostrarPaso();
}

export function alMostrar() {
  const [, id, paso] = (location.hash || '').slice(1).split('/');
  const k = id ? lista().findIndex(p => p.id === id) : -1;
  if (k < 0) return;
  const n = paso && !Number.isNaN(+paso) ? Math.max(0, Math.min(PASOS.length - 1, +paso - 1)) : 0;
  if (k !== estado.i || n !== estado.paso) { estado.i = k; estado.paso = n; dibujarSelector(); dibujarPasos(); mostrarPaso(); }
}

function irA(i, paso = 0) {
  estado.i = i; estado.paso = paso;
  history.replaceState(null, '', `#lab/${actual().id}/${estado.paso + 1}`);
  dibujarSelector(); dibujarPasos(); mostrarPaso();
}
function irPaso(n) {
  estado.paso = Math.max(0, Math.min(PASOS.length - 1, n));
  history.replaceState(null, '', `#lab/${actual().id}/${estado.paso + 1}`);
  dibujarPasos(); mostrarPaso();
}

function dibujarSelector() {
  $('#lab-sel', raiz).innerHTML = lista().map((p, k) => {
    const ap = estado.apuestas[p.id];
    return `<button role="tab" aria-selected="${k === estado.i}" data-k="${k}"><b>Caso ${k + 1}</b><span>${esc(p.paciente)} · ${p.tipo === 'mass' ? 'masa' : 'calc.'}</span>${ap ? '<i title="Ya apostaste" class="apostado"></i>' : ''}</button>`;
  }).join('');
  $$('#lab-sel button', raiz).forEach(b => b.addEventListener('click', () => irA(+b.dataset.k, 0)));
}

function dibujarPasos() {
  $('#lab-pasos', raiz).innerHTML = `<button class="btn sec" id="lab-ant" ${estado.paso === 0 ? 'disabled' : ''}>Anterior</button>
    <ol>${PASOS.map((n, k) => `<li><button data-k="${k}" ${k === estado.paso ? 'aria-current="step"' : ''} class="${k < estado.paso ? 'hecho' : ''}"><span>${k + 1}</span><em>${esc(n)}</em></button></li>`).join('')}</ol>
    <button class="btn" id="lab-sig">${estado.paso === PASOS.length - 1 ? 'Siguiente paciente' : 'Siguiente'}</button>`;
  $('#lab-ant', raiz).addEventListener('click', () => irPaso(estado.paso - 1));
  $('#lab-sig', raiz).addEventListener('click', () => {
    if (estado.paso === PASOS.length - 1) irA((estado.i + 1) % lista().length, 0); else irPaso(estado.paso + 1);
  });
  $$('#lab-pasos ol button', raiz).forEach(b => b.addEventListener('click', () => irPaso(+b.dataset.k)));
}

function mostrarPaso() {
  const p = actual(); const fn = PASO_FN[estado.paso];
  const r = fn(p);
  $('#lab-texto', raiz).innerHTML = `<h2>${esc(PASOS[estado.paso])}</h2>${r.texto}`;
  const v = $('#lab-visual', raiz); v.className = 'lab-visual ' + (r.clase || ''); v.innerHTML = r.visual;
  r.despues?.(v, $('#lab-texto', raiz));
  $('#lab-texto', raiz).scrollTop = 0;
}

// ---------------------------------------------------------------------------------------------
const nombreRed = (p) => p.red.pred === 'MALIGNANT' ? 'sospechosa' : 'benigna';
const verdad = (p) => esMaligna(p.patologia) ? 'MALIGNANT' : 'BENIGN';
const textoVer = (v) => v === 'MALIGNANT' ? 'sospechosa' : 'benigna';
const textoBiopsia = (v) => v === 'MALIGNANT' ? 'maligna' : 'benigna';

function ficha(p) {
  const forma = descriptor(p.forma), marg = descriptor(p.margenes);
  const hallazgo = p.tipo === 'mass'
    ? `una <b>masa</b>${forma ? ` de forma <b>${forma}</b>` : ''}${marg ? ` y bordes <b>${marg}</b>` : ''}`
    : `<b>calcificaciones</b>${forma ? ` <b>${forma}</b>` : ''}${marg ? `, de distribución <b>${marg}</b>` : ''}`;
  return hallazgo;
}

const PASO_FN = [
  // 1 ------------------------------------------------------------------------------------------
  (p) => {
    const ap = estado.apuestas[p.id];
    return {
      texto: `<p>Ella es <b>${esc(p.paciente)}</b>. Es un código del conjunto público CBIS-DDSM, no una persona identificable. En su mamografía el radiólogo marcó ${ficha(p)}.</p>
        <p>Lo que <i>no</i> te contamos todavía: qué dijo el radiólogo en su informe y qué encontró la biopsia. Los vemos al final.</p>
        <div class="apuesta" role="group" aria-label="Tu apuesta">
          <p><b>Antes de seguir: ¿qué dirías tú?</b></p>
          <div class="apuesta-botones">
            <button class="btn sec ${ap === 'BENIGN' ? 'on benigno' : ''}" data-ap="BENIGN" aria-pressed="${ap === 'BENIGN'}">Yo diría: benigna</button>
            <button class="btn sec ${ap === 'MALIGNANT' ? 'on sospechoso' : ''}" data-ap="MALIGNANT" aria-pressed="${ap === 'MALIGNANT'}">Yo diría: sospechosa</button>
          </div>
          <p class="nota">Apostar es opcional. Puedes cambiar tu respuesta hasta el último paso.</p>
        </div>`,
      clase: 'luz',
      visual: `<div class="img-marco">${imagen(p, 'original', 'Mamografía de ' + p.paciente)}</div>
        <p class="nota pie-img">Mamografía de ${esc(p.paciente)} reducida a 224×224 píxeles, que es lo único que ve la red.</p>`,
      despues: (v, t) => {
        $$('.apuesta-botones button', t).forEach(b => b.addEventListener('click', () => {
          estado.apuestas[p.id] = b.dataset.ap; guardar('apuestas', estado.apuestas);
          $$('.apuesta-botones button', t).forEach(x => { const on = x === b; x.classList.toggle('on', on); x.classList.toggle('benigno', on && x.dataset.ap === 'BENIGN'); x.classList.toggle('sospechoso', on && x.dataset.ap === 'MALIGNANT'); x.setAttribute('aria-pressed', on); });
          dibujarSelector();
        }));
      },
    };
  },
  // 2 ------------------------------------------------------------------------------------------
  (p) => ({
    texto: `<p>La imagen entra a una red neuronal (una ResNet-18 reentrenada con mamografías). A la salida, la red da un número: qué tan sospechosa le parece. Para esta imagen la red dice <b>${pc(p.red.p, 0)}</b>, es decir, “${nombreRed(p)}”.</p>
      <p>El mapa de calor marca las zonas que más pesaron en esa cuenta.</p>
      <div class="aviso"><p><b>Ojo con los mapas de calor.</b> Parecen explicar la decisión, pero en la tesis medimos si caen sobre la lesión real y no lo hicieron mejor que el azar (Pointing Game, ${S.tesis.ebpg.n} casos). Míralo como una pista, no como una explicación.</p></div>`,
    clase: 'luz',
    visual: `<div class="img-marco">${imagen(p, 'cam', 'Mapa de activación de la red sobre la mamografía')}</div>
      <p class="nota pie-img">Mapa de la red (Grad-CAM). Rojo y amarillo: zonas que más pesaron.</p>`,
  }),
  // 3 ------------------------------------------------------------------------------------------
  (p) => ({
    texto: `<p>Justo antes de dar su veredicto, la red resume la imagen en <b>512 números</b>. Nadie les puso nombre: son lo que la red encontró útil para distinguir una mamografía de otra.</p>
      <p>Cada casilla de la derecha es uno de esos números. En violeta están los que quedan por debajo de lo normal en las imágenes de entrenamiento; en amarillo, los que quedan por encima. Las casillas con borde blanco son las cinco que más mueven el resultado final de la fórmula.</p>
      <p class="nota">Fíjate en que no hay una casilla que diga “aquí hay un tumor”. La información está repartida.</p>`,
    clase: 'luz',
    visual: `<canvas id="gap-lienzo" width="640" height="320" role="img" aria-label="Los 512 descriptores de la imagen como una cuadrícula de colores"></canvas>
      <div class="leyenda-gap"><span><i class="bajo"></i> por debajo de lo normal</span><span><i class="medio"></i> normal</span><span><i class="alto"></i> por encima</span></div>`,
    despues: (v) => pintarGap($('#gap-lienzo', v), p),
  }),
  // 4 ------------------------------------------------------------------------------------------
  (p) => ({
    texto: `<p>Con 512 números es difícil razonar, así que un método llamado <b>S-PCA</b> los comprime a 64 combinaciones, ordenadas por lo útiles que son para separar benignas de malignas.</p>
      <p>Y aquí viene lo curioso: la fórmula final usa <b>solo la primera</b>. Para esta paciente vale <b>u₀ = ${f(p.u0, 2, true)}</b>. Las otras 63 no entran.</p>
      <div class="aviso neutro"><p><b>Cómo leerlo:</b> cuanto <i>más bajo</i> es u₀, más sospechosa. Por eso en toda la app la parte sospechosa está a la izquierda.</p></div>`,
    clase: 'hoja',
    visual: `<svg class="svg-grafica" viewBox="0 0 640 300" role="img" aria-label="Las 64 componentes; solo la primera, u0, entra a la fórmula">${barras64(p)}</svg>
      <p class="nota pie-img">Las 64 componentes de esta imagen. Solo la primera (u₀) llega a la fórmula.</p>`,
  }),
  // 5 ------------------------------------------------------------------------------------------
  (p) => {
    const vec = p.vecindad;
    return {
      texto: `<p>Aquí está u₀ = <b>${f(p.u0, 2, true)}</b> sobre la regla, junto con las ${S.tesis.poblacion.n} imágenes del conjunto de prueba. Las malignas se amontonan a la izquierda y las benignas a la derecha, pero <b>se solapan mucho</b>: ese solape es el error inevitable de la fórmula.</p>
        <p>Esta imagen queda a la derecha del ${pc(p.percentil.maligna_u0_menor_o_igual, 0)} de las malignas y del ${pc(p.percentil.benigna_u0_menor_o_igual, 0)} de las benignas.</p>
        <p>De las <b>${vec.n}</b> imágenes de prueba con un u₀ parecido (±${f(vec.ancho, 1)}), el <b>${pc(vec.frac_mal, 0)}</b> era maligno (IC 95 %: ${pc(vec.ic95[0], 0)} a ${pc(vec.ic95[1], 0)}). Guarda ese número: lo vamos a comparar con lo que dice la fórmula.</p>`,
      clase: 'hoja',
      visual: regla({ marcador: p.u0, etiqueta: 'u₀ = ' + f(p.u0, 2, true), id: 'rl' }),
    };
  },
  // 6 ------------------------------------------------------------------------------------------
  (p) => {
    const t = S.tesis.formula;
    return {
      texto: `<p>La fórmula son dos funciones de u₀: <b>f_ben</b> (qué tan benigna parece) y <b>f_mal</b> (qué tan sospechosa). El resultado es su diferencia: <b>g = f_mal − f_ben</b>.</p>
        <p>Para u₀ = ${f(p.u0, 2, true)}: f_mal = ${f(p.f_mal, 2, true)} y f_ben = ${f(p.f_ben, 2, true)}, así que <b>g = ${f(p.g, 2, true)}</b>. Después se pasa por la sigmoide y sale <b>p_sym = ${pc(p.p_sym)}</b>.</p>
        <p>No hay magia: son sumas de senos, cosenos y un término lineal (${t.caracteres} caracteres en total).</p>
        <details class="mas"><summary>Ver la fórmula completa</summary>
          <p class="formula-txt"><b>f_ben(x) =</b> ${esc(t.expr_ben)}</p><p class="formula-txt"><b>f_mal(x) =</b> ${esc(t.expr_mal)}</p><p class="nota">con x = u₀.</p></details>
        <details class="mas"><summary>Cuánto aporta cada término en este caso</summary>${tablaTerminos(p)}</details>`,
      clase: 'hoja',
      visual: curvas({ u: p.u0 }) + `<p class="nota pie-img">La línea gruesa es g. Los puntos marcan dónde cae esta paciente en cada curva.</p>`,
    };
  },
  // 7 ------------------------------------------------------------------------------------------
  (p) => {
    const e = S.tesis.ebpg, lh = S.tesis.localizador_448;
    return {
      texto: `<p>Como u₀ es una combinación de los 512 números, que a su vez salen de la imagen, se puede preguntar qué zonas empujan g hacia arriba (más sospechosa). Eso es el <b>mapa de la fórmula</b>.</p>
        <div class="aviso"><p><b>Qué tanto creérselo:</b> medimos si este mapa cae sobre la lesión real. Calculado sobre la imagen de 224 píxeles que ve la red, solo el ${pc(e.formula.v)} de su energía cae dentro de la lesión, frente al ${pc(e.azar_mama.v)} que caería si se repartiera al azar por la mama: no localiza mejor que el azar. Por eso la estación lo calcula con la imagen a 448 píxeles; así su foco acierta la región de la lesión en el ${pc(lh.acierto_malignas_448)} de las malignas, frente al ${pc(lh.centro_tejido_malignas)} del centro del tejido. Mejora, pero sigue siendo una ayuda para mirar, no una prueba de que la fórmula “vea” el tumor.</p></div>
        <p class="nota">Los canales que más pesan aquí: ${p.canales_top.slice(0, 3).map(c => `#${c.canal}`).join(', ')} (de los 512 descriptores).</p>`,
      clase: 'luz',
      visual: `<div class="img-marco" id="m7">${imagen(p, 'psikan', 'Mapa de la fórmula sobre la mamografía')}</div>
        <div class="alterna" role="group" aria-label="Capa"><button class="btn sec chico on" data-c="psikan">Mapa de la fórmula</button><button class="btn sec chico" data-c="lesion">Dónde está la lesión de verdad</button></div>
        <p class="nota pie-img" id="m7-pie">Mapa de la fórmula (ψ_KAN). Más cálido = empuja g hacia sospechosa.</p>`,
      despues: (v) => {
        const marco = $('#m7', v);
        $$('.alterna button', v).forEach(b => b.addEventListener('click', () => {
          $$('.alterna button', v).forEach(x => x.classList.toggle('on', x === b));
          marco.innerHTML = imagen(p, b.dataset.c, b.dataset.c === 'lesion' ? 'Mamografía con el contorno de la lesión real' : 'Mapa de la fórmula sobre la mamografía');
          $('#m7-pie', v).textContent = b.dataset.c === 'lesion' ? 'Contorno de la lesión real según la máscara de CBIS-DDSM. Compáralo con el mapa.' : 'Mapa de la fórmula (ψ_KAN). Más cálido = empuja g hacia sospechosa.';
        }));
      },
    };
  },
  // 8 ------------------------------------------------------------------------------------------
  (p) => {
    const dist = p.u0_para_cambiar;
    return {
      texto: `<p>El veredicto es una comparación: ¿g llega al <b>umbral</b> (${f(tau(), 2)}, que equivale a p_sym = ${pc(S.tesis.formula.tau_prob)})? Ese umbral se fijó en validación, sin mirar el conjunto de prueba.</p>
        <p>Aquí g = <b>${f(p.g, 2, true)}</b>, así que el veredicto de la fórmula es <b>${textoVer(p.veredicto)}</b>. Para que cambiara, u₀ tendría que moverse ${f(Math.abs(dist), 2)} (de ${f(p.u0, 2, true)} a ${f(uUmbral(), 2, true)}).</p>
        <div class="juega"><label for="mueve">¿Y si u₀ fuera otro? Muévelo y mira:</label>
          <input class="rango" type="range" id="mueve" min="${soporte()[0]}" max="${soporte()[1]}" step="0.01" value="${p.u0}">
          <p id="mueve-lectura" class="lectura-mini"></p></div>
        <p>La herramienta también da una categoría <b>BI-RADS</b> orientativa: aquí sale <b>${p.birads_herramienta}</b>. Sale del mismo veredicto (2 o 3 si es benigna, 4 o 5 si es sospechosa) y de dos cortes de p_sym que se ajustaron en validación para parecerse a la categoría que puso el radiólogo. No es un riesgo de malignidad.</p>`,
      clase: 'hoja',
      visual: escalaG({ gv: p.g, id: 'eg' }) + franjaBirads(p),
      despues: (v, t) => {
        const ctl = $('#mueve', t); const svg = $('#eg', v);
        const act = () => {
          const u = +ctl.value; const gv = g(u), pp = sigmoide(gv), ver = veredicto(gv);
          moverEscalaG(svg, gv);
          $('#mueve-lectura', t).innerHTML = `u₀ = <b>${f(u, 2, true)}</b> → g = ${f(gv, 2, true)} → p_sym = ${pc(pp)} → <span class="etiqueta ${claseVeredicto(ver)}">${textoVer(ver)}</span> · BI-RADS de la herramienta: ${biradsHerramienta(pp)}`;
        };
        ctl.addEventListener('input', act); act();
      },
    };
  },
  // 9 ------------------------------------------------------------------------------------------
  (p) => {
    const txt = informe(p);
    return {
      texto: `<p>Con todo lo anterior se arma un informe de apoyo. Este es un <b>borrador por plantilla</b>: no lo escribió un médico ni un modelo de lenguaje, solo junta los números del recorrido con redacción prudente.</p>
        <p>En la app local, el mismo caso puede convertirse en un informe en PDF (y, si tienes un modelo de lenguaje local instalado, en uno redactado por el modelo; en el estudio de informes de la tesis, con esta evidencia los modelos dejaron casi de inventar rasgos de la lesión, aunque copian la región y la categoría con sus errores).</p>
        <div class="acciones">
          <button class="btn sec" id="inf-copiar">Copiar el informe</button>
          <button class="btn" id="inf-pdf" ${S.backend ? '' : 'disabled'} title="${S.backend ? '' : 'Necesita la app local de Prometeo'}">Crear el PDF con la app</button>
        </div>
        ${S.backend ? '' : '<p class="nota">El PDF solo está en la app local, porque se genera con los datos de CBIS-DDSM.</p>'}`,
      clase: 'hoja',
      visual: `<div class="informe" tabindex="0"><pre id="inf-txt">${esc(txt)}</pre></div>`,
      despues: (v, t) => {
        $('#inf-copiar', t).addEventListener('click', async (e) => { try { await navigator.clipboard.writeText(txt); e.target.textContent = 'Copiado'; } catch (_) { e.target.textContent = 'No se pudo copiar'; } setTimeout(() => { e.target.textContent = 'Copiar el informe'; }, 1800); });
        if (S.backend) $('#inf-pdf', t).addEventListener('click', async (e) => {
          const b = e.target; b.disabled = true; b.textContent = 'Creando…';
          try {
            const rep = await api.generateReport(p.idx, false); const pdf = await api.generatePDF(p.idx, rep.report_text);
            abrirModal('Informe creado', rep.report_text + '\n\nArchivo: ' + pdf.pdf_path);
          } catch (err) { abrirModal('No se pudo crear el PDF', String(err.message || err)); }
          b.disabled = false; b.textContent = 'Crear el PDF con la app';
        });
      },
    };
  },
  // 10 -----------------------------------------------------------------------------------------
  (p) => {
    const v = verdad(p); const ap = estado.apuestas[p.id];
    const fila = (quien, dice, ok) => `<tr><th scope="row">${quien}</th><td>${dice}</td><td>${ok === null ? '' : `<span class="etiqueta ${ok ? 'benigno' : 'sospechoso'}">${ok ? 'acierta' : 'falla'}</span>`}</td></tr>`;
    const tally = tabla();
    return {
      texto: `<p>La biopsia dijo: <b>${textoBiopsia(v)}</b>. El radiólogo le había puesto <b>BI-RADS ${p.birads_radiologo}</b>.</p>
        <p class="caso-titulo"><b>${esc(p.titulo)}.</b> ${esc(p.gancho)}</p>
        <p>${esc(p.leccion)}</p>
        ${tally}`,
      clase: 'hoja',
      visual: `<table class="tabla comparacion"><thead><tr><th>Quién</th><th>Qué dice</th><th></th></tr></thead><tbody>
          ${fila('Tú', ap ? textoVer(ap) : '<span class="nota">no apostaste</span>', ap ? ap === v : null)}
          ${fila('La fórmula', `${textoVer(p.veredicto)} (p_sym ${pc(p.p_sym, 0)})`, p.acierto_formula)}
          ${fila('La red completa', `${nombreRed(p)} (${pc(p.red.p, 0)})`, p.acierto_red)}
          ${fila('El radiólogo', `BI-RADS ${p.birads_radiologo}`, null)}
          ${fila('La biopsia', `<b>${textoBiopsia(v)}</b>`, null)}
        </tbody></table>
        <p class="nota pie-img">Un caso no demuestra nada. Las cifras de verdad están en Resultados.</p>`,
    };
  },
];

// ---- piezas -----------------------------------------------------------------------------------
function tabla() {
  const hechas = lista().filter(p => estado.apuestas[p.id]);
  if (!hechas.length) return '';
  const tu = hechas.filter(p => estado.apuestas[p.id] === verdad(p)).length;
  const fo = hechas.filter(p => p.acierto_formula).length, re = hechas.filter(p => p.acierto_red).length;
  return `<div class="aviso neutro"><p><b>Tu marcador:</b> en los ${hechas.length} casos en que apostaste, acertaste ${tu}; la fórmula acertó ${fo} y la red completa ${re}. Son muy pocos casos para sacar conclusiones.</p></div>`;
}

function pintarGap(cv, p) {
  const ctx = cv.getContext('2d'); const cols = 32, filas = 16; const w = cv.width / cols, h = cv.height / filas;
  const top = new Set(p.canales_top.map(c => c.canal));
  const col = (z) => {
    const t = Math.max(-1, Math.min(1, z / 2.5)); const base = [27, 38, 44];
    const dest = t < 0 ? [123, 95, 208] : [242, 217, 139]; const a = Math.abs(t);
    return `rgb(${base.map((b, k) => Math.round(b + (dest[k] - b) * a)).join(',')})`;
  };
  ctx.clearRect(0, 0, cv.width, cv.height);
  p.gap_z.forEach((z, i) => { const x = (i % cols) * w, y = Math.floor(i / cols) * h; ctx.fillStyle = col(z); ctx.fillRect(x + 1, y + 1, w - 2, h - 2); });
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 2.4;
  top.forEach(i => { const x = (i % cols) * w, y = Math.floor(i / cols) * h; ctx.strokeRect(x + 1.5, y + 1.5, w - 3, h - 3); });
}

function barras64(p) {
  const z = p.z64; const L = 30, B = 40, T = 14, W = 640, H = 300; const m = Math.max(4, ...z.map(Math.abs));
  const x = (i) => L + i * ((W - L - 8) / 64), y = (v) => T + (H - T - B) * (0.5 - v / (2 * m));
  let s = `<line x1="${L}" x2="${W - 8}" y1="${y(0)}" y2="${y(0)}" stroke="var(--tinta-3)"/>`;
  z.forEach((v, i) => { const y0 = y(0), y1 = y(v); s += `<rect x="${x(i) + 1}" y="${Math.min(y0, y1)}" width="${(W - L - 8) / 64 - 2}" height="${Math.max(1, Math.abs(y1 - y0))}" fill="${i === 0 ? 'var(--tinta)' : 'var(--linea)'}"/>`; });
  s += `<text x="${x(0) + 8}" y="${y(z[0]) + (z[0] >= 0 ? -8 : 20)}" style="fill:var(--tinta);font-weight:700">u₀ = ${f(z[0], 2, true)}</text>`;
  s += `<text x="${L}" y="${H - 14}">componente 1</text><text x="${W - 8}" y="${H - 14}" text-anchor="end">componente 64</text>`;
  s += `<text x="4" y="${T + 10}">+</text><text x="6" y="${H - B - 4}">−</text>`;
  return s;
}

function tablaTerminos(p) {
  const maxv = Math.max(...p.terminos.map(t => Math.abs(t.valor)), 0.01);
  const nombre = (t) => t.k === 'const' ? 'constante' : t.k === 'lin' ? 'término lineal' : `${t.k === 'cos' ? 'coseno' : 'seno'} (frecuencia ${f(t.w, 2)})`;
  return `<table class="tabla"><thead><tr><th>Término</th><th>Lado</th><th class="num">Aporte a g</th></tr></thead><tbody>${p.terminos.map(t =>
    `<tr><td>${nombre(t)}</td><td>${t.lado === 'mal' ? 'f_mal' : 'f_ben (resta)'}</td><td class="num"><span class="mini-barra ${t.valor >= 0 ? 'mas' : 'menos'}" style="width:${Math.round(Math.abs(t.valor) / maxv * 70) + 2}px"></span> ${f(t.valor, 3, true)}</td></tr>`).join('')}</tbody>
    <tfoot><tr><td colspan="2"><b>Suma = g</b></td><td class="num"><b>${f(p.g, 3, true)}</b></td></tr></tfoot></table>`;
}

function franjaBirads(p) {
  // Cortes de la herramienta (desktop_app/ui/data/tesis.json, generados desde src/reporting/birads_herramienta.py)
  const cs = S.tesis.birads_cortes; const cortes = [0, ...cs.map(c => c.hasta === null ? 1 : c.hasta)]; const cats = cs.map(c => c.cat);
  return `<div class="birads" role="img" aria-label="Categorías BI-RADS de la herramienta sobre p_sym"><div class="birads-tira">${cats.map((c, k) =>
    `<div class="${c === p.birads_herramienta ? 'on' : ''}" style="flex:${(cortes[k + 1] - cortes[k]) * 100}"><b>${c}</b><span>${Math.round(cortes[k] * 100)}–${Math.round(cortes[k + 1] * 100)} %</span></div>`).join('')}</div>
    <p class="nota">Categoría BI-RADS orientativa según p_sym = ${pc(p.p_sym, 0)}. El corte entre 3 y 4 es el umbral del veredicto (${pc(S.tesis.formula.tau_prob, 0)}).</p></div>`;
}

function informe(p) {
  const t = S.tesis;
  return `INFORME DE APOYO (borrador por plantilla · no es un informe clínico)
Caso: ${p.paciente} · ${TIPO[p.tipo]} · conjunto de prueba de CBIS-DDSM

HALLAZGO DESCRITO EN LOS DATOS
${p.tipo === 'mass' ? 'Masa' : 'Calcificaciones'}${p.forma ? ', ' + descriptor(p.forma) : ''}${p.margenes ? ', ' + descriptor(p.margenes) : ''}.

LO QUE CALCULA EL MODELO
u₀ = ${f(p.u0, 2, true)}  →  g(u₀) = ${f(p.g, 2, true)}  →  p_sym = ${pc(p.p_sym)}
Umbral de decisión (Youden en validación): g ≥ ${f(tau(), 2)}  (p_sym ≥ ${pc(t.formula.tau_prob)})
Veredicto de la fórmula: ${p.veredicto === 'MALIGNANT' ? 'COMPATIBLE CON LESIÓN SOSPECHOSA' : 'COMPATIBLE CON LESIÓN BENIGNA'}.
Categoría BI-RADS orientativa de la herramienta: ${p.birads_herramienta} (anclada en el veredicto y ajustada a la valoración del radiólogo; no es un riesgo de malignidad).
Red neuronal completa, como contraste: ${pc(p.red.p, 0)} de sospecha.

LIMITACIONES
La fórmula tiene AUC ${f(t.auc.formula.auc, 2)} en el conjunto de prueba (sensibilidad ${pc(t.poblacion.sens)}, especificidad ${pc(t.poblacion.esp)}).
En imágenes con un u₀ parecido, ${pc(p.vecindad.frac_mal, 0)} eran malignas; p_sym no es una probabilidad calibrada.
No se evaluó de forma prospectiva ni con lectores humanos. No sustituye el juicio del radiólogo.`;
}
