// Explorador (solo con la app local): recorre la cohorte CBIS-DDSM de prueba y ve el veredicto de la fórmula en cada caso.
import { S, tau } from './data.js';
import { regla } from './charts.js';
import { $, $$, f, pc, esc, TIPO, esMaligna, claseVeredicto } from './util.js';
import { abrirModal } from './app.js';
import * as api from './api.js';

const CAPAS = [['clean', 'Original'], ['cam', 'Mapa de la red'], ['wsss', 'Foco'], ['psikan', 'Mapa de la fórmula'], ['zoom', 'Lupa']];
let casos = [], activo = null, capa = 'clean', verEtiqueta = false, revelado = false, informe = null;

export function necesitaApp(el, titulo, texto) {
  el.innerHTML = `<h1 class="vista-titulo">${esc(titulo)}</h1><div class="vacio"><h3>Esta sección necesita la app local</h3><p>${texto}</p>
    <p>Para abrirla, desde la raíz del repositorio: <code>python desktop_app/run_prometeo.py</code> (o <code>desktop_app\\run_prometeo.bat</code> en Windows). Necesita las mamografías de CBIS-DDSM y los puntos de control del modelo; la guía está en <code>desktop_app/README.md</code>.</p>
    <p>Mientras tanto, el <a href="#lab">Laboratorio</a>, la <a href="#formula">Fórmula</a> y los <a href="#resultados">Resultados</a> funcionan sin ella.</p></div>`;
}

export async function montar(el) {
  if (!S.backend) return necesitaApp(el, 'Explorador', 'Aquí se recorren los 368 casos de prueba con máscara de lesión, con las capas de la red y de la fórmula calculadas en el momento.');
  el.innerHTML = `
    <h1 class="vista-titulo">Explorador</h1>
    <p class="vista-intro">Los casos de prueba de CBIS-DDSM con máscara de lesión. Por defecto la biopsia está oculta en la lista para que puedas apostar antes de mirar.</p>
    <div class="explorador">
      <aside class="hoja exp-lista">
        <input type="search" id="ex-buscar" placeholder="Buscar paciente (P_00141)" aria-label="Buscar paciente">
        <div class="exp-filtros" role="group" aria-label="Tipo de lesión">
          <button class="btn sec chico on" data-t="all">Todas</button><button class="btn sec chico" data-t="mass">Masas</button><button class="btn sec chico" data-t="calc">Calcificaciones</button>
        </div>
        <label class="exp-check"><input type="checkbox" id="ex-etq"> Mostrar la biopsia en la lista</label>
        <ul id="ex-ul"></ul>
      </aside>
      <section class="exp-centro">
        <div class="luz exp-visor"><div id="ex-img"></div></div>
        <div class="exp-capas" role="group" aria-label="Capa">${CAPAS.map(([k, n]) => `<button class="btn sec chico ${k === 'clean' ? 'on' : ''}" data-capa="${k}">${n}</button>`).join('')}</div>
        <p class="nota" id="ex-pie"></p>
      </section>
      <aside class="exp-panel" id="ex-panel"><div class="vacio"><p>Elige un caso de la lista.</p></div></aside>
    </div>`;
  try { casos = await api.getCohort(); } catch (e) { $('#ex-ul', el).innerHTML = `<li class="nota">${esc(e.message)}</li>`; return; }
  lista(el);
  $('#ex-buscar', el).addEventListener('input', () => lista(el));
  $('#ex-etq', el).addEventListener('change', (e) => { verEtiqueta = e.target.checked; lista(el); });
  $$('.exp-filtros button', el).forEach(b => b.addEventListener('click', () => { $$('.exp-filtros button', el).forEach(x => x.classList.toggle('on', x === b)); lista(el); }));
  $$('.exp-capas button', el).forEach(b => b.addEventListener('click', () => { capa = b.dataset.capa; $$('.exp-capas button', el).forEach(x => x.classList.toggle('on', x === b)); if (activo !== null) pintar(el); }));
  const p141 = casos.find(c => (c.patient_id || '').includes('P_00141')) || casos[0];
  if (p141) elegir(el, p141.index);
}

function lista(el) {
  const q = ($('#ex-buscar', el).value || '').toLowerCase().trim(); const t = ($('.exp-filtros .on', el) || {}).dataset?.t || 'all';
  const fl = casos.filter(c => (c.patient_id || '').toLowerCase().includes(q) && (t === 'all' || c.type === t));
  $('#ex-ul', el).innerHTML = fl.length ? fl.map(c => `<li><button data-i="${c.index}" class="${c.index === activo ? 'sel' : ''}"><b>${esc(c.patient_id)}</b><span>${TIPO[c.type] || c.type} · ${esc(c.laterality || '')} ${esc(c.view || '')}</span>${verEtiqueta ? `<em class="etiqueta ${esMaligna(c.pathology) ? 'sospechoso' : 'benigno'}">${esMaligna(c.pathology) ? 'maligna' : 'benigna'}</em>` : ''}</button></li>`).join('') : '<li class="nota">Ningún caso coincide.</li>';
  $$('#ex-ul button', el).forEach(b => b.addEventListener('click', () => elegir(el, +b.dataset.i)));
}

async function elegir(el, i) {
  activo = i; revelado = false; informe = null; lista(el);
  $('#ex-panel', el).innerHTML = '<div class="vacio"><p>Calculando…</p></div>';
  try {
    const d = await api.getCase(i); el.__caso = d; panel(el, d); pintar(el);
  } catch (e) { $('#ex-panel', el).innerHTML = `<div class="aviso sospechoso"><p>${esc(e.message)}</p></div>`; }
}

async function pintar(el) {
  const cont = $('#ex-img', el); cont.style.opacity = .5;
  try { const r = await api.getRender(activo, capa); cont.innerHTML = `<img class="lab-img" alt="Capa ${esc(capa)}" src="${r.image_b64}">`; }
  catch (e) { cont.innerHTML = `<p class="nota" style="padding:20px">${esc(e.message)}</p>`; }
  cont.style.opacity = 1;
}

function panel(el, d) {
  const p = d.prediction, m = d.metadata, v = p.verdict;
  const fallback = p.decision_source !== 'formula';
  $('#ex-panel', el).innerHTML = `
    <div class="hoja panel">
      <h2>${esc(d.patient_id)}</h2>
      <p class="nota">${TIPO[m.abnormality_type] || m.abnormality_type} · ${esc(m.laterality)} ${esc(m.view)}</p>
      ${fallback ? '<div class="aviso"><p><b>Sin la fórmula.</b> No se encontraron los artefactos de la fórmula; el veredicto sale de la red completa.</p></div>' : ''}
      <div class="lectura-cifras">
        <div><span>u₀</span><b>${f(p.u0, 2, true)}</b></div><div><span>g</span><b>${f(p.g_logit, 2, true)}</b></div><div><span>p_sym</span><b>${pc(p.p_sym)}</b></div>
        <div class="veredicto ${claseVeredicto(v)}"><span>veredicto</span><b>${v === 'MALIGNANT' ? 'sospechosa' : 'benigna'}</b></div>
      </div>
      <p class="nota">Umbral: g ≥ ${f(p.threshold_logit, 2)} (p_sym ≥ ${pc(p.threshold_prob)}). Red completa: ${pc(p.prob_net_robust, 0)} de sospecha.</p>
      ${p.u0 !== null && p.u0 !== undefined ? regla({ marcador: p.u0, etiqueta: 'u₀ = ' + f(p.u0, 2, true), id: 'rx', alto: 200 }) : ''}
      <p><b>${esc(p.birads_label)}</b><br><span class="nota">${esc(p.birads_note || '')}</span></p>
      <div class="acciones">
        <button class="btn sec chico" id="ex-rev">${revelado ? 'Ocultar la verdad' : 'Revelar biopsia y BI-RADS del radiólogo'}</button>
      </div>
      <div id="ex-verdad" ${revelado ? '' : 'hidden'}><p>Biopsia: <b>${esMaligna(m.pathology_gt) ? 'maligna' : 'benigna'}</b>. Radiólogo: BI-RADS <b>${m.assessment_gt || '—'}</b>.</p></div>
      <div class="acciones">
        <button class="btn sec" id="ex-inf">Ver informe</button><button class="btn" id="ex-pdf">Crear PDF</button>
      </div>
    </div>`;
  $('#ex-rev', el).addEventListener('click', () => { revelado = !revelado; panel(el, d); });
  $('#ex-inf', el).addEventListener('click', async () => { try { const r = await api.generateReport(activo, false); informe = r.report_text; abrirModal('Informe (' + r.engine + ')', r.report_text); } catch (e) { abrirModal('No se pudo crear el informe', e.message); } });
  $('#ex-pdf', el).addEventListener('click', async (e) => {
    e.target.disabled = true;
    try { if (!informe) informe = (await api.generateReport(activo, false)).report_text; const r = await api.generatePDF(activo, informe); abrirModal('PDF creado', 'Archivo: ' + r.pdf_path); }
    catch (err) { abrirModal('No se pudo crear el PDF', err.message); }
    e.target.disabled = false;
  });
}
