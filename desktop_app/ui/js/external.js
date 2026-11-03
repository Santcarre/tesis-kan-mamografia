// Estudio externo (solo con la app local): sube un DICOM o una imagen y mira el veredicto de la fórmula.
import { S } from './data.js';
import { regla } from './charts.js';
import { $, f, pc, esc, claseVeredicto } from './util.js';
import { necesitaApp } from './cohort.js';
import * as api from './api.js';

export async function montar(el) {
  if (!S.backend) return necesitaApp(el, 'Estudio externo', 'Aquí puedes subir una mamografía propia (DICOM, JPG o PNG) y ver qué calcula la fórmula.');
  el.innerHTML = `
    <h1 class="vista-titulo">Estudio externo</h1>
    <p class="vista-intro">Sube una mamografía (.dcm, .jpg o .png). Se estandariza a 224×224, pasa por la red y por la fórmula, y verás el mismo recorrido del Laboratorio. Los archivos se procesan en tu computador y no salen de él.</p>
    <div class="aviso"><p><b>Importante:</b> el modelo se entrenó con mamografías digitalizadas de película (CBIS-DDSM). Con imágenes de otros equipos no se ha validado y el resultado puede no significar nada.</p></div>
    <label class="dropzone" id="ext-drop" tabindex="0"><input type="file" id="ext-file" accept=".dcm,.dicom,.jpg,.jpeg,.png" hidden>
      <b>Suelta aquí un archivo o haz clic para elegirlo</b><span class="nota">DICOM, JPG o PNG</span></label>
    <div id="ext-demos" class="exp-capas"></div>
    <div id="ext-res" aria-live="polite"></div>`;
  const drop = $('#ext-drop', el), inp = $('#ext-file', el);
  drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { inp.click(); e.preventDefault(); } });
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('sobre'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('sobre'));
  drop.addEventListener('drop', (e) => { e.preventDefault(); drop.classList.remove('sobre'); if (e.dataTransfer.files[0]) subir(el, e.dataTransfer.files[0]); });
  inp.addEventListener('change', () => { if (inp.files[0]) subir(el, inp.files[0]); });
  try {
    const demos = await api.getDemos();
    $('#ext-demos', el).innerHTML = demos.length ? '<span class="nota">O prueba un ejemplo:</span> ' + demos.map(d => `<button class="btn sec chico" data-id="${esc(d.id)}">${esc(d.title || d.filename)}</button>`).join('') : '';
    el.querySelectorAll('#ext-demos button').forEach(b => b.addEventListener('click', () => correr(el, () => api.runDemo(b.dataset.id), b.textContent)));
  } catch (_) { /* sin ejemplos */ }
}

const subir = (el, file) => correr(el, () => api.uploadStudy(file), file.name);

async function correr(el, fn, nombre) {
  const res = $('#ext-res', el); res.innerHTML = '<div class="vacio"><p>Analizando…</p></div>';
  try { mostrar(res, await fn(), nombre); }
  catch (e) { res.innerHTML = `<div class="aviso sospechoso"><p><b>No se pudo analizar el archivo.</b> ${esc(e.message)}</p></div>`; }
}

function mostrar(res, r, nombre) {
  const v = r.verdict;
  const capas = [['image_b64', 'Original'], ['cam_b64', 'Mapa de la red'], ['wsss_b64', 'Foco'], ['zoom_b64', 'Lupa'], ['plots_b64', 'Curvas']].filter(([k]) => r[k]);
  res.innerHTML = `<div class="ext-res">
    <div class="luz exp-visor"><img class="lab-img" id="ext-img" alt="Estudio analizado" src="${r.image_b64}"></div>
    <div class="hoja panel">
      <h2>${esc(nombre)}</h2>
      <p class="nota">${esc(r.file_format || '')} ${r.orig_resolution ? '· ' + esc(r.orig_resolution) : ''}</p>
      ${r.decision_source !== 'formula' ? '<div class="aviso"><p><b>Sin la fórmula.</b> No se encontraron sus artefactos; el veredicto sale de la red completa.</p></div>' : ''}
      <div class="lectura-cifras">
        <div><span>u₀</span><b>${f(r.u0, 2, true)}</b></div><div><span>g</span><b>${f(r.g_logit, 2, true)}</b></div><div><span>p_sym</span><b>${pc(r.p_sym)}</b></div>
        <div class="veredicto ${claseVeredicto(v)}"><span>veredicto</span><b>${v === 'MALIGNANT' ? 'sospechosa' : 'benigna'}</b></div>
      </div>
      <p class="nota">Red completa: ${pc(r.prob_net_robust, 0)} de sospecha. ${esc(r.birads_label || '')}. ${esc(r.birads_note || '')}</p>
      ${r.u0 !== null && r.u0 !== undefined ? regla({ marcador: r.u0, etiqueta: 'u₀ = ' + f(r.u0, 2, true), id: 'rext', alto: 200 }) : ''}
      <div class="exp-capas" id="ext-capas">${capas.map(([k, n], i) => `<button class="btn sec chico ${i === 0 ? 'on' : ''}" data-k="${k}">${n}</button>`).join('')}</div>
    </div></div>`;
  res.querySelectorAll('#ext-capas button').forEach(b => b.addEventListener('click', () => {
    res.querySelectorAll('#ext-capas button').forEach(x => x.classList.toggle('on', x === b)); $('#ext-img', res).src = r[b.dataset.k];
  }));
}
