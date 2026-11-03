// Arranque: carga de datos, navegación por hash y montaje perezoso de cada vista.
import { $, $$, esc } from './util.js';
import { cargar, S } from './data.js';

const VISTAS = {
  inicio: () => import('./home.js'),
  lab: () => import('./lab.js'),
  formula: () => import('./formula.js'),
  resultados: () => import('./thesis.js'),
  explorador: () => import('./cohort.js'),
  externo: () => import('./external.js'),
};
const montadas = new Set();

export function abrirModal(titulo, texto, pie = '') {
  $('#modal-titulo').textContent = titulo; $('#modal-cuerpo').textContent = texto; $('#modal-pie').innerHTML = pie;
  $('#modal').hidden = false; $('#modal-cerrar').focus();
}

async function mostrar(nombre) {
  if (!VISTAS[nombre]) nombre = 'inicio';
  $$('.vista').forEach(v => { v.hidden = v.id !== 'vista-' + nombre; });
  $$('.nav button').forEach(b => { if (b.dataset.vista === nombre) b.setAttribute('aria-current', 'page'); else b.removeAttribute('aria-current'); });
  const el = $('#vista-' + nombre);
  if (!montadas.has(nombre)) {
    montadas.add(nombre);
    try { const m = await VISTAS[nombre](); await m.montar(el); }
    catch (e) { console.error(e); el.innerHTML = `<div class="vacio"><h3>Esta sección no pudo cargarse</h3><p>${esc(e.message)}</p></div>`; montadas.delete(nombre); }
  } else {
    const m = await VISTAS[nombre](); m.alMostrar?.(el);
  }
  window.scrollTo({ top: 0 });
}

function ruta() { return (location.hash || '#inicio').slice(1).split('/')[0]; }

async function arrancar() {
  try {
    await cargar();
  } catch (e) {
    $('#contenido').innerHTML = `<div class="vacio"><h3>No se pudieron leer los datos</h3>
      <p>La interfaz necesita los archivos de <code>data/</code>. Si abriste <code>index.html</code> con doble clic, el navegador lo bloquea: sírvela con un servidor local.</p>
      <p><code>python -m http.server -d desktop_app/ui 8080</code> y abre <code>http://localhost:8080</code>. Si faltan los JSON, genera con <code>python scripts/generar_datos_ui.py</code>.</p>
      <p>${esc(e.message)}</p></div>`;
    $('#estado span').textContent = 'Sin datos';
    return;
  }
  const est = $('#estado');
  est.classList.toggle('en-linea', S.backend);
  $('span', est).textContent = S.backend ? 'App local' : 'Modo web';
  est.title = S.backend ? 'El servidor local de Prometeo responde: están disponibles el Explorador y el Estudio externo.' : 'Estás viendo la versión estática: el Laboratorio, la Fórmula y los Resultados funcionan completos. El Explorador y el Estudio externo necesitan la app local.';
  $$('.nav .local').forEach(s => { s.hidden = S.backend; });
  $$('.nav button').forEach(b => b.addEventListener('click', () => { location.hash = '#' + b.dataset.vista; }));
  window.addEventListener('hashchange', () => mostrar(ruta()));
  $('#modal-cerrar').addEventListener('click', () => { $('#modal').hidden = true; });
  $('#modal').addEventListener('click', (e) => { if (e.target.id === 'modal') $('#modal').hidden = true; });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') $('#modal').hidden = true; });
  mostrar(ruta());
}
arrancar();
