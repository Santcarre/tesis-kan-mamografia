// Utilidades de formato y DOM. Números con coma decimal y signo menos tipográfico.
export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
export const MENOS = '−';

export function f(x, d = 2, conSigno = false) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  const abs = Math.abs(x).toFixed(d).replace('.', ',');
  const esCero = Number(Math.abs(x).toFixed(d)) === 0;
  if (x < 0 && !esCero) return MENOS + abs;
  return (conSigno && !esCero ? '+' : '') + abs;
}
export const pc = (x, d = 1) => (x === null || x === undefined || Number.isNaN(x)) ? '—' : (x * 100).toFixed(d).replace('.', ',') + ' %';
export const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
export const clamp = (x, a, b) => Math.min(b, Math.max(a, x));

export function guardar(clave, valor) { try { localStorage.setItem('prometeo.' + clave, JSON.stringify(valor)); } catch (_) { /* sin almacenamiento */ } }
export function leer(clave, porDefecto) {
  try { const v = localStorage.getItem('prometeo.' + clave); return v === null ? porDefecto : JSON.parse(v); } catch (_) { return porDefecto; }
}

export const TIPO = { mass: 'masa', calc: 'calcificaciones' };
export const PATOLOGIA = (p) => String(p || '').toUpperCase().includes('MALIGNANT') ? 'maligna' : 'benigna';
export const esMaligna = (p) => String(p || '').toUpperCase().startsWith('MALIGNANT');
export const VEREDICTO = (v) => v === 'MALIGNANT' ? 'sospechosa' : 'benigna';
export const claseVeredicto = (v) => v === 'MALIGNANT' ? 'sospechoso' : 'benigno';

// Traducciones breves de los descriptores BI-RADS que trae CBIS-DDSM
const DESC = {
  AMORPHOUS: 'amorfas', PLEOMORPHIC: 'pleomórficas', FINE_LINEAR_BRANCHING: 'lineales finas ramificadas',
  ROUND_AND_REGULAR: 'redondas y regulares', PUNCTATE: 'puntiformes', CLUSTERED: 'agrupadas', LINEAR: 'lineales',
  REGIONAL: 'regionales', SEGMENTAL: 'segmentarias', DIFFUSELY_SCATTERED: 'difusas',
  IRREGULAR: 'irregular', LOBULATED: 'lobulada', OVAL: 'ovalada', ROUND: 'redonda', LYMPH_NODE: 'ganglio linfático',
  FOCAL_ASYMMETRIC_DENSITY: 'densidad focal asimétrica', ARCHITECTURAL_DISTORTION: 'distorsión arquitectural',
  ASYMMETRIC_BREAST_TISSUE: 'tejido asimétrico',
  SPICULATED: 'espiculados', ILL_DEFINED: 'mal definidos', CIRCUMSCRIBED: 'bien delimitados', MICROLOBULATED: 'microlobulados',
  OBSCURED: 'oscurecidos',
};
export function descriptor(s) {
  if (!s) return null;
  return String(s).split(/[-]/).map(t => DESC[t.trim()] || t.trim().toLowerCase().replace(/_/g, ' ')).join(' / ');
}
