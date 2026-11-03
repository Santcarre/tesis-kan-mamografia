// Carga de datos y aritmética de la fórmula canónica. Todo corre en el navegador:
// la fórmula depende solo de u0 y está guardada como lista de términos (tesis.json).
import { f, pc } from './util.js';

export const S = { tesis: null, glob: null, pac: null, backend: false, gArr: null };

async function json(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`No se pudo leer ${url} (${r.status})`);
  return r.json();
}

export async function cargar() {
  const [tesis, glob, pac] = await Promise.all([
    json('data/tesis.json'), json('data/lab_global.json'), json('data/lab_pacientes.json'),
  ]);
  S.tesis = tesis; S.glob = glob; S.pac = pac.pacientes;
  S.gArr = glob.u0.map(u => g(u));
  S.backend = await detectarBackend();
  return S;
}

async function detectarBackend() {
  try {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), 1500);
    const r = await fetch('/api/health', { signal: ctl.signal });
    clearTimeout(t);
    if (!r.ok) return false;
    const j = await r.json();
    return !!j && typeof j === 'object' && 'status' in j;
  } catch (_) { return false; }
}

// ---- fórmula -------------------------------------------------------------
function evalTerminos(ts, x) {
  let t = 0;
  for (const q of ts) {
    if (q.k === 'const') t += q.a;
    else if (q.k === 'lin') t += q.a * x;
    else if (q.k === 'cos') t += q.a * Math.cos(q.w * x + q.p);
    else t += q.a * Math.sin(q.w * x + q.p);
  }
  return t;
}
export const fBen = (u) => evalTerminos(S.tesis.formula.f_ben, u);
export const fMal = (u) => evalTerminos(S.tesis.formula.f_mal, u);
export const g = (u) => fMal(u) - fBen(u);
export const sigmoide = (z) => 1 / (1 + Math.exp(-z));
export const tau = () => S.tesis.formula.tau_logit;
export const uUmbral = () => S.tesis.formula.u0_umbral;
export const soporte = () => S.tesis.formula.u0_soporte;

export function veredicto(gv, t = tau()) { return gv >= t ? 'MALIGNANT' : 'BENIGN'; }

// Categoría BI-RADS orientativa: cortes de tesis.json (src/reporting/birads_herramienta.py); el corte 3|4 es el umbral
export function biradsHerramienta(p) {
  for (const c of S.tesis.birads_cortes) if (c.hasta === null || p < c.hasta) return c.cat;
  return '5';
}

// ---- población de prueba ---------------------------------------------------
export function matrizEn(t) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  const y = S.glob.y;
  for (let i = 0; i < y.length; i++) {
    const pos = S.gArr[i] >= t;
    if (y[i] === 1) pos ? tp++ : fn++; else pos ? fp++ : tn++;
  }
  return { tp, fp, tn, fn, sens: tp / (tp + fn), esp: tn / (tn + fp), ppv: tp / Math.max(1, tp + fp), npv: tn / Math.max(1, tn + fn) };
}

export function curvaROC() {
  const y = S.glob.y; const idx = y.map((_, i) => i).sort((a, b) => S.gArr[b] - S.gArr[a]);
  const P = y.reduce((a, b) => a + b, 0), N = y.length - P;
  let tp = 0, fp = 0, auc = 0, prevF = 0, prevT = 0; const pts = [[0, 0, Infinity]];
  for (let k = 0; k < idx.length; k++) {
    const i = idx[k]; if (y[i] === 1) tp++; else fp++;
    const nextSame = k + 1 < idx.length && S.gArr[idx[k + 1]] === S.gArr[i];
    if (!nextSame) {
      const fx = fp / N, ty = tp / P; auc += (fx - prevF) * (ty + prevT) / 2; prevF = fx; prevT = ty;
      pts.push([fx, ty, S.gArr[i]]);
    }
  }
  return { pts, auc };
}

// Densidad suavizada por clase (normalizada por tamaño de clase) para la regla de u0
export function densidades(umin, umax, paso = 0.2) {
  const n = Math.round((umax - umin) / paso); const ben = new Array(n).fill(0), mal = new Array(n).fill(0);
  const { u0, y } = S.glob; let nb = 0, nm = 0;
  for (let i = 0; i < u0.length; i++) {
    const b = Math.floor((u0[i] - umin) / paso); if (b < 0 || b >= n) continue;
    if (y[i] === 1) { mal[b]++; nm++; } else { ben[b]++; nb++; }
  }
  const suave = (a) => a.map((_, i) => {
    let s = 0, w = 0; for (let d = -2; d <= 2; d++) { const k = i + d; if (k >= 0 && k < n) { const wt = 3 - Math.abs(d); s += a[k] * wt; w += wt; } } return s / w;
  });
  const sb = suave(ben).map(v => v / nb), sm = suave(mal).map(v => v / nm);
  return { n, paso, umin, ben: sb, mal: sm, max: Math.max(...sb, ...sm) };
}

// Fracción empírica de malignas entre imágenes de prueba con u0 cercano
export function vecindad(u, ancho = 0.3) {
  let n = 0, k = 0; const { u0, y } = S.glob;
  for (let i = 0; i < u0.length; i++) if (Math.abs(u0[i] - u) <= ancho) { n++; k += y[i]; }
  return { n, k, frac: n ? k / n : null };
}

// u0 donde g cruza un umbral dado (g es estrictamente decreciente en u0)
export function uParaTau(t) {
  let lo = -10, hi = 10;
  for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (g(m) > t) lo = m; else hi = m; }
  return (lo + hi) / 2;
}
// Umbral (en g) que consigue una sensibilidad dada, o una especificidad dada
export function tauParaSens(obj) {
  const gm = S.gArr.filter((_, i) => S.glob.y[i] === 1).sort((a, b) => b - a);
  const k = Math.min(gm.length - 1, Math.max(0, Math.ceil(obj * gm.length) - 1));
  return gm[k];
}
export function tauParaEsp(obj) {
  const gb = S.gArr.filter((_, i) => S.glob.y[i] === 0).sort((a, b) => a - b);
  const k = Math.min(gb.length - 1, Math.ceil(obj * gb.length));
  return (gb[k - 1] + gb[k]) / 2;
}

export function pacientePorId(id) { return S.pac.find(p => p.id === id); }
export { f, pc };
