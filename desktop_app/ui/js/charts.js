// Gráficas en SVG sin dependencias. Convención en toda la app: izquierda = lado sospechoso (u0 bajo),
// derecha = lado benigno (u0 alto). Teal = benigno, baya = sospechoso, ámbar = umbral.
import { S, tau, uUmbral, densidades, fBen, fMal, g, sigmoide } from './data.js';
import { f, pc, esc } from './util.js';

const C = { ben: 'var(--benigno)', mal: 'var(--sospechoso)', umb: 'var(--umbral)', tin: 'var(--tinta)', gr: 'var(--linea)' };
const lin = (d0, d1, r0, r1) => (v) => r0 + (v - d0) / (d1 - d0) * (r1 - r0);
const ticks = (a, b, paso) => { const o = []; for (let v = Math.ceil(a / paso) * paso; v <= b + 1e-9; v += paso) o.push(+v.toFixed(6)); return o; };

// ---- regla de u0 -------------------------------------------------------------------------------
export const REGLA = { W: 1000, H: 270, L: 34, R: 34, T: 78, B: 72, UMIN: -3.5, UMAX: 3.5 };

export function regla({ marcador = null, etiqueta = '', mostrarUmbral = true, id = 'regla', alto = REGLA.H, aria = '', umbralU = uUmbral() } = {}) {
  const { W, L, R, T, B, UMIN, UMAX } = REGLA; const H = alto;
  const x = lin(UMIN, UMAX, L, W - R); const base = H - B; const plotH = base - T;
  const d = densidades(UMIN, UMAX, 0.2);
  const y = (v) => base - (v / (d.max * 1.08)) * plotH;
  const area = (arr) => {
    let p = `M ${x(UMIN)} ${base}`;
    arr.forEach((v, i) => { p += ` L ${x(UMIN + (i + 0.5) * d.paso)} ${y(v)}`; });
    return p + ` L ${x(UMAX)} ${base} Z`;
  };
  const xt = x(umbralU);
  let s = `<svg id="${id}" class="svg-grafica regla" style="--fs:21px" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(aria || 'Regla de u0: distribución de las imágenes benignas y malignas del conjunto de prueba')}">`;
  s += `<rect x="${L}" y="${T}" width="${xt - L}" height="${plotH}" fill="var(--sospechoso-suave)" opacity=".6"/>`;
  s += `<rect x="${xt}" y="${T}" width="${W - R - xt}" height="${plotH}" fill="var(--benigno-suave)" opacity=".55"/>`;
  s += `<path d="${area(d.ben)}" fill="var(--benigno)" fill-opacity=".38" stroke="var(--benigno)" stroke-width="1.6"/>`;
  s += `<path d="${area(d.mal)}" fill="var(--sospechoso)" fill-opacity=".34" stroke="var(--sospechoso)" stroke-width="1.6"/>`;
  s += `<line x1="${L}" x2="${W - R}" y1="${base}" y2="${base}" stroke="var(--tinta-2)" stroke-width="1.2"/>`;
  for (const t of ticks(-3, 3, 1)) s += `<line x1="${x(t)}" x2="${x(t)}" y1="${base}" y2="${base + 6}" stroke="var(--tinta-2)"/><text x="${x(t)}" y="${base + 30}" text-anchor="middle">${f(t, 0)}</text>`;
  s += `<text x="${L}" y="${H - 8}" font-weight="600">← más sospechosa</text><text x="${W - R}" y="${H - 8}" text-anchor="end" font-weight="600">más benigna →</text>`;
  s += `<text x="${(W) / 2}" y="${H - 8}" text-anchor="middle">u₀</text>`;
  if (mostrarUmbral) {
    s += `<line x1="${xt}" x2="${xt}" y1="${T - 6}" y2="${base}" stroke="var(--umbral)" stroke-width="2" stroke-dasharray="5 4"/>`;
    const izq = marcador !== null && marcador > umbralU;
    s += `<text x="${izq ? xt - 8 : xt + 8}" y="${T - 8}" text-anchor="${izq ? 'end' : 'start'}" style="fill:#7a4b00;font-weight:600">umbral (u₀ = ${f(umbralU, 2)})</text>`;
  }
  // leyenda
  s += `<g transform="translate(${L + 4} 22)"><rect width="16" height="16" y="-13" fill="var(--benigno)" opacity=".7"/><text x="24" y="1">Benignas (${S.tesis.poblacion.n_ben})</text>`;
  s += `<rect x="230" width="16" height="16" y="-13" fill="var(--sospechoso)" opacity=".7"/><text x="254" y="1">Malignas (${S.tesis.poblacion.n_mal})</text></g>`;
  if (marcador !== null) {
    const mx = x(Math.max(UMIN + .02, Math.min(UMAX - .02, marcador)));
    s += `<g id="${id}-m" transform="translate(${mx} 0)"><line y1="${T - 2}" y2="${base}" stroke="var(--tinta)" stroke-width="2.4"/>`;
    s += `<circle cy="${base}" r="9" fill="var(--tinta)" stroke="var(--hoja)" stroke-width="3"/>`;
    s += `<text id="${id}-t" y="${T - 34}" text-anchor="middle" style="fill:var(--tinta);font-weight:700;font-size:24px">${esc(etiqueta)}</text></g>`;
  }
  return s + '</svg>';
}

export function moverMarcador(svg, u, etiqueta) {
  const { UMIN, UMAX, L, W, R } = REGLA; const x = lin(UMIN, UMAX, L, W - R);
  const m = svg.querySelector('[id$="-m"]'); const t = svg.querySelector('[id$="-t"]');
  m.setAttribute('transform', `translate(${x(Math.max(UMIN + .02, Math.min(UMAX - .02, u)))} 0)`);
  if (t && etiqueta !== undefined) t.textContent = etiqueta;
}

// Arrastre sobre la regla: devuelve función para soltar los listeners
export function hacerArrastrable(svg, alCambiar) {
  const { UMIN, UMAX, L, W, R } = REGLA;
  const aU = (ev) => {
    const r = svg.getBoundingClientRect(); const px = (ev.clientX - r.left) / r.width * W;
    return Math.max(UMIN + .02, Math.min(UMAX - .02, UMIN + (px - L) / (W - L - R) * (UMAX - UMIN)));
  };
  let activo = false;
  const down = (e) => { activo = true; svg.setPointerCapture?.(e.pointerId); alCambiar(aU(e)); e.preventDefault(); };
  const move = (e) => { if (activo) alCambiar(aU(e)); };
  const up = () => { activo = false; };
  svg.style.touchAction = 'none'; svg.style.cursor = 'ew-resize';
  svg.addEventListener('pointerdown', down); svg.addEventListener('pointermove', move);
  svg.addEventListener('pointerup', up); svg.addEventListener('pointercancel', up);
  svg.tabIndex = 0;
  svg.addEventListener('keydown', (e) => {
    const paso = e.shiftKey ? 0.5 : 0.1; const cur = svg.__u ?? 0;
    if (e.key === 'ArrowLeft') { alCambiar(Math.max(UMIN + .02, cur - paso)); e.preventDefault(); }
    if (e.key === 'ArrowRight') { alCambiar(Math.min(UMAX - .02, cur + paso)); e.preventDefault(); }
  });
}

// ---- curvas f_ben, f_mal, g ----------------------------------------------------------------------
export function curvas({ u = null, ancho = 680, alto = 360, tauV = tau() } = {}) {
  const L = 46, R = 18, T = 20, B = 44; const [a, b] = [-3.4, 3.4];
  const xs = []; for (let v = a; v <= b + 1e-9; v += 0.05) xs.push(v);
  const FB = xs.map(fBen), FM = xs.map(fMal), G = xs.map(g);
  const all = [...FB, ...FM, ...G, tauV]; const lo = Math.floor(Math.min(...all) * 2) / 2 - 0.1, hi = Math.ceil(Math.max(...all) * 2) / 2 + 0.1;
  const X = lin(a, b, L, ancho - R), Y = lin(lo, hi, alto - B, T);
  const path = (arr) => arr.map((v, i) => `${i ? 'L' : 'M'} ${X(xs[i]).toFixed(1)} ${Y(v).toFixed(1)}`).join(' ');
  let s = `<svg class="svg-grafica" style="--fs:14px" viewBox="0 0 ${ancho} ${alto}" role="img" aria-label="Curvas f_ben, f_mal y g en función de u0">`;
  for (const t of ticks(lo, hi, 1)) s += `<line x1="${L}" x2="${ancho - R}" y1="${Y(t)}" y2="${Y(t)}" stroke="var(--linea-suave)"/><text x="${L - 8}" y="${Y(t) + 4}" text-anchor="end">${f(t, 0)}</text>`;
  for (const t of ticks(-3, 3, 1)) s += `<text x="${X(t)}" y="${alto - B + 20}" text-anchor="middle">${f(t, 0)}</text>`;
  s += `<text x="${(L + ancho - R) / 2}" y="${alto - 6}" text-anchor="middle">u₀ (← más sospechosa · más benigna →)</text>`;
  // y cero
  s += `<line x1="${L}" x2="${ancho - R}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--tinta-3)"/>`;
  // umbral en g
  s += `<line x1="${L}" x2="${ancho - R}" y1="${Y(tauV)}" y2="${Y(tauV)}" stroke="var(--umbral)" stroke-width="1.8" stroke-dasharray="5 4"/>`;
  s += `<text x="${ancho - R - 4}" y="${Y(tauV) - 6}" text-anchor="end" style="fill:#7a4b00;font-weight:600">umbral de g = ${f(tauV, 2)}</text>`;
  s += `<path d="${path(FB)}" fill="none" stroke="var(--benigno)" stroke-width="2.2"/>`;
  s += `<path d="${path(FM)}" fill="none" stroke="var(--sospechoso)" stroke-width="2.2"/>`;
  s += `<path d="${path(G)}" fill="none" stroke="var(--tinta)" stroke-width="3.2"/>`;
  s += `<text x="${X(-3.3)}" y="${Y(FB[1]) + 18}" style="fill:var(--benigno-oscuro);font-weight:600">f_ben</text>`;
  s += `<text x="${X(-3.3)}" y="${Y(FM[1]) - 8}" style="fill:var(--sospechoso-oscuro);font-weight:600">f_mal</text>`;
  s += `<text x="${X(3.3)}" y="${Y(G[xs.length - 4]) + 22}" text-anchor="end" style="fill:var(--tinta);font-weight:700">g = f_mal − f_ben</text>`;
  if (u !== null) {
    const uu = Math.max(a, Math.min(b, u));
    s += `<line x1="${X(uu)}" x2="${X(uu)}" y1="${T}" y2="${alto - B}" stroke="var(--tinta)" stroke-width="1.6" stroke-dasharray="2 3"/>`;
    for (const [v, col] of [[fBen(uu), 'var(--benigno)'], [fMal(uu), 'var(--sospechoso)'], [g(uu), 'var(--tinta)']]) s += `<circle cx="${X(uu)}" cy="${Y(v)}" r="6" fill="${col}" stroke="var(--hoja)" stroke-width="2.4"/>`;
  }
  return s + '</svg>';
}

// ---- curva ROC con punto de operación ------------------------------------------------------------
export function roc({ pts, op = null, tam = 340 }) {
  const L = 44, R = 12, T = 12, B = 44; const X = lin(0, 1, L, tam - R), Y = lin(0, 1, tam - B, T);
  let s = `<svg class="svg-grafica" style="--fs:12.5px" viewBox="0 0 ${tam} ${tam}" role="img" aria-label="Curva ROC de la fórmula en el conjunto de prueba">`;
  for (const t of [0, .25, .5, .75, 1]) s += `<line x1="${L}" x2="${tam - R}" y1="${Y(t)}" y2="${Y(t)}" stroke="var(--linea-suave)"/><text x="${L - 7}" y="${Y(t) + 4}" text-anchor="end">${t * 100}</text><text x="${X(t)}" y="${tam - B + 18}" text-anchor="middle">${t * 100}</text>`;
  s += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" stroke="var(--tinta-3)" stroke-dasharray="4 4"/>`;
  s += `<path d="${pts.map((p, i) => `${i ? 'L' : 'M'} ${X(p[0]).toFixed(1)} ${Y(p[1]).toFixed(1)}`).join(' ')}" fill="none" stroke="var(--tinta)" stroke-width="2.6"/>`;
  s += `<text x="${(L + tam - R) / 2}" y="${tam - 6}" text-anchor="middle">Falsas alarmas (1 − especificidad), %</text>`;
  s += `<text transform="translate(12 ${(T + tam - B) / 2}) rotate(-90)" text-anchor="middle">Sensibilidad, %</text>`;
  if (op) s += `<circle cx="${X(op.fpr)}" cy="${Y(op.tpr)}" r="7.5" fill="var(--umbral)" stroke="var(--hoja)" stroke-width="2.4"/>`;
  return s + '</svg>';
}

// ---- puntos con intervalo (AUC, EBPG) -------------------------------------------------------------
export function puntosIC({ filas, dominio, ancho = 700, ref = [], fmt = (v) => f(v, 3), paso = 0.05 }) {
  const L = 190, R = 24, T = 14; const fh = 40; const alto = T + filas.length * fh + 34;
  const X = lin(dominio[0], dominio[1], L, ancho - R);
  let s = `<svg class="svg-grafica" style="--fs:14px" viewBox="0 0 ${ancho} ${alto}" role="img">`;
  for (const t of ticks(dominio[0], dominio[1], paso)) s += `<line x1="${X(t)}" x2="${X(t)}" y1="${T - 4}" y2="${alto - 30}" stroke="var(--linea-suave)"/><text x="${X(t)}" y="${alto - 12}" text-anchor="middle">${fmt(t)}</text>`;
  for (const r of ref) {
    s += `<line x1="${X(r.v)}" x2="${X(r.v)}" y1="${T - 4}" y2="${alto - 30}" stroke="${r.color || 'var(--umbral)'}" stroke-width="1.8" stroke-dasharray="5 4"/>`;
    s += `<text x="${X(r.v) + 5}" y="${T + 8}" style="fill:#7a4b00;font-weight:600">${esc(r.label)}</text>`;
  }
  filas.forEach((r, i) => {
    const cy = T + i * fh + fh / 2;
    s += `<text x="${L - 12}" y="${cy + 4}" text-anchor="end" style="fill:var(--tinta);font-weight:${r.fuerte ? 700 : 500}">${esc(r.label)}</text>`;
    (r.ic || []).forEach((ic, k) => {
      const dy = (r.ic.length > 1 ? (k === 0 ? -5 : 5) : 0);
      s += `<line x1="${X(ic.lo)}" x2="${X(ic.hi)}" y1="${cy + dy}" y2="${cy + dy}" stroke="${r.color || 'var(--tinta)'}" stroke-width="${k === 0 ? 3.2 : 1.6}" opacity="${k === 0 ? 1 : .55}"/>`;
    });
    s += `<circle cx="${X(r.v)}" cy="${cy}" r="5.5" fill="${r.color || 'var(--tinta)'}" stroke="var(--hoja)" stroke-width="2"/>`;
    s += `<text x="${Math.min(ancho - 4, X(Math.max(...(r.ic || [{ hi: r.v }]).map(q => q.hi), r.v)) + 10)}" y="${cy + 4}" style="fill:var(--tinta);font-weight:600">${fmt(r.v)}</text>`;
  });
  return s + '</svg>';
}

// ---- escala de g (invertida: sospechosa a la izquierda, igual que la regla de u0) ------------------
export function escalaG({ gv, id = 'escalag', ancho = 1000, alto = 200 }) {
  const L = 34, R = 34, T = 90, B = 62; const [a, b] = [3, -3];  // eje invertido: g alto a la izquierda
  const X = lin(a, b, L, ancho - R); const base = alto - B;
  const xt = X(tau());
  let s = `<svg id="${id}" class="svg-grafica" style="--fs:21px" viewBox="0 0 ${ancho} ${alto}" role="img" aria-label="Valor de g frente al umbral">`;
  s += `<rect x="${L}" y="${T}" width="${xt - L}" height="${base - T}" fill="var(--sospechoso-suave)" opacity=".7"/><rect x="${xt}" y="${T}" width="${ancho - R - xt}" height="${base - T}" fill="var(--benigno-suave)" opacity=".65"/>`;
  s += `<line x1="${L}" x2="${ancho - R}" y1="${base}" y2="${base}" stroke="var(--tinta-2)"/>`;
  for (const t of [3, 2, 1, 0, -1, -2, -3]) s += `<line x1="${X(t)}" x2="${X(t)}" y1="${base}" y2="${base + 6}" stroke="var(--tinta-2)"/><text x="${X(t)}" y="${base + 28}" text-anchor="middle">${f(t, 0)}</text>`;
  s += `<line x1="${xt}" x2="${xt}" y1="${T - 8}" y2="${base}" stroke="var(--umbral)" stroke-width="2.2" stroke-dasharray="5 4"/><text x="${xt + 8}" y="${T - 14}" style="fill:#7a4b00;font-weight:600">umbral ${f(tau(), 2)}</text>`;
  s += `<text x="${L}" y="${alto - 6}" font-weight="600">← g alto: más sospechosa</text><text x="${ancho - R}" y="${alto - 6}" text-anchor="end" font-weight="600">g bajo: más benigna →</text>`;
  const gx = X(Math.max(-3, Math.min(3, gv)));
  s += `<g id="${id}-m" transform="translate(${gx} 0)"><line y1="${T}" y2="${base}" stroke="var(--tinta)" stroke-width="2.6"/><circle cy="${base}" r="8.5" fill="var(--tinta)" stroke="var(--hoja)" stroke-width="3"/><text id="${id}-t" y="${T - 50}" text-anchor="middle" style="fill:var(--tinta);font-weight:700;font-size:24px">g = ${f(gv, 2, true)}</text></g>`;
  return s + '</svg>';
}
export function moverEscalaG(svg, gv) {
  const ancho = 1000, L = 34, R = 34; const X = lin(3, -3, L, ancho - R);
  svg.querySelector('[id$="-m"]').setAttribute('transform', `translate(${X(Math.max(-3, Math.min(3, gv)))} 0)`);
  svg.querySelector('[id$="-t"]').textContent = `g = ${f(gv, 2, true)}`;
}

export { pc };
