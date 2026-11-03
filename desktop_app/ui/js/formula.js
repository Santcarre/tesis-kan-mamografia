// Fórmula: juega con u0 y con el umbral y mira qué pasa con las imágenes de prueba.
import { S, g, fBen, fMal, sigmoide, tau, uUmbral, soporte, veredicto, biradsHerramienta, matrizEn, curvaROC, uParaTau, tauParaSens, tauParaEsp } from './data.js';
import { regla, moverMarcador, hacerArrastrable, curvas, roc } from './charts.js';
import { $, f, pc, claseVeredicto } from './util.js';

export async function montar(el) {
  const T = S.tesis; const sop = soporte();
  el.innerHTML = `
    <h1 class="vista-titulo">La fórmula</h1>
    <p class="vista-intro">Dos cosas se pueden mover: <b>u₀</b> (qué le “dice” la imagen a la fórmula) y el <b>umbral</b> (qué tan fácil es decir “sospechosa”). Lo primero cambia el caso; lo segundo cambia el equilibrio entre aciertos y errores en las ${T.poblacion.n} imágenes de prueba.</p>

    <div class="formula-rejilla">
      <section class="hoja panel">
        <h2>Mueve u₀</h2>
        <label class="solo-lectores" for="f-u">u0</label>
        <input class="rango" type="range" id="f-u" min="${sop[0]}" max="${sop[1]}" step="0.01" value="0.55">
        <div class="lectura-cifras" id="f-lectura" aria-live="polite"></div>
        <div id="f-curvas"></div>
      </section>

      <section class="hoja panel">
        <h2>Mueve el umbral</h2>
        <label for="f-t">Umbral de g (el de la tesis es ${f(T.formula.tau_logit, 2)})</label>
        <input class="rango" type="range" id="f-t" min="-2" max="2.4" step="0.01" value="${T.formula.tau_logit}">
        <div class="atajos">
          <button class="btn sec chico" data-a="youden">El de la tesis (Youden)</button>
          <button class="btn sec chico" data-a="sens90">Que detecte el 90 % de las malignas</button>
          <button class="btn sec chico" data-a="esp90">Que casi no dé falsas alarmas (esp. 90 %)</button>
        </div>
        <div id="f-regla"></div>
        <div class="umbral-res" id="f-res" aria-live="polite"></div>
      </section>
    </div>

    <section class="hoja panel ancho" id="f-matriz-sec">
      <h2>Qué pasa con las ${T.poblacion.n} imágenes de prueba</h2>
      <div class="matriz-roc">
        <div id="f-matriz"></div>
        <div id="f-roc"></div>
      </div>
      <p class="nota">La curva ROC dibuja todos los umbrales posibles; el punto ámbar es el que elegiste. El área bajo la curva (AUC) es ${f(T.auc.formula.auc, 4)}: si tomas una maligna y una benigna al azar, la fórmula pone a la maligna como más sospechosa el ${pc(T.auc.formula.auc, 0)} de las veces. El azar daría 50 %.</p>
    </section>`;

  const R = curvaROC();
  let u = 0.55, t = T.formula.tau_logit;

  const uSvg = () => {
    $('#f-regla', el).innerHTML = regla({ marcador: u, etiqueta: 'u₀ = ' + f(u, 2, true), id: 'rf', umbralU: uParaTau(t) });
    const svg = $('#rf', el); svg.__u = u;
    hacerArrastrable(svg, (nu) => { u = nu; $('#f-u', el).value = u; svg.__u = u; moverMarcador(svg, u, 'u₀ = ' + f(u, 2, true)); caso(); });
  };
  const caso = () => {
    const gv = g(u), p = sigmoide(gv), v = veredicto(gv, t);
    $('#f-lectura', el).innerHTML = `
      <div><span>f_ben</span><b>${f(fBen(u), 2, true)}</b></div><div><span>f_mal</span><b>${f(fMal(u), 2, true)}</b></div>
      <div><span>g</span><b>${f(gv, 2, true)}</b></div><div><span>p_sym</span><b>${pc(p)}</b></div>
      <div class="veredicto ${claseVeredicto(v)}"><span>veredicto</span><b>${v === 'MALIGNANT' ? 'sospechosa' : 'benigna'}</b></div>`;
    $('#f-curvas', el).innerHTML = curvas({ u, tauV: t }) + `<p class="nota">BI-RADS orientativa de la herramienta: <b>${biradsHerramienta(p)}</b> (anclada en el veredicto; no es un riesgo de malignidad).</p>`;
  };
  const umbral = () => {
    const m = matrizEn(t); const uu = uParaTau(t);
    $('#f-res', el).innerHTML = `<p>Con este umbral, la fórmula dice “sospechosa” cuando <b>u₀ ≤ ${f(uu, 2, true)}</b> (p_sym ≥ ${pc(sigmoide(t), 1)}).</p>`;
    const total = m.tp + m.fp + m.tn + m.fn;
    $('#f-matriz', el).innerHTML = `
      <table class="tabla matriz" aria-label="Matriz de confusión">
        <thead><tr><th></th><th>Fórmula: sospechosa</th><th>Fórmula: benigna</th></tr></thead>
        <tbody>
          <tr><th scope="row">Biopsia: maligna</th><td class="ok">${m.tp}<small>detectadas</small></td><td class="mal">${m.fn}<small>se le escapan</small></td></tr>
          <tr><th scope="row">Biopsia: benigna</th><td class="mal">${m.fp}<small>falsas alarmas</small></td><td class="ok">${m.tn}<small>bien descartadas</small></td></tr>
        </tbody></table>
      <ul class="metricas">
        <li><b>${pc(m.sens)}</b> sensibilidad: de cada 100 malignas, detecta ${Math.round(m.sens * 100)}.</li>
        <li><b>${pc(m.esp)}</b> especificidad: de cada 100 benignas, descarta ${Math.round(m.esp * 100)}.</li>
        <li><b>${pc(m.ppv)}</b> de lo que marca como sospechoso es de verdad maligno.</li>
      </ul>`;
    $('#f-roc', el).innerHTML = roc({ pts: R.pts, op: { fpr: 1 - m.esp, tpr: m.sens } });
    uSvg(); caso();
  };

  $('#f-u', el).addEventListener('input', (e) => { u = +e.target.value; const svg = $('#rf', el); moverMarcador(svg, u, 'u₀ = ' + f(u, 2, true)); svg.__u = u; caso(); });
  $('#f-t', el).addEventListener('input', (e) => { t = +e.target.value; umbral(); });
  el.querySelectorAll('.atajos button').forEach(b => b.addEventListener('click', () => {
    t = b.dataset.a === 'youden' ? T.formula.tau_logit : b.dataset.a === 'sens90' ? tauParaSens(0.9) : tauParaEsp(0.9);
    $('#f-t', el).value = t.toFixed(2); umbral();
  }));
  umbral();
}
