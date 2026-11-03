// Inicio: la regla de u0 arrastrable como primera idea de la tesis.
import { S, g, sigmoide, tau, uUmbral, veredicto, biradsHerramienta, vecindad } from './data.js';
import { regla, moverMarcador, hacerArrastrable } from './charts.js';
import { $, f, pc, claseVeredicto } from './util.js';

export async function montar(el) {
  const t = S.tesis;
  el.innerHTML = `
    <div class="portada">
      <div class="portada-texto">
        <h1>Un solo número decide el veredicto</h1>
        <p class="lede">En esta tesis, una red neuronal resume cada mamografía en cientos de números y una fórmula cierra la cuenta con uno solo, que llamamos <b>u₀</b>. Arrastra el punto por la regla y mira cuándo cambia de bando.</p>
      </div>
      <div class="hoja portada-regla">
        <div id="home-regla"></div>
        <p class="regla-pista">Arrastra el punto, o usa las flechas del teclado.</p>
        <div class="lectura" id="home-lectura" aria-live="polite"></div>
      </div>
    </div>

    <div class="dos-col">
      <section>
        <h2>Qué es esto</h2>
        <p>Es el laboratorio de una tesis de pregrado: se entrenó una red neuronal con mamografías del conjunto público CBIS-DDSM y después se le <em>destiló</em> una fórmula matemática cerrada de 749 caracteres que depende de una sola variable.</p>
        <p>Lo interesante no es que acierte mucho (acierta lo justo: AUC ${f(t.auc.formula.auc, 2)}). Lo interesante es que puedes <b>ver y tocar cada paso</b> entre la imagen y el veredicto, algo que con una red neuronal normal no se puede.</p>
        <p>Y también puedes ver dónde se equivoca. Eso es parte del hallazgo.</p>
      </section>
      <section>
        <h2>Por dónde empezar</h2>
        <ul class="ruta">
          <li><a href="#lab"><b>Laboratorio</b></a><span>Nueve casos de prueba (ocho pacientes), paso a paso, desde la imagen hasta el informe. Puedes apostar antes de ver el resultado.</span></li>
          <li><a href="#formula"><b>Fórmula</b></a><span>Mueve u₀ y el umbral y mira cómo cambian los aciertos y los errores en las ${t.poblacion.n} imágenes de prueba.</span></li>
          <li><a href="#resultados"><b>Resultados</b></a><span>Las cifras de la tesis con su “qué significa” y su “qué no significa”.</span></li>
        </ul>
      </section>
    </div>

    <div class="aviso neutro" style="margin-top:8px"><p><b>Antes de seguir:</b> esto es investigación, no un dispositivo médico. Con un AUC de ${f(t.auc.formula.auc, 2)}, la fórmula discrimina mejor que el azar pero está lejos de ser útil como prueba por sí sola. Detecta ${pc(t.poblacion.sens, 0)} de las malignas y deja pasar ${pc(1 - t.poblacion.esp, 0)} de las benignas como sospechosas.</p></div>`;

  const cont = $('#home-regla', el);
  let u = 0.55;
  const dibujar = () => {
    cont.innerHTML = regla({ marcador: u, etiqueta: 'u₀ = ' + f(u, 2, true), id: 'rh' });
    const svg = $('#rh', cont); svg.__u = u;
    hacerArrastrable(svg, (nu) => { u = nu; svg.__u = u; moverMarcador(svg, u, 'u₀ = ' + f(u, 2, true)); leer(); });
  };
  const leer = () => {
    const gv = g(u), p = sigmoide(gv), v = veredicto(gv), vec = vecindad(u, 0.3);
    $('#home-lectura', el).innerHTML = `
      <div class="lectura-cifras">
        <div><span>u₀</span><b>${f(u, 2, true)}</b></div>
        <div><span>g(u₀)</span><b>${f(gv, 2, true)}</b></div>
        <div><span>p_sym</span><b>${pc(p)}</b></div>
        <div class="veredicto ${claseVeredicto(v)}"><span>veredicto</span><b>${v === 'MALIGNANT' ? 'sospechosa' : 'benigna'}</b></div>
      </div>
      <p>Con <b>u₀ = ${f(u, 2, true)}</b> la fórmula da g = ${f(gv, 2, true)} y p_sym = ${pc(p)}. El umbral de decisión es ${pc(t.formula.tau_prob)} (u₀ = ${f(uUmbral(), 2)}), así que el veredicto es <b>${v === 'MALIGNANT' ? 'sospechosa' : 'benigna'}</b> (categoría BI-RADS de la herramienta: ${biradsHerramienta(p)}).
      ${vec.n ? `En el conjunto de prueba, de las ${vec.n} imágenes con un u₀ parecido, el <b>${pc(vec.frac, 0)}</b> era maligno. ${Math.abs(vec.frac - p) > 0.08 ? 'Fíjate que no coincide con p_sym: p_sym no es una probabilidad calibrada.' : ''}` : ''}</p>`;
  };
  dibujar(); leer();
}
