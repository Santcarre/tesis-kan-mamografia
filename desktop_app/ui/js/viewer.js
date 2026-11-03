// El visor de imágenes vive ahora dentro de cohort.js (Explorador) y external.js (Estudio externo).
// Se conserva este módulo como utilidad de inversión fotométrica para quien quiera reutilizarlo.
export function invertir(img) { img.style.filter = img.style.filter ? '' : 'invert(1)'; }
