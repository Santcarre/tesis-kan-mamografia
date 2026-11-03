// Cliente del backend local (FastAPI). Solo se usa cuando la app local está conectada.
const API = '/api';
async function res(r, msg) {
  if (!r.ok) { let m = msg; try { const j = await r.json(); if (j && j.detail) m = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch (_) { /* sin cuerpo */ } throw new Error(m); }
  return r.json();
}
export const getCohort = (pat = 'all', tipo = 'all') => fetch(`${API}/cohort?pathology=${pat}&type=${tipo}`).then(r => res(r, 'No se pudo leer la cohorte.'));
export const getCase = (i) => fetch(`${API}/cases/${i}`).then(r => res(r, `No se pudo cargar el caso ${i}.`));
export const getRender = (i, capa) => fetch(`${API}/cases/${i}/render?layer=${capa}`).then(r => res(r, `No se pudo dibujar la capa ${capa}.`));
export const getDemos = () => fetch(`${API}/inference/demos`).then(r => res(r, 'No se pudieron listar los casos de ejemplo.'));
export const runDemo = (id) => fetch(`${API}/inference/demo/${id}`, { method: 'POST' }).then(r => res(r, 'No se pudo analizar el caso de ejemplo.'));
export const uploadStudy = (file) => { const fd = new FormData(); fd.append('file', file); return fetch(`${API}/inference/upload`, { method: 'POST', body: fd }).then(r => res(r, 'No se pudo analizar el archivo.')); };
export const generateReport = (i, llm = false) => fetch(`${API}/reports/generate`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ case_index: i, use_llm: llm }) }).then(r => res(r, 'No se pudo generar el informe.'));
export const generatePDF = (i, texto) => fetch(`${API}/reports/pdf`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ case_index: i, report_text: texto }) }).then(r => res(r, 'No se pudo crear el PDF.'));
