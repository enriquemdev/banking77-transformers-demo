'use strict';
(() => {
  const $ = (id) => document.getElementById(id);
  const examples = {
    lost: 'I lost my card. What should I do?',
    transfer: 'My bank transfer is still pending. When will it arrive?',
    cash: 'The ATM did not give me cash, but the money was taken from my account.'
  };
  const config = window.BANKING77_CONFIG || {};
  let apiBase = String(config.apiBase || '').replace(/\/$/, '');
  let ready = false;
  let falconAvailable = false;
  let requiresAccess = false;
  let currentResult = null;
  let currentQuery = '';
  let controller = null;
  let requestId = 0;
  let busy = false;
  function message(id, text, kind = 'error') {
    $(id).textContent = text;
    $(id).className = `message${kind === 'info' ? ' info' : ''}`;
    $(id).hidden = !text;
  }
  function setBusy(value, action = 'classify') {
    busy = value;
    $('classify').disabled = value || !ready;
    $('explain').disabled = value || !falconAvailable;
    $('cancel').hidden = !value;
    $('classify').querySelector('.loader').hidden = !(value && action === 'classify');
    $('classify').querySelector('span').textContent = value && action === 'classify' ? 'Clasificando…' : 'Clasificar';
    $('explain').textContent = value && action === 'explain' ? 'Generando…' : 'Explicar con Falcon';
    $('query-form').setAttribute('aria-busy', String(value));
    $('result').setAttribute('aria-busy', String(value && action === 'explain'));
  }
  function clearResult() {
    currentResult = null;
    currentQuery = '';
    $('prediction').hidden = true;
    $('result').hidden = true;
    $('empty-result').hidden = false;
    $('result').setAttribute('aria-labelledby', 'result-title');
    $('explanation').hidden = true;
    $('explanation-text').textContent = '';
    message('explain-message', '');
  }
  function invalidate() {
    requestId += 1;
    if (controller) controller.abort();
    controller = null;
    clearResult();
    setBusy(false);
    if (ready) message('form-message', '');
  }
  async function request(path, body, signal) {
    const headers = { 'Content-Type': 'application/json' };
    if (requiresAccess) headers.Authorization = `Bearer ${$('access-code').value.trim()}`;
    const response = await fetch(`${apiBase}/api/${path}`, { method: 'POST', headers, body: JSON.stringify(body), signal, credentials: 'omit', cache: 'no-store' });
    let payload;
    try { payload = await response.json(); } catch { throw new Error('El servidor devolvió una respuesta incompleta. Inténtalo de nuevo.'); }
    if (!response.ok) {
      const fallback = response.status === 429 ? 'Se alcanzó el límite de consultas. Inténtalo más tarde.' : 'No se pudo completar la consulta. Inténtalo de nuevo.';
      if (response.status === 401) { $('access-field').hidden = false; $('access-code').focus(); }
      throw new Error(typeof payload.detail === 'string' ? payload.detail : fallback);
    }
    return payload;
  }
  function validPrediction(data) {
    return data && typeof data.label === 'string' && typeof data.display_name === 'string' && Array.isArray(data.top3) && data.top3.length === 3 && data.top3.every((x) => typeof x.label === 'string' && typeof x.display_name === 'string' && Number.isFinite(x.score) && x.score >= 0 && x.score <= 1);
  }
  function renderPrediction(data, query) {
    if (!validPrediction(data)) throw new Error('La predicción está incompleta. No se mostrará un resultado no verificable.');
    currentResult = data;
    currentQuery = query;
    $('predicted-title').textContent = data.display_name;
    $('predicted-label').textContent = data.label;
    $('truncation').hidden = !data.truncated;
    $('top-three').replaceChildren();
    data.top3.forEach((item) => {
      const row = document.createElement('li');
      const label = document.createElement('span');
      const score = document.createElement('span');
      label.textContent = item.display_name;
      score.textContent = `${(item.score * 100).toLocaleString('es', { minimumFractionDigits: 1, maximumFractionDigits: 1 })} %`;
      row.append(label, score);
      $('top-three').append(row);
    });
    $('empty-result').hidden = true;
    $('result').hidden = false;
    $('prediction').hidden = false;
    $('result').setAttribute('aria-labelledby', 'predicted-title');
    $('explanation').hidden = true;
    $('falcon-status').textContent = falconAvailable ? 'El primer arranque puede tardar.' : 'Falcon aún no está conectado.';
    $('predicted-title').focus({ preventScroll: true });
  }
  async function run(action) {
    if (busy || !ready) return;
    const query = $('query').value.trim();
    if (!query) { message('form-message', 'Escribe una consulta o elige un ejemplo.'); $('query').focus(); return; }
    if (query.length > 1200) { message('form-message', 'Usa una consulta de hasta 1.200 caracteres.'); return; }
    if (requiresAccess && !$('access-code').value.trim()) { message('form-message', 'Introduce el código de acceso a la demo.'); $('access-code').focus(); return; }
    if (action === 'explain' && (!currentResult || currentQuery !== query || !falconAvailable)) return;
    const id = ++requestId;
    controller = new AbortController();
    const timeout = setTimeout(() => { if (id === requestId && controller) controller.abort('timeout'); }, Number(config.requestTimeoutMs) || 240000);
    message('form-message', action === 'classify' ? 'Consultando el modelo. El primer arranque puede tardar.' : '', 'info');
    message('explain-message', action === 'explain' ? 'Cargando Falcon y generando la explicación…' : '', 'info');
    if (action === 'classify') clearResult();
    if (action === 'explain') { $('explanation').hidden = true; $('explanation-text').textContent = ''; }
    setBusy(true, action);
    try {
      const body = action === 'classify' ? { query } : { query, label: currentResult.label };
      const data = await request(action === 'classify' ? 'classify' : 'explain', body, controller.signal);
      if (id !== requestId || query !== $('query').value.trim()) return;
      if (action === 'classify') { renderPrediction(data, query); message('form-message', ''); }
      else {
        if (typeof data.text !== 'string' || !data.text.trim()) throw new Error('Falcon devolvió una respuesta vacía. Puedes volver a intentarlo.');
        $('explanation-text').textContent = data.text;
        $('explanation').hidden = false;
        const flags = [];
        if (data.hit_token_limit) flags.push('La generación alcanzó el límite de tokens y puede estar incompleta.');
        if (data.sentence_count > 2) flags.push('Esta respuesta supera las dos oraciones solicitadas. Se conserva sin recortar.');
        $('format-warning').textContent = flags.join(' ');
        $('format-warning').hidden = !flags.length;
        message('explain-message', '');
      }
    } catch (error) {
      if (id !== requestId) return;
      const text = controller.signal.aborted ? 'Se agotó el tiempo de espera. Puedes volver a intentarlo; el servidor podría seguir procesando la consulta anterior.' : error.message === 'Failed to fetch' ? 'No pudimos conectar con el servicio. Comprueba tu conexión y vuelve a intentarlo.' : error.message;
      message(action === 'classify' ? 'form-message' : 'explain-message', text);
    } finally {
      clearTimeout(timeout);
      if (id === requestId) { controller = null; setBusy(false); }
    }
  }
  $('query-form').addEventListener('submit', (event) => { event.preventDefault(); void run('classify'); });
  $('query').addEventListener('input', invalidate);
  $('access-code').addEventListener('input', invalidate);
  $('query').addEventListener('keydown', (event) => { if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) { event.preventDefault(); void run('classify'); } });
  document.querySelectorAll('[data-example]').forEach((button) => button.addEventListener('click', () => { invalidate(); $('query').value = examples[button.dataset.example]; $('query').focus(); }));
  $('cancel').addEventListener('click', () => { invalidate(); message('form-message', 'Se canceló la espera. Si el servidor ya recibió la consulta, podría seguir procesándola.', 'info'); });
  $('explain').addEventListener('click', () => void run('explain'));
  async function initialize() {
    setBusy(false);
    try {
      if (apiBase) {
        const url = new URL(apiBase);
        const local = ['localhost', '127.0.0.1'].includes(url.hostname);
        if (url.protocol !== 'https:' && !(local && url.protocol === 'http:')) throw new Error('El servicio de inferencia debe usar HTTPS.');
      }
      if (location.protocol === 'file:') throw new Error('Abre esta página desde el servidor local para conectar el modelo.');
      const healthController = new AbortController();
      const timer = setTimeout(() => healthController.abort(), 20000);
      let response;
      try { response = await fetch(`${apiBase}/api/health`, { signal: healthController.signal, cache: 'no-store', credentials: 'omit' }); } finally { clearTimeout(timer); }
      if (!response.ok) throw new Error('La página está lista, pero el servicio de inferencia aún no está conectado.');
      const data = await response.json();
      ready = data.classifier_ready === true;
      falconAvailable = data.falcon_available === true;
      requiresAccess = data.requires_access_code === true;
      $('access-field').hidden = !requiresAccess;
      if (!ready) message('form-message', 'Modelo entrenado pendiente de conexión.', 'info');
      setBusy(false);
    } catch (error) { message('form-message', error.name === 'AbortError' ? 'El servicio tarda en arrancar. Recarga la página en unos momentos.' : error.message, 'info'); }
  }
  void initialize();
})();
