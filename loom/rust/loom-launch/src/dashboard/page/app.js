'use strict';
(function () {
  const root = document.documentElement;
  const $ = (id) => document.getElementById(id);
  const THEME_KEY = 'loom-launch-theme';
  const TOKEN_KEY = 'loom-launch-token';
  const WINDOW = 120;
  const HISTORY = 600;
  const PANELS = ['overview', 'models', 'chat', 'benchmarks', 'logs'];
  // Install, launch and benchmark share one slot on the server.
  const EXCLUSIVE = ['install', 'serve', 'bench'];

  const state = {
    token: sessionStorage.getItem(TOKEN_KEY) || '',
    status: null,
    mode: null,
    launch: null,
    // The launcher service this page belongs to (null on a launch's own page).
    launcher: null,
    launchKey: null,
    metrics: null,
    samples: [],
    // Every action the server keeps, newest first (from /api/events).
    jobs: [],
    jobsLoaded: false,
    // id -> state, to notice an action finishing while the page watches.
    jobStates: new Map(),
    // id -> DOM refs of its activity row.
    rows: new Map(),
    expanded: new Set(),
    // Requests in flight from this page, so a double click sends one.
    pending: new Set(),
    // Per-row refusals (409/400) a retry or install met, shown in place.
    refusals: new Map(),
    installs: [],
    installsLoaded: false,
    runs: [],
    runsLoaded: false,
    selectedRun: null,
    suites: [],
    search: { controller: null, timer: 0, last: null, profiles: [], loading: true, errors: new Map() },
    chat: { messages: [], busy: false, controller: null },
    log: { kind: 'serve', run: null, lines: { serve: [], events: [] } },
    source: null,
    connected: false,
    hover: {},
    tab: 'overview',
    stopRequested: false,
    launchView: '',
  };
  // ---------------------------------------------------------------- helpers

  function el(tag, props, ...children) {
    const node = document.createElement(tag);
    if (props) {
      for (const [key, value] of Object.entries(props)) {
        if (value == null || value === false) continue;
        if (key === 'class') node.className = value;
        else if (key === 'text') node.textContent = value;
        else if (key === 'dataset') Object.assign(node.dataset, value);
        else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
        else node.setAttribute(key, value === true ? '' : String(value));
      }
    }
    for (const child of children.flat()) {
      if (child == null || child === false) continue;
      node.append(child instanceof Node ? child : String(child));
    }
    return node;
  }

  const int = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 });
  const one = new Intl.NumberFormat('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
  const fmtInt = (v) => (isNum(v) ? int.format(v) : '—');
  function fmtCompact(v) {
    if (!isNum(v)) return '—';
    const a = Math.abs(v);
    if (a >= 1e9) return (v / 1e9).toFixed(1) + 'B';
    if (a >= 1e6) return (v / 1e6).toFixed(1) + 'M';
    if (a >= 1e4) return (v / 1e3).toFixed(1) + 'K';
    return int.format(v);
  }
  function fmtRate(v) {
    if (!isNum(v)) return '—';
    if (Math.abs(v) >= 100) return int.format(v);
    return one.format(v);
  }
  function fmtSeconds(v) {
    if (!isNum(v)) return '—';
    if (v < 1) return Math.round(v * 1000) + ' ms';
    if (v < 60) return v.toFixed(2) + ' s';
    return fmtDuration(v);
  }
  function fmtDuration(v) {
    if (!isNum(v)) return '—';
    const s = Math.max(0, Math.round(v));
    const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
    const pad = (n) => String(n).padStart(2, '0');
    if (d) return `${d}d ${h}h`;
    if (h) return `${h}h ${pad(m)}m`;
    if (m) return `${m}m ${pad(r)}s`;
    return `${r}s`;
  }
  function fmtBytes(v) {
    if (!isNum(v)) return '—';
    const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
    let i = 0, n = v;
    while (n >= 1024 && i < units.length - 1) { n /= 1024; i += 1; }
    return (i === 0 ? int.format(n) : n.toFixed(1)) + ' ' + units[i];
  }
  function fmtContext(v) {
    if (!isNum(v)) return '—';
    if (v >= 1024 && v % 1024 === 0) return int.format(v / 1024) + 'k';
    return int.format(v);
  }
  function fmtDate(iso) {
    if (!iso) return '—';
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return iso;
    return date.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  }
  function fmtAgo(iso) {
    const date = new Date(iso);
    if (!iso || Number.isNaN(date.getTime())) return '';
    return fmtDuration((Date.now() - date.getTime()) / 1000) + ' ago';
  }
  // `40k`, or `40k of 256k native` when the preset serves less than the model holds.
  function contextFacts(served, native) {
    if (!isNum(served)) return fmtContext(native);
    if (isNum(native) && native !== served) return `${fmtContext(served)} of ${fmtContext(native)} native`;
    return fmtContext(served);
  }
  function splitRepo(repo) {
    const index = (repo || '').indexOf('/');
    return index < 0 ? ['', repo || ''] : [repo.slice(0, index + 1), repo.slice(index + 1)];
  }
  function css(name) {
    return getComputedStyle(root).getPropertyValue(name).trim();
  }

  class ApiError extends Error {
    constructor(message, status, body) { super(message); this.status = status; this.body = body; }
  }

  async function api(path, options = {}) {
    const headers = { Accept: 'application/json' };
    if (state.token) headers.Authorization = 'Bearer ' + state.token;
    let body;
    if (options.json !== undefined) {
      headers['Content-Type'] = 'application/json';
      body = JSON.stringify(options.json);
    }
    const response = await fetch(path, { method: options.method || 'GET', headers, body, signal: options.signal, cache: 'no-store' });
    if (response.status === 401) {
      requireToken();
      throw new ApiError('This launch requires an API key.', 401, null);
    }
    if (options.raw) return response;
    const text = await response.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch (_) { data = null; }
    if (!response.ok) {
      const message = (data && data.error && data.error.message) || `HTTP ${response.status}`;
      throw new ApiError(message, response.status, data);
    }
    return data;
  }

  // A passing confirmation (a copied command); never the only feedback for an action.
  function toast(message, tone) {
    const node = el('div', { class: 'toast', dataset: { tone: tone || 'accent' } }, el('span', { class: 'dot', 'aria-hidden': 'true' }), el('span', { text: message }));
    $('toasts').append(node);
    setTimeout(() => node.remove(), 3500);
  }

  function setTone(node, tone) { node.dataset.tone = tone; }

  // ---------------------------------------------------------------- theme

  function effectiveTheme() {
    return root.dataset.theme || (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  }
  function syncThemeButton() {
    const theme = effectiveTheme();
    const button = $('theme-toggle');
    button.setAttribute('aria-pressed', String(theme === 'light'));
    button.setAttribute('aria-label', theme === 'light' ? 'Use dark theme' : 'Use light theme');
  }
  function initTheme() {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === 'light' || saved === 'dark') root.dataset.theme = saved;
    syncThemeButton();
    $('theme-toggle').addEventListener('click', () => {
      const next = effectiveTheme() === 'light' ? 'dark' : 'light';
      root.dataset.theme = next;
      localStorage.setItem(THEME_KEY, next);
      syncThemeButton();
      drawSparks();
    });
    matchMedia('(prefers-color-scheme: light)').addEventListener('change', () => { syncThemeButton(); drawSparks(); });
  }

  // ---------------------------------------------------------------- token

  function requireToken() {
    const form = $('token-form');
    if (!form.hidden) return;
    form.hidden = false;
    $('token-input').focus();
  }
  // A dashboard link printed with ?access_token= signs the page in once; the
  // key moves to this tab's session storage and leaves the address bar.
  function tokenFromAddress() {
    const params = new URLSearchParams(location.search);
    const token = params.get('access_token');
    if (!token) return;
    state.token = token;
    sessionStorage.setItem(TOKEN_KEY, token);
    params.delete('access_token');
    const rest = params.toString();
    history.replaceState(null, '', location.pathname + (rest ? '?' + rest : '') + location.hash);
  }

  function initToken() {
    $('token-form').addEventListener('submit', (event) => {
      event.preventDefault();
      state.token = $('token-input').value.trim();
      sessionStorage.setItem(TOKEN_KEY, state.token);
      $('token-form').hidden = true;
      connect();
      loadAll();
    });
  }

  // ---------------------------------------------------------------- small UI helpers

  const ICONS = {
    install: '<path d="M12 4v11m0 0l-4.5-4.5M12 15l4.5-4.5M5 19.5h14"/>',
    serve: '<path d="M7 5.5v13l11-6.5z"/>',
    stop: '<rect x="6.5" y="6.5" width="11" height="11" rx="1.5"/>',
    unload: '<rect x="6.5" y="6.5" width="11" height="11" rx="1.5"/>',
    exit: '<circle cx="12" cy="12" r="8"/><path d="M12 7.5v5.5M12 16.2v.3"/>',
    bench: '<path d="M5 19V11m7 8V5m7 14v-6"/>',
    fetch: '<circle cx="12" cy="12" r="8"/><path d="M4 12h16M12 4c2.5 2.6 2.5 13.4 0 16M12 4c-2.5 2.6-2.5 13.4 0 16"/>',
    copy: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/>',
  };

  function icon(kind) {
    // Parsed from the static markup above, never from page input.
    const template = document.createElement('template');
    template.innerHTML = `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true">${ICONS[kind] || ICONS.fetch}</svg>`;
    return template.content.firstChild;
  }

  function showError(node, message) {
    if (!node) return;
    node.textContent = message || '';
    node.hidden = !message;
  }

  // A button that is waiting: disabled, announced as busy, labelled for it.
  function setBusy(button, busy, label) {
    if (!button) return;
    if (!button.dataset.label) button.dataset.label = button.textContent;
    button.setAttribute('aria-busy', String(!!busy));
    if (busy) {
      button.disabled = true;
      if (label) button.textContent = label;
    } else {
      button.textContent = button.dataset.label;
    }
  }

  // One request per key at a time: a second click while the first is in
  // flight does nothing. Refusals and failures land in `errorNode`.
  async function act(key, button, request, errorNode, busyLabel) {
    if (state.pending.has(key)) return null;
    state.pending.add(key);
    showError(errorNode, '');
    setBusy(button, true, busyLabel);
    try {
      const data = await request();
      refreshJobs();
      return data;
    } catch (error) {
      if (error.status === 409) showError(errorNode, `Refused: ${error.message}`);
      else if (error.status === 401) showError(errorNode, 'This launch requires an API key.');
      else showError(errorNode, error.message || 'The request failed.');
      return null;
    } finally {
      state.pending.delete(key);
      setBusy(button, false);
      applyCapabilities();
    }
  }

  async function copyText(text, button) {
    let copied = false;
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        copied = true;
      }
    } catch (_) { /* fall back */ }
    if (!copied) {
      const area = el('textarea', { class: 'sr-only', 'aria-hidden': 'true' });
      area.value = text;
      document.body.append(area);
      area.select();
      try { copied = document.execCommand('copy'); } catch (_) { copied = false; }
      area.remove();
    }
    toast(copied ? 'Copied to the clipboard' : 'Copy failed; select the text instead', copied ? 'good' : 'warning');
    if (button) button.focus();
  }

  function progressBar(progress, jobState) {
    const bar = el('span', { class: 'progress', role: 'progressbar' }, el('span', { class: 'progress-fill' }));
    updateBar(bar, progress, jobState);
    return bar;
  }

  // Determinate when the job measured an extent, indeterminate otherwise.
  function updateBar(bar, progress, jobState, label) {
    const determinate = !!(progress && isNum(progress.total) && progress.total > 0);
    const active = jobState === 'running' || jobState === 'queued';
    bar.dataset.indeterminate = String(!determinate && active);
    bar.dataset.state = jobState || 'running';
    const fill = bar.firstChild;
    if (determinate) {
      const fraction = Math.max(0, Math.min(1, progress.done / progress.total));
      fill.style.width = (jobState === 'done' ? 100 : Math.round(fraction * 1000) / 10) + '%';
      bar.setAttribute('aria-valuemin', '0');
      bar.setAttribute('aria-valuemax', '100');
      bar.setAttribute('aria-valuenow', String(Math.round(fraction * 100)));
    } else {
      fill.style.width = jobState === 'done' ? '100%' : '';
      bar.removeAttribute('aria-valuenow');
      bar.removeAttribute('aria-valuemin');
      bar.removeAttribute('aria-valuemax');
    }
    bar.setAttribute('aria-valuetext', label || (determinate ? `${Math.round((progress.done / progress.total) * 100)} %` : (active ? 'in progress, no measurable extent' : jobState || '')));
  }

  function stepList(steps, large) {
    const list = el('ol', { class: 'steps' + (large ? ' large' : '') });
    for (const step of steps || []) {
      const time = isNum(step.elapsed_seconds) ? fmtDuration(step.elapsed_seconds) : '';
      list.append(el('li', { class: 'step', dataset: { state: step.state } },
        el('span', { class: 'step-mark', 'aria-hidden': 'true' }),
        el('span', {},
          el('span', { class: 'step-label', text: step.label }),
          el('span', { class: 'sr-only', text: ` (${step.state})` }),
          step.detail ? el('span', { class: 'step-detail', text: step.detail }) : null),
        el('span', { class: 'step-time', text: time })));
    }
    return list;
  }

  // ---------------------------------------------------------------- tabs

  function selectTab(name, focus) {
    if (!PANELS.includes(name)) name = 'overview';
    state.tab = name;
    for (const panel of PANELS) {
      const tab = $('tab-' + panel);
      const active = panel === name;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
      $(panel).hidden = !active;
    }
    if (location.hash !== '#' + name) history.replaceState(null, '', '#' + name);
    if (focus) $('tab-' + name).focus();
    if (name === 'overview') drawSparks();
    if (name === 'logs') renderLog();
  }

  function initTabs() {
    const tabs = PANELS.map((name) => $('tab-' + name));
    for (const tab of tabs) {
      tab.addEventListener('click', () => selectTab(tab.dataset.panel));
      tab.addEventListener('keydown', (event) => {
        const index = tabs.indexOf(tab);
        let next = null;
        if (event.key === 'ArrowRight') next = tabs[(index + 1) % tabs.length];
        else if (event.key === 'ArrowLeft') next = tabs[(index - 1 + tabs.length) % tabs.length];
        else if (event.key === 'Home') next = tabs[0];
        else if (event.key === 'End') next = tabs[tabs.length - 1];
        if (!next) return;
        event.preventDefault();
        selectTab(next.dataset.panel, true);
      });
    }
    const fromHash = () => {
      const name = location.hash.replace(/^#/, '');
      // Links from the earlier one-page layout land on the panel that holds them.
      const alias = { traffic: 'overview', job: 'overview' }[name];
      selectTab(alias || name);
    };
    window.addEventListener('hashchange', fromHash);
    fromHash();
    $('go-models').addEventListener('click', () => { selectTab('models'); $('model-q').focus(); });
    $('go-chat').addEventListener('click', () => { selectTab('chat'); $('chat-input').focus(); });
    $('go-bench').addEventListener('click', () => selectTab('benchmarks', true));
  }

  // The sticky rail sits below the header, whose height depends on wrapping.
  function trackHeader() {
    const set = () => root.style.setProperty('--header-h', $('topbar').offsetHeight + 'px');
    set();
    if ('ResizeObserver' in window) new ResizeObserver(set).observe($('topbar'));
    else window.addEventListener('resize', set);
  }

  // ---------------------------------------------------------------- header

  function renderHeader(launch) {
    const serveJob = activeJob('serve');
    // A launch this page just started is preparing before the next event says so.
    const preparing = serveJob && (!launch || launch.state === 'none');
    const none = (!launch || launch.state === 'none') && !preparing;
    $('launch-repo').textContent = preparing
      ? serveJob.target
      : none ? (launcherMode() ? 'No model loaded' : 'Nothing running') : (launch.repo || 'launch');
    // A launcher's endpoint answers whether or not a model is loaded.
    const address = (launch && launch.endpoint) || (launcherMode() && state.launcher && state.launcher.endpoint) || '';
    const endpoint = address ? address.replace(/^[a-z]+:\/\//, '') : '';
    $('launch-endpoint').textContent = endpoint || (preparing ? 'preparing…' : none ? (canStartHere() ? 'pick a profile to launch' : '—') : '—');
    $('copy-endpoint').hidden = !endpoint;
    const hardware = state.status && state.status.hardware;
    $('chip-device').textContent = (launch && launch.device) || (hardware && hardware.default_device_uri) || '—';
    $('chip-device').title = launch && launch.launch_kind === 'scenario'
      ? `The package's own deployment ${launch.deployment || ''} across ${launch.device || ''}` : '';
    $('chip-context').textContent = launch && isNum(launch.context_length) ? fmtContext(launch.context_length) : '—';
    $('chip-sessions').textContent = launch && isNum(launch.max_sessions) ? String(launch.max_sessions) : '—';
    $('chip-uptime').textContent = launch && isNum(launch.uptime_seconds) ? fmtDuration(launch.uptime_seconds) : '—';
    let tone = 'neutral', text = launcherMode() ? 'no model' : 'no launch', dot = 'none';
    if (preparing) { tone = 'warning'; text = 'preparing'; dot = 'starting'; }
    else if (launch) {
      if (launch.state === 'starting') { tone = 'warning'; text = 'preparing'; dot = 'starting'; }
      else if (launch.state === 'stopping') { tone = 'warning'; text = 'stopping'; dot = 'stopping'; }
      else if (launch.state === 'serving') {
        dot = 'serving';
        if (launch.up === true) { tone = 'good'; text = 'upstream up'; }
        else if (launch.up === false) { tone = 'critical'; text = 'upstream down'; dot = 'down'; }
        else { text = 'checking'; }
      }
    }
    setTone($('upstream-pill'), tone);
    $('upstream-text').textContent = text;
    $('launch-state').dataset.state = dot;
    const title = preparing ? serveJob.target : none ? '' : launch.repo;
    document.title = title ? `${splitRepo(title)[1] || 'launch'} · loom-launch` : 'loom-launch';
  }

  function setLive(connected) {
    state.connected = connected;
    setTone($('live-pill'), connected ? 'good' : 'warning');
    $('live-text').textContent = connected ? 'live' : 'reconnecting';
    if (!connected && state.stopRequested && state.mode === 'hosted') {
      const banner = $('banner');
      banner.textContent = 'The launch stopped, and this dashboard stopped with it: it was served by that launch\'s proxy. Start another with `loom-launch serve` or `loom-launch dashboard`.';
      banner.hidden = false;
    }
  }

  // ---------------------------------------------------------------- activity

  const KINDS = {
    install: { noun: 'Install', verb: 'Installing' },
    serve: { noun: 'Launch', verb: 'Launching' },
    stop: { noun: 'Stop', verb: 'Stopping' },
    unload: { noun: 'Unload', verb: 'Unloading' },
    exit: { noun: 'Model exited', verb: 'Exiting' },
    bench: { noun: 'Benchmark', verb: 'Benchmarking' },
    fetch: { noun: 'Web fetch', verb: 'Fetching' },
  };
  const isActive = (job) => job && (job.state === 'running' || job.state === 'queued');
  const activeJob = (kind) => state.jobs.find((job) => job.kind === kind && isActive(job)) || null;
  const busySlot = () => state.jobs.find((job) => EXCLUSIVE.includes(job.kind) && isActive(job)) || null;
  // A launcher's page loads and unloads models; a launch's own page cannot.
  const launcherMode = () => state.mode === 'launcher';
  const canStartHere = () => state.mode === 'launcher' || state.mode === 'standalone';
  const unloadJob = () => activeJob('unload') || activeJob('stop');
  const firstLine = (text) => (text || 'unknown error').split('\n')[0];

  function jobNoun(job) {
    if (job.kind === 'install') return job.source === 'path' ? 'Register a local package' : 'Install from the hub';
    return (KINDS[job.kind] || { noun: job.kind }).noun;
  }

  function jobMeasure(job) {
    const p = job.progress;
    if (!p) return '';
    const parts = [];
    if (p.unit === 'bytes' && isNum(p.total)) parts.push(`${fmtBytes(p.done)} of ${fmtBytes(p.total)}`);
    else if (p.bytes && isNum(p.bytes.done) && p.bytes.done > 0) parts.push(`${fmtBytes(p.bytes.done)}${isNum(p.bytes.total) ? ' of ' + fmtBytes(p.bytes.total) : ''} seen`);
    if (p.files && isNum(p.files.total)) parts.push(`${p.files.done}/${p.files.total} files`);
    else if (p.unit === 'files' && isNum(p.total)) parts.push(`${fmtInt(p.done)} of ${fmtInt(p.total)} files`);
    if (p.unit === 'requests' && isNum(p.total)) parts.push(`request ${Math.min(p.total, p.done + (isActive(job) ? 1 : 0))} of ${p.total}`);
    if (p.unit === 'stages') parts.push(`${fmtInt(p.done)} stage${p.done === 1 ? '' : 's'} done · no measurable extent until it serves`);
    return parts.join(' · ');
  }

  // What a finished job came to, in one line.
  function jobOutcome(job) {
    const result = job.result || {};
    if (job.state === 'failed') return '';
    if (job.kind === 'install') {
      const install = result.install || {};
      return install.repo ? `Installed ${install.repo}${install.size ? ' · ' + install.size : ''}` : 'Installed';
    }
    if (job.kind === 'serve') return result.endpoint ? `Served at ${String(result.endpoint).replace(/^[a-z]+:\/\//, '')}${isNum(result.prepare_seconds) ? ` after ${fmtDuration(result.prepare_seconds)}` : ''}` : 'Served';
    if (job.kind === 'stop' || job.kind === 'unload') {
      const exit = (job.steps || []).find((s) => s.name === 'exit');
      const done = job.kind === 'unload' ? 'Unloaded; the launcher keeps serving' : 'Stopped';
      return result.cancelled ? 'Load cancelled before it served' : `${done}${exit && exit.detail ? ' · ' + exit.detail : ''}`;
    }
    if (job.kind === 'bench') return result.id ? `Stored run ${result.id}` : 'Stored';
    if (job.kind === 'fetch') {
      const step = (job.steps || [])[0];
      return step && step.detail ? step.detail : 'Fetched';
    }
    return job.state;
  }

  function createRow(job) {
    const refs = {};
    refs.node = el('li', { class: 'job', dataset: { kind: job.kind } });
    refs.title = el('strong');
    refs.target = el('span', { class: 'job-target' });
    refs.badge = el('span', { class: 'badge' });
    refs.time = el('span', { class: 'job-time' });
    refs.step = el('p', { class: 'job-step' });
    refs.bar = progressBar(job.progress, job.state);
    refs.measure = el('p', { class: 'job-measure' });
    refs.steps = el('div');
    refs.lines = el('pre', { class: 'job-lines', 'aria-label': 'Latest output' });
    refs.error = el('p', { class: 'job-error' });
    refs.hint = el('p', { class: 'job-hint' });
    refs.refusal = el('p', { class: 'job-refusal', role: 'alert', hidden: true });
    refs.log = el('pre', { class: 'job-log', id: 'job-log-' + job.id, tabindex: '0', hidden: true, 'aria-label': 'Full output' });
    refs.toggle = el('button', { class: 'btn ghost small', type: 'button', 'aria-expanded': 'false', 'aria-controls': refs.log.id, text: 'Details' });
    refs.retry = el('button', { class: 'btn small', type: 'button', text: 'Retry' });
    refs.dismiss = el('button', { class: 'btn ghost small', type: 'button', text: 'Dismiss' });
    refs.toggle.addEventListener('click', () => toggleRow(job.id));
    refs.retry.addEventListener('click', () => retryJob(job.id));
    refs.dismiss.addEventListener('click', () => dismissJob(job.id));
    refs.node.append(
      el('div', { class: 'job-head' },
        el('span', { class: 'job-icon' }, icon(job.kind)),
        el('div', { class: 'job-title' }, refs.title, refs.target),
        refs.badge, refs.time),
      refs.step, refs.bar, refs.measure, refs.steps, refs.lines, refs.error, refs.hint, refs.refusal,
      el('div', { class: 'job-actions' }, refs.toggle, refs.retry, refs.dismiss),
      refs.log);
    state.rows.set(job.id, refs);
    return refs;
  }

  function updateRow(refs, job) {
    const active = isActive(job);
    const failed = job.state === 'failed';
    const expanded = state.expanded.has(job.id);
    refs.node.dataset.state = job.state;
    refs.node.dataset.active = String(active);
    refs.node.setAttribute('aria-label', `${jobNoun(job)} ${job.target}: ${job.state}`);
    refs.title.textContent = jobNoun(job);
    refs.target.textContent = job.target;
    refs.target.title = job.target;
    refs.badge.textContent = job.state;
    setTone(refs.badge, { queued: 'neutral', running: 'accent', done: 'good', failed: 'critical' }[job.state] || 'neutral');
    refs.time.textContent = fmtDuration(job.elapsed_seconds) + (job.finished_at ? ` · ${fmtAgo(job.finished_at)}` : '');
    const label = active
      ? [job.step, job.progress && job.progress.label && job.progress.label !== job.step ? job.progress.label : null].filter(Boolean).join(' · ') || 'starting…'
      : jobOutcome(job);
    refs.step.textContent = label;
    refs.step.hidden = !label;
    updateBar(refs.bar, job.progress, job.state, label);
    refs.bar.hidden = !active && !failed && job.kind === 'fetch';
    const measure = jobMeasure(job);
    refs.measure.textContent = measure;
    refs.measure.hidden = !measure || (!active && !expanded);
    const showSteps = (active || expanded || failed) && (job.steps || []).length > 0;
    const stepsKey = JSON.stringify(job.steps || []);
    if (showSteps && refs.stepsKey !== stepsKey) {
      refs.steps.replaceChildren(stepList(job.steps));
      refs.stepsKey = stepsKey;
    }
    refs.steps.hidden = !showSteps;
    const tail = (job.lines || []).slice(-3).join('\n');
    refs.lines.textContent = tail;
    refs.lines.hidden = !tail || expanded || !(active || failed);
    refs.error.textContent = failed ? (expanded ? job.error : firstLine(job.error)) : '';
    refs.error.hidden = !failed;
    refs.hint.textContent = job.hint || '';
    refs.hint.hidden = !job.hint;
    const refusal = state.refusals.get(job.id);
    refs.refusal.textContent = refusal || '';
    refs.refusal.hidden = !refusal;
    refs.retry.hidden = !failed || !job.retry;
    refs.retry.title = job.retry ? `Send the same request again (${job.retry.route})` : '';
    refs.dismiss.hidden = active;
    refs.toggle.textContent = expanded ? 'Hide details' : `Details${job.line_count ? ` (${job.line_count} line${job.line_count === 1 ? '' : 's'})` : ''}`;
    refs.toggle.setAttribute('aria-expanded', String(expanded));
    refs.log.hidden = !expanded;
    if (expanded && (active || !refs.logLoaded)) loadJobLog(job.id);
  }

  async function loadJobLog(id) {
    const refs = state.rows.get(id);
    if (!refs || refs.logLoading) return;
    if (refs.logAt && Date.now() - refs.logAt < 2000) return;
    refs.logLoading = true;
    try {
      const job = await api('/api/jobs/' + encodeURIComponent(id));
      const atBottom = refs.log.scrollHeight - refs.log.scrollTop - refs.log.clientHeight < 24;
      const text = (job.line_count > (job.lines || []).length ? `… ${job.line_count - job.lines.length} earlier lines not kept\n` : '')
        + (job.lines || []).join('\n') + (job.error ? '\n' + job.error : '') || '(no output)';
      if (refs.log.textContent !== text) {
        refs.log.textContent = text;
        if (atBottom) refs.log.scrollTop = refs.log.scrollHeight;
      }
      refs.logLoaded = true;
    } catch (error) {
      refs.log.textContent = `The full output could not be read: ${error.message}`;
    } finally {
      refs.logLoading = false;
      refs.logAt = Date.now();
    }
  }

  function toggleRow(id) {
    if (state.expanded.has(id)) state.expanded.delete(id); else state.expanded.add(id);
    const job = state.jobs.find((j) => j.id === id);
    const refs = state.rows.get(id);
    if (job && refs) { refs.logAt = 0; updateRow(refs, job); }
  }

  async function retryJob(id) {
    const job = state.jobs.find((j) => j.id === id);
    const refs = state.rows.get(id);
    if (!job || !job.retry || !refs) return;
    state.refusals.delete(id);
    const answer = await act('retry:' + id, refs.retry,
      () => api(job.retry.route, { method: 'POST', json: job.retry.body || {} }), refs.refusal, 'Retrying…');
    if (!answer) state.refusals.set(id, refs.refusal.textContent);
  }

  async function dismissJob(id) {
    const refs = state.rows.get(id);
    if (!refs) return;
    const answer = await act('dismiss:' + id, refs.dismiss,
      () => api('/api/jobs/dismiss', { method: 'POST', json: { id } }), refs.refusal);
    if (answer) {
      state.jobs = state.jobs.filter((j) => j.id !== id);
      renderActivity();
    }
  }

  function renderActivity() {
    const list = $('activity-list');
    const ids = new Set(state.jobs.map((job) => job.id));
    for (const [id, refs] of state.rows) {
      if (!ids.has(id)) { refs.node.remove(); state.rows.delete(id); state.expanded.delete(id); }
    }
    state.jobs.forEach((job, index) => {
      const refs = state.rows.get(job.id) || createRow(job);
      updateRow(refs, job);
      if (list.children[index] !== refs.node) list.insertBefore(refs.node, list.children[index] || null);
    });
    const active = state.jobs.filter(isActive).length;
    $('activity-empty').hidden = state.jobs.length > 0 || !state.jobsLoaded;
    const count = $('activity-count');
    count.textContent = active ? `${active} active` : String(state.jobs.length);
    setTone(count, active ? 'accent' : 'neutral');
    const more = $('activity-more');
    more.hidden = state.jobs.length === 0 || state.jobs.length === active;
    renderPulse();
  }

  // The header's view of the action under way (or the last one that failed),
  // so it is in sight from every tab without scrolling.
  function renderPulse() {
    const pulse = $('activity-pulse');
    const job = busySlot() || unloadJob() || state.jobs.find((j) => j.state === 'failed' && EXCLUSIVE.concat('stop', 'unload', 'exit').includes(j.kind) && j.finished_at && Date.now() - new Date(j.finished_at).getTime() < 10 * 60 * 1000) || activeJob('fetch');
    if (!job) { pulse.hidden = true; return; }
    pulse.hidden = false;
    pulse.dataset.state = job.state;
    const verb = (KINDS[job.kind] || { verb: job.kind }).verb;
    const p = job.progress;
    const percent = p && isNum(p.total) && p.total > 0 ? ` · ${Math.round((100 * p.done) / p.total)} %` : '';
    const text = job.state === 'failed'
      ? (job.kind === 'exit' ? `${job.target} exited on its own — see Activity` : `${jobNoun(job)} failed: ${job.target} — see Activity`)
      : `${verb} ${job.target}${job.step ? ' · ' + job.step : ''}${percent}`;
    $('pulse-text').textContent = text;
    pulse.title = text;
    pulse.setAttribute('aria-label', `${text}. Show it in the activity list.`);
    updateBar($('pulse-bar'), p, job.state);
  }

  function initActivity() {
    $('activity-pulse').addEventListener('click', () => {
      const aside = $('activity');
      aside.dataset.collapsed = 'false';
      aside.scrollIntoView({ block: 'nearest', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
      const job = busySlot() || unloadJob() || state.jobs[0];
      const refs = job && state.rows.get(job.id);
      if (refs) { refs.node.tabIndex = -1; refs.node.focus({ preventScroll: true }); }
    });
    $('activity-more').addEventListener('click', () => {
      const aside = $('activity');
      const collapsed = aside.dataset.collapsed !== 'true';
      aside.dataset.collapsed = String(collapsed);
      $('activity-more').setAttribute('aria-expanded', String(!collapsed));
      $('activity-more').textContent = collapsed ? 'Show all' : 'Show only active';
    });
  }

  function onJobs(jobs) {
    if (!Array.isArray(jobs)) return;
    const first = !state.jobsLoaded;
    state.jobs = jobs;
    state.jobsLoaded = true;
    for (const job of jobs) {
      const previous = state.jobStates.get(job.id);
      state.jobStates.set(job.id, job.state);
      // Only a change seen live triggers follow-up loads, not a reload's history.
      if (!first && previous && previous !== job.state && !isActive(job)) jobFinished(job);
      if (!first && !previous && !isActive(job)) jobFinished(job);
    }
    renderActivity();
  }

  function jobFinished(job) {
    // A new install changes what the search rows offer (Install → Reinstall, Register → Use).
    if (job.kind === 'install') loadInstalls().then(() => runSearch(state.search.last || '', state.search.lastOffline));
    else if (job.kind === 'bench' && job.state === 'done') {
      const id = job.result && job.result.id;
      loadRuns().then(() => { if (id) selectRun(id); });
    } else if (['serve', 'stop', 'unload', 'exit'].includes(job.kind)) {
      loadStatus();
      loadInstalls();
    }
  }

  async function refreshJobs() {
    try {
      const data = await api('/api/jobs');
      onJobs(data.jobs || []);
      applyCapabilities();
      renderHeader(state.launch);
      renderLaunchCard();
    } catch (_) { /* the event stream catches up */ }
  }

  // ---------------------------------------------------------------- overview

  function launchFacts(launch) {
    const facts = [
      ['Endpoint', launch.endpoint ? launch.endpoint.replace(/^[a-z]+:\/\//, '') : null],
      ['Device', launch.device],
      ['Deployment', launch.deployment ? `${launch.deployment}${launch.launch_kind === 'scenario' ? ' (scenario)' : ''}` : null],
      ['Context', isNum(launch.context_length) ? `${fmtInt(launch.context_length)} tokens` : null],
      ['Max output', isNum(launch.max_tokens) ? `${fmtInt(launch.max_tokens)} tokens` : null],
      ['Sessions', isNum(launch.max_sessions) ? String(launch.max_sessions) : null],
      ['Started', launch.started_at ? fmtDate(launch.started_at) : null],
      ['Run', launch.run_id],
      ['Server pid', isNum(launch.pid) && launch.pid > 0 ? String(launch.pid) : null],
      ['Served as', launch.model],
    ];
    return el('dl', { class: 'facts' }, facts.filter(([, v]) => v).map(([k, v]) => el('div', {}, el('dt', { text: k }), el('dd', { text: v }))));
  }

  function launchTitle(repo, sub) {
    const [owner, name] = splitRepo(repo || '');
    return el('div', { class: 'launch-title' },
      el('strong', {}, owner ? el('span', { class: 'owner', text: owner }) : null, name || repo || 'launch'),
      sub ? el('span', { class: 'hint', text: sub }) : null);
  }

  function renderLaunchCard() {
    const launch = state.launch;
    if (!launch) return;
    const serveJob = activeJob('serve');
    const stopJob = unloadJob();
    const lastServe = state.jobs.find((j) => j.kind === 'serve');
    const lastExit = state.jobs.find((j) => j.kind === 'exit');
    const standalone = canStartHere();
    const launcher = launcherMode();
    $('launch-card-h').textContent = launcher ? 'Model' : 'Current launch';
    const view = {
      launch: [launch.state, launch.kind, launch.repo, launch.endpoint, launch.device, launch.deployment, launch.context_length, launch.max_tokens, launch.max_sessions, launch.run_id, launch.pid, launch.model, launch.started_at, launch.launch_kind],
      serve: serveJob && [serveJob.id, serveJob.steps, serveJob.progress, serveJob.state],
      stop: stopJob && [stopJob.id, stopJob.steps],
      failed: lastServe && lastServe.state === 'failed' ? [lastServe.id, lastServe.error] : null,
      exited: lastExit ? lastExit.id : null,
      installs: launch.state === 'none' ? [state.installsLoaded, state.installs.map((i) => i.repo)] : null,
      mode: state.mode,
    };
    const key = JSON.stringify(view);
    const badge = $('launch-badge');
    let tone = 'neutral', text = launcher ? 'no model' : 'idle';
    if (launch.state === 'starting' || serveJob) { tone = 'warning'; text = 'preparing'; }
    else if (launch.state === 'stopping' || stopJob) { tone = 'warning'; text = 'stopping'; }
    else if (launch.state === 'serving') { tone = launch.up === false ? 'critical' : 'good'; text = launch.up === false ? 'upstream down' : 'serving'; }
    setTone(badge, tone);
    badge.textContent = text;
    const stop = $('stop-button');
    stop.hidden = !(launch.state === 'serving' || launch.state === 'starting' || launch.state === 'stopping' || serveJob);
    if (stopJob || state.pending.has('stop')) setBusy(stop, true, launcher ? 'Unloading…' : 'Stopping…');
    else {
      setBusy(stop, false);
      const preparing = serveJob || launch.state === 'starting';
      stop.textContent = launcher ? (preparing ? 'Cancel load' : 'Unload model') : (preparing ? 'Cancel launch' : 'Stop launch');
      stop.title = launcher ? 'Stops the model server; the launcher, its endpoint and this page keep running.' : '';
      stop.dataset.label = stop.textContent;
      stop.disabled = false;
    }
    $('go-chat').hidden = launch.state !== 'serving';
    $('go-bench').hidden = launch.state !== 'serving';
    $('go-models').hidden = !(launch.state === 'none' && !serveJob && standalone);
    $('go-models').textContent = launcher ? 'More options' : 'Choose a profile';
    if (key === state.launchView) return;
    state.launchView = key;
    const body = $('launch-body');
    body.replaceChildren();
    if (serveJob || launch.state === 'starting') {
      const job = serveJob || {};
      const bar = progressBar(job.progress, job.state || 'running');
      body.append(
        launchTitle(job.target || launch.repo, 'Preparing: the stages below finish one by one; their length depends on the model, so there is no percentage until it serves.'),
        bar,
        stepList(job.steps, true));
      return;
    }
    if (launch.state === 'none') {
      const empty = el('div', { class: 'empty-state' },
        el('strong', { text: launcher ? 'No model loaded' : 'Nothing is running — pick a profile to launch' }),
        el('p', { text: launcher
          ? 'The launcher keeps its endpoint and this page up without one: /v1 answers 503 until a model is loaded. Load an installed profile here, or install one from the Models tab.'
          : standalone
            ? 'Install a profile from the hub or launch a package on this machine from the Models tab. Its preparation shows here and in Activity.'
            : 'This dashboard belongs to a launch that is no longer running.' }));
      body.append(empty);
      if (launcher) body.append(quickLoadForm());
      if (lastExit && (!lastServe || new Date(lastExit.created_at) >= new Date(lastServe.created_at))) {
        body.append(el('p', { class: 'inline-error', text: `${firstLine(lastExit.error)}. Activity has its last output.` }));
      } else if (lastServe && lastServe.state === 'failed') {
        body.append(el('p', { class: 'inline-error', text: `The last launch of ${lastServe.target} failed: ${firstLine(lastServe.error)}. Activity has its stages and a Retry.` }));
      }
      return;
    }
    body.append(launchTitle(launch.repo, launch.launch_kind === 'scenario' ? `The package's own deployment across ${launch.device || 'its devices'}` : null), launchFacts(launch));
    if (stopJob) {
      body.append(progressBar(stopJob.progress, stopJob.state), stepList(stopJob.steps, true));
    }
  }

  // The Overview's own way to load a model while none is loaded: an
  // installed profile with the card's values (the Models tab has the rest).
  function quickLoadForm() {
    const select = el('select', { id: 'quick-select', 'aria-label': 'Installed profile' });
    if (!state.installs.length) select.append(el('option', { value: '', text: state.installsLoaded ? 'Install a profile first' : 'Loading installs…' }));
    else select.append(el('option', { value: '', text: 'Choose an installed profile' }));
    for (const item of state.installs) select.append(el('option', { value: item.repo, text: item.repo }));
    const preferred = $('launch-select').value;
    if (preferred && state.installs.some((i) => i.repo === preferred)) select.value = preferred;
    const button = el('button', { id: 'quick-load', class: 'btn primary', type: 'submit', text: 'Load' });
    const error = el('p', { id: 'quick-error', class: 'inline-error', role: 'alert', hidden: true });
    const form = el('form', { class: 'quick-load needs-js', 'aria-label': 'Load a model' },
      el('div', { class: 'quick-row' }, select, button),
      el('p', { class: 'hint', text: "Loads with the card's own context and sessions; the Models tab sets them, and `loom-launch start PROFILE` does the same from a shell." }),
      error);
    const sync = () => { button.disabled = !select.value || !!busySlot() || state.pending.has('serve'); };
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      if (!select.value) { showError(error, 'Choose an installed profile first.'); return; }
      await startLaunch({ repo: select.value }, button, error);
      sync();
    });
    select.addEventListener('change', sync);
    sync();
    return form;
  }

  // The launcher itself: its endpoint, how long it has run, what it answered
  // on its own, and the last model that left without being unloaded.
  function renderLauncherCard() {
    const card = $('launcher-card');
    const launcher = state.launcher;
    card.hidden = !launcher;
    if (!launcher) return;
    const badge = $('launcher-badge');
    setTone(badge, launcher.state === 'running' ? 'good' : 'warning');
    badge.textContent = launcher.state === 'running' ? 'running' : launcher.state || '—';
    const c = launcher.counters || {};
    const model = { none: 'none loaded', starting: `loading ${launcher.repo || ''}`, serving: `serving ${launcher.repo || ''}`, stopping: `unloading ${launcher.repo || ''}` }[launcher.model] || '—';
    const exit = launcher.last_exit;
    const facts = [
      ['Endpoint', launcher.endpoint ? launcher.endpoint.replace(/^[a-z]+:\/\//, '') : null],
      ['Model', model],
      ['Up', isNum(launcher.uptime_seconds) ? fmtDuration(launcher.uptime_seconds) : null],
      ['Pid', isNum(launcher.pid) ? String(launcher.pid) : null],
      ['Listening', launcher.bind_host ? `${launcher.bind_host}:${launcher.port}${launcher.auth ? ' · API key required' : ''}` : null],
      ['Models on', isNum(launcher.server_port) && launcher.server_port > 0 ? `127.0.0.1:${launcher.server_port}` : 'a free 127.0.0.1 port per load'],
      ['Answered itself', `page ${fmtInt(c.page || 0)} · api ${fmtInt(c.api || 0)} · loom ${fmtInt(c.loom || 0)} · health ${fmtInt(c.health || 0)} · /v1 without a model ${fmtInt(c.no_model || 0)}`],
      ['Last exit', exit && exit.repo ? `${exit.repo} exited on its own (${exit.detail || 'exit status ' + exit.exit_code})${exit.at ? ' · ' + fmtAgo(exit.at) : ''}` : null],
    ];
    const key = JSON.stringify(facts);
    if (key === state.launcherView) return;
    state.launcherView = key;
    $('launcher-body').replaceChildren(
      el('dl', { class: 'facts' }, facts.filter(([, v]) => v).map(([k, v]) => el('div', {}, el('dt', { text: k }), el('dd', { text: v })))),
      el('p', { class: 'hint', text: 'Stop it with `loom-launch launcher stop`; `loom-launch stop` and Unload model leave it running.' }));
  }

  function policyLine(strong, rest, tone) {
    const parts = [el('strong', { text: strong, dataset: tone ? { tone } : null })];
    if (rest) parts.push(el('span', { text: rest }));
    return parts;
  }

  function commandBox(command) {
    const box = el('span', { class: 'command' }, el('code', { text: command }));
    const button = el('button', { class: 'icon-btn small', type: 'button', 'aria-label': 'Copy the command', title: 'Copy the command' });
    button.append(icon('copy'));
    button.addEventListener('click', () => copyText(command, button));
    box.append(button);
    return box;
  }

  function renderPolicy() {
    const status = state.status;
    if (!status) return;
    const launch = state.launch || status.launch || {};
    const serving = launch.state === 'serving';
    const profile = status.profile;
    const caps = status.capabilities || {};
    const thinking = $('policy-thinking');
    const web = $('policy-web');
    const served = $('policy-served');
    const drafter = $('policy-drafter');
    if (!serving || !profile) {
      const why = serving ? 'The launcher has no documents for this launch.' : 'Follows the chat policy of the launch, once one serves.';
      thinking.replaceChildren(...policyLine('—', why));
      served.replaceChildren(...policyLine('—', serving ? '' : 'The preset of the launch, once one serves.'));
      drafter.replaceChildren(...policyLine('—', ''));
    } else {
      const t = profile.thinking || {};
      if (t.available) thinking.replaceChildren(...policyLine('Switchable', `on by default: ${t.default ? 'yes' : 'no'} (thinking: ${t.mode})`, 'good'));
      else if (t.locked === true) thinking.replaceChildren(...policyLine('Always on', t.reason, 'warning'));
      else if (t.locked === false) thinking.replaceChildren(...policyLine('Off', t.reason));
      else thinking.replaceChildren(...policyLine('Unknown', t.reason, 'warning'));
      const s = profile.served || {};
      const bits = [
        s.preset ? `preset ${s.preset}` : null,
        isNum(s.draft_depth) ? `draft depth ${s.draft_depth}` : null,
        isNum(s.rows) ? `${s.rows} row${s.rows === 1 ? '' : 's'}` : null,
        isNum(s.context_length) ? `context ${fmtInt(s.context_length)}` : null,
        isNum(s.max_tokens) ? `max ${fmtInt(s.max_tokens)} tokens` : null,
      ].filter(Boolean);
      served.replaceChildren(...policyLine(bits.length ? bits.join(' · ') : '—', ''));
      const d = profile.drafter;
      if (!d) drafter.replaceChildren(...policyLine('None recommended', 'The chat policy names no drafter.'));
      else if (d.note) drafter.replaceChildren(...policyLine(`${d.kind} at depth ${d.depth} recommended; depth ${d.served_depth} served`, d.note, 'warning'));
      else drafter.replaceChildren(...policyLine(`${d.kind} at depth ${d.depth}`, isNum(d.recommended_tps) ? `as recommended (${fmtRate(d.recommended_tps)} tok/s in the publisher's tuning)` : 'as recommended'));
    }
    if (caps.web_access) web.replaceChildren(...policyLine('On', 'The chat\'s web_fetch tool may fetch public pages through this launcher; it refuses this machine and the local network.', 'good'));
    else {
      web.replaceChildren(...policyLine('Off', caps.web_access_reason || 'web access is off'));
      if (caps.web_access_command) web.append(commandBox(caps.web_access_command));
    }
  }
  // ---------------------------------------------------------------- traffic

  const rate = (field) => (samples) => samples.map((s, i) => {
    if (i === 0) return null;
    const previous = samples[i - 1];
    const dt = s.t - previous.t;
    if (!(dt > 0)) return null;
    return Math.max(0, (s[field] - previous[field]) / dt);
  });
  const level = (field, scale) => (samples) => samples.map((s) => (isNum(s[field]) ? s[field] * (scale || 1) : null));

  const TILES = [
    { id: 'requests', series: rate('requests'), unit: 'req/s',
      value: (m) => fmtCompact(m.requests), sub: (m, last) => `${fmtRate(last)} req/s now · ${fmtInt(m.errors)} error${m.errors === 1 ? '' : 's'}` },
    { id: 'prompt', series: rate('prompt_tokens'), unit: 'tok/s',
      value: (m) => fmtCompact(m.prompt_tokens), sub: (m, last) => `${fmtRate(last)} tok/s now` },
    { id: 'completion', series: rate('completion_tokens'), unit: 'tok/s',
      value: (m) => fmtCompact(m.completion_tokens), sub: (m, last) => `${fmtRate(last)} tok/s now` },
    { id: 'cached', series: rate('cached_tokens'), unit: 'tok/s',
      value: (m) => fmtCompact(m.cached_tokens),
      sub: (m) => (m.prompt_tokens > 0 ? `${Math.round((100 * m.cached_tokens) / m.prompt_tokens)} % of prompt tokens` : 'share of prompt') },
    { id: 'ttft50', series: level('ttft_p50', 1000), unit: 'ms',
      value: (m) => fmtSeconds(m.ttft_seconds && m.ttft_seconds.p50), sub: () => 'time to first token' },
    { id: 'ttft95', series: level('ttft_p95', 1000), unit: 'ms',
      value: (m) => fmtSeconds(m.ttft_seconds && m.ttft_seconds.p95), sub: () => 'slowest 5 %' },
    { id: 'decode', series: level('decode_mean'), unit: 'tok/s',
      value: (m) => fmtRate(m.decode_tokens_per_second && m.decode_tokens_per_second.mean), suffix: 'tok/s',
      sub: (m) => `p50 ${fmtRate(m.decode_tokens_per_second && m.decode_tokens_per_second.p50)} tok/s` },
    { id: 'inflight', series: level('in_flight'), unit: '',
      value: (m) => fmtInt(m.in_flight), sub: (m) => `of ${fmtInt(m.requests)} carried` },
  ];

  function visibleSamples() {
    return state.samples.slice(-WINDOW - 1);
  }

  function renderTraffic() {
    const metrics = state.metrics;
    const samples = visibleSamples();
    for (const tile of TILES) {
      const node = $('tile-' + tile.id);
      const valueNode = node.querySelector('.tile-value');
      const subNode = node.querySelector('.tile-sub');
      const values = tile.series(samples);
      const last = values.length ? values[values.length - 1] : null;
      if (metrics) {
        valueNode.textContent = tile.value(metrics);
        if (tile.suffix && valueNode.textContent !== '—') valueNode.append(el('small', { text: tile.suffix }));
        subNode.textContent = tile.sub(metrics, last);
      } else {
        valueNode.textContent = '—';
        subNode.textContent = state.launch && state.launch.state === 'serving' ? 'waiting for counters' : 'no launch';
      }
    }
    $('traffic-hint').textContent = metrics
      ? (launcherMode()
        ? 'Every request the launcher carried, across the models loaded under it (while none is, /v1 answers 503); sparklines cover the last two minutes.'
        : 'What clients experienced through the instrumented proxy; sparklines cover the last two minutes.')
      : state.launch && state.launch.state === 'serving'
        ? 'This launch runs without the instrumentation proxy, or its counters are not readable.'
        : 'Counters appear while a launch is serving.';
    drawSparks();
  }

  function drawSparks() {
    const samples = state.metrics ? visibleSamples() : [];
    for (const tile of TILES) {
      const node = $('tile-' + tile.id);
      drawSpark(node.querySelector('canvas'), tile.series(samples), state.hover[tile.id]);
    }
  }

  function geometry(canvas) {
    const width = canvas.clientWidth, height = canvas.clientHeight;
    const pad = 5;
    const step = (width - 2 * pad) / (WINDOW - 1);
    return { width, height, pad, step, x: (i, n) => width - pad - (n - 1 - i) * step };
  }

  function drawSpark(canvas, values, hoverIndex) {
    const g = geometry(canvas);
    if (!g.width || !g.height) return;
    const ratio = window.devicePixelRatio || 1;
    const width = Math.round(g.width * ratio), height = Math.round(g.height * ratio);
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, g.width, g.height);
    const top = 6, bottom = g.height - 3;
    ctx.strokeStyle = css('--grid');
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, bottom + 0.5);
    ctx.lineTo(g.width, bottom + 0.5);
    ctx.stroke();
    const n = values.length;
    const finite = values.filter(isNum);
    if (n < 2 || !finite.length) return;
    let max = Math.max(...finite);
    if (!(max > 0)) max = 1;
    const y = (v) => bottom - (v / max) * (bottom - top);
    const accent = css('--accent');
    const segments = [];
    let current = [];
    values.forEach((v, i) => {
      if (isNum(v)) current.push([g.x(i, n), y(v)]);
      else if (current.length) { segments.push(current); current = []; }
    });
    if (current.length) segments.push(current);
    ctx.fillStyle = css('--accent-wash');
    for (const points of segments) {
      if (points.length < 2) continue;
      ctx.beginPath();
      ctx.moveTo(points[0][0], bottom);
      for (const [px, py] of points) ctx.lineTo(px, py);
      ctx.lineTo(points[points.length - 1][0], bottom);
      ctx.closePath();
      ctx.fill();
    }
    ctx.strokeStyle = accent;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    for (const points of segments) {
      ctx.beginPath();
      points.forEach(([px, py], i) => (i ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
      if (points.length === 1) ctx.lineTo(points[0][0] + 0.1, points[0][1]);
      ctx.stroke();
    }
    const dot = (i) => {
      if (!isNum(values[i])) return;
      ctx.beginPath();
      ctx.arc(g.x(i, n), y(values[i]), 4, 0, Math.PI * 2);
      ctx.fillStyle = accent;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = css('--surface-2');
      ctx.stroke();
    };
    if (isNum(hoverIndex) && hoverIndex >= 0 && hoverIndex < n) {
      ctx.strokeStyle = css('--muted');
      ctx.lineWidth = 1;
      ctx.beginPath();
      const hx = Math.round(g.x(hoverIndex, n)) + 0.5;
      ctx.moveTo(hx, top - 4);
      ctx.lineTo(hx, bottom);
      ctx.stroke();
      dot(hoverIndex);
    } else {
      dot(n - 1);
    }
  }

  function initSparkHover() {
    for (const tile of TILES) {
      const node = $('tile-' + tile.id);
      const canvas = node.querySelector('canvas');
      const tip = node.querySelector('.spark-tip');
      canvas.addEventListener('pointermove', (event) => {
        if (!state.metrics) return;
        const samples = visibleSamples();
        const values = tile.series(samples);
        const n = values.length;
        if (n < 2) return;
        const g = geometry(canvas);
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        let index = n - 1 - Math.round((g.width - g.pad - x) / g.step);
        index = Math.max(0, Math.min(n - 1, index));
        state.hover[tile.id] = index;
        const value = values[index];
        const age = samples[n - 1].t - samples[index].t;
        const shown = tile.unit === 'ms' ? fmtSeconds(isNum(value) ? value / 1000 : null) : `${fmtRate(value)}${tile.unit ? ' ' + tile.unit : ''}`;
        tip.textContent = `${shown} · ${age < 1 ? 'now' : fmtDuration(age) + ' ago'}`;
        tip.hidden = false;
        tip.style.left = Math.max(30, Math.min(g.width - 30, g.x(index, n))) + 'px';
        drawSpark(canvas, values, index);
      });
      canvas.addEventListener('pointerleave', () => {
        delete state.hover[tile.id];
        tip.hidden = true;
        drawSpark(canvas, tile.series(visibleSamples()), undefined);
      });
    }
    if ('ResizeObserver' in window) {
      const observer = new ResizeObserver(() => drawSparks());
      observer.observe($('traffic'));
    } else {
      window.addEventListener('resize', drawSparks);
    }
  }

  function renderLastRequest(request) {
    const node = $('last-request');
    node.replaceChildren();
    if (!request) {
      node.append(el('span', { class: 'k', text: state.metrics ? 'No requests yet.' : 'No live traffic.' }));
      return;
    }
    const ok = request.status && request.status < 400 && !request.error;
    const parts = [
      el('span', { class: 'k', text: 'Last request' }),
      el('code', { text: `${request.method || ''} ${request.path || ''}` }),
      el('span', { class: 'badge', dataset: { tone: ok ? 'good' : 'critical' } }, el('span', { class: 'dot', 'aria-hidden': 'true' }), String(request.status || 'error')),
      request.stream ? el('span', { text: 'streamed' }) : null,
      isNum(request.prompt_tokens) || isNum(request.completion_tokens)
        ? el('span', { text: `${fmtInt(request.prompt_tokens)} → ${fmtInt(request.completion_tokens)} tokens` }) : null,
      isNum(request.ttft_seconds) ? el('span', { text: `TTFT ${fmtSeconds(request.ttft_seconds)}` }) : null,
      isNum(request.decode_tokens_per_second) ? el('span', { text: `${fmtRate(request.decode_tokens_per_second)} tok/s` }) : null,
      request.error ? el('span', { text: request.error }) : null,
      el('span', { class: 'k', text: fmtAgo(request.time) }),
    ];
    node.append(...parts.filter(Boolean));
  }

  // ---------------------------------------------------------------- events

  function connect() {
    if (!('EventSource' in window)) return;
    if (state.source) state.source.close();
    const url = '/api/events' + (state.token ? '?access_token=' + encodeURIComponent(state.token) : '');
    const source = new EventSource(url);
    state.source = source;
    source.addEventListener('snapshot', (event) => {
      setLive(true);
      try { onSnapshot(JSON.parse(event.data)); } catch (error) { console.error(error); }
    });
    source.addEventListener('open', () => setLive(true));
    source.addEventListener('error', () => setLive(false));
  }

  function onSnapshot(event) {
    if (event.mode) state.mode = event.mode;
    if (Array.isArray(event.samples) && event.samples.length) {
      for (const sample of event.samples) {
        const last = state.samples[state.samples.length - 1];
        if (last && last.key !== sample.key) state.samples = [];
        if (!last || sample.t > last.t || last.key !== sample.key) state.samples.push(sample);
      }
      if (state.samples.length > HISTORY) state.samples.splice(0, state.samples.length - HISTORY);
    }
    state.metrics = event.metrics || null;
    const launch = event.launch || null;
    const key = launch ? [launch.kind, launch.state, launch.repo, launch.endpoint].join('|') : '';
    const changed = key !== state.launchKey;
    state.launch = launch;
    if ('launcher' in event) state.launcher = event.launcher || null;
    onJobs(event.jobs);
    renderHeader(launch);
    renderLauncherCard();
    renderLaunchCard();
    if (state.tab === 'overview') renderTraffic();
    renderLastRequest(event.last_request);
    onLog(event.log);
    if (changed) {
      const first = state.launchKey === null;
      state.launchKey = key;
      if (!first) {
        loadStatus();
        loadInstalls();
        loadLogs();
      }
    }
    applyCapabilities();
  }

  // ---------------------------------------------------------------- capabilities

  // Every control's enabled state follows the server's picture: what runs,
  // which job holds the slot, and which request this page has in flight.
  function applyCapabilities() {
    const launch = state.launch || { state: 'none' };
    const serving = launch.state === 'serving';
    const slot = busySlot();
    const serveJob = activeJob('serve');
    const benchJob = activeJob('bench');
    const standalone = canStartHere();
    const canLaunch = standalone && launch.state === 'none' && !slot && !state.pending.has('serve');
    const caps = (state.status && state.status.capabilities) || {};

    const launchButton = $('launch-button');
    if (serveJob || state.pending.has('serve')) setBusy(launchButton, true, 'Launching…');
    else {
      setBusy(launchButton, false);
      launchButton.disabled = !canLaunch || !$('launch-select').value;
    }
    let note = '';
    if (!standalone && state.mode) note = 'This dashboard belongs to the running launch; load other models from `loom-launch launcher start`.';
    else if (serveJob) note = `Preparing ${serveJob.target}; its stages are on the Overview and in Activity.`;
    else if (serving) note = `Serving ${launch.repo}; ${launcherMode() ? 'unload it' : 'stop it'} to launch another profile.`;
    else if (slot) note = `Waiting for the ${jobNoun(slot).toLowerCase()} of ${slot.target} to finish.`;
    else if (!$('launch-select').value) note = 'Choose an installed profile.';
    $('launch-note').textContent = note;
    launchButton.title = note;

    const webField = $('launch-web-field');
    const webBox = $('launch-allow-web');
    webField.hidden = !caps.launch_allow_web;
    if (caps.launch_allow_web && launch.state === 'none') {
      const already = !!caps.web_access;
      webBox.disabled = already;
      if (already) webBox.checked = true;
      webField.title = already ? 'This dashboard runs with --allow-web, so every launch it starts has web access.' : 'Lets the chat\'s web_fetch tool fetch public pages through this dashboard while the launch serves.';
    }

    for (const button of document.querySelectorAll('.suite')) {
      const mine = benchJob && benchJob.suite === button.dataset.suite;
      if (mine || state.pending.has('bench:' + button.dataset.suite)) {
        button.setAttribute('aria-busy', 'true');
        button.disabled = true;
      } else {
        button.setAttribute('aria-busy', 'false');
        button.disabled = !serving || !!slot || !!state.pending.size && [...state.pending].some((k) => k.startsWith('bench:'));
      }
      button.title = !serving ? (caps.bench_reason || 'Suites run against a serving launch.') : slot ? `Waiting for the ${jobNoun(slot).toLowerCase()} of ${slot.target}.` : '';
    }
    const noModel = caps.bench_reason ? caps.bench_reason.charAt(0).toUpperCase() + caps.bench_reason.slice(1) + '.' : null;
    $('bench-note').textContent = serving
      ? (benchJob ? `Running the ${benchJob.suite || ''} suite; per-request progress is in Activity.` : slot ? `Waiting for the ${jobNoun(slot).toLowerCase()} of ${slot.target}.` : `Runs against ${launch.repo} at ${(launch.endpoint || '').replace(/^[a-z]+:\/\//, '')}; results are stored with the CLI's records.`)
      : launcherMode() ? `Benchmarks are off. ${noModel || 'No model is loaded.'}` : 'Suites run against a serving launch. Launch a profile first.';

    $('chat-send').disabled = !serving || state.chat.busy;
    $('chat-input').disabled = !serving && !state.chat.busy;
    $('chat-hint').textContent = serving
      ? `Streaming from ${launch.model || launch.repo} through the instrumented ${launcherMode() ? 'launcher' : 'proxy'}.`
      : launcherMode() ? 'No model is loaded, so there is nothing to chat with yet.' : 'Chat opens when a launch is serving.';
    applyChatEmpty(serving);
    for (const button of document.querySelectorAll('[data-install]')) {
      const mine = activeJob('install') && activeJob('install').target === button.dataset.install;
      if (mine || state.pending.has('install:' + button.dataset.install)) setBusy(button, true, 'Installing…');
      else { setBusy(button, false); button.disabled = !!slot; }
      button.title = slot && !mine ? `Waiting for the ${jobNoun(slot).toLowerCase()} of ${slot.target}.` : '';
    }
    for (const button of document.querySelectorAll('[data-launch-local]')) {
      if (serveJob && state.pending.has('serve')) continue;
      button.disabled = !canLaunch;
      button.title = canLaunch ? button.dataset.launchLocal : (standalone ? (launcherMode() ? 'Unload the running model first.' : 'Stop the running launch first.') : 'Load from `loom-launch launcher start`.');
    }
    for (const button of document.querySelectorAll('[data-register]')) {
      const mine = activeJob('install') && activeJob('install').target === button.dataset.register;
      if (mine || state.pending.has('install:' + button.dataset.register)) setBusy(button, true, 'Registering…');
      else { setBusy(button, false); button.disabled = !!slot || !standalone; }
    }
    applyChatSwitches();
  }

  function applyChatEmpty(serving) {
    const log = $('chat-log');
    const empty = log.querySelector('.empty-state');
    if (!empty || state.chat.messages.length) return;
    const view = `${serving}|${state.mode}`;
    if (empty.dataset.view === view) return;
    empty.dataset.view = view;
    if (!serving && launcherMode()) {
      empty.replaceChildren(
        el('strong', { text: 'No model is loaded' }),
        el('p', { text: 'The launcher is up, but nothing answers chats until a model is loaded. Load an installed profile from the Overview or the Models tab, or run `loom-launch start PROFILE`; the chat opens as soon as it serves, without a reload.' }),
        el('button', { class: 'btn primary small', type: 'button', text: 'Load a model', onclick: () => selectTab('overview', true) }));
      return;
    }
    empty.replaceChildren(
      el('strong', { text: serving ? 'Say hello' : 'Chat opens when a launch is serving.' }),
      el('p', { text: serving ? 'Enter sends; Shift+Enter starts a new line.' : 'Launch a profile from the Models tab; its progress shows in Activity.' }));
  }

  // The thinking and web switches say why they are off, in place.
  function applyChatSwitches() {
    const status = state.status;
    const launch = state.launch || {};
    const serving = launch.state === 'serving';
    const toggle = $('chat-thinking');
    const note = $('thinking-note');
    const lock = $('thinking-lock');
    const profile = status && status.profile;
    const thinking = profile && profile.thinking;
    let reason = '';
    if (!serving) {
      toggle.disabled = true;
      toggle.checked = false;
      reason = 'Thinking follows the chat policy of the running launch.';
      lock.textContent = '';
      note.hidden = true;
    } else if (!thinking) {
      toggle.disabled = true;
      toggle.checked = false;
      reason = 'The launcher has no documents for this launch, so it cannot tell whether the model thinks.';
      lock.textContent = 'unknown';
      note.hidden = false;
    } else if (thinking.available) {
      toggle.disabled = false;
      if (toggle.dataset.repo !== profile.repo) {
        toggle.checked = !!thinking.default;
        toggle.dataset.repo = profile.repo;
      }
      reason = `Thinking is ${thinking.default ? 'on' : 'off'} by default for ${profile.repo} (thinking: ${thinking.mode}).`;
      lock.textContent = '';
      note.hidden = true;
    } else {
      toggle.disabled = true;
      toggle.checked = thinking.locked === true;
      reason = thinking.reason || 'Thinking cannot be switched for this package.';
      lock.textContent = thinking.locked === true ? 'always on' : thinking.locked === false ? 'off' : 'unknown';
      note.hidden = false;
    }
    note.textContent = reason;
    $('thinking-switch').title = reason;

    const caps = (status && status.capabilities) || {};
    const web = $('chat-tool-web');
    const webNote = $('web-note');
    if (caps.web_access) {
      if (web.disabled) { web.disabled = false; web.checked = true; }
      $('web-switch').title = 'Let the model fetch public web pages through this launcher (it refuses this machine and the local network).';
      webNote.hidden = true;
    } else {
      web.disabled = true;
      web.checked = false;
      const why = caps.web_access_reason || 'Web access is off for this launch.';
      $('web-switch').title = why;
      $('web-note-text').textContent = why.charAt(0).toUpperCase() + why.slice(1) + (caps.web_access_command ? ':' : '.');
      $('web-command').textContent = caps.web_access_command || '';
      $('web-command-box').hidden = !caps.web_access_command;
      webNote.hidden = !status;
    }
  }

  // ---------------------------------------------------------------- status

  async function loadStatus() {
    try {
      const status = await api('/api/status');
      state.status = status;
      state.mode = status.mode;
      state.launcher = status.launcher || null;
      renderLauncherCard();
      if (Array.isArray(status.suites)) state.suites = status.suites;
      $('footer-version').textContent = `loom-launch ${status.version || ''}`;
      $('footer-state').textContent = `· state ${status.state_directory || ''}`;
      $('chat-system').placeholder = status.profile && status.profile.chat && status.profile.chat.default_system
        ? `Default: ${status.profile.chat.default_system}` : "The package's default system prompt applies when empty";
      if (!state.launch) {
        state.launch = status.launch;
        renderHeader(status.launch);
      }
      if (!state.jobsLoaded) onJobs(status.jobs || []);
      renderSuites();
      renderLaunchCard();
      renderPolicy();
      applyCapabilities();
    } catch (error) {
      if (error.status !== 401) {
        const banner = $('banner');
        banner.textContent = `The dashboard's status could not be read: ${error.message}`;
        banner.hidden = false;
      }
    }
  }

  // ---------------------------------------------------------------- installs and launch

  async function loadInstalls() {
    try {
      const data = await api('/api/installs');
      state.installs = data.installs || [];
    } catch (error) {
      state.installs = [];
    }
    state.installsLoaded = true;
    $('installed').setAttribute('aria-busy', 'false');
    renderInstalls();
  }

  function policyChip(policy) {
    if (!policy) return null;
    if (policy.available) return el('span', { class: 'badge', dataset: { tone: 'accent' }, title: `thinking: ${policy.mode}, on by default: ${policy.default ? 'yes' : 'no'}`, text: 'thinking: switchable' });
    if (policy.locked === true) return el('span', { class: 'badge', title: policy.reason, text: 'thinking: always' });
    if (policy.locked === false) return el('span', { class: 'badge', title: policy.reason, text: 'thinking: never' });
    return el('span', { class: 'badge', dataset: { tone: 'warning' }, title: policy.reason, text: 'no chat policy' });
  }

  function renderInstalls() {
    const list = $('installed');
    list.replaceChildren();
    const select = $('launch-select');
    const previous = select.value;
    select.replaceChildren();
    if (!state.installs.length) {
      list.append(el('li', {}, el('div', { class: 'empty-state small' },
        el('strong', { text: 'Nothing installed yet' }),
        el('p', { text: 'Search the hub and install a profile, or register a package found on this machine.' }))));
      select.append(el('option', { value: '', text: 'Install a profile first' }));
    } else {
      select.append(el('option', { value: '', text: 'Choose a profile' }));
    }
    for (const item of state.installs) {
      const [owner, name] = splitRepo(item.repo);
      const policy = item.policy;
      list.append(el('li', { class: 'installed-item', dataset: { running: String(!!item.running) } },
        el('span', { class: 'name', title: item.repo }, el('span', { class: 'owner', text: owner }), el('strong', { text: name }),
          item.running ? el('span', { class: 'badge', dataset: { tone: 'good' } }, el('span', { class: 'dot', 'aria-hidden': 'true' }), 'serving') : null,
          item.scenario ? el('span', { class: 'badge', dataset: { tone: 'accent' }, title: 'Carries its own deployment.json and devices.json', text: `scenario · ${item.scenario_devices || ''}` }) : null,
          item.source === 'local' ? el('span', { class: 'badge', title: item.path, text: 'local' }) : null,
          policyChip(policy)),
        el('div', { class: 'result-actions' },
          el('button', { class: 'btn small', type: 'button', text: 'Use', 'aria-label': `Use ${item.repo} in the launch form`, onclick: () => chooseForLaunch(item.repo) })),
        el('div', { class: 'facts-row' },
          item.quant ? el('span', {}, el('span', { class: 'k', text: 'quant' }), item.quant) : null,
          isNum(item.context_length) ? el('span', {}, el('span', { class: 'k', text: 'ctx' }), contextFacts(item.context_length, item.native_context_length)) : null,
          el('span', {}, el('span', { class: 'k', text: 'size' }), item.size || '—'),
          isNum(item.draft_depth) ? el('span', {}, el('span', { class: 'k', text: 'draft' }), String(item.draft_depth)) : null,
          item.installed_at ? el('span', { class: 'k', text: `installed ${fmtAgo(item.installed_at)}` }) : null,
          item.card_exists === false ? el('span', { class: 'error', text: 'card missing' }) : null),
        policy && !policy.available && policy.locked == null && policy.reason ? el('p', { class: 'note', dataset: { tone: 'warning' }, text: policy.reason }) : null,
      ));
      select.append(el('option', { value: item.repo, text: item.repo }));
    }
    if (state.installs.some((i) => i.repo === previous)) select.value = previous;
    updateLaunchPlaceholders();
    applyCapabilities();
  }

  function chooseForLaunch(repo) {
    const select = $('launch-select');
    if (![...select.options].some((o) => o.value === repo)) return;
    select.value = repo;
    updateLaunchPlaceholders();
    applyCapabilities();
    select.focus();
  }

  // The capacity fields: the card's served values as placeholders, the model's
  // native context as the ceiling; a scenario owns all of them.
  function updateLaunchPlaceholders() {
    const item = state.installs.find((i) => i.repo === $('launch-select').value);
    const scenario = !!(item && item.scenario);
    for (const id of ['launch-context', 'launch-max-tokens', 'launch-parallel', 'launch-draft']) {
      $(id).disabled = scenario;
      if (scenario) $(id).value = '';
      $(id).title = scenario ? 'A scenario serves exactly as prepared; the launcher refuses capacity overrides for it.' : '';
    }
    const ceiling = item && (isNum(item.native_context_length) ? item.native_context_length : item.context_length);
    $('launch-context').placeholder = scenario ? 'scenario' : (item && isNum(item.context_length) ? `card ${fmtInt(item.context_length)}` : 'card');
    if (isNum(ceiling)) $('launch-context').max = String(ceiling); else $('launch-context').removeAttribute('max');
    if (!scenario && isNum(ceiling)) $('launch-context').title = `At most the model's native ${fmtInt(ceiling)} tokens`;
    $('launch-max-tokens').placeholder = scenario ? 'scenario' : (item && isNum(item.max_tokens) ? `card ${item.max_tokens}` : 'card');
    $('launch-parallel').placeholder = scenario ? 'scenario' : (item && isNum(item.rows) ? `card ${item.rows}` : 'card');
    $('launch-draft').placeholder = scenario ? 'scenario' : (item && isNum(item.draft_depth) ? `card ${item.draft_depth}` : 'card');
  }

  function numberInput(id) {
    const text = $(id).value.trim();
    return text === '' ? null : Number(text);
  }

  async function startLaunch(body, button, errorNode) {
    const answer = await act('serve', button, () => api('/api/serve', { method: 'POST', json: body }), errorNode, 'Launching…');
    if (answer) selectTab('overview');
    return answer;
  }

  function initLaunch() {
    $('launch-select').addEventListener('change', () => { updateLaunchPlaceholders(); applyCapabilities(); });
    $('launch-form').addEventListener('submit', async (event) => {
      event.preventDefault();
      const repo = $('launch-select').value;
      if (!repo) { showError($('launch-error'), 'Choose an installed profile first.'); return; }
      const item = state.installs.find((i) => i.repo === repo);
      const body = item && item.scenario ? { repo } : {
        repo,
        context: numberInput('launch-context'),
        max_tokens: numberInput('launch-max-tokens'),
        parallel: numberInput('launch-parallel'),
        draft_depth: numberInput('launch-draft'),
      };
      if (!$('launch-web-field').hidden && $('launch-allow-web').checked && !$('launch-allow-web').disabled) body.allow_web = true;
      await startLaunch(body, $('launch-button'), $('launch-error'));
    });
    $('stop-button').addEventListener('click', async () => {
      state.stopRequested = true;
      const route = launcherMode() ? '/api/unload' : '/api/stop';
      await act('stop', $('stop-button'), () => api(route, { method: 'POST', json: {} }), $('stop-error'), launcherMode() ? 'Unloading…' : 'Stopping…');
      renderLaunchCard();
    });
    $('copy-endpoint').addEventListener('click', () => copyText($('launch-endpoint').textContent, $('copy-endpoint')));
    $('copy-web-command').addEventListener('click', () => copyText($('web-command').textContent, $('copy-web-command')));
  }

  // ---------------------------------------------------------------- search

  function initSearch() {
    const input = $('model-q');
    input.addEventListener('input', () => {
      clearTimeout(state.search.timer);
      state.search.timer = setTimeout(() => runSearch(input.value.trim()), 400);
    });
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') { event.preventDefault(); clearTimeout(state.search.timer); runSearch(input.value.trim()); }
    });
    $('model-offline').addEventListener('change', () => runSearch(input.value.trim()));
  }

  function searchSkeleton() {
    const list = $('model-results');
    list.setAttribute('aria-busy', 'true');
    list.replaceChildren(...[0, 1, 2].map(() => el('li', { class: 'skeleton tall', 'aria-hidden': 'true' })));
  }

  async function runSearch(query, forceOffline) {
    if (state.search.controller) state.search.controller.abort();
    const controller = new AbortController();
    state.search.controller = controller;
    const offline = forceOffline || $('model-offline').checked;
    const status = $('model-status');
    delete status.dataset.tone;
    status.textContent = offline ? 'Reading cached and local profiles…' : (query ? `Searching the hub for “${query}”…` : 'Listing Loom profiles on the hub…');
    if (!state.search.profiles.length) searchSkeleton();
    try {
      const params = new URLSearchParams({ q: query, limit: '24' });
      if (offline) params.set('offline', '1');
      const data = await api('/api/models?' + params.toString(), { signal: controller.signal });
      state.search.profiles = data.profiles || [];
      state.search.last = query;
      state.search.lastOffline = !!forceOffline;
      renderResults();
      const hardware = data.hardware || {};
      const machine = [hardware.gpu, hardware.memory].filter(Boolean).join(', ');
      if (data.warning) { status.dataset.tone = 'warning'; status.textContent = data.warning; }
      else status.textContent = `${data.count} profile${data.count === 1 ? '' : 's'}${data.offline ? ' from the cache and this machine' : ''} · sorted by fit for this machine${machine ? ` (${machine})` : ''}`;
    } catch (error) {
      if (error.name === 'AbortError') return;
      status.dataset.tone = 'warning';
      status.textContent = `Search failed: ${error.message}. Tick “Cached only” to list what this machine already has.`;
      $('model-results').setAttribute('aria-busy', 'false');
      if (!state.search.profiles.length) $('model-results').replaceChildren();
    }
  }

  function fitTone(score) {
    if (!isNum(score)) return 'neutral';
    if (score >= 0.6) return 'accent';
    if (score >= 0.4) return 'warning';
    return 'critical';
  }

  function renderResults() {
    const list = $('model-results');
    list.setAttribute('aria-busy', 'false');
    list.replaceChildren();
    const installed = new Set(state.installs.map((i) => i.repo));
    for (const row of state.search.profiles) {
      const [owner, name] = splitRepo(row.repo);
      const local = row.source === 'local';
      const isInstalled = installed.has(row.repo) || row.installed;
      const fit = row.fit;
      const meter = el('span', { class: 'meter', dataset: { tone: fitTone(fit) }, 'aria-hidden': 'true' }, el('span'));
      if (isNum(fit)) meter.firstChild.style.width = Math.round(Math.max(0, Math.min(1, fit)) * 100) + '%';
      const key = local ? row.path : row.repo;
      const error = el('p', { class: 'inline-error', role: 'alert', hidden: !state.search.errors.get(key), text: state.search.errors.get(key) || '' });
      const actions = [];
      if (local) {
        actions.push(el('button', {
          class: 'btn small primary', type: 'button', 'data-launch-local': row.path, text: 'Launch',
          'aria-label': `Launch the local profile at ${row.path}`,
          onclick: (event) => startLaunch({ path: row.path }, event.currentTarget, error),
        }));
        if (!row.installed) {
          actions.push(el('button', {
            class: 'btn small', type: 'button', 'data-register': row.path, text: 'Register',
            'aria-label': `Register ${row.path} as an install`, title: 'Record it as an install (in place), adding its chat policy from local copies',
            onclick: (event) => register(row.path, event.currentTarget, error),
          }));
        }
      } else {
        actions.push(el('button', {
          class: 'btn small' + (isInstalled ? '' : ' primary'), type: 'button', 'data-install': row.repo,
          text: isInstalled ? 'Reinstall' : 'Install', 'aria-label': `${isInstalled ? 'Reinstall' : 'Install'} ${row.repo}`,
          onclick: (event) => install(row.repo, event.currentTarget, error),
        }));
      }
      if (isInstalled && installed.has(row.repo)) {
        actions.push(el('button', { class: 'btn small ghost', type: 'button', text: 'Use', 'aria-label': `Use ${row.repo} in the launch form`, onclick: () => chooseForLaunch(row.repo) }));
      }
      list.append(el('li', { class: 'result' },
        el('div', { class: 'result-title', title: row.repo },
          el('span', { class: 'owner', text: owner }), el('strong', { text: name }),
          local ? el('span', { class: 'badge', title: row.path || '', text: 'local' }) : null,
          row.scenario ? el('span', { class: 'badge', dataset: { tone: 'accent' }, title: 'Carries its own deployment.json and devices.json; served exactly as prepared', text: row.scenario_devices ? `scenario · ${row.scenario_devices}` : 'scenario' }) : null,
          isInstalled ? el('span', { class: 'badge', dataset: { tone: 'good' } }, el('span', { class: 'dot', 'aria-hidden': 'true' }), 'installed') : null,
          row.running ? el('span', { class: 'badge', dataset: { tone: 'good' }, text: 'serving' }) : null,
          row.gated ? el('span', { class: 'badge', dataset: { tone: 'warning' }, text: 'gated' }) : null),
        el('div', { class: 'result-actions' }, actions),
        el('div', { class: 'facts-row' },
          el('span', {}, el('span', { class: 'k', text: 'quant' }), row.quant || '—'),
          el('span', { title: isNum(row.native_context_length) && row.native_context_length !== row.context_length ? `served ${fmtInt(row.context_length)} of the model's native ${fmtInt(row.native_context_length)}` : '' }, el('span', { class: 'k', text: 'ctx' }), contextFacts(row.context_length, row.native_context_length)),
          el('span', {}, el('span', { class: 'k', text: 'memory' }), row.memory || '—'),
          el('span', {}, el('span', { class: 'k', text: 'decode' }), isNum(row.decode) ? `${fmtRate(row.decode)} tok/s` : '—'),
          el('span', { class: 'fit', title: (row.fit_notes || []).join('\n') || 'no evidence recorded' },
            el('span', { class: 'k', text: 'fit' }), meter, isNum(fit) ? fit.toFixed(2) : '—'),
          local && row.path ? el('span', { class: 'k', title: row.path, text: row.path }) : null,
          row.error ? el('span', { class: 'error', text: row.error }) : null),
        error,
      ));
    }
    if (!state.search.profiles.length && state.search.last !== null) {
      list.append(el('li', {}, el('div', { class: 'empty-state small' },
        el('strong', { text: 'No matching profiles' }),
        el('p', { text: 'Try a shorter query, an author followed by “/”, or “Cached only” to list what this machine holds.' }))));
    }
    applyCapabilities();
  }

  async function install(repo, button, errorNode) {
    state.search.errors.delete(repo);
    const answer = await act('install:' + repo, button, () => api('/api/install', { method: 'POST', json: { repo } }), errorNode, 'Installing…');
    if (!answer && errorNode.textContent) state.search.errors.set(repo, errorNode.textContent);
  }

  async function register(path, button, errorNode) {
    state.search.errors.delete(path);
    const answer = await act('install:' + path, button, () => api('/api/install', { method: 'POST', json: { path } }), errorNode, 'Registering…');
    if (!answer && errorNode.textContent) state.search.errors.set(path, errorNode.textContent);
  }

  // ---------------------------------------------------------------- tools

  // At most this many rounds of tool calls answer one message.
  const TOOL_ROUNDS = 6;
  // Characters of tool output handed back to the model.
  const TOOL_OUTPUT_LIMIT = 8192;
  const SANDBOX_TIMEOUT_MS = 2000;
  // Output lines a tool card shows before it collapses.
  const CARD_LINES = 6;

  const TOOLS = {
    run_javascript: {
      type: 'function',
      function: {
        name: 'run_javascript',
        description: 'Run a snippet of JavaScript in a sandboxed worker with no DOM, no network and a 2 s limit; return the value of the last expression (use `return` explicitly or leave an expression; console.log output is captured)',
        parameters: {
          type: 'object',
          properties: {
            code: { type: 'string', description: 'The JavaScript to run, for example: return [3, 1, 2].sort((a, b) => a - b)' },
          },
          required: ['code'],
        },
      },
    },
    web_fetch: {
      type: 'function',
      function: {
        name: 'web_fetch',
        description: 'Fetch one public http/https URL and return its text content (HTML reduced to text), for facts you cannot answer otherwise',
        parameters: {
          type: 'object',
          properties: {
            url: { type: 'string', description: 'The absolute http or https URL of a public page, scheme included' },
          },
          required: ['url'],
        },
      },
    },
  };

  const encoder = new TextEncoder();

  // `text` cut to at most `limit` UTF-8 bytes, with a marker saying so.
  function capText(text, limit) {
    const bytes = encoder.encode(text);
    if (bytes.length <= limit) return text;
    const marker = `\n…[truncated: ${bytes.length} bytes, showing the first ${limit - 64}]`;
    const kept = new TextDecoder().decode(bytes.slice(0, limit - 64)).replace(/�+$/, '');
    return kept + marker;
  }

  // Runs inside the sandbox worker before the snippet: removes every way out
  // (network, storage, nested workers, raw postMessage), captures the console
  // and reports exactly one outcome. Serialized with toString(), so it may only
  // use its own parameters.
  function sandboxPrelude(limit) {
    'use strict';
    const post = self.postMessage.bind(self);
    post({ type: 'ready' });
    const scrub = (target, name) => {
      for (let o = target; o; o = Object.getPrototypeOf(o)) {
        const descriptor = Object.getOwnPropertyDescriptor(o, name);
        if (descriptor && descriptor.configurable) {
          try { delete o[name]; } catch (_) { /* keep going */ }
        }
      }
    };
    const refusal = (name, why) => new Error(`${name} is not available in the sandbox: ${why}`);
    const block = (target, name, why, isObject) => {
      if (!target) return;
      scrub(target, name);
      const descriptor = isObject
        ? { get() { throw refusal(name, why); }, configurable: false, enumerable: false }
        : { value: function blocked() { throw refusal(name, why); }, writable: false, configurable: false, enumerable: false };
      try { Object.defineProperty(target, name, descriptor); } catch (_) { /* already gone */ }
    };
    for (const name of ['fetch', 'XMLHttpRequest', 'WebSocket', 'WebSocketStream', 'EventSource', 'WebTransport',
      'importScripts', 'BroadcastChannel', 'RTCPeerConnection']) {
      block(self, name, 'it has no network access');
    }
    for (const name of ['Worker', 'SharedWorker']) block(self, name, 'it cannot start other workers');
    for (const name of ['indexedDB', 'caches']) block(self, name, 'it keeps no storage', true);
    block(self.navigator, 'sendBeacon', 'it has no network access');
    block(self.navigator, 'storage', 'it keeps no storage', true);
    block(self, 'postMessage', 'return a value or use console.log instead');
    block(self, 'close', 'the run ends when the snippet returns');

    // A JSON-safe copy: cycles, functions, symbols, bigints, non-finite
    // numbers, Maps, Sets, Dates, errors and typed arrays become stand-ins.
    const plain = (value, depth, seen) => {
      switch (typeof value) {
        case 'undefined': return undefined;
        case 'string': return value.length > limit ? value.slice(0, limit) + '…' : value;
        case 'boolean': return value;
        case 'number': return Number.isFinite(value) ? value : String(value);
        case 'bigint': return value.toString() + 'n';
        case 'symbol': return value.toString();
        case 'function': return `[Function${value.name ? ': ' + value.name : ''}]`;
        default: break;
      }
      if (value === null) return null;
      if (seen.has(value)) return '[Circular]';
      if (depth > 6) return Array.isArray(value) ? '[Array]' : '[Object]';
      seen.add(value);
      try {
        if (value instanceof Error) return { error: value.name, message: value.message };
        if (value instanceof Date) return Number.isNaN(value.getTime()) ? 'Invalid Date' : value.toISOString();
        if (value instanceof RegExp) return String(value);
        if (value instanceof Map) return { Map: Array.from(value).slice(0, 200).map(([k, v]) => [plain(k, depth + 1, seen), plain(v, depth + 1, seen)]) };
        if (value instanceof Set) return { Set: Array.from(value).slice(0, 200).map((v) => plain(v, depth + 1, seen)) };
        if (value instanceof DataView || value instanceof ArrayBuffer) return `[${value.constructor.name} ${value.byteLength} bytes]`;
        if (ArrayBuffer.isView(value)) value = Array.from(value.subarray(0, 1000), Number);
        if (typeof value.then === 'function') return '[Promise]';
        if (Array.isArray(value)) {
          const items = value.slice(0, 1000).map((v) => { const p = plain(v, depth + 1, seen); return p === undefined ? null : p; });
          if (value.length > 1000) items.push(`… ${value.length - 1000} more`);
          return items;
        }
        const out = {};
        const keys = Object.keys(value);
        for (const key of keys.slice(0, 200)) {
          try { out[key] = plain(value[key], depth + 1, seen); } catch (error) { out[key] = `[threw ${error && error.message}]`; }
        }
        if (keys.length > 200) out['…'] = `${keys.length - 200} more keys`;
        return out;
      } catch (error) {
        return '[unreadable]';
      } finally {
        seen.delete(value);
      }
    };
    const show = (value) => {
      if (typeof value === 'string') return value;
      try {
        const text = JSON.stringify(plain(value, 0, new WeakSet()));
        return text === undefined ? String(value) : text;
      } catch (_) {
        return String(value);
      }
    };

    let used = 0;
    let dropped = 0;
    for (const level of ['log', 'info', 'debug', 'warn', 'error']) {
      console[level] = (...args) => {
        const line = (level === 'warn' || level === 'error' ? level + ': ' : '') + args.map(show).join(' ');
        if (used + line.length > limit) { dropped += 1; return; }
        used += line.length;
        post({ type: 'log', line });
      };
    }

    let reported = false;
    const report = (message) => {
      if (reported) return;
      reported = true;
      message.dropped = dropped;
      post(message);
    };
    const kind = (value) => (value === null ? 'null' : Array.isArray(value) ? 'array' : typeof value);
    Object.defineProperty(self, '__loomSandbox', {
      value: Object.freeze({
        run(body) {
          Promise.resolve().then(body).then(
            (value) => report({
              type: 'done', ok: true, valueType: kind(value), value: plain(value, 0, new WeakSet()),
              length: typeof value === 'string' ? value.length : undefined,
            }),
            (error) => report({ type: 'done', ok: false, error: error instanceof Error ? `${error.name}: ${error.message}` : `Uncaught ${show(error)}` }),
          );
        },
      }),
      configurable: false,
      writable: false,
    });
  }

  // Where the last top-level statement of `source` starts, or -1: after the
  // last `;`, `}` or line break outside brackets, strings and comments.
  function lastStatement(source) {
    let depth = 0;
    let last = -1;
    for (let i = 0; i < source.length; i += 1) {
      const c = source[i];
      if (c === '/' && source[i + 1] === '/') {
        const end = source.indexOf('\n', i);
        if (end < 0) break;
        i = end - 1;
        continue;
      }
      if (c === '/' && source[i + 1] === '*') {
        const end = source.indexOf('*/', i + 2);
        if (end < 0) break;
        i = end + 1;
        continue;
      }
      if (c === '"' || c === "'" || c === '`') {
        let j = i + 1;
        while (j < source.length && source[j] !== c) {
          if (source[j] === '\\') j += 1;
          else if (c !== '`' && source[j] === '\n') break;
          j += 1;
        }
        i = j;
        continue;
      }
      if (c === '(' || c === '[' || c === '{') depth += 1;
      else if (c === ')' || c === ']' || c === '}') {
        depth = Math.max(0, depth - 1);
        if (depth === 0 && c === '}') last = i + 1;
      } else if (depth === 0 && (c === ';' || c === '\n')) last = i + 1;
    }
    return last > 0 && source.slice(last).trim() ? last : -1;
  }

  const STATEMENT = /^(?:return|const|let|var|if|for|while|do|function|class|throw|try|switch|import|export|break|continue|debugger|async\s+function)\b|^}/;

  // The function bodies tried in turn, the first that parses wins: the whole
  // snippet as one expression, its last statement returned, then as written.
  // The worker inherits the page's policy (no eval), so a body that does not
  // parse shows up as a worker that never says it is ready.
  function sandboxBodies(code) {
    const source = String(code).replace(/\s+$/, '');
    const bare = source.replace(/[;\s]+$/, '');
    const bodies = [];
    if (bare) bodies.push(`return (\n${bare}\n);`);
    const split = lastStatement(bare);
    if (split > 0) {
      const head = bare.slice(0, split);
      const tail = bare.slice(split).trim();
      const before = head.replace(/\s+$/, '');
      const ended = before === '' || /[;}]$/.test(before);
      const continues = /[-+*/%=<>&|^?:.,([{!~]$/.test(before) || (!ended && /^[-+*/%=<>&|^?:.,([`]/.test(tail));
      if (tail && !STATEMENT.test(tail) && !continues) bodies.push(`${head}\nreturn (\n${tail}\n);`);
    }
    bodies.push(source);
    return bodies;
  }

  function sandboxSource(body) {
    return `(${sandboxPrelude.toString()})(${TOOL_OUTPUT_LIMIT});\n__loomSandbox.run(async function () {\n${body}\n});\n`;
  }

  // What run_javascript hands back to the model.
  function sandboxText(outcome) {
    const document = { ok: outcome.ok };
    if (outcome.ok) {
      document.type = outcome.type;
      // Ahead of the result, so it survives when the text is cut.
      if (outcome.type === 'string' && outcome.length > TOOL_OUTPUT_LIMIT) document.length = outcome.length;
      if (outcome.type !== 'undefined') document.result = outcome.result;
      else if (!outcome.console.length) document.note = 'The snippet returned nothing; use `return` or end with an expression to get a value.';
    } else {
      document.error = outcome.error;
    }
    if (outcome.console.length) document.console = outcome.console;
    if (outcome.dropped) document.console_dropped = outcome.dropped;
    let text;
    try { text = JSON.stringify(document); } catch (error) { text = JSON.stringify({ ok: false, error: 'the result could not be serialized' }); }
    return capText(text, TOOL_OUTPUT_LIMIT);
  }

  // Runs `code` in a worker built from a fresh Blob and terminated afterwards,
  // so nothing survives between runs; a run past `timeout` is terminated.
  function runJavaScript(code, options) {
    const { timeout = SANDBOX_TIMEOUT_MS, signal } = options || {};
    const started = performance.now();
    const bodies = sandboxBodies(code);
    return new Promise((resolve) => {
      let attempt = 0;
      let worker = null;
      let url = null;
      let timer = 0;
      let parseError = null;
      let settled = false;
      const logs = [];
      const release = () => {
        clearTimeout(timer);
        if (worker) worker.terminate();
        if (url) URL.revokeObjectURL(url);
        worker = null;
        url = null;
      };
      const finish = (outcome) => {
        if (settled) return;
        settled = true;
        release();
        if (signal) signal.removeEventListener('abort', onAbort);
        const result = Object.assign({ ok: false, type: null, result: undefined, error: null, dropped: 0, timedOut: false, cancelled: false }, outcome);
        result.console = logs;
        result.ms = Math.round(performance.now() - started);
        result.text = sandboxText(result);
        resolve(result);
      };
      const onAbort = () => finish({ cancelled: true, error: 'Cancelled: the conversation was stopped.' });
      if (signal) {
        if (signal.aborted) { onAbort(); return; }
        signal.addEventListener('abort', onAbort);
      }
      const next = () => {
        if (attempt >= bodies.length) {
          const message = (parseError || '').replace(/^Uncaught /, '');
          finish({ error: /Error/.test(message) ? message : 'The sandbox worker could not start' + (message ? `: ${message}` : '') });
          return;
        }
        const body = bodies[attempt];
        attempt += 1;
        let ready = false;
        url = URL.createObjectURL(new Blob([sandboxSource(body)], { type: 'text/javascript' }));
        try {
          worker = new Worker(url);
        } catch (error) {
          finish({ error: `The sandbox worker could not start: ${error.message}` });
          return;
        }
        timer = setTimeout(() => finish({ timedOut: true, error: `Timed out after ${timeout} ms; the worker was terminated` }), timeout);
        worker.onmessage = (event) => {
          const data = event.data || {};
          if (data.type === 'ready') ready = true;
          else if (data.type === 'log') { if (logs.length < 1000) logs.push(String(data.line)); }
          else if (data.type === 'done') finish({ ok: !!data.ok, type: data.valueType || null, result: data.value, length: data.length, error: data.error || null, dropped: data.dropped || 0 });
        };
        worker.onerror = (event) => {
          event.preventDefault();
          if (!ready) {
            parseError = event.message || 'SyntaxError';
            release();
            next();
            return;
          }
          finish({ error: (event.message || 'Uncaught error').replace(/^Uncaught /, '') });
        };
      };
      next();
    });
  }

  // POST /api/fetch through this page's own session.
  async function webFetch(url, signal) {
    const started = performance.now();
    const done = (outcome) => Object.assign(outcome, { ms: Math.round(performance.now() - started) });
    try {
      const data = await api('/api/fetch', { method: 'POST', json: { url }, signal });
      return done({ ok: true, data, text: capText(JSON.stringify(data), TOOL_OUTPUT_LIMIT + 4096) });
    } catch (error) {
      if (error.name === 'AbortError') return done({ ok: false, cancelled: true, error: 'Cancelled: the conversation was stopped.', text: JSON.stringify({ ok: false, error: 'Cancelled by the user' }) });
      return done({ ok: false, error: error.message, text: JSON.stringify({ ok: false, error: error.message }) });
    }
  }

  // Tool-call arguments as an object; text that is not JSON is kept as `raw`.
  function callArguments(text) {
    if (!text || !String(text).trim()) return {};
    try {
      const value = JSON.parse(text);
      if (value && typeof value === 'object' && !Array.isArray(value)) return value;
      return { raw: typeof value === 'string' ? value : String(text) };
    } catch (_) {
      return { raw: String(text) };
    }
  }

  function firstString(...values) {
    return values.find((v) => typeof v === 'string' && v.trim()) || null;
  }

  async function executeTool(call, offered, signal) {
    const args = callArguments(call.arguments);
    if (!offered.has(call.name)) {
      const error = `Unknown tool ${call.name || '(unnamed)'}; available: ${[...offered].join(', ') || 'none'}`;
      return { ok: false, error, ms: 0, text: JSON.stringify({ ok: false, error }) };
    }
    if (call.name === 'run_javascript') {
      const code = firstString(args.code, args.script, args.source, args.javascript, args.raw);
      if (!code) {
        const error = 'run_javascript needs {"code": "<JavaScript>"}';
        return { ok: false, error, ms: 0, text: JSON.stringify({ ok: false, error }) };
      }
      return runJavaScript(code, { signal });
    }
    const url = firstString(args.url, args.href, args.link, args.raw);
    if (!url) {
      const error = 'web_fetch needs {"url": "<absolute http or https URL>"}';
      return { ok: false, error, ms: 0, text: JSON.stringify({ ok: false, error }) };
    }
    return webFetch(url.trim(), signal);
  }

  // What a card shows for a call's arguments: the code or the URL itself.
  function argumentsView(call) {
    const args = callArguments(call.arguments);
    const shown = call.name === 'run_javascript'
      ? firstString(args.code, args.script, args.source, args.javascript, args.raw)
      : call.name === 'web_fetch' ? firstString(args.url, args.href, args.link, args.raw) : null;
    return shown || call.arguments || '{}';
  }

  // What a card shows for an outcome, readable rather than raw JSON.
  function outcomeView(call, outcome) {
    if (call.name === 'web_fetch' && outcome.ok && outcome.data) {
      const d = outcome.data;
      const head = [`HTTP ${d.status}`, d.content_type || null, isNum(d.bytes) ? fmtBytes(d.bytes) : null, d.truncated ? 'truncated' : null].filter(Boolean).join(' · ');
      return [head, d.location ? `Location: ${d.location}` : null, d.title ? `Title: ${d.title}` : null, '', d.text || ''].filter((line) => line !== null).join('\n');
    }
    if (call.name === 'run_javascript' && 'console' in outcome) {
      const lines = outcome.console.slice();
      if (outcome.dropped) lines.push(`(${outcome.dropped} more console lines dropped)`);
      if (outcome.ok) {
        if (outcome.type === 'undefined') lines.push(lines.length ? '→ (no value)' : '(no value returned; the model was told to use return)');
        else {
          let shown;
          // Multi-line strings read better as text than as one escaped line.
          if (typeof outcome.result === 'string') shown = outcome.result.includes('\n') ? outcome.result : JSON.stringify(outcome.result);
          else {
            try { shown = JSON.stringify(outcome.result, null, 2); } catch (_) { shown = String(outcome.result); }
          }
          lines.push(`→ ${shown}`);
        }
      } else {
        lines.push(`✗ ${outcome.error}`);
      }
      return lines.join('\n');
    }
    return outcome.ok ? outcome.text : `✗ ${outcome.error}`;
  }

  function toolCard(container, call) {
    const badge = el('span', { class: 'badge', dataset: { tone: 'accent' } }, el('span', { class: 'dot', 'aria-hidden': 'true' }), 'running');
    const time = el('span', { class: 'tool-time', text: '' });
    const output = el('pre', { class: 'tool-output', hidden: true, tabindex: '0', 'aria-label': `${call.name} output` });
    const more = el('button', { class: 'btn ghost small tool-more', type: 'button', hidden: true, 'aria-expanded': 'false' });
    const card = el('div', { class: 'tool-card', role: 'group', 'aria-label': `Tool call: ${call.name}`, dataset: { state: 'running' } },
      el('div', { class: 'tool-head' },
        el('span', { class: 'tool-glyph', 'aria-hidden': 'true', text: call.name === 'web_fetch' ? '↧' : '{ }' }),
        el('code', { class: 'tool-name', text: call.name || '(unnamed tool)' }),
        badge, el('span', { class: 'grow' }), time),
      el('pre', { class: 'tool-args', text: argumentsView(call) }),
      output, more);
    container.append(card);
    const started = performance.now();
    return {
      finish(outcome) {
        const state = outcome.ok ? 'done' : 'failed';
        card.dataset.state = state;
        badge.dataset.tone = outcome.ok ? 'good' : 'critical';
        badge.lastChild.textContent = outcome.cancelled ? 'cancelled' : outcome.timedOut ? 'timed out' : state;
        time.textContent = fmtSeconds((isNum(outcome.ms) ? outcome.ms : performance.now() - started) / 1000);
        const text = outcomeView({ name: card.querySelector('.tool-name').textContent }, outcome);
        output.textContent = text;
        output.hidden = false;
        const lines = text.split('\n').length;
        if (lines > CARD_LINES || text.length > 800) {
          output.classList.add('clamped');
          more.hidden = false;
          const label = () => (output.classList.contains('clamped') ? `Show all ${lines} line${lines === 1 ? '' : 's'}` : 'Show less');
          more.textContent = label();
          more.onclick = () => {
            output.classList.toggle('clamped');
            more.setAttribute('aria-expanded', String(!output.classList.contains('clamped')));
            more.textContent = label();
          };
        }
      },
    };
  }

  // ---------------------------------------------------------------- chat

  function chatLog() { return $('chat-log'); }

  function nearBottom(node) { return node.scrollHeight - node.scrollTop - node.clientHeight < 60; }

  function addTurn(role) {
    const log = chatLog();
    for (const empty of log.querySelectorAll('.empty, .empty-state')) empty.remove();
    const turn = el('div', { class: 'turn ' + role });
    const bubble = el('div', { class: 'bubble' });
    turn.append(bubble);
    log.append(turn);
    return { turn, bubble };
  }

  function usageChips(parts) {
    return el('div', { class: 'usage' }, parts.filter(Boolean).map((text) => el('span', { class: 'badge', text })));
  }

  // The tools this conversation offers the model, by the panel's toggles.
  function enabledTools() {
    const tools = [];
    if ($('chat-tool-js').checked) tools.push(TOOLS.run_javascript);
    if ($('chat-tool-web').checked && !$('chat-tool-web').disabled) tools.push(TOOLS.web_fetch);
    return tools;
  }

  // Folds one streamed tool_calls fragment into the calls seen so far.
  function mergeToolCall(calls, part) {
    const index = isNum(part.index) ? part.index : (part.id ? calls.findIndex((c) => c && c.id === part.id) : -1);
    const at = index >= 0 ? index : calls.length;
    const slot = calls[at] || (calls[at] = { id: '', name: '', arguments: '' });
    if (part.id) slot.id = part.id;
    const fn = part.function || {};
    if (typeof fn.name === 'string' && fn.name && fn.name !== slot.name) slot.name = slot.name ? slot.name + fn.name : fn.name;
    if (typeof fn.arguments === 'string') slot.arguments += fn.arguments;
    else if (fn.arguments && typeof fn.arguments === 'object') slot.arguments = JSON.stringify(fn.arguments);
  }

  function chatPayload(tools) {
    const messages = [];
    const system = $('chat-system').value.trim();
    if (system) messages.push({ role: 'system', content: system });
    messages.push(...state.chat.messages);
    const payload = { messages, stream: true, stream_options: { include_usage: true } };
    const maxTokens = Number($('chat-max-tokens').value);
    if (maxTokens > 0) payload.max_tokens = maxTokens;
    const temperature = Number($('chat-temperature').value);
    if ($('chat-temperature').value !== '' && temperature >= 0) payload.temperature = temperature;
    if (!$('chat-thinking').disabled) payload.enable_thinking = $('chat-thinking').checked;
    if (tools.length) payload.tools = tools;
    return payload;
  }

  // One assistant turn: streamed into a new bubble, recorded in the history.
  async function streamTurn(tools, signal) {
    const log = chatLog();
    const reply = addTurn('assistant');
    reply.bubble.classList.add('pending');
    let thinking = null;
    const payload = chatPayload(tools);
    log.scrollTop = log.scrollHeight;
    const started = performance.now();
    let ttft = null, usage = null, finish = null, content = '', reasoning = '', failure = null, stopped = false;
    const calls = [];
    const seen = () => { if (ttft === null) ttft = (performance.now() - started) / 1000; };
    const take = (delta) => {
      if (delta.reasoning_content) {
        seen();
        reasoning += delta.reasoning_content;
        if (!thinking) {
          thinking = el('details', { class: 'thinking', open: true }, el('summary', { text: 'Thinking' }), el('div'));
          reply.turn.insertBefore(thinking, reply.bubble);
        }
        thinking.lastChild.textContent = reasoning;
      }
      if (delta.content) {
        seen();
        content += delta.content;
        if (thinking && thinking.open && reasoning) { thinking.open = false; }
        reply.bubble.textContent = content;
      }
      if (Array.isArray(delta.tool_calls)) {
        seen();
        for (const part of delta.tool_calls) if (part) mergeToolCall(calls, part);
        if (!content) reply.bubble.textContent = `Calling ${calls.filter(Boolean).map((c) => c.name || 'a tool').join(', ')}…`;
      }
    };
    const handle = (line) => {
      if (!line.startsWith('data:')) return;
      const data = line.slice(5).trim();
      if (!data || data === '[DONE]') return;
      let event;
      try { event = JSON.parse(data); } catch (_) { return; }
      if (event.error) { failure = event.error.message || 'the server reported an error'; return; }
      for (const choice of event.choices || []) {
        take(choice.delta || choice.message || {});
        if (choice.finish_reason) finish = choice.finish_reason;
      }
      if (event.usage) usage = event.usage;
    };
    try {
      const follow = nearBottom(log);
      const response = await api('/api/chat', { method: 'POST', json: payload, raw: true, signal });
      const type = response.headers.get('content-type') || '';
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error((body && body.error && body.error.message) || `HTTP ${response.status}`);
      }
      if (!type.includes('text/event-stream')) {
        const body = await response.json();
        seen();
        const choice = (body.choices && body.choices[0]) || {};
        take(choice.message || {});
        usage = body.usage || null;
        finish = choice.finish_reason || null;
      } else {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let index;
          while ((index = buffer.indexOf('\n')) >= 0) {
            handle(buffer.slice(0, index).replace(/\r$/, ''));
            buffer = buffer.slice(index + 1);
          }
          if (follow || nearBottom(log)) log.scrollTop = log.scrollHeight;
        }
        if (buffer) handle(buffer);
      }
    } catch (error) {
      if (error.name === 'AbortError') stopped = true;
      else failure = error.message;
    }
    const duration = (performance.now() - started) / 1000;
    reply.bubble.classList.remove('pending');
    // A turn cut short never runs the calls it was still writing.
    const toolCalls = stopped || failure ? [] : calls.filter((c) => c && (c.name || c.arguments)).map((c, i) => ({
      id: c.id || `call_${Date.now().toString(36)}_${i}`, name: c.name, arguments: c.arguments || '{}',
    }));
    if (failure && !content && !toolCalls.length) {
      reply.bubble.classList.add('error');
      reply.bubble.textContent = failure;
      return { failed: true, reply, toolCalls: [] };
    }
    if (toolCalls.length && !content) reply.bubble.hidden = true;
    else if (!content) reply.bubble.textContent = reasoning ? '' : (stopped ? '(stopped)' : '(no content)');
    const message = { role: 'assistant', content: toolCalls.length && !content ? null : content };
    if (toolCalls.length) {
      message.tool_calls = toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.arguments } }));
    }
    state.chat.messages.push(message);
    const box = toolCalls.length ? el('div', { class: 'tool-calls' }) : null;
    if (box) reply.turn.append(box);
    const completion = usage && usage.completion_tokens;
    const rateValue = isNum(completion) && completion > 1 && ttft !== null && duration > ttft ? (completion - 1) / (duration - ttft) : null;
    const cached = usage && usage.prompt_tokens_details && usage.prompt_tokens_details.cached_tokens;
    reply.turn.append(usageChips([
      usage ? `${fmtInt(usage.prompt_tokens)} → ${fmtInt(usage.completion_tokens)} tokens` : null,
      isNum(cached) && cached > 0 ? `${fmtInt(cached)} cached` : null,
      rateValue !== null ? `${fmtRate(rateValue)} tok/s` : null,
      ttft !== null ? `TTFT ${fmtSeconds(ttft)}` : null,
      stopped ? 'stopped' : (finish && finish !== 'stop' && finish !== 'tool_calls' ? `finish: ${finish}` : null),
      failure ? `error: ${failure}` : null,
    ]));
    return { failed: false, stopped, reply, box, toolCalls };
  }

  // Runs one round of calls in order, answering each in the history.
  async function runToolCalls(turn, tools, signal) {
    const offered = new Set(tools.map((tool) => tool.function.name));
    const log = chatLog();
    for (const call of turn.toolCalls) {
      const card = toolCard(turn.box, call);
      log.scrollTop = log.scrollHeight;
      const outcome = signal.aborted
        ? { ok: false, cancelled: true, error: 'Cancelled: the conversation was stopped.', ms: 0, text: JSON.stringify({ ok: false, error: 'Cancelled by the user' }) }
        : await executeTool(call, offered, signal);
      card.finish(outcome);
      state.chat.messages.push({ role: 'tool', tool_call_id: call.id, content: outcome.text });
      log.scrollTop = log.scrollHeight;
    }
  }

  // Answers calls that will not run, so the history stays well formed.
  function answerUnrun(turn, reason) {
    for (const call of turn.toolCalls) {
      const card = toolCard(turn.box, call);
      card.finish({ ok: false, error: reason, ms: 0, text: '' });
      state.chat.messages.push({ role: 'tool', tool_call_id: call.id, content: JSON.stringify({ ok: false, error: reason }) });
    }
  }

  async function sendChat(text) {
    const launch = state.launch;
    if (!text.trim() || state.chat.busy || !launch || launch.state !== 'serving') return;
    const log = chatLog();
    const base = state.chat.messages.length;
    const user = addTurn('user');
    user.bubble.textContent = text;
    state.chat.messages.push({ role: 'user', content: text });
    const controller = new AbortController();
    state.chat.busy = true;
    state.chat.controller = controller;
    $('chat-stop').hidden = false;
    applyCapabilities();
    const tools = enabledTools();
    try {
      for (let round = 0; ; round += 1) {
        const turn = await streamTurn(tools, controller.signal);
        if (turn.failed) {
          // Nothing of a failed exchange stays in the history.
          state.chat.messages.length = base;
          break;
        }
        if (turn.stopped || !turn.toolCalls.length) break;
        if (round >= TOOL_ROUNDS) {
          answerUnrun(turn, `Not run: this message already used its ${TOOL_ROUNDS} rounds of tool calls.`);
          turn.reply.turn.append(el('p', { class: 'tool-note', role: 'note', text: `Stopped after ${TOOL_ROUNDS} rounds of tool calls. Send another message to let the model continue.` }));
          break;
        }
        await runToolCalls(turn, tools, controller.signal);
        if (controller.signal.aborted) break;
      }
    } finally {
      state.chat.busy = false;
      state.chat.controller = null;
      $('chat-stop').hidden = true;
      applyCapabilities();
      log.scrollTop = log.scrollHeight;
      $('chat-input').focus();
    }
  }

  function initChat() {
    const input = $('chat-input');
    $('chat-form').addEventListener('submit', (event) => {
      event.preventDefault();
      const text = input.value;
      if (!text.trim()) return;
      input.value = '';
      sendChat(text);
    });
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        $('chat-form').requestSubmit();
      }
    });
    $('chat-stop').addEventListener('click', () => { if (state.chat.controller) state.chat.controller.abort(); });
    $('chat-clear').addEventListener('click', () => {
      if (state.chat.controller) state.chat.controller.abort();
      state.chat.messages = [];
      chatLog().replaceChildren(el('div', { class: 'empty-state small' }, el('strong', { text: 'Conversation cleared' }), el('p', { text: 'The next message starts a new conversation.' })));
    });
  }

  // ---------------------------------------------------------------- benchmarks

  const SUITE_HINTS = { quick: 'decode, prefill 1k, 2k', standard: 'adds decode 2k, prefill 8k, 4 streams', full: 'every row up to 32k' };

  function renderSuites() {
    if (!state.suites.length) return;
    const group = $('suites');
    group.replaceChildren();
    for (const suite of state.suites) {
      const name = suite.name;
      group.append(el('button', {
        class: 'suite', type: 'button', 'data-suite': name, title: (suite.metrics || []).join(', '),
        onclick: (event) => runSuite(name, event.currentTarget),
      }, el('strong', { text: name.charAt(0).toUpperCase() + name.slice(1) }), el('span', { text: SUITE_HINTS[name] || `${(suite.metrics || []).length} rows` })));
    }
    applyCapabilities();
  }

  async function runSuite(suite, button) {
    const launch = state.launch;
    if (!launch || launch.state !== 'serving') return;
    const repeats = Number($('bench-repeats').value) || 1;
    await act('bench:' + suite, button, () => api('/api/bench', { method: 'POST', json: { suite, repeats, repo: launch.repo } }), $('bench-error'));
  }

  async function loadRuns() {
    try {
      const data = await api('/api/benchmarks');
      state.runs = data.runs || [];
      if (!state.suites.length && Array.isArray(data.suites)) { state.suites = data.suites; renderSuites(); }
    } catch (error) {
      state.runs = [];
    }
    state.runsLoaded = true;
    renderRuns();
    if (!state.selectedRun && state.runs.length) selectRun(state.runs[0].id, true);
  }

  function headlineDecode(run) {
    const row = (run.headline || []).find((r) => r.metric === 'decode');
    return row && isNum(row.value) ? `${fmtRate(row.value)} tok/s` : '—';
  }

  function renderRuns() {
    const body = $('runs-body');
    body.replaceChildren();
    if (!state.runs.length) {
      body.append(el('tr', {}, el('td', { colspan: '5' }, el('div', { class: 'empty-state small' }, el('strong', { text: 'No stored runs yet' }), el('p', { text: 'Launch a profile, then run a suite; each run is stored where `loom-launch bench --list` finds it.' })))));
      return;
    }
    for (const run of state.runs) {
      const selected = run.id === state.selectedRun;
      const [, name] = splitRepo(run.profile);
      const row = el('tr', { 'aria-selected': String(selected), onclick: () => selectRun(run.id) },
        el('td', {}, el('button', { class: 'row-button', type: 'button', text: fmtDate(run.started_at), 'aria-label': `Show the ${run.suite} run of ${run.profile} from ${fmtDate(run.started_at)}`, 'aria-pressed': String(selected), onclick: (event) => { event.stopPropagation(); selectRun(run.id); } })),
        el('td', { class: 'profile', title: run.profile, text: name || run.profile }),
        el('td', { text: run.suite || '—' }),
        el('td', { class: 'num', text: headlineDecode(run) }),
        el('td', {}, el('span', { class: 'status-ok', dataset: { tone: run.ok ? 'good' : 'critical' } }, el('span', { class: 'dot', 'aria-hidden': 'true' }), run.missing ? 'missing' : (run.ok ? 'ok' : 'failed'))),
      );
      body.append(row);
    }
  }

  function ratioChip(ratio) {
    if (!isNum(ratio)) return el('span', { class: 'empty', text: '—' });
    let tone = 'neutral', glyph = '≈', words = 'near the published row';
    if (ratio >= 1) { tone = 'good'; glyph = '▲'; words = 'at or above the published row'; }
    else if (ratio < 0.9) { tone = 'serious'; glyph = '▼'; words = 'below the published row'; }
    return el('span', { class: 'ratio', dataset: { tone }, title: words },
      el('span', { class: 'glyph', 'aria-hidden': 'true', text: glyph }), `${ratio.toFixed(2)}×`, el('span', { class: 'sr-only', text: ` (${words})` }));
  }

  function rooflineCell(fraction, value, ceiling, unit) {
    if (!isNum(fraction)) return el('span', { class: 'empty', text: '—' });
    const meter = el('span', { class: 'meter', 'aria-hidden': 'true' }, el('span'));
    meter.firstChild.style.width = Math.round(Math.max(0, Math.min(1, fraction)) * 100) + '%';
    const label = `${Math.round(fraction * 100)}%`;
    return el('span', { class: 'roofline', title: isNum(ceiling) ? `${fmtRate(value)} of a ${fmtRate(ceiling)} ${unit} ceiling` : '' },
      meter, el('span', { class: 'pct', text: label }), el('span', { class: 'sr-only', text: ' of the hardware ceiling' }));
  }

  async function selectRun(id, quiet) {
    state.selectedRun = id;
    renderRuns();
    try {
      const record = await api('/api/benchmarks/' + encodeURIComponent(id));
      if (state.selectedRun === id) renderRecord(record);
    } catch (error) {
      if (!quiet) $('compare-body').replaceChildren(el('tr', {}, el('td', { colspan: '5' }, el('p', { class: 'inline-error', text: `Run ${id} could not be read: ${error.message}` }))));
    }
  }

  function renderRecord(record) {
    const harness = record.harness || {};
    $('detail-title').textContent = `${record.profile || 'run'} · ${record.suite || ''} suite`;
    const meta = $('detail-meta');
    meta.replaceChildren();
    const fact = (k, v) => (v == null || v === '' ? null : el('li', {}, el('span', { class: 'k', text: k }), String(v)));
    meta.append(...[
      fact('started', fmtDate(record.started_at)),
      fact('elapsed', isNum(record.elapsed_seconds) ? fmtDuration(record.elapsed_seconds) : null),
      fact('device', record.device),
      fact('model', record.model),
      fact('repeats', harness.repeats),
      fact('parallel', harness.parallel),
      fact('thinking', harness.thinking === null || harness.thinking === undefined ? null : (harness.thinking ? 'on' : 'off')),
      fact('workload', harness.workload),
      fact('id', record.id),
    ].filter(Boolean));
    const body = $('compare-body');
    body.replaceChildren();
    const metrics = record.metrics || {};
    let rows = Array.isArray(record.comparison) ? record.comparison.slice() : [];
    const compared = new Set(rows.map((r) => r.metric));
    for (const [metric, entry] of Object.entries(metrics)) {
      if (!compared.has(metric)) rows.push({ metric, label: entry.label, value: entry.value, note: entry.note });
    }
    if (!rows.length) {
      body.append(el('tr', {}, el('td', { colspan: '5', class: 'empty', text: 'This run measured nothing.' })));
    }
    for (const row of rows) {
      const entry = metrics[row.metric] || {};
      const value = isNum(row.value) ? row.value : (isNum(row.measured) ? row.measured : entry.value);
      const unit = entry.unit || 'tok/s';
      const label = row.label || entry.label || row.metric;
      body.append(el('tr', {},
        el('td', {}, el('span', { class: 'metric-label', text: label }), el('span', { class: 'metric-code', text: row.metric })),
        el('td', { class: 'num', title: entry.note || row.note || '' }, isNum(value) ? `${fmtRate(value)} ${unit}` : (entry.note || row.note || '—')),
        el('td', { class: 'num' }, isNum(row.published) ? fmtRate(row.published) : '—'),
        el('td', {}, ratioChip(row.ratio)),
        el('td', {}, rooflineCell(row.roofline_fraction, value, row.ceiling, unit)),
      ));
    }
    const notes = $('detail-notes');
    notes.replaceChildren(...(record.notes || []).map((note) => el('li', { text: note })));
  }

  // ---------------------------------------------------------------- logs

  function logLimit() { return Number($('log-lines').value) || 200; }

  async function loadLogs() {
    try {
      const data = await api('/api/logs?lines=' + logLimit());
      state.log.run = data.run;
      state.log.lines.serve = data.serve_log || [];
      state.log.lines.events = data.events || [];
      $('logs-hint').textContent = data.run
        ? `Run ${data.run}${data.current ? ' (current launch)' : ' (latest recorded run)'}.`
        : 'No runs recorded yet.';
    } catch (error) {
      state.log.lines.serve = [];
      state.log.lines.events = [];
      $('logs-hint').textContent = error.message;
    }
    renderLog();
  }

  function lineNode(text, kind) {
    const node = el('span', { class: 'line' });
    if (kind === 'events' && text.startsWith('{')) {
      try {
        const event = JSON.parse(text);
        const rest = Object.assign({}, event);
        delete rest.time; delete rest.event;
        const time = event.time ? String(event.time).replace(/^.*T/, '').replace(/Z$/, '') : '';
        node.append(el('span', { class: 'ts', text: time.padEnd(10) }), ' ', el('span', { class: 'ev', text: String(event.event || '?') }), ' ', JSON.stringify(rest));
        if (event.event === 'failed' || event.error) node.dataset.level = 'error';
        return node;
      } catch (_) { /* fall through */ }
    }
    node.textContent = text;
    if (/\b(error|fatal|panic|failed)\b/i.test(text)) node.dataset.level = 'error';
    else if (/\bwarn(ing)?\b/i.test(text)) node.dataset.level = 'warn';
    return node;
  }

  function renderLog() {
    const view = $('log-view');
    const lines = state.log.lines[state.log.kind].slice(-logLimit());
    view.replaceChildren(...lines.map((line) => lineNode(line, state.log.kind)));
    if (!lines.length) view.append(el('span', { class: 'line empty', text: state.log.run ? 'Nothing logged yet.' : 'No runs recorded yet: a launch writes serve.log and events.jsonl here.' }));
    if ($('log-follow').checked) view.scrollTop = view.scrollHeight;
  }

  function onLog(log) {
    if (!log) return;
    if (log.reset || (log.run && log.run !== state.log.run)) { loadLogs(); return; }
    let appended = false;
    for (const kind of ['serve', 'events']) {
      const fresh = log[kind] || [];
      if (!fresh.length) continue;
      const lines = state.log.lines[kind];
      lines.push(...fresh);
      if (lines.length > 5000) lines.splice(0, lines.length - 5000);
      if (kind === state.log.kind && $('log-follow').checked) {
        const view = $('log-view');
        const empty = view.querySelector('.empty');
        if (empty) empty.remove();
        for (const line of fresh) view.append(lineNode(line, kind));
        while (view.childElementCount > logLimit()) view.firstElementChild.remove();
        appended = true;
      }
    }
    if (appended) $('log-view').scrollTop = $('log-view').scrollHeight;
  }

  function initLogs() {
    const tabs = [$('log-tab-serve'), $('log-tab-events')];
    const select = (tab) => {
      for (const other of tabs) {
        const active = other === tab;
        other.setAttribute('aria-selected', String(active));
        other.tabIndex = active ? 0 : -1;
      }
      $('log-view').setAttribute('aria-labelledby', tab.id);
      state.log.kind = tab.dataset.log;
      renderLog();
    };
    for (const tab of tabs) {
      tab.addEventListener('click', () => select(tab));
      tab.addEventListener('keydown', (event) => {
        if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
        event.preventDefault();
        const next = tabs[(tabs.indexOf(tab) + 1) % tabs.length];
        select(next);
        next.focus();
      });
    }
    $('log-follow').addEventListener('change', () => { if ($('log-follow').checked) renderLog(); });
    $('log-lines').addEventListener('change', loadLogs);
    $('log-reload').addEventListener('click', loadLogs);
  }

  // ---------------------------------------------------------------- start

  function loadAll() {
    loadStatus();
    loadInstalls().then(() => runSearch('', true));
    loadRuns();
    loadLogs();
    refreshJobs();
  }

  function start() {
    tokenFromAddress();
    trackHeader();
    initTheme();
    initToken();
    initTabs();
    initActivity();
    initSparkHover();
    initLaunch();
    initSearch();
    initChat();
    initLogs();
    renderLastRequest(null);
    applyCapabilities();
    loadAll();
    connect();
  }

  // Hooks for scripted checks of the chat tools (the page itself does not use them).
  window.loomLaunch = Object.freeze({ tools: TOOLS, runJavaScript, webFetch, sandboxBodies, capText });

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
