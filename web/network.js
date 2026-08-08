/* Terllama Network panel — node/peer/cluster management UI.
 * Requires: web/index.html (provides #netToggle button, #netPanel container,
 * global apiBase, window.TerllamaFetchModels). Optional: anime.min.js (global `anime`).
 * Serve proxies /network/* to the node daemon; run serve with --node for node mode.
 */
(function () {
  'use strict';

  var NET_STATUS = '/network/status';
  var NET_START = '/network/start-model';
  var NET_STOP = '/network/stop-model';
  var POLL_MS = 4000;
  var STALE_MS = 8000;
  var MAX_CLUSTERS = 3;

  var btn = document.getElementById('netToggle');
  var panel = document.getElementById('netPanel');
  if (!btn || !panel) return;

  var nodeRunning = false;
  var lastStatus = null;
  var runningCache = [];
  var startingModel = null;
  var pollTimer = null;
  var clusterUrls = {}; // port -> cluster_url (host may be a remote node)

  var canvas = null;
  var ctx = null;
  var tl = null;
  var prog = { p: 0 };
  var nodes = [];

  function $id(s) { return document.getElementById(s); }
  function el(tag, cls) { var n = document.createElement(tag); if (cls) n.className = cls; return n; }

  /* ── Styling (injected; keeps index.html diff minimal) ── */
  function injectCss() {
    if (document.getElementById('net-style')) return;
    var st = document.createElement('style');
    st.id = 'net-style';
    var css = '.net-toggle{position:fixed;left:16px;bottom:16px;z-index:1000;width:44px;height:44px;border-radius:var(--radius-sm);border:1px solid var(--border);background:var(--surface);color:var(--accent);cursor:pointer;display:flex;align-items:center;justify-content:center;box-shadow:var(--shadow-lg);transition:background var(--transition),color var(--transition),transform .1s ease}' +
      '.net-toggle:hover{background:var(--surface-hover);color:var(--accent-hover)}' +
      '.net-toggle:active{transform:scale(.94)}' +
      '.net-panel{position:fixed;left:16px;bottom:72px;width:360px;max-width:calc(100vw - 32px);max-height:calc(100vh - 96px);overflow-y:auto;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow-lg);z-index:999}' +
      '.net-panel.hidden{display:none}' +
      '.net-canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;opacity:.55;z-index:0}' +
      '.net-panel>*:not(.net-canvas){position:relative;z-index:1}' +
      '.net-header{display:flex;align-items:center;justify-content:space-between;padding:12px 14px;font-weight:700;font-size:14px;border-bottom:1px solid var(--border)}' +
      '.net-close{background:transparent;border:none;cursor:pointer;color:var(--text-secondary);font-size:14px;padding:2px 8px;border-radius:4px;transition:background var(--transition),color var(--transition)}' +
      '.net-close:hover{background:var(--surface-hover);color:var(--text)}' +
      '.net-section{padding:10px 14px;border-top:1px solid var(--border)}' +
      '.net-section-title{font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--text-secondary);margin-bottom:8px}' +
      '.net-section-body{display:flex;flex-direction:column;gap:6px;font-size:13px;color:var(--text)}' +
      '.net-empty{color:var(--text-secondary);font-size:12px;padding:4px 0}' +
      '.net-row{display:flex;align-items:center;justify-content:space-between;gap:8px}' +
      '.net-row-info{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}' +
      '.net-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;display:inline-block;margin-right:8px}' +
      '.net-dot.fresh{background:#22c55e}' +
      '.net-dot.stale{background:var(--danger)}' +
      '.net-btn{padding:3px 10px;font-size:12px;border:1px solid var(--border);border-radius:var(--radius-sm);background:var(--surface);color:var(--text);cursor:pointer;transition:background var(--transition),border-color var(--transition),color var(--transition)}' +
      '.net-btn:hover:not(:disabled){background:var(--surface-hover);border-color:var(--accent)}' +
      '.net-btn:disabled{opacity:.5;cursor:not-allowed}' +
      '.net-btn.start{border-color:var(--accent);color:var(--accent)}' +
      '.net-btn.start:hover:not(:disabled){border-color:var(--accent-hover);color:var(--accent-hover)}' +
      '.net-btn.stop{border-color:var(--danger);color:var(--danger)}' +
      '.net-btn.stop:hover:not(:disabled){background:rgba(239,68,68,.08)}' +
      '.net-count{font-weight:700;color:var(--accent)}' +
      '.net-count.max{color:var(--danger)}' +
      '.net-status{padding:8px 14px;font-size:12px;color:var(--text-secondary);border-top:1px solid var(--border);line-height:1.4}' +
      '.net-status.error{color:var(--danger)}';
    st.textContent = css;
    document.head.appendChild(st);
  }

  /* ── helpers ── */
  function fmtMB(mb) {
    if (mb == null || isNaN(mb)) return '?';
    if (mb >= 1024) return (mb / 1024).toFixed(1) + ' GB';
    return Math.round(mb) + ' MB';
  }
  function ago(ms) {
    if (ms == null || isNaN(ms) || ms < 0) return 'unknown';
    if (ms < 1000) return 'just now';
    var s = Math.floor(ms / 1000);
    if (s < 60) return s + 's ago';
    var m = Math.floor(s / 60);
    if (m < 60) return m + 'm ago';
    return Math.floor(m / 60) + 'h ago';
  }
  function cssVar(name) {
    try { return (getComputedStyle(document.documentElement).getPropertyValue(name) || '').trim(); } catch (_) { return ''; }
  }
  function setStatus(msg, isError) {
    var box = $id('netStatus');
    if (!box) return;
    box.textContent = msg || '';
    box.classList.toggle('error', !!isError);
  }

  /* ── Panel build ── */
  function section(title, bodyId, countId) {
    var wrap = el('div', 'net-section');
    var t = el('div', 'net-section-title');
    t.appendChild(document.createTextNode(title));
    if (countId) {
      t.appendChild(document.createTextNode(' '));
      var c = el('span', 'net-count');
      c.id = countId;
      t.appendChild(c);
    }
    var body = el('div', 'net-section-body');
    body.id = bodyId;
    wrap.appendChild(t);
    wrap.appendChild(body);
    return wrap;
  }
  function buildPanel() {
    panel.innerHTML = '';
    var header = el('div', 'net-header');
    header.appendChild(document.createTextNode('Terllama Network'));
    var close = el('button', 'net-close');
    close.textContent = '\u2715';
    close.title = 'Close';
    close.setAttribute('aria-label', 'Close network panel');
    close.addEventListener('click', closePanel);
    header.appendChild(close);
    panel.appendChild(header);
    panel.appendChild(section('This node', 'netNode'));
    panel.appendChild(section('Peers', 'netPeers'));
    panel.appendChild(section('Models', 'netModels'));
    panel.appendChild(section('Running clusters', 'netClusters', 'netClusterCount'));
    var status = el('div', 'net-status');
    status.id = 'netStatus';
    panel.appendChild(status);
  }

  /* ── Open / close / poll ── */
  function openPanel() {
    panel.classList.remove('hidden');
    resizeCanvas();
    startAnim();
    fetchStatus();
    if (!pollTimer) pollTimer = setInterval(fetchStatus, POLL_MS);
  }
  function closePanel() {
    panel.classList.add('hidden');
    stopAnim();
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }
  function togglePanel() {
    if (panel.classList.contains('hidden')) openPanel();
    else closePanel();
  }

  /* ── Status fetch + render ── */
  function fetchStatus() {
    fetch(NET_STATUS)
      .then(function (res) {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.json();
      })
      .then(function (data) {
        lastStatus = data || {};
        runningCache = (data && data.running || []).slice();
        nodeRunning = true;
        renderAll();
        syncClusterOptions();
      })
      .catch(function () {
        nodeRunning = false;
        lastStatus = null;
        runningCache = [];
        renderAll();
        syncClusterOptions();
      });
  }

  function renderAll() {
    var s = lastStatus;
    renderNode(s);
    renderPeers(s ? s.peers : null);
    renderModels(s ? s.models : null);
    renderClusters(s ? s.running : null);
    if (!nodeRunning) {
      setStatus('Node daemon not running (start serve with --node)', true);
    }
    if (!panel.classList.contains('hidden')) resizeCanvas();
  }

  function renderNode(s) {
    var box = $id('netNode');
    if (!box) return;
    if (!s || (!s.name && !s.node_id)) { box.innerHTML = ''; return; }
    var name = s.name || s.node_id;
    var parts = [name];
    if (s.http_port != null) parts.push('http:' + s.http_port);
    if (s.ram_available_mb != null) parts.push(fmtMB(s.ram_available_mb) + ' RAM');
    box.textContent = parts.join(' \u00b7 ');
  }

  function renderPeers(peers) {
    var box = $id('netPeers');
    if (!box) return;
    box.innerHTML = '';
    if (!peers || !peers.length) {
      var e = el('div', 'net-empty');
      e.textContent = 'No peers connected.';
      box.appendChild(e);
      return;
    }
    peers.forEach(function (p) {
      var row = el('div', 'net-row');
      var left = el('div', 'net-row-info');
      left.style.display = 'flex';
      left.style.alignItems = 'center';
      var dot = el('span', 'net-dot');
      var age = typeof p.last_seen_ms === 'number' ? p.last_seen_ms : NaN;
      dot.classList.add((isNaN(age) || age > STALE_MS) ? 'stale' : 'fresh');
      left.appendChild(dot);
      left.appendChild(document.createTextNode(
        (p.name || 'peer') + ' \u00b7 ' + fmtMB(p.ram_available_mb) + ' \u00b7 ' + ago(age)
      ));
      row.appendChild(left);
      box.appendChild(row);
    });
  }

  function renderModels(models) {
    var box = $id('netModels');
    if (!box) return;
    box.innerHTML = '';
    if (!models || !models.length) {
      var e = el('div', 'net-empty');
      e.textContent = 'No models available.';
      box.appendChild(e);
      return;
    }
    models.forEach(function (m) {
      var name = m.name || m.model || String(m);
      var row = el('div', 'net-row');
      var info = el('div', 'net-row-info');
      info.textContent = name + ' \u00b7 ' + fmtMB(m.size_mb);
      row.appendChild(info);
      var b = el('button', 'net-btn start');
      if (startingModel === name) {
        b.textContent = 'Starting\u2026';
        b.disabled = true;
      } else {
        b.textContent = 'Start';
        b.addEventListener('click', function () { startModel(name); });
      }
      if (!nodeRunning) b.disabled = true;
      row.appendChild(b);
      box.appendChild(row);
    });
  }

  function renderClusters(running) {
    var box = $id('netClusters');
    if (!box) return;
    box.innerHTML = '';
    var count = running ? running.length : 0;
    var cnt = $id('netClusterCount');
    if (cnt) {
      cnt.textContent = count + '/' + MAX_CLUSTERS;
      cnt.classList.toggle('max', count >= MAX_CLUSTERS);
    }
    if (!count) {
      var e = el('div', 'net-empty');
      e.textContent = 'No clusters running.';
      box.appendChild(e);
      return;
    }
    running.forEach(function (r) {
      var row = el('div', 'net-row');
      var info = el('div', 'net-row-info');
      info.textContent = (r.model || '?') + ' \u00b7 :' + r.cluster_port;
      row.appendChild(info);
      var b = el('button', 'net-btn stop');
      b.textContent = 'Stop';
      if (!nodeRunning) b.disabled = true;
      b.addEventListener('click', function () { stopCluster(r.cluster_port); });
      row.appendChild(b);
      box.appendChild(row);
    });
  }

  /* ── Start / stop clusters ── */
  function startModel(name) {
    if (!nodeRunning) return;
    startingModel = name;
    renderModels(lastStatus ? lastStatus.models : null);
    setStatus('Starting ' + name + '\u2026');
    fetch(NET_START, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_name: name })
    })
      .then(function (res) { return res.json().catch(function () { return {}; }).then(function (d) { return { ok: res.ok, status: res.status, d: d }; }); })
      .then(function (r) {
        var d = r.d;
        if (!r.ok || d.ok === false) {
          setStatus(d.error || ('Failed to start ' + name + ' (HTTP ' + r.status + ')'), true);
        } else {
          if (d.port != null) {
            runningCache = runningCache.filter(function (x) { return x.cluster_port !== d.port; });
            runningCache.push({ model: d.model || name, cluster_port: d.port });
            if (d.cluster_url) clusterUrls[d.port] = d.cluster_url;
          }
          setStatus(name + ' started on port ' + (d.port != null ? d.port : '?'));
          syncClusterOptions();
        }
        fetchStatus();
      })
      .catch(function () {
        nodeRunning = false;
        setStatus('Cannot reach node daemon (start serve with --node)', true);
        renderAll();
      })
      .finally(function () {
        startingModel = null;
        renderModels(lastStatus ? lastStatus.models : null);
      });
  }

  function stopCluster(port) {
    if (!nodeRunning) return;
    setStatus('Stopping cluster on port ' + port + '\u2026');
    fetch(NET_STOP, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cluster_port: port })
    })
      .then(function (res) {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        runningCache = runningCache.filter(function (x) { return x.cluster_port !== port; });
        delete clusterUrls[port];
        syncClusterOptions();
        if (apiBaseMatchesPort(port)) revertToLocal();
        setStatus('Cluster on port ' + port + ' stopped');
        fetchStatus();
      })
      .catch(function () {
        setStatus('Failed to stop cluster on port ' + port, true);
        fetchStatus();
      });
  }

  function apiBaseMatchesPort(port) {
    if (!apiBase) return false;
    var local = 'http://127.0.0.1:' + port;
    var viaUrl = (clusterUrls[port] || '').replace(/\/v1\/?$/, '').replace(/\/+$/, '');
    return apiBase === local || (viaUrl && apiBase === viaUrl);
  }

  function revertToLocal() {
    apiBase = '';
    if (window.TerllamaFetchModels) window.TerllamaFetchModels();
  }

  /* ── Model dropdown integration ── */
  function syncClusterOptions() {
    var sel = document.getElementById('modelSelect');
    if (!sel) return;
    var prev = sel.value;
    Array.prototype.forEach.call(sel.querySelectorAll('option[value^="cluster:"]'), function (o) { o.remove(); });
    runningCache.forEach(function (r) {
      if (r.cluster_port == null) return;
      var opt = document.createElement('option');
      opt.value = 'cluster:' + r.cluster_port;
      opt.textContent = (r.model || 'model') + ' (cluster)';
      sel.appendChild(opt);
    });
    var stillThere = Array.prototype.some.call(sel.options, function (o) { return o.value === prev; });
    if (prev && stillThere) sel.value = prev;
  }
  window.TerllamaNetSync = syncClusterOptions;

  window.TerllamaClusterUrl = function (port) {
    var u = clusterUrls[port];
    if (!u) return '';
    return u.replace(/\/v1\/?$/, '').replace(/\/+$/, '');
  };

  /* ── anime.js background: network of computers ── */
  function seedNodes() {
    if (!canvas) return;
    var w = canvas.width, h = canvas.height;
    if (!w || !h) return;
    nodes.length = 0;
    nodes.push({ hx: 16, hy: h - 16, ph: 0, dr: 0, s: 9, me: true }); // this node, bottom-left
    for (var i = 1; i < 10; i++) {
      nodes.push({
        hx: 24 + Math.random() * Math.max(20, w - 48),
        hy: 28 + Math.random() * Math.max(24, h - 52),
        ph: Math.random() * Math.PI * 2,
        dr: 6 + Math.random() * 10,
        s: 4 + Math.random() * 3,
        me: false
      });
    }
  }
  function resizeCanvas() {
    if (!canvas) return;
    var w = panel.clientWidth, h = panel.clientHeight;
    if (w && h && (canvas.width !== w || canvas.height !== h)) {
      canvas.width = w;
      canvas.height = h;
      seedNodes();
    }
  }
  function draw() {
    if (!ctx || !canvas || !canvas.width || !canvas.height) return;
    var w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    var t = (prog.p || 0) * Math.PI * 2;
    var accent = cssVar('--accent') || '#3b82f6';
    var dim = cssVar('--text-secondary') || '#9aa0a6';
    var pts = nodes.map(function (n) {
      return {
        x: n.hx + Math.sin(t * 0.7 + n.ph * 2.4) * n.dr,
        y: n.hy + Math.cos(t * 0.55 + n.ph * 3.1) * n.dr,
        n: n
      };
    });
    ctx.lineWidth = 1;
    for (var i = 0; i < pts.length; i++) {
      for (var j = i + 1; j < pts.length; j++) {
        var dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y;
        var d = Math.sqrt(dx * dx + dy * dy);
        if (d < 90) {
          ctx.globalAlpha = 0.22 * (1 - d / 90);
          ctx.strokeStyle = accent;
          ctx.beginPath();
          ctx.moveTo(pts[i].x, pts[i].y);
          ctx.lineTo(pts[j].x, pts[j].y);
          ctx.stroke();
        }
      }
    }
    pts.forEach(function (p) {
      var n = p.n;
      var pulse = n.me ? 0.95 : 0.45 + 0.35 * (0.5 + 0.5 * Math.sin(t * 1.1 + n.ph * 5));
      ctx.globalAlpha = pulse;
      ctx.fillStyle = n.me ? accent : dim;
      var r = n.s / 2;
      if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(p.x - r, p.y - r, n.s, n.s, 2); ctx.fill(); }
      else ctx.fillRect(p.x - r, p.y - r, n.s, n.s);
    });
    ctx.globalAlpha = 1;
  }
  function initAnimation() {
    canvas = document.createElement('canvas');
    canvas.className = 'net-canvas';
    panel.appendChild(canvas);
    ctx = canvas.getContext('2d');
    if (window.anime) {
      tl = anime.timeline({ loop: true, autoplay: false });
      tl.add({ targets: prog, p: 1, duration: 18000, easing: 'linear', update: draw });
    }
  }
  function startAnim() {
    resizeCanvas();
    draw();
    if (tl) tl.play();
  }
  function stopAnim() {
    if (tl) tl.pause();
  }
  window.addEventListener('resize', function () { if (!panel.classList.contains('hidden')) resizeCanvas(); });

  /* ── Init ── */
  injectCss();
  buildPanel();
  initAnimation();
  btn.addEventListener('click', togglePanel);
})();
