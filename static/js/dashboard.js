/**
 * dashboard.js — Research Agent Dashboard
 * ────────────────────────────────────────
 * Handles all panel navigation, API calls, result rendering,
 * Plotly chart drawing, file upload, chat, and report download.
 *
 * No external dependencies beyond Bootstrap 5 and Plotly (loaded in template).
 */

/* ═══════════════════════════════════════════════════════════════════
   Constants
═══════════════════════════════════════════════════════════════════ */
const API = {
  upload:      '/api/upload',
  chat:        '/api/chat',
  summary:     '/api/summary',
  citation:    '/api/citation',
  trend:       '/api/trend',
  knowledge:   '/api/knowledge',
  insight:     '/api/insight',
  orchestrate: '/api/orchestrate',
  dashboard:   '/api/dashboard',
};

const CHART_LAYOUT = {
  margin:  { t: 20, r: 10, b: 40, l: 40 },
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor:  'rgba(0,0,0,0)',
  font:    { family: '-apple-system,"Segoe UI",system-ui,sans-serif', size: 12 },
};

/* ═══════════════════════════════════════════════════════════════════
   Sidebar navigation
═══════════════════════════════════════════════════════════════════ */
document.querySelectorAll('.sidebar-link').forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    const targetId = link.dataset.panel;
    if (!targetId) return;

    // Update active state
    document.querySelectorAll('.sidebar-link').forEach(l => l.classList.remove('active'));
    link.classList.add('active');

    // Show / hide panels
    document.querySelectorAll('.panel').forEach(p => p.classList.add('d-none'));
    document.getElementById(targetId)?.classList.remove('d-none');

    // Update topbar title
    const icon  = link.querySelector('i')?.className.replace('me-2', '').trim() || '';
    const label = link.textContent.trim();
    document.getElementById('panel-title').textContent = label;

    // Close sidebar on mobile
    document.getElementById('sidebar')?.classList.remove('open');
  });
});

// Mobile toggle
document.getElementById('sidebarToggle')?.addEventListener('click', () => {
  document.getElementById('sidebar')?.classList.toggle('open');
});

/* ═══════════════════════════════════════════════════════════════════
   Utility helpers
═══════════════════════════════════════════════════════════════════ */

/** POST JSON to an API endpoint, return parsed response. */
async function apiPost(url, body) {
  const res = await fetch(url, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
  });
  return res.json();
}

/** Show a spinner inside *container*. */
function showSpinner(container, message = 'Processing…') {
  container.innerHTML = `
    <div class="loading-overlay">
      <div class="spinner-border spinner-border-sm text-primary"></div>
      <span>${message}</span>
    </div>`;
}

/** Render section-based agent output (literature, citation, trend, insight). */
function renderSections(container, data) {
  if (!data || data.error) {
    container.innerHTML = `<div class="alert alert-danger small">${data?.error || 'Unknown error'}</div>`;
    return;
  }
  const sections = data.sections || {};
  if (!Object.keys(sections).length) {
    container.innerHTML = '<p class="text-muted small">No output generated.</p>';
    return;
  }
  container.innerHTML = Object.entries(sections).map(([label, content]) => `
    <div class="result-section">
      <h6>${label}</h6>
      ${escapeHtml(content)}
    </div>`).join('');
}

/** Escape HTML entities to prevent XSS from model output. */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n/g, '<br>');
}

/** Alert helper (success / danger). */
function alertBox(message, type = 'success') {
  return `<div class="alert alert-${type} py-2 small mt-2">${message}</div>`;
}

/* ═══════════════════════════════════════════════════════════════════
   Dashboard stats (load on page init)
═══════════════════════════════════════════════════════════════════ */
async function loadDashboardStats() {
  try {
    const res = await fetch(API.dashboard);
    const json = await res.json();
    if (!json.ok) return;

    const d = json.data;
    document.getElementById('stat-vectors').textContent  = d.vector_store.total_vectors;
    document.getElementById('stat-pdfs').textContent     = d.uploaded_pdfs;
    document.getElementById('stat-model').textContent    = d.model_id || 'IBM Granite';
    document.getElementById('sidebar-model-id').textContent = d.model_id || 'IBM Granite';
    document.getElementById('sidebar-vectors').textContent  = d.vector_store.total_vectors;

    // Sources list
    const sourcesList = document.getElementById('sources-list');
    if (d.vector_store.sources?.length) {
      sourcesList.innerHTML = d.vector_store.sources.map(s =>
        `<li class="list-group-item small py-1">
           <i class="bi bi-file-earmark-pdf text-danger me-2"></i>${s}
         </li>`
      ).join('');
    }

    drawPipelineChart();
    drawEntityPlaceholderChart();

  } catch (err) {
    console.warn('[Dashboard] Could not load stats:', err);
  }
}

/* ═══════════════════════════════════════════════════════════════════
   Plotly charts
═══════════════════════════════════════════════════════════════════ */
function drawPipelineChart() {
  const stages = ['RAG Retrieval', 'Literature Review', 'Citation Analysis',
                  'Trend Prediction', 'Knowledge Graph', 'Insight Generation'];
  const colors = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6','#06b6d4'];

  Plotly.newPlot('chart-pipeline', [{
    type:        'bar',
    x:           stages,
    y:           [1, 1, 1, 1, 1, 1],
    marker:      { color: colors },
    hoverinfo:   'x',
    showlegend:  false,
  }], {
    ...CHART_LAYOUT,
    yaxis: { visible: false },
    xaxis: { tickfont: { size: 10 } },
  }, { responsive: true, displayModeBar: false });
}

function drawEntityPlaceholderChart() {
  Plotly.newPlot('chart-entities', [{
    type:   'pie',
    labels: ['Authors', 'Institutions', 'Concepts', 'Methods', 'Keywords', 'Datasets'],
    values: [3, 2, 8, 5, 10, 2],
    hole:   0.45,
    marker: { colors: ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6','#06b6d4'] },
    textinfo: 'label+percent',
    textfont: { size: 11 },
  }], {
    ...CHART_LAYOUT,
    showlegend: false,
  }, { responsive: true, displayModeBar: false });
}

function drawEntityChart(entityCounts) {
  const labels = Object.keys(entityCounts);
  const values = Object.values(entityCounts);
  Plotly.newPlot('chart-kg', [{
    type:   'bar',
    x:      labels,
    y:      values,
    marker: { color: '#3b82f6' },
  }], {
    ...CHART_LAYOUT,
    yaxis: { title: 'Count' },
  }, { responsive: true, displayModeBar: false });
}

function drawTrendChart(sections) {
  const labels  = Object.keys(sections);
  const lengths = labels.map(l => sections[l].length);
  Plotly.newPlot('chart-trend', [{
    type:        'bar',
    orientation: 'h',
    y:           labels,
    x:           lengths,
    marker:      { color: '#3b82f6' },
    hovertemplate: '%{y}: %{x} chars<extra></extra>',
  }], {
    ...CHART_LAYOUT,
    xaxis: { title: 'Content length (chars)' },
  }, { responsive: true, displayModeBar: false });
}

/* ═══════════════════════════════════════════════════════════════════
   PDF Upload
═══════════════════════════════════════════════════════════════════ */
const pdfInput  = document.getElementById('pdfFileInput');
const uploadBtn = document.getElementById('uploadBtn');
const fileInfo  = document.getElementById('fileInfo');
const dropZone  = document.getElementById('dropZone');

pdfInput?.addEventListener('change', () => {
  const file = pdfInput.files[0];
  if (file) {
    fileInfo.textContent = `Selected: ${file.name}  (${(file.size/1024).toFixed(1)} KB)`;
    fileInfo.classList.remove('d-none');
    uploadBtn.disabled = false;
  }
});

// Drag-and-drop
dropZone?.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone?.addEventListener('dragleave', ()  => dropZone.classList.remove('drag-over'));
dropZone?.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer?.files[0];
  if (file && file.name.toLowerCase().endsWith('.pdf')) {
    pdfInput.files = e.dataTransfer.files;
    fileInfo.textContent = `Selected: ${file.name}  (${(file.size/1024).toFixed(1)} KB)`;
    fileInfo.classList.remove('d-none');
    uploadBtn.disabled = false;
  }
});

uploadBtn?.addEventListener('click', async () => {
  const file = pdfInput.files[0];
  if (!file) return;

  const result = document.getElementById('uploadResult');
  uploadBtn.disabled = true;
  uploadBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Uploading…';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res  = await fetch(API.upload, { method: 'POST', body: formData });
    const json = await res.json();

    if (json.ok) {
      const d = json.data;
      result.innerHTML = alertBox(
        `✓ <strong>${d.filename}</strong> uploaded — ` +
        `${d.pages_loaded} pages, ${d.chunks_added} chunks, ` +
        `${d.total_vectors} total vectors indexed.`
      );
      loadDashboardStats(); // refresh counts
    } else {
      result.innerHTML = alertBox(`Upload failed: ${json.error}`, 'danger');
    }
  } catch (err) {
    result.innerHTML = alertBox(`Network error: ${err.message}`, 'danger');
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.innerHTML = '<i class="bi bi-upload me-2"></i>Upload &amp; Index';
  }
});

/* ═══════════════════════════════════════════════════════════════════
   AI Chat
═══════════════════════════════════════════════════════════════════ */
function appendChatBubble(role, html) {
  const wrap = document.getElementById('chatMessages');
  const div  = document.createElement('div');
  div.className = `chat-bubble ${role}`;
  div.innerHTML = `<strong>${role === 'user' ? 'You' : 'ResearchAgent'}</strong>
                   <p class="mb-0 mt-1">${html}</p>`;
  wrap.appendChild(div);
  wrap.scrollTop = wrap.scrollHeight;
}

async function sendChat() {
  const input = document.getElementById('chatInput');
  const query = input.value.trim();
  if (!query) return;

  appendChatBubble('user', escapeHtml(query));
  input.value = '';

  // Thinking bubble
  const thinkId = 'think-' + Date.now();
  const wrap = document.getElementById('chatMessages');
  wrap.insertAdjacentHTML('beforeend', `
    <div id="${thinkId}" class="chat-bubble assistant">
      <strong>ResearchAgent</strong>
      <p class="mb-0 mt-1 text-muted">
        <span class="spinner-border spinner-border-sm me-2"></span>Thinking…
      </p>
    </div>`);
  wrap.scrollTop = wrap.scrollHeight;

  try {
    const json = await apiPost(API.chat, { query });
    document.getElementById(thinkId)?.remove();

    if (json.ok) {
      appendChatBubble('assistant', escapeHtml(json.data.answer));
    } else {
      appendChatBubble('assistant', `<span class="text-danger">Error: ${json.error}</span>`);
    }
  } catch (err) {
    document.getElementById(thinkId)?.remove();
    appendChatBubble('assistant', `<span class="text-danger">Network error: ${err.message}</span>`);
  }
}

document.getElementById('chatSendBtn')?.addEventListener('click', sendChat);
document.getElementById('chatInput')?.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
});

/* ═══════════════════════════════════════════════════════════════════
   Literature Review
═══════════════════════════════════════════════════════════════════ */
document.getElementById('literatureBtn')?.addEventListener('click', async () => {
  const query  = document.getElementById('literatureQuery').value.trim();
  const result = document.getElementById('literatureResult');
  if (!query) return;

  showSpinner(result, 'Generating literature review…');
  const json = await apiPost(API.summary, { query });
  renderSections(result, json.ok ? json.data : json);
});

/* ═══════════════════════════════════════════════════════════════════
   Citation Analysis
═══════════════════════════════════════════════════════════════════ */
document.getElementById('citationBtn')?.addEventListener('click', async () => {
  const query  = document.getElementById('citationQuery').value.trim();
  const result = document.getElementById('citationResult');
  if (!query) return;

  showSpinner(result, 'Analysing citations…');
  const json = await apiPost(API.citation, { query });
  renderSections(result, json.ok ? json.data : json);
});

/* ═══════════════════════════════════════════════════════════════════
   Trend Analysis
═══════════════════════════════════════════════════════════════════ */
document.getElementById('trendBtn')?.addEventListener('click', async () => {
  const query    = document.getElementById('trendQuery').value.trim();
  const result   = document.getElementById('trendResult');
  const chartWrap = document.getElementById('trendChartWrap');
  if (!query) return;

  showSpinner(result, 'Predicting trends…');
  const json = await apiPost(API.trend, { query });

  if (json.ok) {
    chartWrap.classList.remove('d-none');
    drawTrendChart(json.data.sections || {});
    renderSections(result, json.data);
  } else {
    result.innerHTML = alertBox(json.error || 'Trend prediction failed.', 'danger');
  }
});

/* ═══════════════════════════════════════════════════════════════════
   Knowledge Graph
═══════════════════════════════════════════════════════════════════ */
document.getElementById('knowledgeBtn')?.addEventListener('click', async () => {
  const query    = document.getElementById('knowledgeQuery').value.trim();
  const result   = document.getElementById('knowledgeResult');
  const chartWrap = document.getElementById('kgChartWrap');
  if (!query) return;

  showSpinner(result, 'Extracting knowledge graph…');
  const json = await apiPost(API.knowledge, { query });

  if (!json.ok) {
    result.innerHTML = alertBox(json.error || 'Extraction failed.', 'danger');
    return;
  }

  const { graph, entity_counts } = json.data;

  // Draw entity count bar chart
  if (entity_counts && Object.keys(entity_counts).length) {
    chartWrap.classList.remove('d-none');
    drawEntityChart(entity_counts);
  }

  // Render entity tables
  const entityTypes = ['authors','institutions','concepts','methods','keywords','datasets'];
  const icons = { authors:'person', institutions:'building', concepts:'lightbulb',
                  methods:'gear', keywords:'tags', datasets:'database' };

  result.innerHTML = entityTypes.map(type => {
    const items = graph?.[type] || [];
    if (!items.length) return '';
    return `
      <div class="result-section">
        <h6><i class="bi bi-${icons[type] || 'list'} me-1"></i>${type.charAt(0).toUpperCase()+type.slice(1)}</h6>
        ${items.map(item => `<span class="entity-badge">${escapeHtml(item)}</span>`).join('')}
      </div>`;
  }).join('');

  // Relationships
  const rels = graph?.relationships || [];
  if (rels.length) {
    result.innerHTML += `
      <div class="result-section">
        <h6><i class="bi bi-diagram-3 me-1"></i>Relationships</h6>
        <table class="table table-sm table-borderless small mb-0">
          <thead><tr><th>Source</th><th>Relation</th><th>Target</th></tr></thead>
          <tbody>
            ${rels.map(r => `<tr>
              <td>${escapeHtml(r.source||'')}</td>
              <td class="text-muted">${escapeHtml(r.relation||'')}</td>
              <td>${escapeHtml(r.target||'')}</td>
            </tr>`).join('')}
          </tbody>
        </table>
      </div>`;
  }
});

/* ═══════════════════════════════════════════════════════════════════
   Insight Generation
═══════════════════════════════════════════════════════════════════ */
document.getElementById('insightBtn')?.addEventListener('click', async () => {
  const query  = document.getElementById('insightQuery').value.trim();
  const result = document.getElementById('insightResult');
  if (!query) return;

  showSpinner(result, 'Generating research insights…');
  const json = await apiPost(API.insight, { query });
  renderSections(result, json.ok ? json.data : json);
});

/* ═══════════════════════════════════════════════════════════════════
   Download Report (full pipeline)
═══════════════════════════════════════════════════════════════════ */
let _reportData = null;

document.getElementById('reportBtn')?.addEventListener('click', async () => {
  const query    = document.getElementById('reportQuery').value.trim();
  const progress = document.getElementById('reportProgress');
  const bar      = document.getElementById('reportProgressBar');
  const text     = document.getElementById('reportProgressText');
  const dlWrap   = document.getElementById('reportDownloadWrap');
  if (!query) return;

  progress.classList.remove('d-none');
  dlWrap.classList.add('d-none');
  document.getElementById('reportBtn').disabled = true;
  _reportData = null;

  // Fake progress animation while waiting for pipeline
  let pct = 0;
  const ticker = setInterval(() => {
    pct = Math.min(pct + 2, 90);
    bar.style.width = pct + '%';
  }, 400);

  const stages = [
    'Retrieving context…',
    'Running Literature Review…',
    'Running Citation Analysis…',
    'Running Trend Prediction…',
    'Running Knowledge Graph…',
    'Running Insight Generation…',
  ];
  let stageIdx = 0;
  const stageTicker = setInterval(() => {
    if (stageIdx < stages.length) {
      text.textContent = stages[stageIdx++];
    }
  }, 5000);

  try {
    const json = await apiPost(API.orchestrate, { query });
    clearInterval(ticker);
    clearInterval(stageTicker);

    if (json.ok) {
      _reportData = json.data;
      bar.style.width = '100%';
      bar.classList.remove('progress-bar-animated');
      text.textContent = 'Complete!';
      dlWrap.classList.remove('d-none');
    } else {
      text.textContent = 'Pipeline failed: ' + json.error;
      bar.classList.add('bg-danger');
    }
  } catch (err) {
    clearInterval(ticker);
    clearInterval(stageTicker);
    text.textContent = 'Network error: ' + err.message;
    bar.classList.add('bg-danger');
  } finally {
    document.getElementById('reportBtn').disabled = false;
  }
});

document.getElementById('reportDownloadBtn')?.addEventListener('click', () => {
  if (!_reportData) return;

  // Build a plain-text report from all sections
  const lines = [
    `RESEARCH AGENT — FULL PIPELINE REPORT`,
    `Query: ${_reportData.query}`,
    `${'='.repeat(70)}`,
    '',
  ];

  const sectionGroups = [
    ['LITERATURE REVIEW',   _reportData.literature],
    ['CITATION ANALYSIS',   _reportData.citation],
    ['TREND PREDICTION',    _reportData.trend],
    ['INSIGHT GENERATION',  _reportData.insight],
  ];

  sectionGroups.forEach(([title, data]) => {
    lines.push(`\n${'='.repeat(70)}\n${title}\n${'='.repeat(70)}\n`);
    if (data?.sections) {
      Object.entries(data.sections).forEach(([label, content]) => {
        lines.push(`\n${label}\n${'-'.repeat(label.length)}\n${content}`);
      });
    } else if (data?.error) {
      lines.push(`[Error: ${data.error}]`);
    }
  });

  // Knowledge graph
  const kg = _reportData.knowledge_graph;
  if (kg?.graph) {
    lines.push(`\n${'='.repeat(70)}\nKNOWLEDGE GRAPH ENTITIES\n${'='.repeat(70)}\n`);
    Object.entries(kg.graph).forEach(([type, items]) => {
      if (Array.isArray(items) && items.length) {
        lines.push(`${type.toUpperCase()}: ${items.join(', ')}`);
      }
    });
  }

  if (_reportData.timing) {
    lines.push(`\n--- Timing ---`);
    Object.entries(_reportData.timing).forEach(([k,v]) => lines.push(`${k}: ${v}s`));
  }

  const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `research_report_${Date.now()}.txt`;
  a.click();
  URL.revokeObjectURL(url);
});

/* ═══════════════════════════════════════════════════════════════════
   Init
═══════════════════════════════════════════════════════════════════ */
loadDashboardStats();
