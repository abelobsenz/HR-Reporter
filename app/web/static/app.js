const form = document.getElementById('analysis-form');
const fileDrop = document.getElementById('file-drop');
const fileInput = document.getElementById('files');
const browseBtn = document.getElementById('browse-btn');
const fileList = document.getElementById('file-list');
const clearBtn = document.getElementById('clear-btn');
const submitBtn = document.getElementById('submit-btn');
const statusEl = document.getElementById('status');
const outputPanel = document.getElementById('output-panel');
const reportPreview = document.getElementById('report-preview');
const meta = document.getElementById('meta');
const copyMdBtn = document.getElementById('copy-md');
const downloadJsonBtn = document.getElementById('download-json');
const downloadDocxBtn = document.getElementById('download-docx');
const downloadPdfBtn = document.getElementById('download-pdf');
const urlsField = document.getElementById('urls');
const configBadges = document.getElementById('config-badges');
const progressWrap = document.getElementById('progress-wrap');
const progressFill = document.getElementById('progress-fill');
const progressSteps = Array.from(document.querySelectorAll('#progress-steps li'));

let selectedFiles = [];
let lastPayload = null;
let progressPollTimer = null;
let activeProgressToken = null;

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.style.color = isError ? 'var(--warn)' : '#1f4738';
}

function renderFiles() {
  fileList.innerHTML = '';
  selectedFiles.forEach((f) => {
    const pill = document.createElement('span');
    pill.className = 'file-pill';
    pill.textContent = `${f.name} (${Math.max(1, Math.round(f.size / 1024))} KB)`;
    fileList.appendChild(pill);
  });
}

function addFiles(fileListLike) {
  const incoming = Array.from(fileListLike || []);
  if (!incoming.length) return;

  const seen = new Set(selectedFiles.map((f) => `${f.name}:${f.size}:${f.lastModified}`));
  incoming.forEach((file) => {
    const key = `${file.name}:${file.size}:${file.lastModified}`;
    if (!seen.has(key)) {
      seen.add(key);
      selectedFiles.push(file);
    }
  });
  renderFiles();
}

function parseUrlsFromText(raw) {
  if (!raw) return [];
  const matches = raw.match(/https?:\/\/[^\s,]+|https?:\/[^\s,]+/g) || [];
  return matches
    .map((url) => url.replace(/^https:\/(?!\/)/i, 'https://').replace(/^http:\/(?!\/)/i, 'http://'))
    .map((url) => url.replace(/[),;]+$/, ''));
}

function appendUrls(urls) {
  if (!urls.length) return;
  const existing = parseUrlsFromText(urlsField.value);
  const deduped = Array.from(new Set([...existing, ...urls]));
  urlsField.value = deduped.join('\n');
}

function setupDropzone(zone, onDrop) {
  ['dragenter', 'dragover'].forEach((evt) => {
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      zone.classList.add('drag');
    });
  });
  ['dragleave', 'drop'].forEach((evt) => {
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      zone.classList.remove('drag');
    });
  });
  zone.addEventListener('drop', onDrop);
}

function addBadge(label, value) {
  const badge = document.createElement('span');
  badge.className = 'badge';
  badge.textContent = `${label}: ${value}`;
  configBadges.appendChild(badge);
}

function setProgress(stepIndex, pct) {
  const boundedPct = Math.max(0, Math.min(100, Number(pct) || 0));
  progressFill.style.width = `${boundedPct}%`;
  progressSteps.forEach((step, idx) => {
    step.classList.toggle('active', idx <= stepIndex);
  });
}

function stageToStep(stage, pct) {
  const normalized = String(stage || '').toLowerCase();
  if (normalized === 'ingest' || normalized === 'chunk' || normalized === 'queued') return 0;
  if (normalized === 'retrieve' || normalized === 'snapshot') return 1;
  if (normalized === 'evaluate' || normalized === 'discovery') return 2;
  if (normalized === 'render' || normalized === 'complete') return 3;

  if (pct >= 80) return 3;
  if (pct >= 55) return 2;
  if (pct >= 25) return 1;
  return 0;
}

function stopProgressPolling() {
  if (progressPollTimer) {
    window.clearInterval(progressPollTimer);
    progressPollTimer = null;
  }
  activeProgressToken = null;
}

function startProgressTracking(progressToken) {
  stopProgressPolling();
  activeProgressToken = progressToken;
  progressWrap.hidden = false;
  setProgress(0, 2);

  const poll = async () => {
    if (!activeProgressToken) return;
    try {
      const resp = await fetch(`/api/progress/${encodeURIComponent(activeProgressToken)}`);
      if (!resp.ok) return;
      const progress = await resp.json();
      const pct = Number(progress.percent ?? 0);
      const step = stageToStep(progress.stage, pct);
      setProgress(step, pct);

      if (progress.message) {
        setStatus(progress.message, progress.state === 'failed');
      }

      if (progress.state === 'completed') {
        stopProgressPolling();
      } else if (progress.state === 'failed') {
        stopProgressPolling();
      }
    } catch {
      // Non-blocking; keep polling while the analysis request is in-flight.
    }
  };

  poll();
  progressPollTimer = window.setInterval(poll, 900);
}

function stopProgressTracking(success = true) {
  stopProgressPolling();
  setProgress(3, success ? 100 : 0);
  window.setTimeout(() => {
    progressWrap.hidden = !success;
    if (!success) setProgress(0, 0);
  }, 500);
}

function escapeHtml(text) {
  return String(text || '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function formatInline(text) {
  let out = escapeHtml(text);
  out = out.replace(/`([^`]+)`/g, '<code>$1</code>');
  out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  out = out.replace(
    /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>',
  );
  return out;
}

function isTableSeparator(line) {
  return /^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$/.test(line || '');
}

function parseTableCells(line) {
  return String(line || '')
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map((cell) => formatInline(cell.trim()));
}

function renderMarkdown(markdown) {
  const lines = String(markdown || '').replace(/\r\n/g, '\n').split('\n');
  const html = [];
  let paragraph = [];
  let inCode = false;
  let codeLang = '';
  let codeLines = [];
  let inUl = false;
  let inOl = false;

  const closeLists = () => {
    if (inUl) {
      html.push('</ul>');
      inUl = false;
    }
    if (inOl) {
      html.push('</ol>');
      inOl = false;
    }
  };

  const flushParagraph = () => {
    if (!paragraph.length) return;
    html.push(`<p>${formatInline(paragraph.join(' '))}</p>`);
    paragraph = [];
  };

  for (let idx = 0; idx < lines.length; idx += 1) {
    const line = lines[idx];
    const trimmed = line.trim();

    if (inCode) {
      if (trimmed.startsWith('```')) {
        const langClass = codeLang ? ` class="language-${escapeHtml(codeLang)}"` : '';
        html.push(`<pre><code${langClass}>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
        inCode = false;
        codeLang = '';
        codeLines = [];
      } else {
        codeLines.push(line);
      }
      continue;
    }

    if (trimmed.startsWith('```')) {
      flushParagraph();
      closeLists();
      inCode = true;
      codeLang = trimmed.slice(3).trim();
      codeLines = [];
      continue;
    }

    if (!trimmed) {
      flushParagraph();
      closeLists();
      continue;
    }

    const headingMatch = trimmed.match(/^(#{1,6})\s+(.+)$/);
    if (headingMatch) {
      flushParagraph();
      closeLists();
      const level = headingMatch[1].length;
      html.push(`<h${level}>${formatInline(headingMatch[2])}</h${level}>`);
      continue;
    }

    const nextLine = idx + 1 < lines.length ? lines[idx + 1].trim() : '';
    if (trimmed.includes('|') && isTableSeparator(nextLine)) {
      flushParagraph();
      closeLists();

      const headerCells = parseTableCells(trimmed);
      idx += 1; // Skip separator line.
      const bodyRows = [];
      while (idx + 1 < lines.length) {
        const rowLine = lines[idx + 1].trim();
        if (!rowLine || !rowLine.includes('|')) break;
        bodyRows.push(parseTableCells(rowLine));
        idx += 1;
      }

      const headHtml = `<thead><tr>${headerCells.map((cell) => `<th>${cell}</th>`).join('')}</tr></thead>`;
      let bodyHtml = '';
      if (bodyRows.length) {
        bodyHtml = `<tbody>${bodyRows
          .map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join('')}</tr>`)
          .join('')}</tbody>`;
      }
      html.push(`<table>${headHtml}${bodyHtml}</table>`);
      continue;
    }

    const ulMatch = trimmed.match(/^[-*]\s+(.+)$/);
    if (ulMatch) {
      flushParagraph();
      if (inOl) {
        html.push('</ol>');
        inOl = false;
      }
      if (!inUl) {
        html.push('<ul>');
        inUl = true;
      }
      html.push(`<li>${formatInline(ulMatch[1])}</li>`);
      continue;
    }

    const olMatch = trimmed.match(/^\d+\.\s+(.+)$/);
    if (olMatch) {
      flushParagraph();
      if (inUl) {
        html.push('</ul>');
        inUl = false;
      }
      if (!inOl) {
        html.push('<ol>');
        inOl = true;
      }
      html.push(`<li>${formatInline(olMatch[1])}</li>`);
      continue;
    }

    const quoteMatch = trimmed.match(/^>\s?(.+)$/);
    if (quoteMatch) {
      flushParagraph();
      closeLists();
      html.push(`<blockquote>${formatInline(quoteMatch[1])}</blockquote>`);
      continue;
    }

    closeLists();
    paragraph.push(trimmed);
  }

  if (inCode) {
    const langClass = codeLang ? ` class="language-${escapeHtml(codeLang)}"` : '';
    html.push(`<pre><code${langClass}>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
  }
  flushParagraph();
  closeLists();

  return html.join('\n');
}

async function loadConfig() {
  try {
    const resp = await fetch('/api/config');
    if (!resp.ok) return;
    const cfg = await resp.json();
    addBadge('Profile', cfg.profile || 'default');
    addBadge('Model', cfg.model || 'n/a');
    const timeoutBadge = cfg.timeout_seconds == null ? 'none' : `${cfg.timeout_seconds}s`;
    addBadge('Timeout', timeoutBadge);
  } catch {
    // Non-blocking
  }
}

setupDropzone(fileDrop, (e) => {
  addFiles(e.dataTransfer.files);
});

setupDropzone(urlsField, (e) => {
  const text = e.dataTransfer.getData('text/plain') || '';
  appendUrls(parseUrlsFromText(text));
});

browseBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => addFiles(fileInput.files));
fileDrop.addEventListener('click', () => fileInput.click());

window.addEventListener('paste', (e) => {
  const items = e.clipboardData?.items || [];
  const files = [];
  for (const item of items) {
    if (item.kind === 'file') {
      const file = item.getAsFile();
      if (file) files.push(file);
    }
  }

  if (files.length) {
    addFiles(files);
    setStatus('Added file(s) from clipboard.');
    return;
  }

  const text = e.clipboardData?.getData('text/plain') || '';
  const urls = parseUrlsFromText(text);
  if (urls.length) {
    appendUrls(urls);
    setStatus('Detected URL(s) from clipboard text.');
  }
});

clearBtn.addEventListener('click', () => {
  form.reset();
  selectedFiles = [];
  renderFiles();
  outputPanel.hidden = true;
  stopProgressPolling();
  progressWrap.hidden = true;
  setProgress(0, 0);
  setStatus('');
});

copyMdBtn.addEventListener('click', async () => {
  if (!lastPayload) return;
  await navigator.clipboard.writeText(lastPayload.result.report_markdown || '');
  setStatus('Markdown copied.');
});

downloadJsonBtn.addEventListener('click', () => {
  if (!lastPayload) return;
  const blob = new Blob([JSON.stringify(lastPayload.result.report, null, 2)], {
    type: 'application/json',
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'report.json';
  a.click();
  URL.revokeObjectURL(url);
});

async function downloadExport(format) {
  if (!lastPayload) return;
  const markdown = (lastPayload.result.report_markdown || '').trim();
  if (!markdown) {
    setStatus('Nothing to export yet.', true);
    return;
  }

  const endpoint = format === 'pdf' ? '/api/export/pdf' : '/api/export/docx';
  const extension = format === 'pdf' ? 'pdf' : 'docx';
  const mime =
    format === 'pdf'
      ? 'application/pdf'
      : 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';

  try {
    setStatus(`Preparing ${extension.toUpperCase()} export...`);
    const resp = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        markdown,
        filename: 'hr_assessment_report',
      }),
    });
    if (!resp.ok) {
      const payload = await resp.json().catch(() => ({}));
      throw new Error(payload.detail || 'Export failed');
    }

    const blob = await resp.blob();
    const typedBlob = new Blob([blob], { type: mime });
    const url = URL.createObjectURL(typedBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `hr_assessment_report.${extension}`;
    a.click();
    URL.revokeObjectURL(url);
    setStatus(`${extension.toUpperCase()} downloaded.`);
  } catch (err) {
    setStatus(`Export error: ${err.message}`, true);
  }
}

downloadDocxBtn.addEventListener('click', async () => {
  await downloadExport('docx');
});

downloadPdfBtn.addEventListener('click', async () => {
  await downloadExport('pdf');
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  submitBtn.disabled = true;
  setStatus('Submitting request...');

  const formData = new FormData();
  formData.append('text', document.getElementById('text').value || '');
  formData.append('urls', urlsField.value || '');
  selectedFiles.forEach((f) => formData.append('files', f));
  const progressToken =
    (typeof crypto !== 'undefined' && crypto.randomUUID)
      ? crypto.randomUUID()
      : `p_${Date.now()}_${Math.random().toString(16).slice(2)}`;
  formData.append('progress_token', progressToken);
  startProgressTracking(progressToken);

  try {
    const resp = await fetch('/api/analyze', {
      method: 'POST',
      body: formData,
    });
    const payload = await resp.json();
    if (!resp.ok) {
      throw new Error(payload.detail || 'Request failed');
    }

    lastPayload = payload;
    stopProgressTracking(true);
    outputPanel.hidden = false;

    const report = payload.result.report || {};
    const outputDir = payload.result.output_dir || payload.result.output || '';
    reportPreview.innerHTML = renderMarkdown(payload.result.report_markdown || '');

    meta.textContent = `Stage: ${report.stage || 'n/a'} | Output: ${outputDir} | Inputs: ${payload.request_dir}`;
    setStatus('Assessment complete.');
  } catch (err) {
    stopProgressTracking(false);
    setStatus(`Error: ${err.message}`, true);
  } finally {
    submitBtn.disabled = false;
  }
});

loadConfig();
