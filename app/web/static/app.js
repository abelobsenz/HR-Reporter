const form = document.getElementById('analysis-form');
const fileDrop = document.getElementById('file-drop');
const urlDrop = document.getElementById('url-drop');
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
const urlsField = document.getElementById('urls');
const configBadges = document.getElementById('config-badges');

let selectedFiles = [];
let lastPayload = null;

function setStatus(text) {
  statusEl.textContent = text;
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

async function loadConfig() {
  try {
    const resp = await fetch('/api/config');
    if (!resp.ok) return;
    const cfg = await resp.json();
    addBadge('Pack', cfg.pack);
    addBadge('Model', cfg.model);
    addBadge('Timeout', `${cfg.timeout_seconds}s`);
    addBadge('Mode', 'Single Production Flow');
  } catch {
    // Non-blocking UI enhancement.
  }
}

setupDropzone(fileDrop, (e) => {
  addFiles(e.dataTransfer.files);
});

setupDropzone(urlDrop, (e) => {
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

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  submitBtn.disabled = true;
  setStatus('Running assessment...');

  const formData = new FormData();
  formData.append('text', document.getElementById('text').value || '');
  formData.append('urls', urlsField.value || '');
  selectedFiles.forEach((f) => formData.append('files', f));

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
    outputPanel.hidden = false;
    meta.textContent = `Output: ${payload.result.output_dir} | Inputs: ${payload.request_dir}`;
    reportPreview.textContent = payload.result.report_markdown || '';
    setStatus('Assessment complete.');
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  } finally {
    submitBtn.disabled = false;
  }
});

loadConfig();
