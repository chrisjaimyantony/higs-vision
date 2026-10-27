const DATA_URL = new URL('./data/research.json', import.meta.url);
const FEATURE_NAMES = [
  'lepton_pT', 'lepton_eta', 'lepton_phi', 'missing_energy_mag', 'missing_energy_phi',
  'jet1_pT', 'jet1_eta', 'jet1_phi', 'jet1_b-tag', 'jet2_pT', 'jet2_eta', 'jet2_phi',
  'jet2_b-tag', 'jet3_pT', 'jet3_eta', 'jet3_phi', 'jet3_b-tag', 'jet4_pT', 'jet4_eta',
  'jet4_phi', 'jet4_b-tag', 'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb',
];

const state = { data: null, eventIndex: 0, threshold: 0.5, contributionMode: 'shap', sortMode: 'magnitude', storyStage: 0, storyScenario: 'signal' };
const byId = (id) => document.getElementById(id);
const storySteps = [
  {
    title: 'Two protons meet. Their inner parts collide.',
    text: 'A proton is made of quarks and gluons. At the Large Hadron Collider, one of these constituents from each proton interacts. The experiment records the resulting spray of particles—not a photograph of a Higgs.',
    kicker: 'THE SETTING · 1 OF 3',
  },
  {
    title: 'The detector sees the debris, not the Higgs itself.',
    text: 'A Higgs boson exists too briefly to reach the detector. Physicists reconstruct possible decay products from jets, leptons, and missing momentum. A neutrino is invisible; its presence is inferred from an energy imbalance.',
    kicker: 'WHAT IS MEASURED · 2 OF 3',
  },
  {
    title: 'Twenty-eight clues become a score.',
    text: 'The network looks for patterns that appeared in simulated Higgs examples. A higher score means this collision looks more signal-like to the model. SHAP can then show which clues nudged the score.',
    kicker: 'THE MODEL · 3 OF 3',
  },
];
const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (char) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[char]));
const fixed = (value, digits = 3) => Number(value).toFixed(digits);
const percent = (value) => `${(Number(value) * 100).toFixed(1)}%`;

function setStatus(message, failed = false) {
  const status = byId('data-status');
  status.textContent = message;
  status.classList.toggle('data-status-error', failed);
}

function setFailure() {
  const loader = byId('load-state');
  setStatus('Examples could not be loaded', true);
  loader.innerHTML = '<div class="small-label">EXAMPLES NOT READY</div><h2>The collision examples could not be loaded.</h2><p>Refresh the page. If the problem continues, open the dashboard from its local server.</p><button class="button" id="retry-load" type="button">Try again</button>';
  byId('retry-load').addEventListener('click', loadResearchData);
}

function validateData(data) {
  if (!data || data.schemaVersion !== 1 || !Array.isArray(data.events) || data.events.length === 0) {
    throw new Error('The research file is empty or has an unsupported schema. Regenerate it with dashboard/export_data.py.');
  }
  if (!Array.isArray(data.features) || data.features.length !== 28 ||
      data.events.some((event) => !Array.isArray(event.values) || !Array.isArray(event.shap) || event.values.length !== 28 || event.shap.length !== 28)) {
    throw new Error('The research file does not contain the expected 28-feature event explanations. Regenerate it with dashboard/export_data.py.');
  }
  return data;
}

async function loadResearchData() {
  const loader = byId('load-state');
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 12000);
  loader.hidden = false;
  loader.innerHTML = '<div class="loader"></div><h2>Getting collisions ready…</h2><p>Loading example scores and the clues behind them.</p>';
  document.querySelectorAll('.tab-panel').forEach((panel) => { panel.hidden = true; });
  setStatus('Preparing examples');
  try {
    const response = await fetch(DATA_URL, { cache: 'no-store', signal: controller.signal });
    if (!response.ok) throw new Error(`Could not load ${DATA_URL.pathname} (HTTP ${response.status}).`);
    state.data = validateData(await response.json());
    renderDashboard();
    loader.hidden = true;
    byId('panel-events').hidden = false;
    setStatus(`${state.data.events.length} collision examples ready`);
    byId('validation-auc').textContent = fixed(state.data.model.validationAuc, 4);
    byId('announcement').textContent = 'Research data loaded.';
  } catch (error) {
    setFailure(error?.name === 'AbortError'
      ? new Error(`Loading ${DATA_URL.pathname} timed out. Keep the local server running and open this dashboard through its server address.`)
      : error);
  } finally {
    window.clearTimeout(timeout);
  }
}

function sectionIntro(label, title, description, badge = '') {
  return `<div class="section-intro"><div><span class="small-label">${label}</span><h2>${title}</h2><p>${description}</p></div>${badge ? `<span class="small-label outline-label">${badge}</span>` : ''}</div>`;
}

function cardHeader(title, subtitle, number = '') {
  return `<div class="card-header"><div><h3>${title}</h3>${subtitle ? `<p>${subtitle}</p>` : ''}</div>${number ? `<span class="card-number">${number}</span>` : ''}</div>`;
}

function describeFeature(name) {
  if (name === 'lepton_pT' || /^jet\d+_pT$/.test(name)) return 'How much momentum the particle carries across the direction of the proton beams.';
  if (name === 'lepton_eta' || /^jet\d+_eta$/.test(name)) return 'Where the particle travels in relation to the proton beams.';
  if (name === 'lepton_phi' || /^jet\d+_phi$/.test(name)) return 'The particle’s direction as it circles around the proton beams.';
  if (name === 'missing_energy_mag') return 'The size of the momentum imbalance, which can hint at particles the detector cannot see.';
  if (name === 'missing_energy_phi') return 'The direction of the momentum imbalance from particles the detector cannot see.';
  if (/^jet\d+_b-tag$/.test(name)) return 'A detector clue about whether this particle spray may have come from a bottom quark.';
  if (name === 'm_jj') return 'A combined mass estimate for two particle sprays called jets.';
  if (name === 'm_jjj') return 'A combined mass estimate for three jets.';
  if (name === 'm_lv') return 'A combined mass estimate for a lepton and missing momentum.';
  if (name === 'm_jlv') return 'A combined mass estimate for a jet, a lepton, and missing momentum.';
  if (name === 'm_bb') return 'A combined mass estimate for two jets tagged as possibly containing bottom quarks.';
  if (name === 'm_wbb') return 'A combined mass estimate for a W boson candidate and two bottom-quark jets.';
  if (name === 'm_wwbb') return 'A combined mass estimate for two W boson candidates and two bottom-quark jets.';
  return 'A value measured by the detector or calculated from the particles it observes.';
}

function friendlyFeatureName(name) {
  if (name === 'lepton_pT') return 'Lepton momentum';
  if (name === 'lepton_eta') return 'Lepton along beam';
  if (name === 'lepton_phi') return 'Lepton around beam';
  if (name === 'missing_energy_mag') return 'Missing momentum · size';
  if (name === 'missing_energy_phi') return 'Missing momentum · direction';
  const jet = name.match(/^jet(\d+)_(pT|eta|phi|b-tag)$/);
  if (jet) {
    const [, number, clue] = jet;
    const labels = { pT: 'momentum', eta: 'along beam', phi: 'around beam', 'b-tag': 'bottom tag' };
    return `Jet ${number} ${labels[clue]}`;
  }
  const massNames = {
    m_jj: 'Mass · 2 jets', m_jjj: 'Mass · 3 jets', m_lv: 'Mass · lepton + unseen',
    m_jlv: 'Mass · jet + lepton + unseen', m_bb: 'Mass · 2 bottom jets',
    m_wbb: 'Mass · W + 2 bottom jets', m_wwbb: 'Mass · 2 W + 2 bottom jets',
  };
  return massNames[name] || name;
}

function contributionRows(items, max, mode) {
  return items.map((item, rowIndex) => {
    const isShap = mode === 'shap';
    const number = isShap ? item.shap : item.value;
    const shownNumber = isShap ? number * 100 : number;
    const width = Math.max(1, Math.abs(number) / max * 48);
    const left = number < 0 ? 50 - width : 50;
    const sign = isShap && shownNumber > 0 ? '+' : '';
    const valueClass = isShap && number < 0 ? 'negative' : '';
    const barClass = isShap ? valueClass : 'feature-value';
    const displayValue = isShap ? `${sign}${fixed(shownNumber, 1)} pts` : fixed(number);
    return `<button type="button" class="contribution-row" data-feature="${item.index}" style="--row-index:${rowIndex}" title="Inspect ${escapeHtml(item.name)}"><span class="feature-label">${escapeHtml(item.name)}</span><span class="bar-area"><span class="shap-bar ${barClass}" style="left:${left}%;width:${width}%;transform-origin:${number < 0 ? 'right' : 'left'}"></span></span><span class="bar-value ${valueClass}">${displayValue}</span></button>`;
  }).join('');
}

function animateScore(element) {
  if (!element) return;
  const target = Number(element.dataset.scoreTarget);
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    element.textContent = target.toFixed(1);
    return;
  }
  const start = performance.now();
  const duration = 900;
  const tick = (now) => {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - (1 - progress) ** 3;
    element.textContent = (target * eased).toFixed(1);
    if (progress < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function renderDashboard() {
  renderEvents();
  renderGlobal();
  renderTrust();
  renderScience();
}

function renderEvents() {
  const panel = byId('panel-events');
  const events = state.data.events;
  const event = events[state.eventIndex];
  const contributions = FEATURE_NAMES.map((name, index) => ({
    name, shap: event.shap[index], value: event.values[index], index,
  }));
  const mode = state.contributionMode;
  const amount = (item) => mode === 'shap' ? item.shap : item.value;
  contributions.sort((a, b) => state.sortMode === 'magnitude'
    ? Math.abs(amount(b)) - Math.abs(amount(a))
    : amount(b) - amount(a));
  const shown = contributions.slice(0, 10);
  const max = Math.max(...contributions.map((item) => Math.abs(amount(item))), 0.001);
  const predictedSignal = event.probability >= state.threshold;
  const label = event.label === 1 ? 'Signal' : 'Background';
  const base = Number(state.data.baseValue);
  const threshold = state.threshold;
  const isShap = mode === 'shap';
  const directionLeft = isShap ? '← Lowers score' : '← Below average';
  const directionRight = isShap ? 'Raises score →' : 'Above average →';
  const chartExplanation = isShap
    ? 'SHAP shares the score across the clues: green nudges it toward signal and purple nudges it toward background. Each point is one point on the 0–100 score. These are clues the network used, not proof of what caused a collision.'
    : 'These are the scaled readings the network receives. Zero is the usual average; positive is above it and negative is below. A reading alone does not show which way it moved the score.';

  panel.innerHTML = `${sectionIntro('ONE COLLISION', 'What did the model see?', 'The network turns 28 clues into one signal score. SHAP shows how much each clue moved that score up or down.', `EXAMPLE ${event.id}`)}
    <div class="event-toolbar"><div class="toolbar-group"><span class="small-label">CHOOSE A SIMULATED COLLISION</span><select id="event-select" aria-label="Select simulated collision">${events.map((item, index) => `<option value="${index}" ${index === state.eventIndex ? 'selected' : ''}>#${item.id} · ${item.label ? 'signal' : 'background'} example · ${percent(item.probability)} score</option>`).join('')}</select><span id="event-counter" class="event-counter">${state.eventIndex + 1} / ${events.length}</span></div><div class="toolbar-group"><span class="small-label">ORDER</span><div class="segmented" aria-label="Contribution order"><button type="button" data-sort="magnitude" aria-pressed="${state.sortMode === 'magnitude'}">Biggest changes</button><button type="button" data-sort="signed" aria-pressed="${state.sortMode === 'signed'}">Score raisers first</button></div></div></div>
    <div class="event-layout"><div class="card">${cardHeader('What moved the score?', 'SHAP estimates how much each clue nudged this collision’s score.', '')}
      <div class="shap-recipe"><div><span>AVERAGE MODEL SCORE</span><strong>${percent(base)}</strong></div><i>+</i><div><span>ALL THE CLUE PUSHES</span><strong>ADDED TOGETHER</strong></div><i>≈</i><div><span>THIS COLLISION’S SCORE</span><strong>${percent(event.probability)}</strong></div><p>Each clue nudges the starting score. Together, the pushes add up to the score shown here.</p></div>
      <div class="chart-topline"><div class="legend"><span><i class="${isShap ? '' : 'feature-value'}"></i>${isShap ? 'Score goes up' : 'Above average'}</span><span><i class="${isShap ? 'negative' : 'feature-value below'}"></i>${isShap ? 'Score goes down' : 'Below average'}</span></div><div class="segmented chart-toggle" role="group" aria-label="Choose score pushes or input values"><button type="button" data-value-mode="shap" aria-pressed="${isShap}">Score pushes</button><button type="button" data-value-mode="value" aria-pressed="${!isShap}">Input values</button></div></div>
      <p class="chart-explanation">${chartExplanation}</p><div class="contribution-chart">${contributionRows(shown, max, mode)}</div><div class="chart-axis"><span>${directionLeft}</span><span>${directionRight}</span></div><div class="chart-footer"><span>${isShap ? 'Each point = 1 on the 0–100 score.' : 'Input values are scaled around the training average.'}</span><button class="subtle-button" id="show-all-features" type="button">Show all 28 clues</button></div><div class="feature-inspector" id="feature-inspector">Choose a clue to see what it is and how it nudged the score.</div></div>
      <aside class="card score-card">${cardHeader('The score', 'How signal-like the clues look to the network.', 'SCORE / 02')}<div class="score-content"><div class="score-line"><div class="big-score"><span class="score-value" data-score-target="${(event.probability * 100).toFixed(1)}">0.0</span><small>/ 100</small></div><span class="prediction-pill ${predictedSignal ? '' : 'background'}">${predictedSignal ? 'SIGNAL-LIKE' : 'BACKGROUND-LIKE'}</span></div><div class="probability-track" role="meter" aria-valuenow="${(event.probability * 100).toFixed(1)}" aria-valuemin="0" aria-valuemax="100" aria-label="Model signal score out of 100"><div class="probability-fill" style="width:${percent(event.probability)}"></div><span class="threshold-marker" style="left:${percent(threshold)}"></span></div><div class="track-labels"><span>0 · background-like</span><span>100 · signal-like</span></div><p class="score-meaning">Higher means the clues look more like the network’s simulated Higgs examples. This is a model score—not the chance of finding a Higgs in real data.</p><p class="score-caption">At the ${percent(threshold)} cutoff, this example is labelled <strong>${predictedSignal ? 'signal-like' : 'background-like'}</strong>. Move the cutoff to change the label; the score itself stays the same. The simulated label is ${label.toLowerCase()}.</p><dl class="score-details"><div><dt>What the simulation says</dt><dd>${label}</dd></div><div><dt>Current cutoff</dt><dd>${percent(threshold)}</dd></div></dl><label class="threshold-control"><span class="range-label">Change the cutoff <output id="threshold-output">${percent(threshold)}</output></span><input id="threshold-range" type="range" min="0.05" max="0.95" step="0.01" value="${threshold}" aria-label="Classification cutoff"></label></div></aside></div>`;

  animateScore(panel.querySelector('[data-score-target]'));

  panel.querySelector('#event-select').addEventListener('change', (event) => {
    state.eventIndex = Number(event.target.value);
    renderEvents();
  });
  panel.querySelectorAll('[data-sort]').forEach((button) => button.addEventListener('click', () => {
    state.sortMode = button.dataset.sort;
    renderEvents();
  }));
  panel.querySelectorAll('[data-value-mode]').forEach((button) => button.addEventListener('click', () => {
    state.contributionMode = button.dataset.valueMode;
    renderEvents();
  }));
  panel.querySelectorAll('[data-feature]').forEach((button) => button.addEventListener('click', () => inspectFeature(Number(button.dataset.feature))));
  panel.querySelector('#show-all-features').addEventListener('click', () => {
    panel.querySelector('.contribution-chart').innerHTML = contributionRows(contributions, max, mode);
    panel.querySelector('#show-all-features').remove();
    panel.querySelectorAll('[data-feature]').forEach((button) => button.addEventListener('click', () => inspectFeature(Number(button.dataset.feature))));
  });
  panel.querySelector('#threshold-range').addEventListener('input', (inputEvent) => {
    state.threshold = Number(inputEvent.target.value);
    const isSignalLike = event.probability >= state.threshold;
    panel.querySelector('#threshold-output').textContent = percent(state.threshold);
    panel.querySelector('.threshold-marker').style.left = percent(state.threshold);
    const prediction = panel.querySelector('.prediction-pill');
    prediction.classList.toggle('background', !isSignalLike);
    prediction.textContent = isSignalLike ? 'SIGNAL-LIKE' : 'BACKGROUND-LIKE';
    panel.querySelector('.score-caption').innerHTML = `At the ${percent(state.threshold)} cutoff, the event is labelled <strong>${isSignalLike ? 'signal-like' : 'background-like'}</strong>. The cutoff changes this label, not the score. The simulation’s answer is ${label.toLowerCase()}.`;
    panel.querySelector('.score-details div:nth-child(2) dd').textContent = percent(state.threshold);
  });
}

function inspectFeature(index) {
  const event = state.data.events[state.eventIndex];
  const feature = state.data.features.find((item) => item.name === FEATURE_NAMES[index]);
  const inspector = byId('feature-inspector');
  if (!inspector || !feature) return;
  inspector.innerHTML = `<strong>${escapeHtml(friendlyFeatureName(feature.name))}</strong> · ${escapeHtml(describeFeature(feature.name))}<br>For this collision, this clue moved the score <code>${event.shap[index] > 0 ? '+' : ''}${fixed(event.shap[index] * 100, 1)} points</code>.`;
}

function renderGlobal() {
  const panel = byId('panel-global');
  const features = [...state.data.features].sort((a, b) => b.importance - a.importance);
  const max = Math.max(...features.map((item) => item.importance), 0.001);
  const selected = features[0];
  panel.innerHTML = `${sectionIntro('MANY COLLISIONS', 'Which clues matter most to the network?', 'SHAP measures each clue’s push across 256 collision examples. Bigger bars mean the network leans on that clue more; the direction can change from one collision to another.', '256 EXAMPLES')}
    <div class="global-layout"><div class="card">${cardHeader('The clues the network uses most', 'Longer bars mean a bigger average nudge to the 0–100 score.', '')}<div class="importance-list">${features.map((item, index) => `<button class="importance-row" type="button" style="--row-index:${index}" data-global-feature="${escapeHtml(item.name)}" aria-pressed="${item.name === selected.name}"><span class="rank-number">${String(index + 1).padStart(2, '0')}</span><span class="feature-label">${escapeHtml(friendlyFeatureName(item.name))}</span><span class="importance-bar-track"><span class="importance-bar" style="display:block;width:${Math.max(1, item.importance / max * 100)}%"></span></span><span class="bar-value">${fixed(item.importance * 100, 1)}pt</span></button>`).join('')}<p class="feature-list-note">This is the average size of each clue’s push. Its direction can change from one collision to another.</p></div></div><div class="card"><div class="card-header"><div><span class="small-label">PICK A CLUE</span><div class="feature-heading" id="global-feature-detail"></div></div></div><div class="scatter-wrap" id="global-scatter"></div><div class="plot-context" id="global-context"></div></div></div>
    <div class="insight-strip"><span>↗</span><div><h4>Read the dots</h4><p>Each dot is one simulated collision. Left to right shows whether a reading is below or above its usual value. Up or down shows how it nudged the score. Color shows the simulation’s label, not proof of a Higgs in real data.</p></div></div><div class="feature-key"><span class="small-label">WHAT THESE CLUES MEAN</span><p><strong>Momentum</strong> = particle motion; <strong>direction</strong> = where it travels; <strong>bottom-quark tag</strong> = a clue about a jet’s origin; <strong>missing momentum</strong> = an imbalance that can hint at an unseen particle; <strong>mass</strong> = an estimate made by combining particles.</p></div><div class="stats-grid"><div class="stat-card"><span class="small-label">SHARE FROM CALCULATED MASSES</span><strong>${percent(state.data.alignment.massShare)}</strong><p>How much of the network’s average clue influence comes from the seven calculated masses.</p></div><div class="stat-card"><span class="small-label">EXPECTED CLUES IN THE TOP FIVE</span><strong>${state.data.alignment.top5Overlap} / 5</strong><p>How many of the project’s expected clues appear in the network’s five strongest clues.</p></div></div>`;
  renderFeatureDetail(selected.name);
  panel.querySelectorAll('[data-global-feature]').forEach((button) => button.addEventListener('click', () => {
    panel.querySelectorAll('[data-global-feature]').forEach((row) => row.setAttribute('aria-pressed', String(row === button)));
    renderFeatureDetail(button.dataset.globalFeature);
  }));
}

function renderFeatureDetail(name) {
  const feature = state.data.features.find((item) => item.name === name);
  const index = FEATURE_NAMES.indexOf(name);
  const detail = byId('global-feature-detail');
  const plot = byId('global-scatter');
  const context = byId('global-context');
  if (!feature || !detail || !plot || !context) return;
  detail.innerHTML = `<h3>${escapeHtml(friendlyFeatureName(name))}</h3><p>Clue ${1 + [...state.data.features].sort((a, b) => b.importance - a.importance).findIndex((item) => item.name === name)} of 28</p><p class="feature-description">${escapeHtml(describeFeature(name))}</p><span class="feature-tag">average score nudge · ${fixed(feature.importance * 100, 1)} points</span>`;
  const values = state.data.events.map((event) => ({ x: event.values[index], y: event.shap[index], label: event.label }));
  const xMin = Math.min(...values.map((point) => point.x));
  const xMax = Math.max(...values.map((point) => point.x));
  const yMax = Math.max(...values.map((point) => Math.abs(point.y)), 0.001);
  const sx = (value) => 42 + ((value - xMin) / Math.max(xMax - xMin, 0.001)) * 330;
  const sy = (value) => 145 - ((value + yMax) / (2 * yMax)) * 112;
  const points = values.map((point, index) => `<circle class="scatter-point" cx="${sx(point.x).toFixed(1)}" cy="${sy(point.y).toFixed(1)}" r="2.1" fill="${point.label ? 'var(--green)' : 'var(--teal)'}" opacity=".68" style="--point-delay:${index % 32 * 16}ms"><title>Input ${fixed(point.x)}; score push ${fixed(point.y * 100, 1)} points; ${point.label ? 'signal' : 'background'} label</title></circle>`).join('');
  plot.innerHTML = `<svg class="plot" viewBox="0 0 400 190" role="img" aria-label="Input value against score push for collision examples"><line class="grid-line" x1="42" y1="89" x2="372" y2="89"/><line class="grid-line" x1="42" y1="33" x2="42" y2="145"/><line class="grid-line" x1="42" y1="145" x2="372" y2="145"/>${points}<text x="42" y="164">${fixed(xMin, 1)}</text><text x="347" y="164">${fixed(xMax, 1)}</text><text x="6" y="38">${fixed(yMax * 100, 1)}</text><text x="8" y="145">-${fixed(yMax * 100, 1)}</text><text class="axis-title" x="160" y="183">scaled input (0 = average)</text><text class="axis-title" transform="translate(12 116) rotate(-90)">score push (points)</text></svg>`;
  context.innerHTML = '<span class="legend"><span><i></i> Signal example</span><span><i class="teal"></i> Background example</span></span><br>Each dot is one collision. Zero on the horizontal axis is the usual reading; above or below zero means higher or lower. The vertical axis shows how this clue moved the score.';
}

function renderTrust() {
  const panel = byId('panel-trust');
  const rows = state.data.faithfulness;
  const maxK = Math.max(...rows.map((row) => row.k), 1);
  const points = (key) => rows.map((row) => `${50 + row.k / maxK * 440},${225 - row[key] * 190}`).join(' ');
  const shapLine = points('shap');
  const randomLine = points('random');
  panel.innerHTML = `${sectionIntro('CHECK THE CLUES', 'Does hiding SHAP’s top clues trip up the model?', 'We hide the clues SHAP says matter most, then see whether the network still sorts signal from background. The score runs from 0.5 (random guessing) to 1.0 (perfect sorting).', 'THE MODEL IN ACTION')}
    <div class="trust-layout"><div class="card">${cardHeader('What happens when we hide the clues?', 'A sharp drop means the network was relying on those clues.', '')}<div class="chart-padded"><svg class="plot" viewBox="0 0 540 290" role="img" aria-label="Signal and background sorting score as SHAP or random clues are hidden"><line class="grid-line" x1="50" y1="35" x2="500" y2="35"/><line class="grid-line" x1="50" y1="130" x2="500" y2="130"/><line class="grid-line" x1="50" y1="225" x2="500" y2="225"/><line class="grid-line" x1="50" y1="35" x2="50" y2="225"/><line class="grid-line" x1="50" y1="225" x2="500" y2="225"/><polyline points="${shapLine}" fill="none" stroke="var(--green)" stroke-width="2.5"/><polyline points="${randomLine}" fill="none" stroke="var(--teal)" stroke-width="2.5"/>${rows.map((row) => `<circle cx="${50 + row.k / maxK * 440}" cy="${225 - row.shap * 190}" r="3" fill="var(--green)"/><circle cx="${50 + row.k / maxK * 440}" cy="${225 - row.random * 190}" r="3" fill="var(--teal)"/>`).join('')}<text x="8" y="39">1.0</text><text x="8" y="134">0.5</text><text x="8" y="229">0.0</text><text x="50" y="246">0</text><text x="468" y="246">${maxK}</text><text class="axis-title" x="204" y="273">number of hidden clues</text><text class="axis-title" transform="translate(15 157) rotate(-90)">sorting score</text></svg><div class="legend"><span><i></i> Hide SHAP’s top clues</span><span><i class="teal"></i> Hide random clues</span></div></div></div><aside class="card card-pad"><span class="small-label">AFTER ALL CLUES ARE HIDDEN</span><div class="comparison-numbers"><div><span>Hide SHAP clues</span><strong>${fixed(rows[rows.length - 1].shap, 3)}</strong></div><div><span>Hide random clues</span><strong>${fixed(rows[rows.length - 1].random, 3)}</strong></div></div><p class="body-copy">A lower score means the network has a harder time separating the two groups. Hiding clues replaces each with its average value.</p><hr class="divider"><span class="small-label">WHAT TO TAKE AWAY</span><p class="body-copy">This checks whether SHAP has found clues the network actually uses. It does not prove the network understands particle physics; hiding related clues can create unusual combinations.</p></aside></div>`;
}

function renderScience() {
  const panel = byId('panel-science');
  const model = state.data.model;
  panel.innerHTML = `${sectionIntro('THE PROJECT', 'From collision clues to an explanation', 'A neural network learns to sort simulated collisions. This page lets you see one score, the clues behind it, and whether the model actually depends on those clues.', 'SIMULATION STUDY')}
    <div class="project-flow"><article class="flow-card"><span>01</span><div><h3>Start with simulated collisions</h3><p>The project uses one million examples: some are generated as Higgs signal, and others as background processes.</p></div></article><i>→</i><article class="flow-card"><span>02</span><div><h3>Turn 28 clues into a score</h3><p>The network learns patterns in particle directions, energy, jets, and calculated masses.</p></div></article><i>→</i><article class="flow-card"><span>03</span><div><h3>Show which clues mattered</h3><p>SHAP breaks one score into smaller pushes from the input clues—up toward signal or down toward background.</p></div></article></div>
    <div class="about-results"><div class="about-result-number"><span class="small-label">HOW WELL IT RANKED THE EXAMPLES</span><strong>${fixed(model.validationAuc, 2)}<small> / 1.00</small></strong><p>0.50 is random ranking; 1.00 is perfect. This result uses separate simulated examples.</p></div><div class="about-boundary"><span class="plain-icon">!</span><p><strong>What these results mean</strong><br>The project explains a network trained with simulations. It does not analyze real LHC data or discover a Higgs boson.</p></div></div>
    <a class="source-chip" href="https://archive.ics.uci.edu/dataset/280/higgs" target="_blank" rel="noreferrer">See where the simulated collision examples come from <span>↗</span></a>`;
}

function wireStory() {
  const tabs = [...document.querySelectorAll('[data-stage]')];
  const panel = byId('story-caption');
  const back = byId('story-back');
  const next = byId('story-next');
  const visual = document.querySelector('.story-visual');
  const stageTabs = document.querySelector('.story-stage-tabs');
  const modeButtons = [...document.querySelectorAll('[data-scenario]')];
  const scenarioCaption = byId('scenario-caption');
  const announcement = byId('announcement');
  const playButton = byId('story-play');
  const progress = document.querySelector('.story-progress');
  const progressFill = byId('story-progress-fill');
  const presentationButton = byId('presentation-toggle');
  let walkthroughTimer = null;
  let fullscreenSession = false;
  const signalCaption = 'A simplified signal example includes a Higgs-producing process. Some events in the dataset have products like the b-jets drawn here; the picture is not any one real or selected event.';
  const backgroundCaption = 'Background events are other simulated processes. Some can leave similar jets, leptons, and missing momentum, so one detector clue alone cannot identify a Higgs.';

  const updateProgress = () => {
    const current = state.storyStage + 1;
    progressFill.style.width = `${current / tabs.length * 100}%`;
    progress.setAttribute('aria-valuenow', String(current));
    progress.setAttribute('aria-valuetext', `Step ${current} of ${tabs.length}`);
  };
  const stopWalkthrough = () => {
    window.clearInterval(walkthroughTimer);
    walkthroughTimer = null;
    playButton.setAttribute('aria-pressed', 'false');
    playButton.textContent = '▶ Play walkthrough';
  };
  const showStage = (index, fromPlayback = false) => {
    if (!fromPlayback) stopWalkthrough();
    state.storyStage = Math.max(0, Math.min(storySteps.length - 1, index));
    const step = storySteps[state.storyStage];
    visual.dataset.stage = String(state.storyStage);
    byId('story-kicker').textContent = step.kicker;
    byId('story-heading').textContent = step.title;
    byId('story-description').textContent = step.text;
    byId('story-step-count').textContent = `${String(state.storyStage + 1).padStart(2, '0')} / 03`;
    panel.classList.remove('story-step-enter');
    void panel.offsetWidth;
    panel.classList.add('story-step-enter');
    panel.setAttribute('aria-labelledby', `story-tab-${state.storyStage}`);
    tabs.forEach((tab, tabIndex) => {
      const active = tabIndex === state.storyStage;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
    });
    back.disabled = state.storyStage === 0;
    next.disabled = state.storyStage === storySteps.length - 1;
    updateProgress();
    announcement.textContent = step.title;
  };
  tabs.forEach((tab) => tab.addEventListener('click', () => showStage(Number(tab.dataset.stage))));
  stageTabs.addEventListener('keydown', (event) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const nextIndex = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1
      : (state.storyStage + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    tabs[nextIndex].focus();
    showStage(nextIndex);
  });
  back.addEventListener('click', () => showStage(state.storyStage - 1));
  next.addEventListener('click', () => showStage(state.storyStage + 1));
  playButton.addEventListener('click', () => {
    if (walkthroughTimer) {
      stopWalkthrough();
      announcement.textContent = 'Walkthrough paused.';
      return;
    }
    if (state.storyStage === tabs.length - 1) showStage(0, true);
    playButton.setAttribute('aria-pressed', 'true');
    playButton.textContent = 'Ⅱ Pause walkthrough';
    announcement.textContent = 'Playing the three-step collision walkthrough.';
    walkthroughTimer = window.setInterval(() => {
      if (state.storyStage >= tabs.length - 1) {
        stopWalkthrough();
        announcement.textContent = 'Walkthrough complete. Use the steps to revisit any part.';
      } else {
        showStage(state.storyStage + 1, true);
      }
    }, 6200);
  });
  modeButtons.forEach((button) => button.addEventListener('click', () => {
    state.storyScenario = button.dataset.scenario;
    visual.dataset.scenario = state.storyScenario;
    modeButtons.forEach((candidate) => candidate.setAttribute('aria-pressed', String(candidate === button)));
    scenarioCaption.textContent = state.storyScenario === 'signal' ? signalCaption : backgroundCaption;
    announcement.textContent = state.storyScenario === 'signal' ? 'Showing a schematic Higgs signal.' : 'Showing a schematic background look-alike.';
  }));
  const setPresentationMode = (active) => {
    document.body.classList.toggle('presentation-mode', active);
    presentationButton.setAttribute('aria-pressed', String(active));
    presentationButton.textContent = active ? '× Exit presentation' : '⛶ Presentation view';
  };
  presentationButton.addEventListener('click', async () => {
    const entering = !document.body.classList.contains('presentation-mode');
    let fullscreenUnavailable = false;
    setPresentationMode(entering);
    if (entering && document.documentElement.requestFullscreen) {
      try {
        await document.documentElement.requestFullscreen();
        fullscreenSession = true;
      } catch {
        fullscreenUnavailable = true;
      }
    } else if (!entering && document.fullscreenElement && document.exitFullscreen) {
      await document.exitFullscreen();
    } else if (entering) {
      fullscreenUnavailable = true;
    }
    if (entering) announcement.textContent = fullscreenUnavailable
      ? 'Presentation view enabled. Full screen is unavailable, but the larger layout is active.'
      : 'Presentation view enabled. Use the button again or press Escape to leave full screen.';
  });
  document.addEventListener('fullscreenchange', () => {
    if (document.fullscreenElement) {
      fullscreenSession = true;
    } else if (fullscreenSession) {
      fullscreenSession = false;
      setPresentationMode(false);
    }
  });
  visual.dataset.stage = String(state.storyStage);
  visual.dataset.scenario = state.storyScenario;
  updateProgress();
  scenarioCaption.textContent = signalCaption;
}

function wireTabs() {
  const tabs = [...document.querySelectorAll('[data-tab]')];
  tabs.forEach((tab) => {
    tab.addEventListener('click', () => activateTab(tab.dataset.tab));
    tab.addEventListener('keydown', (event) => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const current = tabs.indexOf(tab);
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1
        : (current + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
      tabs[next].focus();
      activateTab(tabs[next].dataset.tab);
    });
  });
  document.querySelectorAll('[data-nav]').forEach((link) => link.addEventListener('click', () => activateTab(link.dataset.nav)));
}

function activateTab(name) {
  document.querySelectorAll('[data-tab]').forEach((tab) => {
    const active = tab.dataset.tab === name;
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
  });
  document.querySelectorAll('.tab-panel').forEach((panel) => { panel.hidden = panel.id !== `panel-${name}`; });
}

wireTabs();
wireStory();
loadResearchData();
