import { MARKETS, fetchLatestPredictions, formatLong } from './predictions.js';

const STORAGE_KEY = 'molina:market';
const DEFAULT_SYMBOL = 'XAUUSD';

const DESCRIPTIONS = {
  SPX: 'S&P 500 index',
  NDX: 'NASDAQ index',
  DJI: 'Dow Jones Industrial Average',
  XAUUSD: 'Gold spot / US dollar',
  DXY: 'US Dollar Index',
};

const SIGNAL_META = {
  'BOS Bull':       { tone: 'up',      arrow: '▲', description: 'Bullish break of structure' },
  'BOS Bear':       { tone: 'down',    arrow: '▼', description: 'Bearish break of structure' },
  'Bull FVG':       { tone: 'up',      arrow: '▲', description: 'Bullish fair value gap' },
  'Bear FVG':       { tone: 'down',    arrow: '▼', description: 'Bearish fair value gap' },
  'Bull IFVG':      { tone: 'up',      arrow: '▲', description: 'Bullish inverse fair value gap' },
  'Bear IFVG':      { tone: 'down',    arrow: '▼', description: 'Bearish inverse fair value gap' },
  'Bull Breaker':   { tone: 'up',      arrow: '▲', description: 'Bullish breaker block' },
  'Bear Breaker':   { tone: 'down',    arrow: '▼', description: 'Bearish breaker block' },
  'Bull OB':        { tone: 'up',      arrow: '▲', description: 'Bullish order block' },
  'Bear OB':        { tone: 'down',    arrow: '▼', description: 'Bearish order block' },
  'Liquidity Sweep':{ tone: 'neutral', arrow: '◆', description: 'Liquidity taken at recent extreme' },
};

const LEVEL_PREFIX = {}; // no currency symbol on trade-setup prices

const isSymbol = (s) => MARKETS.some((m) => m.symbol === s);

/* ---------- Which market is open ---------- */

export function selectMarket(symbol) {
  if (isSymbol(symbol)) sessionStorage.setItem(STORAGE_KEY, symbol);
}

export function selectedSymbol() {
  const fromState = history.state?.symbol;
  if (isSymbol(fromState)) return fromState;
  const stored = sessionStorage.getItem(STORAGE_KEY);
  return isSymbol(stored) ? stored : DEFAULT_SYMBOL;
}

/* ---------- Small helpers ---------- */

const $ = (id) => document.getElementById(id);

function setText(id, text) {
  const el = $(id);
  if (el) el.textContent = text;
}

const fmt = (n) => n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

const level = (symbol, n) => (n === null ? '—' : `${LEVEL_PREFIX[symbol] || ''}${fmt(n)}`);

const distance = (symbol, n) =>
  n === null ? '—' : LEVEL_PREFIX[symbol] ? `${LEVEL_PREFIX[symbol]}${fmt(n)}` : `${fmt(n)} pts`;

const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

function formatCandleLabel(c, timeframe) {
  if (timeframe === '1H') {
    const d = new Date(c.time);
    if (isNaN(d.getTime())) return c.time || '';
    return `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()} ${String(d.getUTCHours()).padStart(2, '0')}:00`;
  }
  if (!c.date) return '';
  const [, mo, day] = c.date.split('-').map(Number);
  return `${MONTHS[(mo || 1) - 1]} ${day}`;
}

/* ---------- Header ---------- */

function paintHeader(symbol, m) {
  setText('mk-symbol', symbol);
  setText('mk-sub', `${DESCRIPTIONS[symbol]} · Daily`);
  setText('mk-price', m ? m.price : '—');

  const delta = $('mk-delta');
  if (!delta) return;
  if (m && m.change !== null) {
    const up = m.change >= 0;
    const tone = Math.abs(m.change) < 0.1 ? 'flat' : up ? 'up' : 'down';
    delta.className = `delta ${tone}`;
    delta.textContent = `${up ? '▲' : '▼'} ${m.changeText}`;
  } else {
    delta.className = 'delta';
    delta.textContent = '';
  }
}

/* ---------- Prediction card ---------- */

const TAG_CLASS = { up: 'tag--up', down: 'tag--down', flat: 'tag--flat', none: 'tag--neutral' };
const ICON = { up: '▲', down: '▼', flat: '◆', none: '–' };

function paintPrediction(symbol, m, live) {
  const dir = m?.dir ?? 'none';
  const has = Boolean(m?.hasForecast);
  const confidence = has ? m.confidence : null;

  const icon = $('pred-icon');
  if (icon) {
    icon.textContent = ICON[dir];
    icon.className = `prediction__icon prediction__icon--${dir}`;
  }

  setText(
    'pred-title',
    has
      ? `${symbol} — Predicted ${m.direction.label} with ${confidence ?? '—'}% confidence`
      : `${symbol} — No forecast yet`
  );

  const conviction = m?.trade?.conviction;
  setText(
    'pred-sub',
    has
      ? `Forecast for ${formatLong(live.forecastDay)} · daily model${conviction ? ` · ${conviction} conviction` : ''}`
      : 'Waiting for the next daily run'
  );

  const tag = $('pred-tag');
  if (tag) {
    tag.className = `tag ${TAG_CLASS[dir]}`;
    tag.textContent = has
      ? `${m.direction.arrow} ${m.direction.label}`
      : 'NO FORECAST';
  }

  setText('pred-conf', confidence === null ? '—' : `${confidence}%`);

  const bar = $('pred-conf-bar');
  if (bar) bar.style.width = `${confidence ?? 0}%`;

  const probs = $('pred-probs');

  if (probs) {
    const probabilities = m?.probabilities;

    if (has && probabilities) {
      const outcomes = [
        { key: 'up',            label: 'UP',            arrow: '▲', value: probabilities.up,            className: 'up'   },
        { key: 'down',          label: 'DOWN',          arrow: '▼', value: probabilities.down,          className: 'down' },
        { key: 'consolidation', label: 'CONSOLIDATION', arrow: '◆', value: probabilities.consolidation, className: 'flat' },
      ];

      probs.innerHTML = outcomes.map(outcome => {
        const value = outcome.value;
        const percentage = value ?? 0;

        return `
          <li>
            <span class="probs__label ${outcome.className}">
              ${outcome.arrow} ${outcome.label}
            </span>
            <div class="bar">
              <div class="bar__fill bar__fill--${outcome.className}" style="width:${percentage}%"></div>
            </div>
            <span class="probs__val">${value === null ? '—' : `${value}%`}</span>
          </li>
        `;
      }).join('');
    } else {
      probs.innerHTML = '';
    }
  }
}

/* ---------- Trade setup card ---------- */

function paintTrade(symbol, m) {
  const t = m?.trade ?? null;

  const tag = $('setup-tag');
  if (tag) {
    const action = t?.action;
    const cls = action === 'LONG' ? 'tag--up' : action === 'SHORT' ? 'tag--down' : 'tag--neutral';
    tag.className = `tag ${cls} tag--sm`;
    tag.textContent = action === 'LONG' ? '▲ LONG' : action === 'SHORT' ? '▼ SHORT' : action || 'NO SETUP';
  }

  setText('setup-entry', level(symbol, t?.entry ?? null));
  setText('setup-stop', level(symbol, t?.stopLoss ?? null));
  setText('setup-target', level(symbol, t?.takeProfit ?? null));

  const rr =
    t?.riskReward ??
    (t && t.risk && t.reward !== null ? t.reward / t.risk : null);
  setText('setup-rr', rr === null ? '—' : `1 : ${Number.isInteger(rr) ? rr : rr.toFixed(2)}`);

  const riskBar = $('setup-rr-risk');
  const rewardBar = $('setup-rr-reward');
  if (riskBar) riskBar.style.flex = '1';
  if (rewardBar) rewardBar.style.flex = rr === null ? '1' : String(rr);

  setText('setup-risk', `Risk ${distance(symbol, t?.risk ?? null)}`);
  setText('setup-reward', `Reward ${distance(symbol, t?.reward ?? null)}`);

  const notes = $('setup-notes');
  if (notes) {
    const rows = [
      ['note-entry', t?.entryType],
      ['note-stop', t?.stopSource],
      ['note-target', t?.targetSource],
    ];
    rows.forEach(([id, text]) => setText(id, text || '—'));
    notes.hidden = !t || rows.every(([, text]) => !text);
  }
}

/* ==========================================================
   Chart (with timeframe switching)

   How it stays accurate:
   - bad candles (zero / negative / wildly off-scale prices) are
     dropped before the axis is worked out, so one bad row can't
     squash every other candle against the bottom;
   - the SVG viewBox always equals its real pixel size, so nothing
     is stretched (labels were squashed on phones before);
   - the price axis uses "nice" round tick values;
   - entry / stop / target are placed with the same y() function as
     the candles, and nearby levels widen the axis so they sit next
     to the candles instead of being pushed off-screen.
   ========================================================== */

const CHART_FONT = "'Space Mono', monospace";
const CHART_FONT_SIZE = 11;
const CHAR_W = CHART_FONT_SIZE * 0.62; // approx. width of one Space Mono glyph

let currentTimeframe = '1D';
let currentMarket = null;
let currentSymbol = null;
let chartObserver = null;
let lastChartSize = { W: 0, H: 0 };

/** Drop candles that can't be real, then sort and de-duplicate. */
function cleanCandles(raw, timeframe) {
  const keyOf = (c) => (timeframe === '1H' ? c.time : c.date);

  const parsed = (Array.isArray(raw) ? raw : []).map((c) => ({
    ...c,
    open: Number(c.open),
    high: Number(c.high),
    low: Number(c.low),
    close: Number(c.close),
  }));

  // 0. keep only the latest continuous run, so stale history mixed into the
  //    array can't skew the axis or the median check below
  const t = (c) => Date.parse(keyOf(c));
  const sorted = parsed.filter((c) => Number.isFinite(t(c))).sort((a, b) => t(a) - t(b));
  const maxGap = (timeframe === '1H' ? 5 : 14) * 86400000;
  let s = sorted.length - 1;
  while (s > 0 && t(sorted[s]) - t(sorted[s - 1]) <= maxGap) s--;
  const recent = sorted.slice(s);

  // 1. numbers must be finite, positive and high >= low
  let rows = recent.filter(
    (c) =>
      [c.open, c.high, c.low, c.close].every((n) => Number.isFinite(n) && n > 0) &&
      c.high >= c.low
  );

  // 2. every price must sit within +/-50% of the median close
  if (rows.length) {
    const closes = rows.map((c) => c.close).sort((a, b) => a - b);
    const median = closes[Math.floor(closes.length / 2)];
    const ok = (n) => n >= median * 0.5 && n <= median * 1.5;
    rows = rows.filter((c) => ok(c.open) && ok(c.high) && ok(c.low) && ok(c.close));
  }

  // 3. wicks must contain the candle body
  rows = rows.map((c) => ({
    ...c,
    high: Math.max(c.high, c.open, c.close),
    low: Math.min(c.low, c.open, c.close),
  }));

  // 4. one candle per date/time, oldest first
  const byKey = new Map();
  rows.forEach((c, i) => byKey.set(keyOf(c) ?? `#${i}`, c));
  rows = [...byKey.values()];
  rows.sort((a, b) => {
    const ta = Date.parse(keyOf(a));
    const tb = Date.parse(keyOf(b));
    return Number.isFinite(ta) && Number.isFinite(tb) ? ta - tb : 0;
  });

  return { rows, dropped: parsed.length - rows.length };
}

/** Round, evenly spaced axis values (1, 2, 5 x 10^n). */
function niceTicks(lo, hi, target) {
  const raw = (hi - lo) / Math.max(target, 1);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3.5 ? 2 : norm < 7.5 ? 5 : 10) * mag;

  const first = Math.ceil(lo / step - 1e-9);
  const last = Math.floor(hi / step + 1e-9);
  const ticks = [];
  for (let k = first; k <= last; k++) ticks.push(k * step);

  const decimals = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  return { ticks: ticks.length ? ticks : [lo, hi], decimals };
}

/* ==========================================================
   SMC overlay (subtle, optional)

   The model exports a small set of drawings per timeframe
   (m.smc.daily / m.smc.h1): the nearest still-valid zones, the
   nearest unswept liquidity, the last structure break and recent
   sweeps. They are drawn faintly *behind* the candles; hover any
   of them for the details. Everything is toggled from the chart
   header and remembered in localStorage.
   ========================================================== */

const SMC_STORE = 'molina:smc';
const smcState = { on: false, layers: { zones: true, levels: true, structure: true } };

const RGB = { bull: '34,197,94', bear: '239,68,68', mute: '169,178,199', amber: '245,158,11' };
const ZONE_LABEL = { fvg: 'FVG', ob: 'OB', ifvg: 'IFVG', breaker: 'BRK' };
const ZONE_NAME = { fvg: 'fair value gap', ob: 'order block', ifvg: 'inverse FVG', breaker: 'breaker block' };

function loadSmcState() {
  try {
    const s = JSON.parse(localStorage.getItem(SMC_STORE) || 'null');
    if (!s) return;
    if (typeof s.on === 'boolean') smcState.on = s.on;
    Object.keys(smcState.layers).forEach((k) => {
      if (typeof s.layers?.[k] === 'boolean') smcState.layers[k] = s.layers[k];
    });
  } catch { /* storage unavailable - keep defaults */ }
}

function saveSmcState() {
  try { localStorage.setItem(SMC_STORE, JSON.stringify(smcState)); } catch { /* ignore */ }
}

const toMs = (v) =>
  typeof v === 'string' ? Date.parse(/^\d{4}-\d{2}-\d{2} /.test(v) ? v.replace(' ', 'T') : v) : NaN;

/** Build the SVG for the overlay. `back` goes under the candles, `front` over them. */
function buildSmcOverlay({ set, candles, isH1, PAD, plotW, plotH, plotRight, barW, y, lo, hi, layers }) {
  const out = { back: '', front: '', zones: 0, levels: 0 };
  if (!set || !candles.length) return out;

  const keyOf = (c) => (isH1 ? c.time : c.date);
  const tms = candles.map((c) => toMs(keyOf(c)));
  const first = tms[0];
  const last = tms[tms.length - 1];
  const idxAt = (v) => {
    const t = toMs(v);
    if (!Number.isFinite(t) || t > last) return null;
    if (t <= first) return 0;
    const i = tms.findIndex((x) => x >= t);
    return i < 0 ? null : i;
  };
  const xl = (i) => PAD.left + i * barW;
  const xc = (i) => PAD.left + i * barW + barW / 2;
  const inView = (a, b) => b >= lo && a <= hi;
  const unit = isH1 ? 'h' : 'd';

  out.back += `<defs><clipPath id="smc-clip"><rect x="${PAD.left}" y="${PAD.top}" width="${plotW}" height="${plotH}"/></clipPath></defs><g clip-path="url(#smc-clip)">`;

  if (layers.zones) {
    (set.zones || []).slice(0, 5).forEach((z) => {
      if (!inView(z.bottom, z.top)) return;
      const i = idxAt(z.start);
      if (i === null) return;
      const rgb = z.dir === 'bull' ? RGB.bull : RGB.bear;
      const dashed = z.kind === 'ifvg' || z.kind === 'breaker';
      const yT = y(Math.min(z.top, hi));
      const yB = y(Math.max(z.bottom, lo));
      const x0 = xl(i);
      const h = Math.max(yB - yT, 1.5);
      const tip = `${z.dir === 'bull' ? 'Bullish' : 'Bearish'} ${ZONE_NAME[z.kind] || z.kind} · ` +
        `${fmt(z.bottom)} – ${fmt(z.top)} · touched ${z.touches}× · ${z.age}${unit} old`;
      out.back +=
        `<g><title>${tip}</title>` +
        `<rect x="${x0}" y="${yT}" width="${Math.max(plotRight - x0, 1)}" height="${h}" fill="rgba(${rgb},0.07)"` +
        (dashed ? ` stroke="rgba(${rgb},0.35)" stroke-dasharray="3 3"` : '') + '/>' +
        (z.kind === 'ob' ? `<line x1="${x0}" y1="${yT}" x2="${x0}" y2="${yT + h}" stroke="rgba(${rgb},0.45)" stroke-width="2"/>` : '') +
        `<text x="${x0 + 4}" y="${h >= 12 ? yT + 10 : yT - 2}" font-size="9" font-family="${CHART_FONT}" fill="rgba(${rgb},0.75)">${ZONE_LABEL[z.kind] || z.kind}</text>` +
        '</g>';
      out.zones += 1;
    });
  }

  if (layers.levels) {
    (set.levels || []).forEach((l) => {
      if (l.price < lo || l.price > hi) return;
      const i = idxAt(l.start);
      if (i === null) return;
      const yy = y(l.price);
      const bsl = l.type === 'bsl';
      out.back +=
        `<g><title>${bsl ? 'Buy-side liquidity (unswept high)' : 'Sell-side liquidity (unswept low)'} · ${fmt(l.price)}</title>` +
        `<line x1="${xc(i)}" y1="${yy}" x2="${plotRight}" y2="${yy}" stroke="rgba(${RGB.mute},0.4)" stroke-dasharray="1 4" stroke-linecap="round"/>` +
        `<text x="${plotRight - 4}" y="${yy - 3}" text-anchor="end" font-size="9" font-family="${CHART_FONT}" fill="rgba(${RGB.mute},0.6)">${bsl ? 'BSL' : 'SSL'}</text></g>`;
      out.levels += 1;
    });

    (set.sweeps || []).forEach((sw) => {
      if (sw.price < lo || sw.price > hi) return;
      const i = idxAt(sw.time);
      if (i === null) return;
      const cx = xc(i);
      const cy = y(sw.price) + (sw.dir === 'bear' ? -6 : 6);
      out.front +=
        `<g><title>Liquidity sweep (${sw.dir === 'bear' ? 'highs' : 'lows'} taken) · ${fmt(sw.price)}</title>` +
        `<path d="M${cx} ${cy - 3} L${cx + 3} ${cy} L${cx} ${cy + 3} L${cx - 3} ${cy} Z" fill="rgba(${RGB.amber},0.6)"/></g>`;
      out.levels += 1;
    });
  }

  if (layers.structure) {
    (set.structure || []).forEach((st) => {
      if (st.price < lo || st.price > hi) return;
      const i0 = idxAt(st.from);
      const i1 = idxAt(st.to);
      if (i0 === null || i1 === null || i1 <= i0) return;
      const rgb = st.dir === 'bull' ? RGB.bull : RGB.bear;
      const yy = y(st.price);
      out.back +=
        `<g><title>${st.dir === 'bull' ? 'Bullish' : 'Bearish'} ${st.type} · broke ${fmt(st.price)}</title>` +
        `<line x1="${xc(i0)}" y1="${yy}" x2="${xc(i1)}" y2="${yy}" stroke="rgba(${rgb},0.5)" stroke-dasharray="4 3"/>` +
        `<text x="${(xc(i0) + xc(i1)) / 2}" y="${yy - 3}" text-anchor="middle" font-size="9" font-family="${CHART_FONT}" fill="rgba(${rgb},0.7)">${st.type}</text></g>`;
      out.levels += 1;
    });
  }

  out.back += '</g>';
  return out;
}

function syncSmcControls() {
  const btn = $('smc-toggle');
  const layersEl = $('smc-layers');
  const has = Boolean(currentMarket?.smc);
  const active = has && smcState.on;

  if (btn) {
    btn.disabled = !has;
    btn.setAttribute('aria-pressed', String(active));
    btn.title = has ? 'Show model-detected SMC zones on the chart' : 'SMC drawings appear after the next model run';
  }
  if (layersEl) {
    layersEl.hidden = !active;
    layersEl.querySelectorAll('[data-smc-layer]').forEach((b) => {
      const on = smcState.layers[b.dataset.smcLayer] !== false;
      b.classList.toggle('is-active', on);
      b.setAttribute('aria-pressed', String(on));
    });
  }
  if (!active) setText('smc-note', '');
}

function bindSmcControls() {
  const btn = $('smc-toggle');
  if (btn && !btn.dataset.bound) {
    btn.dataset.bound = '1';
    btn.addEventListener('click', () => {
      smcState.on = !smcState.on;
      saveSmcState();
      syncSmcControls();
      paintChart();
    });
  }
  document.querySelectorAll('[data-smc-layer]').forEach((b) => {
    if (b.dataset.bound) return;
    b.dataset.bound = '1';
    b.addEventListener('click', () => {
      const k = b.dataset.smcLayer;
      smcState.layers[k] = !(smcState.layers[k] !== false);
      saveSmcState();
      syncSmcControls();
      paintChart();
    });
  });
}

function paintChart() {
  const svg = $('price-chart');
  const lastDateEl = $('chart-last-date');
  const titleEl = $('chart-title');
  if (!svg) return;

  const wrap = svg.parentElement;
  if (wrap) wrap.style.paddingRight = '0';
  svg.style.width = '100%';
  svg.style.height = '100%';

  const rect = svg.getBoundingClientRect();
  const W = Math.round(rect.width);
  const H = Math.round(rect.height);
  if (W < 50 || H < 50) return;
  lastChartSize = { W, H };
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

  const narrow = W < 520;
  const m = currentMarket;
  const isH1 = currentTimeframe === '1H';

  const { rows, dropped } = cleanCandles(isH1 ? m?.candlesH1 : m?.candles, currentTimeframe);

  if (dropped > 0) {
    console.warn(
      `[chart] ${currentSymbol} ${currentTimeframe}: ignored ${dropped} bad candle(s). ` +
      'Check the source CSV for zero, negative or off-scale prices.'
    );
  }

 const candles = narrow ? rows.slice(-50) : rows.slice(-100);

  if (titleEl) {
    titleEl.textContent = `Price, ${candles.length} ${isH1 ? 'hourly' : 'daily'} candles`;
  }

  if (!candles.length) {
    svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle" fill="#8b95ad" font-size="14" font-family="${CHART_FONT}">No ${isH1 ? 'hourly' : 'daily'} data</text>`;
    if (lastDateEl) lastDateEl.textContent = '';
    return;
  }

  if (lastDateEl) {
    const last = candles[candles.length - 1];
    lastDateEl.textContent =
      `Last: ${formatCandleLabel(last, currentTimeframe)}` +
      (dropped ? ` · ${dropped} bad candle${dropped === 1 ? '' : 's'} hidden` : '');
  }

  /* ----- Price range ----- */
  const lastClose = candles[candles.length - 1].close;
  const minP = Math.min(...candles.map((c) => c.low));
  const maxP = Math.max(...candles.map((c) => c.high));
  const span = maxP - minP || lastClose * 0.01;
  const reach = Math.max(span * 0.75, lastClose * 0.015);

  const t = m?.trade;
  const levels = [
    { key: 'TP',    price: t?.takeProfit, color: '#22c55e' },
    { key: 'Entry', price: t?.entry,      color: '#818cf8' },
    { key: 'SL',    price: t?.stopLoss,   color: '#ef4444' },
  ].filter((l) => Number.isFinite(l.price) && l.price > 0);

  levels.forEach((l) => {
    l.near = l.price >= minP - reach && l.price <= maxP + reach;
  });

  let lo = minP;
  let hi = maxP;
  levels.filter((l) => l.near).forEach((l) => {
    lo = Math.min(lo, l.price);
    hi = Math.max(hi, l.price);
  });

  if (hi - lo < lastClose * 1e-6) {
    lo -= lastClose * 0.005;
    hi += lastClose * 0.005;
  } else {
    const pad = (hi - lo) * 0.06;
    lo -= pad;
    hi += pad;
  }

  /* ----- Layout in real pixels ----- */
  const { ticks, decimals } = niceTicks(lo, hi, H < 260 ? 4 : 6);
  const fmtTick = (v) =>
    v.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });

  const widest = Math.max(...ticks.map((v) => fmtTick(v).length));

  // ----- Reserve a right-side gutter for the level tags -----
  // Compute the widest tag ("Entry 25,291.00") so the gutter fits it comfortably.
  const levelTexts = levels.map((l) => `${l.key} ${fmt(l.price)}`);
  const widestLevel = levelTexts.reduce((max, s) => Math.max(max, s.length), 0);
  const levelTagW = widestLevel ? Math.ceil(widestLevel * CHAR_W) + 12 : 0;
  // Extra breathing room after the tag so it doesn't kiss the SVG edge.
  const rightGutter = widestLevel ? levelTagW + 14 : (narrow ? 8 : 16);

  const PAD = {
    left: Math.ceil(widest * CHAR_W) + 14,
    right: rightGutter,
    top: 14,
    bottom: 28,
  };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const plotRight = PAD.left + plotW;
  const plotBottom = PAD.top + plotH;

  const y = (p) => PAD.top + plotH - ((p - lo) / (hi - lo)) * plotH;
  const barW = plotW / candles.length;
  const bodyW = Math.min(Math.max(barW * 0.62, 1.5), 14);

  let html = '';
  let smcFront = '';

  /* ----- Gridlines + price axis ----- */
  ticks.forEach((price) => {
    const yPix = y(price);
    if (yPix < PAD.top - 1 || yPix > plotBottom + 1) return;
    html += `<line x1="${PAD.left}" y1="${yPix}" x2="${plotRight}" y2="${yPix}" stroke="#1b2540" stroke-width="1"/>`;
    html += `<text x="${PAD.left - 8}" y="${yPix + 4}" text-anchor="end" fill="#8b95ad" font-size="${CHART_FONT_SIZE}" font-family="${CHART_FONT}">${fmtTick(price)}</text>`;
  });

  /* ----- SMC overlay: zones / lines sit behind the candles ----- */
  if (smcState.on && m?.smc) {
    const ov = buildSmcOverlay({
      set: m.smc[isH1 ? 'h1' : 'daily'], candles, isH1,
      PAD, plotW, plotH, plotRight, barW, y, lo, hi, layers: smcState.layers,
    });
    html += ov.back;
    smcFront = ov.front;
    const parts = [];
    if (ov.zones) parts.push(`${ov.zones} zone${ov.zones === 1 ? '' : 's'}`);
    if (ov.levels) parts.push(`${ov.levels} level${ov.levels === 1 ? '' : 's'}`);
    setText('smc-note', parts.length ? parts.join(' · ') : 'Nothing to draw in view');
  }

  /* ----- Candles (now end before the right gutter) ----- */
  candles.forEach((c, i) => {
    const x = PAD.left + i * barW + barW / 2;
    const up = c.close >= c.open;
    const color = up ? '#22c55e' : '#ef4444';

    html += `<line x1="${x}" y1="${y(c.high)}" x2="${x}" y2="${y(c.low)}" stroke="${color}" stroke-width="1"/>`;

    const bodyTop = y(Math.max(c.open, c.close));
    const bodyBot = y(Math.min(c.open, c.close));
    const bodyH = Math.max(bodyBot - bodyTop, 1);
    html += `<rect x="${x - bodyW / 2}" y="${bodyTop}" width="${bodyW}" height="${bodyH}" fill="${color}"/>`;
  });

  html += smcFront;

  /* ----- Entry / stop / target ----- */
  const LABEL_H = 16;
  const plotted = levels.map((l) => ({ ...l, y: y(l.price) }));
  const inside = plotted.filter((l) => l.y >= PAD.top - 1 && l.y <= plotBottom + 1);
  const outside = plotted.filter((l) => !inside.includes(l));

  // Dashed lines stop at the right edge of the plot area (NOT over the gutter).
  inside.forEach((l) => {
    html += `<line x1="${PAD.left}" y1="${l.y}" x2="${plotRight}" y2="${l.y}" stroke="${l.color}" stroke-width="1" stroke-dasharray="6 4"/>`;
  });

  // De-overlap the tag y-positions.
  inside.sort((a, b) => a.y - b.y);
  let prev = -Infinity;
  inside.forEach((l) => {
    l.ly = Math.max(l.y, prev + LABEL_H);
    prev = l.ly;
  });
  if (inside.length) {
    inside[inside.length - 1].ly = Math.min(inside[inside.length - 1].ly, plotBottom - LABEL_H / 2);
    for (let i = inside.length - 2; i >= 0; i--) {
      inside[i].ly = Math.min(inside[i].ly, inside[i + 1].ly - LABEL_H);
    }
  }

  // Tags live in the gutter, right-aligned to W - 6 so there is true
  // right-side breathing room and they never cover the last candle.
  const TAG_RIGHT = W - 6;
  inside.forEach((l) => {
    const text = `${l.key} ${fmt(l.price)}`;
    const w = Math.ceil(text.length * CHAR_W) + 10;
    const tx = TAG_RIGHT - w; // left edge of the tag box
    html += `<rect x="${tx}" y="${l.ly - LABEL_H / 2}" width="${w}" height="${LABEL_H}" rx="3" fill="#111a30" fill-opacity="0.95" stroke="${l.color}" stroke-width="1"/>`;
    html += `<text x="${tx + w / 2}" y="${l.ly + 4}" text-anchor="middle" fill="${l.color}" font-size="${CHART_FONT_SIZE}" font-weight="600" font-family="${CHART_FONT}">${text}</text>`;
  });

  // Far-away levels: compact edge markers, also right-aligned in the gutter.
  let above = 0;
  let below = 0;
  outside.forEach((l) => {
    const isAbove = l.y < PAD.top;
    const edgeY = isAbove ? PAD.top + 10 + above++ * 14 : plotBottom - 6 - below++ * 14;
    html += `<text x="${TAG_RIGHT}" y="${edgeY}" text-anchor="end" fill="${l.color}" font-size="10" font-weight="600" font-family="${CHART_FONT}">${isAbove ? '▲' : '▼'} ${l.key} ${fmt(l.price)}</text>`;
  });

  /* ----- Date axis ----- */
  const labelCount = Math.min(candles.length, W < 380 ? 2 : narrow ? 3 : 5);
  for (let i = 0; i < labelCount; i++) {
    const idx = labelCount === 1 ? 0 : Math.round((candles.length - 1) * (i / (labelCount - 1)));
    const c = candles[idx];
    const x = PAD.left + idx * barW + barW / 2;
    const anchor = i === 0 ? 'start' : i === labelCount - 1 ? 'end' : 'middle';
    const tx = i === 0 ? PAD.left : i === labelCount - 1 ? plotRight : x;
    html += `<text x="${tx}" y="${H - 8}" text-anchor="${anchor}" fill="#8b95ad" font-size="${CHART_FONT_SIZE}" font-family="${CHART_FONT}">${formatCandleLabel(c, currentTimeframe)}</text>`;
  }

  svg.innerHTML = html;
}

/** Redraw when the chart's size changes (rotation, window resize, layout). */
function watchChartSize() {
  const svg = $('price-chart');
  if (!svg || typeof ResizeObserver === 'undefined') return;

  chartObserver?.disconnect();
  chartObserver = new ResizeObserver(() => {
    const el = $('price-chart');
    if (!el || !el.isConnected) {
      chartObserver?.disconnect();
      return;
    }
    const r = el.getBoundingClientRect();
    if (Math.round(r.width) !== lastChartSize.W || Math.round(r.height) !== lastChartSize.H) {
      paintChart();
    }
  });
  chartObserver.observe(svg.parentElement || svg);
}

function bindTimeframeButtons() {
  const buttons = document.querySelectorAll('.tf .chip[data-timeframe]');
  buttons.forEach(btn => {
    if (btn.dataset.bound) return;
    btn.dataset.bound = '1';
    btn.addEventListener('click', () => {
      buttons.forEach(b => {
        b.classList.remove('is-active');
        b.setAttribute('aria-selected', 'false');
      });
      btn.classList.add('is-active');
      btn.setAttribute('aria-selected', 'true');
      currentTimeframe = btn.dataset.timeframe || '1D';
      paintChart();
    });
  });
}

/* ---------- SMC signals ---------- */

function paintSmc(symbol, m) {
  const grid = $('smc-grid');
  const wide = $('smc-wide');
  if (!grid || !wide) return;

  const signals = Array.isArray(m?.signals) ? m.signals : [];

  if (!signals.length) {
    grid.innerHTML = '<p class="muted-light">No active SMC signals for this market.</p>';
    wide.innerHTML = '';
    return;
  }

  const gridItems = [];
  const wideItems = [];

  signals.forEach(raw => {
    const meta = SIGNAL_META[raw] || { tone: 'neutral', arrow: '◆', description: raw };
    const item = {
      label: raw,
      tone: meta.tone,
      arrow: meta.arrow,
      description: meta.description,
    };
    if (raw.includes('Liquidity') || raw.includes('Sweep')) {
      wideItems.push(item);
    } else {
      gridItems.push(item);
    }
  });

  grid.innerHTML = gridItems.map(item => `
    <div class="signal">
      <span class="tag tag--${item.tone} tag--sm">${item.arrow} ${item.label}</span>
      <p>${item.description}</p>
    </div>
  `).join('');

  wide.innerHTML = wideItems.map(item => `
    <div class="signal signal--wide">
      <span class="tag tag--${item.tone} tag--sm">${item.arrow} ${item.label}</span>
      <p>${item.description}</p>
    </div>
  `).join('');
}

/* ---------- Page entry point ---------- */

export async function initMarketPage() {
  const symbol = selectedSymbol();
  loadSmcState();

  setText('mk-symbol', symbol);
  setText('mk-sub', `${DESCRIPTIONS[symbol]} · Daily`);

  try {
    const live = await fetchLatestPredictions();
    if (!$('pred-title') || selectedSymbol() !== symbol) return;

    const m = live?.bySymbol[symbol] ?? null;
    currentMarket = m;
    currentSymbol = symbol;

    paintHeader(symbol, m);
    paintPrediction(symbol, m, live);
    paintTrade(symbol, m);
    paintSmc(symbol, m);
    syncSmcControls();
    bindSmcControls();
    paintChart();
    watchChartSize();
    bindTimeframeButtons();
  } catch (err) {
    console.error('Could not load the market', err);
    if (!$('pred-title')) return;
    setText('pred-title', 'Forecast unavailable');
    setText('pred-sub', 'Could not load the latest forecast.');
  }
}