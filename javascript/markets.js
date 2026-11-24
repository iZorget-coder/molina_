import { MARKETS, fetchLatestPredictions, formatLong, subscribeLiveHourly } from './predictions.js';
import { zoneKey, zoneRgb } from './zones.js';

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
  'CHoCH':          { tone: 'neutral', arrow: '◆', description: 'Change of character — structure flipped' },
  'Bull FVG':       { tone: 'up',      arrow: '▲', description: 'Bullish fair value gap' },
  'Bear FVG':       { tone: 'down',    arrow: '▼', description: 'Bearish fair value gap' },
  'Bull IFVG':      { tone: 'up',      arrow: '▲', description: 'Bullish inverse fair value gap' },
  'Bear IFVG':      { tone: 'down',    arrow: '▼', description: 'Bearish inverse fair value gap' },
  'Bull Breaker':   { tone: 'up',      arrow: '▲', description: 'Bullish breaker block' },
  'Bear Breaker':   { tone: 'down',    arrow: '▼', description: 'Bearish breaker block' },
  'Bull OB':        { tone: 'up',      arrow: '▲', description: 'Bullish order block' },
  'Bear OB':        { tone: 'down',    arrow: '▼', description: 'Bearish order block' },
  'Sweep High':     { tone: 'down',    arrow: '▼', description: 'Buy-side liquidity swept above a recent high' },
  'Sweep Low':      { tone: 'up',      arrow: '▲', description: 'Sell-side liquidity swept below a recent low' },
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

/**
 * Latest price + change since the last daily close, from the live hourly feed.
 * Only used once the live feed has moved on to a trading day AFTER the last daily close (e.g. Monday's
 * hours while the stored close is Friday's); otherwise the stored daily close / change are shown.
 *
 * The prediction payload exposes the daily close as `date` (the YYYY-MM-DD session the close belongs
 * to) plus `close`. `closeDate` is accepted as a fallback so this keeps working if the upstream
 * document ever starts emitting that name.
 */
export function liveHeaderPrice(m, live) {
  const candles = live?.candles;
  const closeDate = m?.closeDate ?? m?.date ?? null;
  if (!m || !candles?.length || !Number.isFinite(m.close) || !closeDate) return null;

  const last = candles.reduce((a, c) => (toMs(c.time) > toMs(a.time) ? c : a), candles[0]);
  const ms = toMs(last.time);
  if (!Number.isFinite(ms) || !Number.isFinite(last.close) || last.close <= 0) return null;
  if (tradingDay(ms) <= closeDate) return null;                    // no new session since the daily close

  const change = (last.close / m.close - 1) * 100;
  const v = Math.round(change * 100) / 100;
  return { price: fmt(last.close), change, changeText: `${v < 0 ? '-' : '+'}${Math.abs(v).toFixed(2)}%` };
}

function paintHeader(symbol, m) {
  setText('mk-symbol', symbol);
  setText('mk-sub', `${DESCRIPTIONS[symbol]} · Daily`);
  const lv = liveHeaderPrice(m, liveData);
  const shown = lv ?? (m ? { price: m.price, change: m.change, changeText: m.changeText } : null);
  setText('mk-price', shown ? shown.price : '—');
  const priceEl = $('mk-price');
  if (priceEl) priceEl.title = lv ? 'Latest hourly price · change since the last daily close' : 'Last daily close';

  const delta = $('mk-delta');
  if (!delta) return;
  if (shown && shown.change !== null && shown.change !== undefined) {
    const up = shown.change >= 0;
    const tone = Math.abs(shown.change) < 0.1 ? 'flat' : up ? 'up' : 'down';
    delta.className = `delta ${tone}`;
    delta.textContent = `${up ? '▲' : '▼'} ${shown.changeText}`;
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
   Live hourly feed

   live_hourly/latest is refreshed every hour by the "Hourly Live
   Candles" workflow (fetch only, the model is not re-run). Those
   candles are merged over the stored 1H candles, and the day's
   entry / stop / target are checked against them to show whether
   the limit order has filled, hit its stop or reached its target.
   ========================================================== */

let liveData = null;          // { candles, source, updatedAt } for the open market, or null
let unsubLive = null;
let forecastDayIso = null;

/** Stored 1H candles + live candles, one entry per hour (live wins), times as ISO strings. */
function mergeLive(raw, live) {
  const byMs = new Map();
  const put = (c, isLive) => {
    const ms = toMs(c?.time);
    if (Number.isFinite(ms)) byMs.set(ms, { ...c, time: new Date(ms).toISOString(), live: isLive });
  };
  (Array.isArray(raw) ? raw : []).forEach((c) => put(c, false));
  (live?.candles || []).forEach((c) => put(c, true));
  return [...byMs.values()];
}

/** Trading day of an hour (futures reopen ~22:00-23:00 UTC, so +2h puts that open on the next day). */
const tradingDay = (ms) => new Date(ms + 2 * 3600000).toISOString().slice(0, 10);

/**
 * Daily candles + the days the live hourly candles add or extend.
 * - a day after the last stored daily candle is built from its hourly candles;
 * - the newest live day that already has a stored candle gets its high / low stretched and its close moved;
 * - older stored days are left exactly as the daily pipeline saved them.
 * The candle for a day that is still trading is flagged `partial` (drawn dotted).
 */
export function mergeDaily(raw, live) {
  const daily = (Array.isArray(raw) ? raw : []).map((c) => ({ ...c }));
  if (!live?.candles?.length) return daily;

  const days = new Map();
  [...live.candles]
    .map((c) => ({ ...c, ms: toMs(c.time) }))
    .filter((c) => Number.isFinite(c.ms))
    .sort((a, b) => a.ms - b.ms)
    .forEach((c) => {
      const d = tradingDay(c.ms);
      const a = days.get(d);
      if (!a) days.set(d, { date: d, open: c.open, high: c.high, low: c.low, close: c.close, partial: Boolean(c.partial), live: true });
      else {
        a.high = Math.max(a.high, c.high);
        a.low = Math.min(a.low, c.low);
        a.close = c.close;
        a.partial = Boolean(c.partial);
      }
    });

  const dates = [...days.keys()].sort();
  const newest = dates[dates.length - 1];

  // While today is still trading, the daily file may hold an unfinished candle for today (or a
  // Sunday one). It is dropped so the chart has ONE current-day candle, built from the hourly bars.
  if (days.get(newest).partial) {
    const cut = Date.parse(`${newest}T00:00:00Z`);
    for (let i = daily.length - 1; i >= 0; i--) {
      const t = Date.parse(daily[i].date);
      if (!Number.isFinite(t)) continue;
      const dow = new Date(t).getUTCDay();
      if (t >= cut || ((dow === 0 || dow === 6) && t >= cut - 3 * 86400000)) daily.splice(i, 1);
    }
  }
  const lastStored = daily.map((c) => c.date).filter(Boolean).sort().pop() || '';

  dates.forEach((d) => {
    const a = days.get(d);
    if (d > lastStored) {
      const dow = new Date(`${d}T00:00:00Z`).getUTCDay();
      if (dow !== 0 && dow !== 6) daily.push(a);       // never a weekend candle
    } else if (d === lastStored && d === newest) {
      const c = daily.find((x) => x.date === d);
      if (c) {
        c.high = Math.max(Number(c.high), a.high);
        c.low = Math.min(Number(c.low), a.low);
        c.close = a.close;
        c.partial = a.partial;
        c.live = true;
      }
    }
  });
  return daily;
}

/** Index / gold futures have no Saturday or Sunday session. A weekend-dated daily candle is only the
 *  first hour or two of Sunday evening's open, so it is dropped (that hour belongs to Monday's candle). */
export const dropWeekendDays = (arr) =>
  (Array.isArray(arr) ? arr : []).filter((c) => {
    const t = Date.parse(c?.date);
    if (!Number.isFinite(t)) return true;
    const dow = new Date(t).getUTCDay();
    return dow !== 0 && dow !== 6;
  });

/** Cleaned candles for a timeframe, with the live feed merged in. */
function chartRows(m, isH1) {
  if (!isH1) {
    const daily = liveData ? mergeDaily(m?.candles, liveData) : m?.candles;
    return cleanCandles(dropWeekendDays(daily), '1D');
  }
  return cleanCandles(liveData ? mergeLive(m?.candlesH1, liveData) : m?.candlesH1, '1H');
}

/**
 * Walks the candles from the start of the forecast day and reports what the limit order did.
 * Same rules as the model's training simulation: the order fills when price reaches the entry,
 * the stop wins a tie, and the fill bar cannot reach the target.
 */
export function evaluateTrade(t, rows, fromMs) {
  if (!t || (t.action !== 'LONG' && t.action !== 'SHORT')) return null;
  if (![t.entry, t.stopLoss, t.takeProfit].every(Number.isFinite)) return null;
  const long = t.action === 'LONG';
  const risk = Math.abs(t.entry - t.stopLoss);
  if (!(risk > 0)) return null;
  const bars = rows.filter((c) => toMs(c.time) >= fromMs);
  if (!bars.length) return { state: 'pending' };

  const hitStop = (c) => (long ? c.low <= t.stopLoss : c.high >= t.stopLoss);
  const hitTarget = (c) => (long ? c.high >= t.takeProfit : c.low <= t.takeProfit);
  const rOf = (p) => (long ? p - t.entry : t.entry - p) / risk;

  let fill = null;
  for (const c of bars) {
    if (!fill) {
      if (long ? c.low <= t.entry : c.high >= t.entry) {
        fill = { time: c.time };
        if (hitStop(c)) return { state: 'stopped', fill, exit: { time: c.time, price: t.stopLoss }, r: -1 };
      } else if (hitTarget(c)) {
        return { state: 'missed', exit: { time: c.time, price: t.takeProfit } };
      }
      continue;
    }
    if (hitStop(c)) return { state: 'stopped', fill, exit: { time: c.time, price: t.stopLoss }, r: -1 };
    if (hitTarget(c)) return { state: 'target', fill, exit: { time: c.time, price: t.takeProfit }, r: rOf(t.takeProfit) };
  }
  const last = bars[bars.length - 1];
  return fill
    ? { state: 'open', fill, r: rOf(last.close), price: last.close }
    : { state: 'waiting', price: last.close, away: Math.abs(last.close - t.entry) };
}

function tradeStatus() {
  const t = currentMarket?.trade;
  if (!liveData || !t || !forecastDayIso) return null;
  const { rows } = chartRows(currentMarket, true);
  return evaluateTrade(t, rows, Date.parse(`${forecastDayIso}T00:00:00Z`));
}

const signedR = (r) => `${r >= 0 ? '+' : '-'}${Math.abs(r).toFixed(1)}R`;

function paintTradeStatus() {
  const el = $('setup-status');
  if (!el) return;
  const st = tradeStatus();
  if (!st) { el.hidden = true; return; }
  const text = {
    pending: () => 'Waiting for the first hourly candle of the forecast day',
    waiting: () => `Limit order waiting · price ${fmt(st.price)}, ${fmt(st.away)} pts from entry`,
    open: () => `Filled · open ${signedR(st.r)} · price ${fmt(st.price)}`,
    target: () => `Target reached · ${signedR(st.r)}`,
    stopped: () => 'Stopped out · -1.0R',
    missed: () => 'Missed · target traded before the entry filled',
  }[st.state]();
  const tone = { target: 'up', stopped: 'down', open: st.r >= 0 ? 'up' : 'down' }[st.state] || 'flat';
  el.dataset.tone = tone;
  el.textContent = text;
  el.hidden = false;
}

/* ==========================================================
   SMC overlay (subtle, optional)

   The model exports a small set of drawings per timeframe
   (m.smc.daily / m.smc.h1): the nearest still-valid zones, the
   nearest unswept liquidity, the last structure break and recent
   sweeps. They are drawn faintly *behind* the candles; hover any
   of them for the details. Everything is toggled from the chart
   header and remembered in localStorage.

   Colours: every zone type has its own colour (see zones.js):
   FVG orange, OB blue, IFVG purple, breaker red, sweep pink,
   BOS teal, CHoCH lime, liquidity slate. Direction is shown by
   the ▲/▼ in the label and the border: solid = bullish,
   dashed = bearish.
   ========================================================== */

const SMC_STORE = 'molina:smc';
const smcState = { on: false, layers: { zones: true, levels: true, structure: true } };

const RGB = { mute: zoneRgb('liq') };
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

const toMs = (v) => {
  if (typeof v !== 'string') return NaN;
  let s = /^\d{4}-\d{2}-\d{2} /.test(v) ? v.replace(' ', 'T') : v;
  if (/T\d{2}:\d{2}(:\d{2}(\.\d+)?)?$/.test(s)) s += 'Z'; // no zone given = UTC
  return Date.parse(s);
};

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
      const bull = z.dir === 'bull';
      const rgb = zoneRgb(zoneKey(z.kind));
      const yT = y(Math.min(z.top, hi));
      const yB = y(Math.max(z.bottom, lo));
      const x0 = xl(i);
      const h = Math.max(yB - yT, 1.5);
      const tip = `${bull ? 'Bullish' : 'Bearish'} ${ZONE_NAME[z.kind] || z.kind} · ` +
        `${fmt(z.bottom)} – ${fmt(z.top)} · touched ${z.touches}× · ${z.age}${unit} old`;
      const labelY = h >= 12 ? yT + 10 : yT - 2;
      out.back +=
        `<g><title>${tip}</title>` +
        `<rect x="${x0}" y="${yT}" width="${Math.max(plotRight - x0, 1)}" height="${h}" fill="rgba(${rgb},0.10)" ` +
        `stroke="rgba(${rgb},0.45)" stroke-width="1"${bull ? '' : ' stroke-dasharray="3 3"'}/>` +
        (z.kind === 'ob' ? `<line x1="${x0}" y1="${yT}" x2="${x0}" y2="${yT + h}" stroke="rgba(${rgb},0.75)" stroke-width="2"/>` : '') +
        `<text x="${plotRight - 4}" y="${labelY}" text-anchor="end" font-size="9" font-family="${CHART_FONT}" fill="rgba(${rgb},0.9)">${bull ? '▲' : '▼'} ${ZONE_LABEL[z.kind] || z.kind}</text>` +
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
        `<line x1="${xc(i)}" y1="${yy}" x2="${plotRight}" y2="${yy}" stroke="rgba(${RGB.mute},0.5)" stroke-dasharray="1 4" stroke-linecap="round"/>` +
        `<text x="${plotRight - 4}" y="${yy - 3}" text-anchor="end" font-size="9" font-family="${CHART_FONT}" fill="rgba(${RGB.mute},0.8)">${bsl ? 'BSL' : 'SSL'}</text></g>`;
      out.levels += 1;
    });

    const sweepRgb = zoneRgb('sweep');
    (set.sweeps || []).forEach((sw) => {
      if (sw.price < lo || sw.price > hi) return;
      const i = idxAt(sw.time);
      if (i === null) return;
      const cx = xc(i);
      const cy = y(sw.price) + (sw.dir === 'bear' ? -6 : 6);
      out.front +=
        `<g><title>Liquidity sweep (${sw.dir === 'bear' ? 'highs' : 'lows'} taken) · ${fmt(sw.price)}</title>` +
        `<path d="M${cx} ${cy - 3.5} L${cx + 3.5} ${cy} L${cx} ${cy + 3.5} L${cx - 3.5} ${cy} Z" fill="rgba(${sweepRgb},0.85)"/></g>`;
      out.levels += 1;
    });
  }

  if (layers.structure) {
    (set.structure || []).forEach((st) => {
      if (st.price < lo || st.price > hi) return;
      const i0 = idxAt(st.from);
      const i1 = idxAt(st.to);
      if (i0 === null || i1 === null || i1 <= i0) return;
      const bull = st.dir === 'bull';
      const rgb = zoneRgb(zoneKey(st.type));
      const yy = y(st.price);
      out.back +=
        `<g><title>${bull ? 'Bullish' : 'Bearish'} ${st.type} · broke ${fmt(st.price)}</title>` +
        `<line x1="${xc(i0)}" y1="${yy}" x2="${xc(i1)}" y2="${yy}" stroke="rgba(${rgb},0.7)" stroke-dasharray="4 3"/>` +
        `<text x="${(xc(i0) + xc(i1)) / 2}" y="${yy - 3}" text-anchor="middle" font-size="9" font-family="${CHART_FONT}" fill="rgba(${rgb},0.9)">${bull ? '▲' : '▼'} ${st.type}</text></g>`;
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

  const { rows, dropped } = chartRows(m, isH1);

  if (dropped > 0) {
    console.warn(
      `[chart] ${currentSymbol} ${currentTimeframe}: ignored ${dropped} bad candle(s). ` +
      'Check the source CSV for zero, negative or off-scale prices.'
    );
  }
  const COUNTS = {
    '1D': { narrow: 20, wide: 50 },
    '1H': { narrow: 20, wide: 100 },
  };
  const n = COUNTS[currentTimeframe][narrow ? 'narrow' : 'wide'];
  const candles = rows.slice(-n);

  if (isH1 && candles.length > 1) {
    const gaps = [];
    for (let i = 1; i < candles.length; i++) {
      const d = (toMs(candles[i].time) - toMs(candles[i - 1].time)) / 3600000;
      if (d > 1) gaps.push(`${candles[i - 1].time} -> ${candles[i].time} (${d - 1}h missing)`);
    }
    console.info(`[chart] ${currentSymbol} 1H last 4:`, candles.slice(-4).map((c) => `${c.time}${c.live ? ' live' : ' stored'}`));
    if (gaps.length) console.info(`[chart] ${currentSymbol} 1H gaps:`, gaps);
  }

  if (titleEl) {
    titleEl.textContent = `Price, ${candles.length} ${isH1 ? 'hourly' : 'daily'} candles`;
  }

  if (!candles.length) {
    svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle" fill="#8b95ad" font-size="14" font-family="${CHART_FONT}">No ${isH1 ? 'hourly' : 'daily'} data</text>`;
    if (lastDateEl) lastDateEl.textContent = '';
    if ($('live-note')) $('live-note').hidden = true;
    return;
  }

  if (lastDateEl) {
    const last = candles[candles.length - 1];
    lastDateEl.textContent =
      `Last: ${formatCandleLabel(last, currentTimeframe)}` +
      (dropped ? ` · ${dropped} bad candle${dropped === 1 ? '' : 's'} hidden` : '');
  }

  const liveEl = $('live-note');
  if (liveEl) {
    const u = liveData?.updatedAt;
    liveEl.hidden = !(isH1 && liveData);
    const shown = u ? new Date(u.getTime() + 3600000) : null;
    liveEl.textContent = shown
      ? `● live ${String(shown.getUTCHours()).padStart(2, '0')}:${String(shown.getUTCMinutes()).padStart(2, '0')} UTC`
      : '● live';
  }

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

  const { ticks, decimals } = niceTicks(lo, hi, H < 260 ? 4 : 6);
  const fmtTick = (v) =>
    v.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });

  const widest = Math.max(...ticks.map((v) => fmtTick(v).length));

  const levelTexts = levels.map((l) => `${l.key} ${fmt(l.price)}`);
  const widestLevel = levelTexts.reduce((max, s) => Math.max(max, s.length), 0);
  const levelTagW = widestLevel ? Math.ceil(widestLevel * CHAR_W) + 12 : 0;
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

  ticks.forEach((price) => {
    const yPix = y(price);
    if (yPix < PAD.top - 1 || yPix > plotBottom + 1) return;
    html += `<line x1="${PAD.left}" y1="${yPix}" x2="${plotRight}" y2="${yPix}" stroke="#1b2540" stroke-width="1"/>`;
    html += `<text x="${PAD.left - 8}" y="${yPix + 4}" text-anchor="end" fill="#8b95ad" font-size="${CHART_FONT_SIZE}" font-family="${CHART_FONT}">${fmtTick(price)}</text>`;
  });

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

  const LABEL_H = 16;
  const plotted = levels.map((l) => ({ ...l, y: y(l.price) }));
  const inside = plotted.filter((l) => l.y >= PAD.top - 1 && l.y <= plotBottom + 1);
  const outside = plotted.filter((l) => !inside.includes(l));

  inside.forEach((l) => {
    html += `<line x1="${PAD.left}" y1="${l.y}" x2="${plotRight}" y2="${l.y}" stroke="${l.color}" stroke-width="1" stroke-dasharray="6 4"/>`;
  });

  const st = isH1 ? tradeStatus() : null;
  if (st && t) {
    const at = (time) => candles.findIndex((c) => toMs(c.time) === toMs(time));
    const mark = (time, price, color, label) => {
      const i = at(time);
      if (i < 0 || price < lo || price > hi) return;
      html += `<g><title>${label} · ${fmt(price)}</title><circle cx="${PAD.left + i * barW + barW / 2}" cy="${y(price)}" r="4.5" fill="none" stroke="${color}" stroke-width="1.6"/></g>`;
    };
    if (st.fill) mark(st.fill.time, t.entry, '#818cf8', 'Order filled');
    if (st.exit) mark(st.exit.time, st.exit.price, st.state === 'stopped' ? '#ef4444' : '#22c55e', st.state === 'stopped' ? 'Stopped out' : st.state === 'target' ? 'Target reached' : 'Target traded before fill');
  }

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

  const TAG_RIGHT = W - 6;
  inside.forEach((l) => {
    const text = `${l.key} ${fmt(l.price)}`;
    const w = Math.ceil(text.length * CHAR_W) + 10;
    const tx = TAG_RIGHT - w;
    html += `<rect x="${tx}" y="${l.ly - LABEL_H / 2}" width="${w}" height="${LABEL_H}" rx="3" fill="#111a30" fill-opacity="0.95" stroke="${l.color}" stroke-width="1"/>`;
    html += `<text x="${tx + w / 2}" y="${l.ly + 4}" text-anchor="middle" fill="${l.color}" font-size="${CHART_FONT_SIZE}" font-weight="600" font-family="${CHART_FONT}">${text}</text>`;
  });

  let above = 0;
  let below = 0;
  outside.forEach((l) => {
    const isAbove = l.y < PAD.top;
    const edgeY = isAbove ? PAD.top + 10 + above++ * 14 : plotBottom - 6 - below++ * 14;
    html += `<text x="${TAG_RIGHT}" y="${edgeY}" text-anchor="end" fill="${l.color}" font-size="10" font-weight="600" font-family="${CHART_FONT}">${isAbove ? '▲' : '▼'} ${l.key} ${fmt(l.price)}</text>`;
  });

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
      zone: zoneKey(raw),
      arrow: meta.arrow,
      description: meta.description,
    };
    if (raw.includes('Liquidity') || raw.includes('Sweep')) {
      wideItems.push(item);
    } else {
      gridItems.push(item);
    }
  });

  const signalHtml = (item, extra = '') => `
    <div class="signal${extra}">
      <span class="zone-tag" data-zone="${item.zone}">${item.arrow} ${item.label}</span>
      <p>${item.description}</p>
    </div>
  `;

  grid.innerHTML = gridItems.map((item) => signalHtml(item)).join('');
  wide.innerHTML = wideItems.map((item) => signalHtml(item, ' signal--wide')).join('');
}

/* ---------- Page entry point ---------- */

export async function initMarketPage() {
  const symbol = selectedSymbol();
  loadSmcState();
  unsubLive?.();
  unsubLive = null;
  liveData = null;
  currentTimeframe = '1D';
  currentMarket = null;
  currentSymbol = null;

  setText('mk-symbol', symbol);
  setText('mk-sub', `${DESCRIPTIONS[symbol]} · Daily`);

  try {
    const live = await fetchLatestPredictions();
    if (!$('pred-title') || selectedSymbol() !== symbol) return;

    const m = live?.bySymbol[symbol] ?? null;
    currentMarket = m;
    currentSymbol = symbol;
    forecastDayIso = live?.forecastDay ?? null;

    paintHeader(symbol, m);
    paintPrediction(symbol, m, live);
    paintTrade(symbol, m);
    paintSmc(symbol, m);
    syncSmcControls();
    bindSmcControls();
    paintChart();
    watchChartSize();
    bindTimeframeButtons();

    unsubLive = subscribeLiveHourly(symbol, (data) => {
      if (!$('price-chart') || currentSymbol !== symbol) { unsubLive?.(); return; }
      liveData = data;
      paintHeader(symbol, currentMarket);
      paintTradeStatus();
      paintChart();
    });
  } catch (err) {
    console.error('Could not load the market', err);
    if (!$('pred-title')) return;
    setText('pred-title', 'Forecast unavailable');
    setText('pred-sub', 'Could not load the latest forecast.');
  }
}