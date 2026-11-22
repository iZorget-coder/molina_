import { collection, query, orderBy, limit, getDocs, doc, onSnapshot } from 'firebase/firestore';
import { auth, onAuthStateChanged, db } from './firebase.js';

/* ---------- The five markets the app supports ---------- */
export const MARKETS = [
  { symbol: 'SPX',    name: 'S&P 500',         type: 'Index' },
  { symbol: 'NDX',    name: 'NASDAQ',          type: 'Index' },
  { symbol: 'DJI',    name: 'DOW JONES',       type: 'Index' },
  { symbol: 'XAUUSD', name: 'Gold',            type: 'Commodity' },
  { symbol: 'DXY',    name: 'US Dollar Index', type: 'FX' },
];

/* ---------- Direction helpers ---------- */
const DIRECTIONS = {
  UP:            { key: 'up',   label: 'UP',            arrow: '▲' },
  DOWN:          { key: 'down', label: 'DOWN',          arrow: '▼' },
  CONSOLIDATION: { key: 'flat', label: 'CONSOLIDATION', arrow: '◆' },
};
const NO_FORECAST = { key: 'none', label: 'NO FORECAST', arrow: '' };

export const directionOf = (bias) =>
  DIRECTIONS[String(bias || '').toUpperCase()] || NO_FORECAST;

export function countDirections(data) {
  const counts = { up: 0, down: 0, flat: 0 };
  if (!data) return counts;
  for (const p of data.list) {
    if (p.dir in counts) counts[p.dir] += 1;
  }
  return counts;
}

/* ---------- Formatting ---------- */
const formatPrice = (n) =>
  typeof n === 'number'
    ? n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
    : '—';

export function formatChange(pct) {
  if (typeof pct !== 'number' || !Number.isFinite(pct)) return '—';
  const v = Math.round(pct * 100) / 100;
  return `${v < 0 ? '-' : '+'}${Math.abs(v).toFixed(2)}%`;
}

// dates are plain YYYY-MM-DD strings
const utc = (iso) => {
  const [y, m, d] = String(iso).split('-').map(Number);
  return new Date(Date.UTC(y, m - 1, d));
};

export const formatDay = (iso) =>
  iso
    ? utc(iso)
        .toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', timeZone: 'UTC' })
        .replace(',', '')
    : '';

export const formatLong = (iso) =>
  iso
    ? utc(iso)
        .toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' })
        .replace(',', '')
    : '';

export const weekdayLong = (iso) =>
  iso ? utc(iso).toLocaleDateString('en-GB', { weekday: 'long', timeZone: 'UTC' }) : '';

// "18 Nov 2026"
export const formatDate = (iso) =>
  iso
    ? utc(iso).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' })
    : '';


export function nextTradingDay(iso) {
  const dt = utc(iso);
  do {
    dt.setUTCDate(dt.getUTCDate() + 1);
  } while (dt.getUTCDay() === 0 || dt.getUTCDay() === 6);
  return dt.toISOString().slice(0, 10);
}

/* ---------- Shape the Firestore documents for the UI ---------- */

function previousClose(olderPredictions, symbol, closeDate) {
  for (const preds of olderPredictions) {
    const p = preds[symbol];
    if (p && typeof p.close === 'number' && p.date && p.date !== closeDate) return p.close;
  }
  return null;
}

// The "trade" map the pipeline stores next to each prediction (entry, stop, target, risk/reward)
function normaliseTrade(t) {
  if (!t || typeof t !== 'object') return null;
  const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);
  return {
    action: t.action ? String(t.action).toUpperCase() : null,
    conviction: t.conviction ?? null,
    entry: num(t.entry),
    stopLoss: num(t.stop_loss),
    takeProfit: num(t.take_profit),
    risk: num(t.risk),
    reward: num(t.reward),
    riskReward: num(t.risk_reward),
    entryType: t.entry_type ?? null,
    stopSource: t.stop_source ?? null,
    targetSource: t.target_source ?? null,
  };
}

// The "smc" map the pipeline stores next to each prediction: chart drawings for the daily and H1
// charts (zones, liquidity levels, last structure break, sweeps). Anything malformed is dropped,
// and older documents without "smc" simply give null (the chart toggle is then disabled).
function normaliseSmcSet(s) {
  if (!s || typeof s !== 'object') return null;
  const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
  const list = (a, ok) => (Array.isArray(a) ? a.filter((x) => x && typeof x === 'object' && ok(x)) : []);
  return {
    zones: list(s.zones, (z) =>
      isNum(z.top) && isNum(z.bottom) && typeof z.start === 'string' &&
      (z.dir === 'bull' || z.dir === 'bear') && typeof z.kind === 'string'
    ).map((z) => ({
      kind: z.kind,
      dir: z.dir,
      top: z.top,
      bottom: z.bottom,
      start: z.start,
      touches: isNum(z.touches) ? z.touches : 0,
      age: isNum(z.age) ? z.age : 0,
    })),
    levels: list(s.levels, (l) => isNum(l.price) && typeof l.start === 'string' && (l.type === 'bsl' || l.type === 'ssl')),
    structure: list(s.structure, (b) =>
      isNum(b.price) && typeof b.from === 'string' && typeof b.to === 'string' &&
      (b.dir === 'bull' || b.dir === 'bear') && typeof b.type === 'string'
    ),
    sweeps: list(s.sweeps, (w) => isNum(w.price) && typeof w.time === 'string' && (w.dir === 'bull' || w.dir === 'bear')),
  };
}

function normaliseSmc(smc) {
  if (!smc || typeof smc !== 'object') return null;
  return { daily: normaliseSmcSet(smc.daily), h1: normaliseSmcSet(smc.h1) };
}


function normalise(docId, data, olderPredictions) {
  const raw = data.predictions || {};
  const bySymbol = {};

  for (const m of MARKETS) {
    const p = raw[m.symbol];
    const direction = directionOf(p?.bias);
    const hasForecast = Boolean(p) && direction.key !== 'none';

    const confidence = typeof p?.confidence === 'number'
      ? Math.round(p.confidence * 100)
      : null;

    // Individual probabilities for all three market directions
    const probabilities = {
      up: typeof p?.probabilities?.UP === 'number'
        ? Math.round(p.probabilities.UP * 100)
        : null,

      down: typeof p?.probabilities?.DOWN === 'number'
        ? Math.round(p.probabilities.DOWN * 100)
        : null,

      consolidation: typeof p?.probabilities?.CONSOLIDATION === 'number'
        ? Math.round(p.probabilities.CONSOLIDATION * 100)
        : null,
    };

    const close = typeof p?.close === 'number' ? p.close : null;
    const closeDate = p?.date ?? null;

    let change = null;
    if (close !== null && closeDate) {
      const prev = previousClose(olderPredictions, m.symbol, closeDate);
      if (prev) change = (close / prev - 1) * 100;
    }

    bySymbol[m.symbol] = {
  ...m,
  sub: `${m.symbol} · ${m.type}`,
  hasForecast,
  bias: p?.bias ?? null,
  direction,
  dir: direction.key,
  confidence,
  probabilities,
  close,
  closeDate,
  trade: normaliseTrade(p?.trade),
  signals: Array.isArray(p?.signals) ? p.signals : [],
  smc: normaliseSmc(p?.smc),
  candles: Array.isArray(p?.candles) ? p.candles : [],
  candlesH1: Array.isArray(p?.candlesH1) ? p.candlesH1 : [],
  change,
  price: formatPrice(close),
  changeText: formatChange(change),
  label: hasForecast
    ? `${direction.arrow} ${direction.label}${confidence !== null ? ` ${confidence}%` : ''}`
    : 'No forecast',
};
  }

  const closeDates = Object.values(raw)
    .map(p => p?.date)
    .filter(Boolean)
    .sort();

  const closeDate = closeDates.length
    ? closeDates[closeDates.length - 1]
    : null;

  return {
    forecastDate: docId,
    forecastDay: closeDate ? nextTradingDay(closeDate) : docId,
    closeDate,
    generatedAt: data.generated_at?.toDate
      ? data.generated_at.toDate()
      : null,
    bySymbol,
    list: MARKETS.map(m => bySymbol[m.symbol]),
  };
}


/* ---------- Fetch the newest forecast documents ---------- */
// Firestore only answers signed-in users, so wait for Firebase to restore the session first.
const authReady = new Promise((resolve) => {
  const stop = onAuthStateChanged(auth, (user) => {
    stop();
    resolve(user);
  });
});

const CACHE_MS = 10 * 60 * 1000;
let cached = null;
let cachedAt = 0;
let inflight = null;

export function fetchLatestPredictions({ force = false } = {}) {
  if (cached && !force && Date.now() - cachedAt < CACHE_MS) return Promise.resolve(cached);
  if (inflight) return inflight;

  inflight = (async () => {
    await authReady;

    const q = query(collection(db, 'daily_predictions'), orderBy('generated_at', 'desc'), limit(5));
    const snap = await getDocs(q);
    if (snap.empty) return null;

    const [latest, ...older] = snap.docs;
    cached = normalise(
      latest.id,
      latest.data(),
      older.map((d) => d.data().predictions || {})
    );
    cachedAt = Date.now();
    return cached;
  })().finally(() => {
    inflight = null;
  });

  return inflight;
}

/* ---------- Live hourly candles ---------- */
// live_hourly/latest is rewritten every hour by the "Hourly Live Candles" workflow (no model run).
// shape: { symbols: { SPX: { source, fetched_at, candles: [{time, open, high, low, close, partial}] } }, updated_at }
function normaliseLive(data, symbol) {
  const s = data?.symbols?.[symbol];
  if (!s || !Array.isArray(s.candles)) return null;
  const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);
  const candles = s.candles
    .map((c) => ({
      time: typeof c?.time === 'string' ? c.time : null,
      open: num(c?.open), high: num(c?.high), low: num(c?.low), close: num(c?.close),
      partial: Boolean(c?.partial),
    }))
    .filter((c) => c.time && c.open !== null && c.high !== null && c.low !== null && c.close !== null);
  if (!candles.length) return null;
  return {
    candles,
    source: s.source ?? null,
    fetchedAt: typeof s.fetched_at === 'string' ? s.fetched_at : null,
    updatedAt: data.updated_at?.toDate ? data.updated_at.toDate() : null,
  };
}

/** Calls onData(liveOrNull) now and after every hourly update. Returns an unsubscribe function. */
export function subscribeLiveHourly(symbol, onData, onError) {
  let stopped = false;
  let unsub = () => {};
  authReady.then(() => {
    if (stopped) return;
    unsub = onSnapshot(
      doc(db, 'live_hourly', 'latest'),
      (snap) => onData(snap.exists() ? normaliseLive(snap.data(), symbol) : null),
      (err) => { console.warn('Live hourly feed unavailable', err); onError?.(err); }
    );
  });
  return () => { stopped = true; unsub(); };
}