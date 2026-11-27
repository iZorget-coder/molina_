import { doc, getDoc, setDoc } from 'firebase/firestore';
import { db, auth, onAuthStateChanged } from './firebase.js';
import { MARKETS, fetchLatestPredictions, subscribeLiveHourly, subscribeLiveFifteen } from './predictions.js';
import { selectMarket, liveHeaderPrice } from './markets.js';

const DEFAULT_WATCHLIST = ['SPX'];

let cachedSymbols = null;
let authReady = null;

// symbol -> latest live snapshot per feed, exactly like markets.js's `liveData` / `liveFifteen`
const liveBySymbol = new Map();      // hourly feed
const fifteenBySymbol = new Map();   // 15-minute feed
const unsubs = new Map();            // symbol -> [unsubscribe hourly, unsubscribe 15-minute]
const fifteenSeenAt = new Map();      // symbol -> Date.now() when its last 15-minute snapshot arrived

function whenAuthReady() {
  if (authReady) return authReady;
  authReady = new Promise(resolve => {
    const unsub = onAuthStateChanged(auth, user => { unsub(); resolve(user); });
  });
  return authReady;
}

function userRef(uid) {
  return doc(db, 'users', uid);
}

async function fetchWatchlist() {
  if (cachedSymbols) return cachedSymbols;

  const user = await whenAuthReady();
  if (!user) {
    cachedSymbols = [...DEFAULT_WATCHLIST];
    return cachedSymbols;
  }

  const ref = userRef(user.uid);
  const snap = await getDoc(ref);

  if (snap.exists() && Array.isArray(snap.data().watchlist)) {
    cachedSymbols = snap.data().watchlist;
  } else {
    cachedSymbols = [...DEFAULT_WATCHLIST];
    await setDoc(ref, {
      watchlist: cachedSymbols,
      updatedAt: new Date().toISOString(),
    }, { merge: true });
  }
  return cachedSymbols;
}

async function saveWatchlist(symbols) {
  const user = await whenAuthReady();
  cachedSymbols = symbols;
  if (!user) return;
  await setDoc(userRef(user.uid), {
    watchlist: symbols,
    updatedAt: new Date().toISOString(),
  }, { merge: true });
}

/* ---------- Live forecasts (Firestore: daily_predictions) ---------- */

let live = null;
let liveLoadedAt = 0;

async function loadLive() {
  live = await fetchLatestPredictions();
  liveLoadedAt = Date.now();
  return live;
}

/* ---------- Live feeds (same pattern markets.js uses with unsubLive / liveData / liveFifteen) ---------- */

// Same timestamp rule as markets.js: a time with no zone is UTC.
const toMs = (v) => {
  if (typeof v !== 'string') return NaN;
  let s = /^\d{4}-\d{2}-\d{2} /.test(v) ? v.replace(' ', 'T') : v;
  if (/T\d{2}:\d{2}(:\d{2}(\.\d+)?)?$/.test(s)) s += 'Z';
  return Date.parse(s);
};

function newestMs(feed) {
  const times = (feed?.candles || []).map(c => toMs(c.time)).filter(Number.isFinite);
  return times.length ? Math.max(...times) : -Infinity;
}

/**
 * The 15-minute feed drives every price on the cards. The hourly feed only stands in when there is no
 * 15-minute data, or when the 15-minute data is more than an hour behind it.
 */
function freshestFeed(symbol) {
  const hourly = liveBySymbol.get(symbol) || null;
  const fifteen = fifteenBySymbol.get(symbol) || null;
  if (fifteen?.candles?.length && newestMs(fifteen) >= newestMs(hourly) - 3600000) return fifteen;
  return hourly || fifteen;
}

/*
 * One shared render for every view that is on screen. Each live update (from either feed, for any
 * symbol) re-renders the dashboard sidebar, the forecast cards and the watchlist table, whichever of
 * them exist right now. Before, every symbol was subscribed once with whichever view happened to call
 * first, so only that view ever refreshed and it kept the list of symbols it had at that moment.
 */
function renderAllViews() {
  if (!cachedSymbols) return;
  renderDashboardWatchlist(cachedSymbols);
  renderDashboardCards(cachedSymbols);
  renderWatchlistPage(cachedSymbols);
}

let renderQueued = false;
function scheduleRender() {
  if (renderQueued) return;
  renderQueued = true;
  // batches bursts of updates (several markets, two feeds) into one paint
  requestAnimationFrame(() => {
    renderQueued = false;
    renderAllViews();
  });
}

const FIFTEEN_MS = 15 * 60 * 1000;
const STALE_MS = 20 * 60 * 1000;      // a 15-minute snapshot older than this counts as stale
let heartbeat = null;

function subscribeFifteen(sym) {
  return subscribeLiveFifteen(sym, (data) => {
    fifteenBySymbol.set(sym, data);
    fifteenSeenAt.set(sym, Date.now());
    console.debug(`[live] ${sym} 15m snapshot, newest candle`,
      Number.isFinite(newestMs(data)) && newestMs(data) > 0 ? new Date(newestMs(data)).toISOString() : 'none');
    scheduleRender();
  });
}

/**
 * Safety net: the feeds are pushed by Firestore, but a PC that slept or lost its connection can miss
 * a push. Any market whose 15-minute snapshot is older than STALE_MS is resubscribed, which delivers
 * the latest snapshot straight away.
 */
function refreshStaleFeeds() {
  const now = Date.now();
  for (const [sym, subs] of unsubs) {
    if (now - (fifteenSeenAt.get(sym) || 0) > STALE_MS) {
      subs[1]?.();
      subs[1] = subscribeFifteen(sym);
    }
  }
}

/** Runs shortly after every quarter hour (:00, :15, :30, :45): repaint and refresh stale feeds. */
function startHeartbeat() {
  if (heartbeat) return;
  const next = () => FIFTEEN_MS - (Date.now() % FIFTEEN_MS) + 45000;   // 45 s after each quarter hour
  const tick = () => {
    heartbeat = setTimeout(tick, next());
    if (!unsubs.size || document.visibilityState !== 'visible') return;
    refreshStaleFeeds();
    scheduleRender();
  };
  heartbeat = setTimeout(tick, next());
}

function trackLive(symbols) {
  // stop watching symbols that left the watchlist
  for (const [sym, subs] of unsubs) {
    if (!symbols.includes(sym)) {
      subs.forEach(unsub => unsub());
      unsubs.delete(sym);
      liveBySymbol.delete(sym);
      fifteenBySymbol.delete(sym);
      fifteenSeenAt.delete(sym);
    }
  }
  // start watching new ones: hourly AND 15-minute, so prices are as fresh as on the markets page
  for (const sym of symbols) {
    if (unsubs.has(sym)) continue;
    liveBySymbol.set(sym, null);
    fifteenBySymbol.set(sym, null);
    unsubs.set(sym, [
      subscribeLiveHourly(sym, (data) => {
        liveBySymbol.set(sym, data);
        scheduleRender();
      }),
      subscribeFifteen(sym),
    ]);
  }
  if (unsubs.size) startHeartbeat();
}

export function disposeWatchlistLive() {
  for (const subs of unsubs.values()) subs.forEach(unsub => unsub());
  unsubs.clear();
  liveBySymbol.clear();
  fifteenBySymbol.clear();
  fifteenSeenAt.clear();
  clearTimeout(heartbeat);
  heartbeat = null;
}

// Coming back to the tab (a PC tab can sit in the background for hours): refresh anything stale,
// repaint straight away and pick up a newer daily forecast if the page has been open a while.
if (typeof document !== 'undefined' && !window.__molinaWatchlistVisBound) {
  window.__molinaWatchlistVisBound = true;
  document.addEventListener('visibilitychange', async () => {
    if (document.visibilityState !== 'visible' || !cachedSymbols) return;
    refreshStaleFeeds();
    if (Date.now() - liveLoadedAt > 5 * 60 * 1000) {
      try { await loadLive(); } catch (err) { console.error('Could not refresh forecasts', err); }
    }
    renderAllViews();
  });
}

 
function getMarket(symbol) {
  const meta = MARKETS.find(m => m.symbol === symbol);
  if (!meta) return null;

  const base = live?.bySymbol[symbol] || {
    ...meta,
    sub: `${meta.symbol} · ${meta.type}`,
    hasForecast: false,
    dir: 'none',
    label: 'No forecast',
    confidence: null,
    price: '—',
    change: null,
    changeText: '—',
  };

  // Same override the markets header uses: the freshest live candle (15-minute, else hourly) wins once
  // a new session has started past the last daily close. `liveHeaderPrice` reads `close` + `closeDate`.
  const fresh = liveHeaderPrice(base, freshestFeed(symbol));
  return fresh ? { ...base, ...fresh, isLive: true } : base;
}

 
function changeClass(market) {
  if (market.change == null || Math.abs(market.change) < 0.1) return 'flat';
  return market.change > 0 ? 'up' : 'down';
}

/* ---------- Navigation to the Markets view ---------- */

 
function openMarket(symbol) {
  selectMarket(symbol);
  location.hash = 'markets';
}

/* ---------- Watchlist page rendering ---------- */

function watchRowMarkup(market) {
  return `
    <tr
      data-category="${market.dir}"
      data-open-market="${market.symbol}"
      tabindex="0"
      role="link"
      aria-label="View ${market.name} market details"
    >
      <td class="col-market">
        <div class="market__name">${market.name}</div>
        <div class="market__type">${market.sub}</div>
      </td>
      <td class="col-close num">${market.price}</td>
      <td class="col-pred">
        <span class="tag tag--${market.dir}">${market.label}</span>
      </td>
      <td class="col-change num ${changeClass(market)}">${market.changeText}</td>
      <td class="col-actions num">
        <button class="btn-remove" data-remove="${market.symbol}" aria-label="Remove ${market.name} from watchlist">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </td>
    </tr>
  `;
}

export async function loadWatchlistPage() {
  const tbody = document.querySelector('#watchlist-table tbody');
  if (!tbody) return;

  tbody.innerHTML = '<tr><td colspan="5" class="view-loading">Loading watchlist…</td></tr>';

  try {
    const [symbols] = await Promise.all([fetchWatchlist(), loadLive()]);
    trackLive(symbols);
    renderWatchlistPage(symbols);
  } catch (err) {
    console.error(err);
    tbody.innerHTML = '<tr><td colspan="5" class="view-error">Could not load watchlist.</td></tr>';
  }
}

function renderWatchlistPage(symbols) {
  const tbody = document.querySelector('#watchlist-table tbody');
  if (!tbody) return;

  const markets = symbols.map(getMarket).filter(Boolean);

  if (!markets.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="view-empty">Your watchlist is empty. Add a market to get started.</td></tr>';
  } else {
    tbody.innerHTML = markets.map(watchRowMarkup).join('');
  }

  const countEl = document.getElementById('watchlist-count');
  if (countEl) countEl.textContent = `${markets.length} ${markets.length === 1 ? 'market' : 'markets'}`;
}

/* ---------- Dashboard sidebar rendering ---------- */

function dashboardItemMarkup(market) {
  return `
    <li class="watch__item" data-market="${market.symbol}">
      <div class="watch__row">
        <div>
          <div class="watch__name">${market.name}</div>
          <div class="watch__sub">${market.sub}</div>
        </div>
        <div class="watch__price">${market.price}</div>
      </div>
      <div class="watch__row">
        <span class="badge badge--${market.dir}">${market.label}</span>
        <span class="change ${changeClass(market)}">${market.changeText}</span>
      </div>
    </li>
  `;
}

export async function loadDashboardWatchlist() {
  const list = document.querySelector('.watch');
  if (!list) return;

  try {
    const [symbols] = await Promise.all([fetchWatchlist(), loadLive()]);
    trackLive(symbols);
    renderDashboardWatchlist(symbols);
  } catch (err) {
    console.error(err);
    list.innerHTML = '<li class="watch__item"><p class="view-error">Could not load watchlist.</p></li>';
  }
}

function renderDashboardWatchlist(symbols) {
  const list = document.querySelector('.watch');
  if (!list) return;

  const markets = symbols.map(getMarket).filter(Boolean);

  list.innerHTML = markets.length
    ? markets.map(dashboardItemMarkup).join('')
    : '<li class="watch__item"><p class="view-empty">No markets yet.</p></li>';

  const countEl = document.querySelector('.sidebar__head .muted');
  if (countEl) countEl.textContent = `${markets.length} ${markets.length === 1 ? 'market' : 'markets'}`;
}

/* ---------- Dashboard forecast cards ---------- */

/* 5-point sparkline built from the last 5 daily closes (oldest -> newest) */

const SPARK_W = 200;       // viewBox width
const SPARK_H = 56;        // viewBox height
const SPARK_PAD_X = 3;     // keeps the line and its round caps inside the box
const SPARK_PAD_Y = 7;     // keeps highs/lows off the top/bottom edge
const SPARK_MAX_GAP_DAYS = 14;

function lastCloses(market, n = 5) {
  const rows = (Array.isArray(market?.candles) ? market.candles : [])
    .map(c => ({ t: Date.parse(c?.date), close: Number(c?.close) }))
    .filter(c => Number.isFinite(c.t) && Number.isFinite(c.close) && c.close > 0)
    .sort((a, b) => a.t - b.t);

  let start = rows.length - 1;
  while (start > 0 && rows[start].t - rows[start - 1].t <= SPARK_MAX_GAP_DAYS * 86400000) start--;

  return rows.slice(start).slice(-n).map(c => c.close);
}

function sparkPolyline(market) {
  const closes = lastCloses(market, 5);
  if (closes.length < 2) return '';

  const lo = Math.min(...closes);
  const hi = Math.max(...closes);
  const span = hi - lo;

  return closes
    .map((v, i) => {
      const x = SPARK_PAD_X + (i / (closes.length - 1)) * (SPARK_W - SPARK_PAD_X * 2);
      const y = span === 0
        ? SPARK_H / 2
        : SPARK_PAD_Y + (1 - (v - lo) / span) * (SPARK_H - SPARK_PAD_Y * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
}

function cardMarkup(market) {
  const pillLabel = {
    up: '▲ UP',
    down: '▼ DOWN',
    flat: '◆ CONSOLIDATION'
  }[market.dir] || 'NO FORECAST';

  const confidence = market.confidence ?? 0;
  const confidenceText = market.confidence === null
    ? '—'
    : `${market.confidence}%`;

  const sparkDir = market.dir === 'none' ? 'flat' : market.dir;
  const sparkPoints = sparkPolyline(market);

  const sparkSvg = (extra = '') => sparkPoints
    ? `<svg viewBox="0 0 200 56" preserveAspectRatio="none" class="spark spark--${sparkDir} ${extra}" aria-hidden="true">
         <polyline points="${sparkPoints}"/>
       </svg>`
    : '';

  return `
    <article
      class="card"
      data-market="${market.symbol}"
      tabindex="0"
      role="link"
      aria-label="View ${market.name} market details"
    >
      <div class="card__id">
        <div class="card__title">${market.name} <span class="ticker">${market.symbol}</span></div>
        <div class="card__signal">
          <span class="pill pill--${market.dir}">${pillLabel}</span>
          ${sparkSvg('spark--mini')}
        </div>
      </div>
      <div class="card__conf">
        <div class="conf__row"><span class="conf__label">Confidence</span><span class="conf__value">${confidenceText}</span></div>
        <div class="progress"><div class="progress__bar" style="width:${confidence}%"></div></div>
        <div class="conf__row conf__row--small">
          <span class="muted-light">${market.isLive ? 'Latest' : 'Last close'} <strong>${market.price}</strong></span>
          <span class="${changeClass(market)}">${market.changeText}</span>
        </div>
      </div>
      <div class="card__chart">
        ${sparkSvg()}
        <span class="chart__label">${sparkPoints ? 'Last 5 sessions' : 'No recent price data'}</span>
      </div>
    </article>
  `;
}

export async function loadDashboardCards() {
  const container = document.getElementById('dashboard-cards');
  if (!container) return;

  try {
    const [symbols] = await Promise.all([fetchWatchlist(), loadLive()]);
    trackLive(symbols);
    renderDashboardCards(symbols);
  } catch (err) {
    console.error(err);
    container.innerHTML = '<p class="view-error">Could not load forecasts.</p>';
  }
}

function renderDashboardCards(symbols) {
  const container = document.getElementById('dashboard-cards');
  if (!container) return;

  const markets = symbols.map(getMarket).filter(Boolean);

  if (!markets.length) {
    container.innerHTML = '<p class="view-empty">Your watchlist is empty. Add a market to see forecasts.</p>';
    return;
  }

  container.innerHTML = markets.map(cardMarkup).join('');
}

/* ---------- Add-market modal ---------- */

/**
 * The add-market dialog is static HTML. Any market in MARKETS that has no row yet (the FX pairs and
 * Bitcoin) gets one, cloned from the first row so it looks identical. DXY is left out (no model).
 */
function ensureModalRows(dialog) {
  const existing = dialog.querySelectorAll('.add-modal__row');
  const tpl = existing[0];
  const tplMeta = tpl && MARKETS.find(m => m.symbol === tpl.dataset.symbol);
  if (!tpl || !tplMeta) return;

  const have = new Set(Array.from(existing, r => r.dataset.symbol));
  let after = existing[existing.length - 1];
  const swap = (html, from, to) => html.split(from).join(to).split(from.replace(/&/g, '&amp;')).join(to.replace(/&/g, '&amp;'));

  MARKETS.filter(m => m.symbol !== 'DXY' && !have.has(m.symbol)).forEach(m => {
    const row = tpl.cloneNode(true);
    let html = row.innerHTML;
    html = swap(html, tplMeta.name, m.name);
    html = swap(html, tplMeta.symbol, m.symbol);
    html = swap(html, tplMeta.type, m.type);
    row.innerHTML = html;
    row.dataset.symbol = m.symbol;
    const input = row.querySelector('input');
    if (input) { input.value = m.symbol; input.checked = false; input.disabled = false; }
    after.insertAdjacentElement('afterend', row);
    after = row;
  });
}

function renderModalRows(dialog, currentSymbols) {
  ensureModalRows(dialog);
  const rows = dialog.querySelectorAll('.add-modal__row');
  rows.forEach(row => {
    const symbol = row.dataset.symbol;
    const input = row.querySelector('input');
    input.disabled = currentSymbols.includes(symbol);
    input.checked = false;

    const market = getMarket(symbol);
    const priceEl = row.querySelector('.add-modal__price');
    const tagEl = row.querySelector('.add-modal__tag');
    if (market && priceEl) priceEl.textContent = market.price;
    if (market && tagEl) {
      tagEl.textContent = market.label;
      tagEl.className = `add-modal__tag add-modal__tag--${market.dir}`;
    }
  });

  const empty = dialog.querySelector('.add-modal__empty');
  const allAdded = rows.length > 0 &&
    Array.from(rows).every(row => currentSymbols.includes(row.dataset.symbol));
  if (empty) empty.hidden = !allAdded;

  const countEl = dialog.querySelector('[data-count]');
  if (countEl) countEl.textContent = '0';

  const addBtn = dialog.querySelector('[data-add-selected]');
  if (addBtn) addBtn.disabled = true;
}

function updateSelectionCount(dialog) {
  const checked = dialog.querySelectorAll('.add-modal__row input:checked:not(:disabled)');
  const countEl = dialog.querySelector('[data-count]');
  if (countEl) countEl.textContent = String(checked.length);
  const addBtn = dialog.querySelector('[data-add-selected]');
  if (addBtn) addBtn.disabled = checked.length === 0;
}

export async function openAddMarketModal() {
  const dialog = document.getElementById('add-market-modal');
  if (!dialog) return;

  const [currentSymbols] = await Promise.all([
    fetchWatchlist(),
    loadLive().catch(err => console.error('Could not load forecasts for the modal', err)),
  ]);
  renderModalRows(dialog, currentSymbols);

  dialog.showModal();

  if (!dialog.dataset.bound) {
    dialog.dataset.bound = '1';

    dialog.addEventListener('change', e => {
      if (e.target.matches('input[name="market"]')) updateSelectionCount(dialog);
    });

    dialog.addEventListener('click', e => {
      if (e.target.closest('[data-close]')) {
        dialog.close();
        return;
      }

      if (e.target.closest('[data-add-selected]')) {
        const selected = Array.from(
          dialog.querySelectorAll('input[name="market"]:checked:not(:disabled)')
        ).map(input => input.value);

        if (!selected.length) return;

        const merged = Array.from(new Set([...(cachedSymbols || []), ...selected]));
        saveWatchlist(merged)
          .then(() => {
            dialog.close();
            loadWatchlistPage();
            loadDashboardWatchlist();
            loadDashboardCards();
          })
          .catch(err => console.error('Could not save watchlist', err));
      }
    });
  }
}

/* ---------- Page + dashboard binding ---------- */

export function bindWatchlistPage() {
  loadWatchlistPage();

  const tbody = document.querySelector('#watchlist-table tbody');
  if (tbody && !tbody.dataset.bound) {
    tbody.dataset.bound = '1';

    tbody.addEventListener('click', e => {
       
      const btn = e.target.closest('[data-remove]');
      if (btn) {
        const symbol = btn.dataset.remove;
        const next = (cachedSymbols || []).filter(s => s !== symbol);
        saveWatchlist(next)
          .then(() => {
            loadWatchlistPage();
            loadDashboardWatchlist();
            loadDashboardCards();
          })
          .catch(err => console.error('Could not remove market', err));
        return;
      }

       
      const row = e.target.closest('[data-open-market]');
      if (row) openMarket(row.dataset.openMarket);
    });

     
    tbody.addEventListener('keydown', e => {
      if (e.key !== 'Enter' && e.key !== ' ') return;
      if (e.target.closest('[data-remove]')) return;  
      const row = e.target.closest('[data-open-market]');
      if (!row) return;
      e.preventDefault();
      openMarket(row.dataset.openMarket);
    });
  }

  const addBtn = document.querySelector('[data-open="add-market-modal"]');
  if (addBtn && !addBtn.dataset.bound) {
    addBtn.dataset.bound = '1';
    addBtn.addEventListener('click', openAddMarketModal);
  }
}

export function bindDashboardWatchlist() {
  loadDashboardWatchlist();
}