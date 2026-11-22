// javascript/zones.js — shared zone colours + helpers
export const ZONE_COLORS = {
  fvg: '#f5a524',      // yellowish orange
  breaker: '#ef4444',  // red
  ifvg: '#a855f7',     // purple
  ob: '#3b82f6',       // blue
  sweep: '#ec4899',    // pink
  bos: '#14b8a6',      // teal
  choch: '#84cc16',    // lime
  liq: '#94a3b8',      // slate (equal highs/lows, anything unknown)
};

// "Bull Breaker", "Bear IFVG", "Sweep Low", "BOS Bull", "H1 FVG edge limit" -> zone key
export function zoneKey(label = '') {
  const s = String(label).toLowerCase();
  if (s.includes('ifvg')) return 'ifvg';          // must be checked before 'fvg'
  if (s.includes('fvg')) return 'fvg';
  if (s.includes('breaker')) return 'breaker';
  if (s.includes('choch')) return 'choch';
  if (s.includes('bos')) return 'bos';
  if (s.includes('sweep')) return 'sweep';
  if (s.includes('order block') || /\bob\b/.test(s)) return 'ob';
  return 'liq';
}

export function zoneTag(label) {
  const el = document.createElement('span');
  el.className = 'zone-tag';
  el.dataset.zone = zoneKey(label);
  el.textContent = label;
  return el;
}

// Fill a container with one tag per signal (e.g. predictions[market].signals)
export function renderSignals(container, signals = []) {
  container.replaceChildren(...signals.map(zoneTag));
}

// "r,g,b" string for building rgba(...) with your own alpha
export function zoneRgb(key) {
  const hex = ZONE_COLORS[key] ?? ZONE_COLORS.liq;
  return [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16)).join(',');
}

// Chart overlays: translucent fill + solid edge, dashed for bearish zones
export function zoneStyle(key, bearish = false) {
  const hex = ZONE_COLORS[key] ?? ZONE_COLORS.liq;
  const [r, g, b] = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
  return { fill: `rgba(${r}, ${g}, ${b}, 0.18)`, stroke: hex, dash: bearish ? [6, 4] : [] };
}