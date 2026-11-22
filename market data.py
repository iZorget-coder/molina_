import pandas as pd
import numpy as np
import os
import re
import time
import json
import asyncio
from pathlib import Path
from datetime import date, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=Warning, message='.*Glyph.*missing.*')
warnings.filterwarnings('ignore', category=Warning, message='.*highly fragmented.*')

# ============================================
# 0. SETUP — MULTI-INDEX CONFIG
# ============================================

INDEX_CONFIGS = [
    ("SP500_1d_data.csv",  "S&P 500"),
    ("NASDAQ_1d_data.csv", "NASDAQ"),
    ("DOW_1d_data.csv",    "DOW JONES"),
    ("XAUUSD_1d_data.csv", "GOLD (XAUUSD)"),
]


MARKET_KEY_MAP = {
    "S&P 500":       "SPX",
    "NASDAQ":        "NDX",
    "DOW JONES":     "DJI",
    "GOLD (XAUUSD)": "XAUUSD",
}

DXY_CSV_PATH = "DXY_1d_data.csv"

HOURLY_CONFIGS = {
    "S&P 500":       "SP500_index_H1.csv",
    "NASDAQ":        "NASDAQ100_index_H1.csv",
    "DOW JONES":     "DOW_index_H1.csv",
    "GOLD (XAUUSD)": "XAUUSD_H1.csv",
}

# H1 learned-model fields (per-symbol tv_symbol, csv_symbol_tag)
HOURLY_FALLBACK = {
    "S&P 500":       ("CME_MINI:ES1!",  "USA500"),
    "NASDAQ":        ("CME_MINI:NQ1!",  "USATECH"),
    "DOW JONES":     ("CBOT_MINI:YM1!", "USA30"),
    "GOLD (XAUUSD)": ("OANDA:XAUUSD",   "XAUUSD"),
}

HOURLY_OPT_ENABLED         = True
H1_MODEL_ENABLED           = True

# ---- Intraday H1 trade model (all distances in H1 ATR unless noted) ----
H1_SWING_LEFT              = 3
H1_SWING_RIGHT             = 3
H1_SWING_LOOKBACK          = 72      # bars back to look for "current" swing highs/lows
H1_ZONE_MAX_AGE            = 36      # zone must have become visible within this many bars
H1_ZONE_MIN_ATR            = 0.10    # ignore FVGs smaller than this
H1_IFVG_WINDOW             = 60
H1_BREAKER_WINDOW          = 30
H1_MAX_ENTRY_DIST_ATR      = 1.5     # entry at most this far from the current close
H1_MIN_RISK_ATR            = 0.4
H1_MAX_RISK_ATR            = 2.0
H1_STOP_BUF_ATR            = 0.10
H1_TARGET_BUF_ATR          = 0.05
H1_MIN_RR                  = 1.0
H1_MAX_RR                  = 4.0
H1_TOP_ZONES               = 5       # nearest N valid zones per decision bar
H1_VALID_BARS              = 8       # limit order lifetime
H1_HOLD_BARS               = 16      # max bars in trade after fill
H1_WARMUP                  = 80
H1_SAMPLE_EVERY            = 3
H1_MIN_ROWS                = 1500
H1_MIN_VAL_GROUPS          = 150
H1_MIN_EDGE_R              = 0.05    # model-picked avg R must beat avg-candidate avg R by this
H1_MIN_EV_R                = 0.05    # live: minimum expected R to take a trade
H1_TRAIN_FRACTION          = 0.70

KILL_ZONES_UTC = {
    "London KZ":     (7, 10),
    "NY AM KZ":      (12, 15),
    "Silver Bullet": (14, 15),
}

USE_JOURNAL = True
JOURNAL_DIR = "journal"
JOURNAL_HOLD_BARS = 1

CAL_MIN_TRADES = 60
CAL_TRAIN_FRACTION = 0.7
CAL_MIN_IMPROVEMENT = 0.02
CAL_COMPLEXITY_PENALTY = 0.01
CAL_SHIFT_GRID = [0.0, 0.10, 0.20, 0.35, 0.50]
CAL_EXTRA_STOP_GRID = [0.0, 0.25, 0.50, 1.00, 1.50]
CAL_TARGET_SCALE_GRID = [0.5, 0.75, 1.0, 1.5]

META_MIN_TRADES = 150
META_MIN_AUC = 0.55
META_MIN_EXPECTED_R = 0.0

os.makedirs(JOURNAL_DIR, exist_ok=True)


def _normalize_path(raw: str) -> str:
    cleaned = "".join(" " if c.isspace() else c for c in raw)
    return " ".join(cleaned.split())


def _parse_dates(series):
    formats = ["%b %d, %Y", "%d %b %Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]
    for fmt in formats:
        try:
            return pd.to_datetime(series, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.to_datetime(series, errors='coerce')


# ============================================
# DXY LOADER
# ============================================

def load_dxy_series(csv_path: str):
    if not os.path.exists(csv_path):
        print(f"ℹ️  DXY file not found at {csv_path} — DXY features will be disabled")
        return None
    try:
        dxy = pd.read_csv(csv_path)
        dxy['Date'] = _parse_dates(dxy['Date'])
        dxy = dxy.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
        close_col = None
        for candidate in ['Close', 'DXY_Close', 'Price', 'close']:
            if candidate in dxy.columns:
                close_col = candidate
                break
        if close_col is None:
            print(f"⚠️  DXY CSV missing Close column — DXY features disabled")
            return None
        dxy = dxy[['Date', close_col]].rename(columns={close_col: 'DXY_Close'})
        dxy['DXY_Close'] = pd.to_numeric(
            dxy['DXY_Close'].astype(str).str.replace(',', ''), errors='coerce')
        dxy = dxy.dropna(subset=['DXY_Close']).reset_index(drop=True)
        dxy['DXY_Return'] = dxy['DXY_Close'].pct_change()
        dxy['DXY_MA20'] = dxy['DXY_Close'].rolling(20).mean()
        dxy['DXY_vs_MA20'] = dxy['DXY_Close'] / dxy['DXY_MA20'] - 1
        _delta = dxy['DXY_Close'].diff()
        _gain = _delta.where(_delta > 0, 0).rolling(14).mean()
        _loss = -_delta.where(_delta < 0, 0).rolling(14).mean()
        _rs = _gain / _loss
        dxy['DXY_RSI'] = 100 - (100 / (1 + _rs))
        dxy['DXY_Return_Lag1'] = dxy['DXY_Return'].shift(1)
        dxy['DXY_Return_Lag5'] = dxy['DXY_Close'].pct_change(5)
        print(f"\n💵 DXY loaded: {len(dxy)} rows "
              f"({dxy['Date'].min().date()} → {dxy['Date'].max().date()})")
        return dxy
    except Exception as e:
        print(f"⚠️  Failed to load DXY ({e}) — DXY features will be disabled")
        return None


# ============================================
# HOURLY DATA — LOADING
# ============================================

def load_hourly_csv(csv_path: str):
    if not os.path.exists(csv_path):
        return None
    try:
        h = pd.read_csv(csv_path)
        needed = {"timestamp", "open", "high", "low", "close"}
        if not needed.issubset(h.columns):
            return None
        h = h.drop_duplicates(subset=["timestamp", "open", "high", "low", "close"],
                              keep="first").reset_index(drop=True)
        h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True, errors="coerce")
        h = h.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        for c in ("open", "high", "low", "close"):
            h[c] = pd.to_numeric(h[c], errors="coerce")
        h["volume"] = pd.to_numeric(h.get("volume"), errors="coerce") if "volume" in h else 0.0
        h = h.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
        h["hour_utc"] = h["timestamp"].dt.hour
        h["date_utc"] = h["timestamp"].dt.date
        return h
    except Exception:
        return None


def _session_from_hour(hour: int) -> str:
    if 0 <= hour < 7:
        return "Asia"
    elif 7 <= hour < 12:
        return "London"
    elif 12 <= hour < 21:
        return "New York"
    return "Asia (late)"


def _kill_zone_name(hour_utc: int):
    for name, (start, end) in KILL_ZONES_UTC.items():
        if start <= hour_utc < end:
            return name
    return "none"


# ============================================
# INTRADAY H1 TRADE MODEL
#   Causal zones -> simulated limit-order candidates (zone x entry x stop x target)
#   -> P(fill) and E[R | filled] forests -> pick best expected-value setup.
#   Replaces the old swing-sized H1 zone model.
# ============================================

KINDS = ["fvg", "ob", "ifvg", "breaker", "sweep"]
SESSIONS = ["asia", "london", "newyork", "late"]

FEATURE_COLUMNS = (
    [f"k_{k}" for k in KINDS] + [f"s_{s}" for s in SESSIONS] +
    ["kz_london", "kz_nyam", "kz_sb", "hour_sin", "hour_cos",
     "zone_age", "zone_form_age", "zone_width_atr", "zone_disp", "inside_now", "confluence_n",
     "e_frac", "entry_dist_atr", "stop_swing", "risk_atr",
     "target_rank", "reward_atr", "rr", "target_eq",
     "trend_aligned", "bos_aligned", "bars_since_bos", "range_pos",
     "mom3", "mom12", "recent_range_atr", "atr_pct", "side_sign",
     "prob_up", "prob_down", "prob_cons", "macro_aligned", "dealing_fav",
     "d_pdh", "d_pdl", "in_daily_zone", "daily_zone_dist_atr",
     "d_sweep", "d_bos", "d_choch", "d_breaker", "d_ifvg"]
)
COL = {c: i for i, c in enumerate(FEATURE_COLUMNS)}


def _sess(hr: int) -> str:
    if 0 <= hr < 7:
        return "asia"
    if 7 <= hr < 12:
        return "london"
    if 12 <= hr < 21:
        return "newyork"
    return "late"


# ============================================
# H1 PREP: swings, structure, causal zones
# ============================================
def prep_hourly(h: pd.DataFrame) -> dict:
    t0 = time.time()
    o = h["open"].to_numpy(float); hi = h["high"].to_numpy(float)
    lo = h["low"].to_numpy(float); c = h["close"].to_numpy(float)
    n = len(h)
    prev_c = np.r_[c[0], c[:-1]]
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_c), np.abs(lo - prev_c)))
    atr = pd.Series(tr).rolling(14, min_periods=5).mean().bfill().to_numpy()
    body = c - o
    avg_body = pd.Series(np.abs(body)).rolling(20).mean().shift(1).to_numpy()
    disp = np.abs(body) > 1.5 * np.nan_to_num(avg_body, nan=np.inf)
    P = {"n": n, "o": o, "h": hi, "l": lo, "c": c, "atr": atr, "body": body,
         "avg_body": avg_body, "disp": disp,
         "hour": h["hour_utc"].to_numpy(int),
         "date": h["timestamp"].dt.tz_convert(None).to_numpy().astype("datetime64[D]")}

    sh, sl = [], []
    for i in range(H1_SWING_LEFT, n - H1_SWING_RIGHT):
        if hi[i] == hi[i - H1_SWING_LEFT:i + H1_SWING_RIGHT + 1].max():
            sh.append((i, hi[i]))
        if lo[i] == lo[i - H1_SWING_LEFT:i + H1_SWING_RIGHT + 1].min():
            sl.append((i, lo[i]))
    for name, lst in (("sh", sh), ("sl", sl)):
        P[name + "_idx"] = np.array([x[0] for x in lst], dtype=int)
        P[name + "_px"] = np.array([x[1] for x in lst], dtype=float)
        P[name + "_conf"] = P[name + "_idx"] + H1_SWING_RIGHT   # known only after `right` bars

    _structure(P)
    P["Z"] = _detect_zones(P)
    nz = sum(len(v) for v in P["Z"].values())
    print(f"   ✅ H1 prep: {n} bars, {nz} causal zones ({time.time() - t0:.1f}s)")
    return P


def _structure(P: dict):
    n, c = P["n"], P["c"]
    trend = np.zeros(n, int); bos_dir = np.zeros(n, int); since = np.full(n, 999)
    cur_sh = cur_sl = np.nan
    ksh = ksl = 0
    tr_ = 0; last_bar = -10 ** 9; last_dir = 0
    for i in range(n):
        while ksh < len(P["sh_conf"]) and P["sh_conf"][ksh] <= i:
            cur_sh = P["sh_px"][ksh]; ksh += 1
        while ksl < len(P["sl_conf"]) and P["sl_conf"][ksl] <= i:
            cur_sl = P["sl_px"][ksl]; ksl += 1
        if not np.isnan(cur_sh) and c[i] > cur_sh:
            tr_ = 1; last_bar = i; last_dir = 1; cur_sh = np.nan
        elif not np.isnan(cur_sl) and c[i] < cur_sl:
            tr_ = -1; last_bar = i; last_dir = -1; cur_sl = np.nan
        trend[i] = tr_; bos_dir[i] = last_dir; since[i] = min(i - last_bar, 999)
    P["trend"], P["bos_dir"], P["since"] = trend, bos_dir, since


def _first_close_through(c, start, end, level, below):
    end = min(end, len(c))
    if start >= end:
        return None
    seg = c[start:end]
    m = seg < level if below else seg > level
    return start + int(m.argmax()) if m.any() else None


def _detect_zones(P: dict) -> dict:
    h, l, c, atr, body = P["h"], P["l"], P["c"], P["atr"], P["body"]
    avg, disp, n = P["avg_body"], P["disp"], P["n"]
    raw = []

    def add(kind, d, top, bottom, formation, known, displacement=0.0):
        raw.append({"kind": kind, "dir": d, "top": float(top), "bottom": float(bottom),
                    "mid": float((top + bottom) / 2), "formation": int(formation),
                    "known": int(known), "disp": float(displacement), "inv": None})

    # FVG + IFVG (inversion = first close through the far edge)
    for i in range(2, n):
        a = atr[i]
        if l[i] > h[i - 2] and (l[i] - h[i - 2]) >= H1_ZONE_MIN_ATR * a:
            top, bot = l[i], h[i - 2]
            add("fvg", "bull", top, bot, i, i)
            j = _first_close_through(c, i + 1, i + 1 + H1_IFVG_WINDOW, bot, True)
            if j is not None:
                add("ifvg", "bear", top, bot, i, j)
        if h[i] < l[i - 2] and (l[i - 2] - h[i]) >= H1_ZONE_MIN_ATR * a:
            top, bot = l[i - 2], h[i]
            add("fvg", "bear", top, bot, i, i)
            j = _first_close_through(c, i + 1, i + 1 + H1_IFVG_WINDOW, top, False)
            if j is not None:
                add("ifvg", "bull", top, bot, i, j)

    # Order blocks + breakers (OB that later fails by a close through it)
    for i in range(21, n):
        if not disp[i]:
            continue
        ob = i - 1
        if body[i] > 0 and body[ob] < 0:
            d = "bull"
        elif body[i] < 0 and body[ob] > 0:
            d = "bear"
        else:
            continue
        top, bot = h[ob], l[ob]
        add("ob", d, top, bot, ob, i, abs(body[i]) / avg[i])
        if d == "bull":
            j = _first_close_through(c, i + 1, i + 1 + H1_BREAKER_WINDOW, bot, True)
            if j is not None:
                add("breaker", "bear", top, bot, ob, j)
        else:
            j = _first_close_through(c, i + 1, i + 1 + H1_BREAKER_WINDOW, top, False)
            if j is not None:
                add("breaker", "bull", top, bot, ob, j)

    # Liquidity sweeps of confirmed, still-unbroken swings
    cur_sh = cur_sl = np.nan
    ksh = ksl = 0
    for i in range(n):
        while ksh < len(P["sh_conf"]) and P["sh_conf"][ksh] <= i - 1:
            cur_sh = P["sh_px"][ksh]; ksh += 1
        while ksl < len(P["sl_conf"]) and P["sl_conf"][ksl] <= i - 1:
            cur_sl = P["sl_px"][ksl]; ksl += 1
        if not np.isnan(cur_sh) and h[i] > cur_sh:
            if c[i] < cur_sh:
                add("sweep", "bear", h[i], cur_sh, i, i)
            cur_sh = np.nan
        if not np.isnan(cur_sl) and l[i] < cur_sl:
            if c[i] > cur_sl:
                add("sweep", "bull", cur_sl, l[i], i, i)
            cur_sl = np.nan

    Z = {}
    for z in raw:
        if z["dir"] == "bull":
            z["inv"] = _first_close_through(c, z["known"] + 1, z["known"] + H1_ZONE_MAX_AGE + 2,
                                            z["bottom"], True)
        else:
            z["inv"] = _first_close_through(c, z["known"] + 1, z["known"] + H1_ZONE_MAX_AGE + 2,
                                            z["top"], False)
        Z.setdefault(z["known"], []).append(z)
    return Z


def _live_swings(P: dict, t: int, kind: str) -> list:
    """Confirmed, still-unbroken swings as of bar t. kind: 'sh' or 'sl'."""
    idx, px, conf = P[kind + "_idx"], P[kind + "_px"], P[kind + "_conf"]
    arr = P["h"] if kind == "sh" else P["l"]
    a = np.searchsorted(idx, t - H1_SWING_LOOKBACK, side="left")
    b = np.searchsorted(conf, t, side="right")
    out = []
    for k in range(a, b):
        seg = arr[idx[k] + 1:t + 1]
        if len(seg) and (seg.max() >= px[k] if kind == "sh" else seg.min() <= px[k]):
            continue
        out.append((int(idx[k]), float(px[k])))
    return out


# ============================================
# DAILY CONTEXT
# ============================================
def daily_base_ctx(frame: pd.DataFrame) -> dict:
    def col(name, default=np.nan):
        return frame[name].to_numpy(float) if name in frame.columns else np.full(len(frame), default)
    return {
        "dates": frame["Date"].to_numpy().astype("datetime64[D]"),
        "dealing_pos": col("Dealing_Pos", 0.5), "macro": col("Macro_Structure", 0.0),
        "pdh": col("Dist_To_PDH", 0.0), "pdl": col("Dist_To_PDL", 0.0), "atr": col("ATR_14"),
        "fvg_bull_top": col("FVG_Bull_Top"), "fvg_bull_bot": col("FVG_Bull_Bottom"),
        "fvg_bear_top": col("FVG_Bear_Top"), "fvg_bear_bot": col("FVG_Bear_Bottom"),
        "ob_bull_top": col("Bull_OB_Body_Top"), "ob_bull_bot": col("Bull_OB_Wick_Bottom"),
        "ob_bear_top": col("Bear_OB_Wick_Top"), "ob_bear_bot": col("Bear_OB_Body_Bottom"),
        "sweep_low": col("Liquidity_Sweep_Low", 0.0), "sweep_high": col("Liquidity_Sweep_High", 0.0),
        "bos_bull": col("BOS_Bull", 0.0), "bos_bear": col("BOS_Bear", 0.0),
        "choch": col("CHoCH", 0.0),
        # Your daily flags are named from the zone's point of view: Bull_BB / Bull_IFVG mean a bull
        # OB / bull FVG was closed THROUGH (bearish inversion). So bullish breaker/IFVG = Bear_*.
        "bb_bull": col("Bear_BB", 0.0), "bb_bear": col("Bull_BB", 0.0),
        "ifvg_bull": col("Bear_IFVG", 0.0), "ifvg_bear": col("Bull_IFVG", 0.0),
    }


def daily_ctx_at(B: dict, i: int, probs: dict, side: str) -> dict:
    long = side == "LONG"
    nz = lambda v: 0.0 if v is None or np.isnan(v) else float(v)
    if long:
        ranges = [(B["fvg_bull_bot"][i], B["fvg_bull_top"][i]), (B["ob_bull_bot"][i], B["ob_bull_top"][i])]
    else:
        ranges = [(B["fvg_bear_bot"][i], B["fvg_bear_top"][i]), (B["ob_bear_bot"][i], B["ob_bear_top"][i])]
    return {
        "prob_up": float(probs.get("UP", 0.33)), "prob_down": float(probs.get("DOWN", 0.33)),
        "prob_cons": float(probs.get("CONSOLIDATION", 0.34)),
        "dealing_pos": nz(B["dealing_pos"][i]) if not np.isnan(B["dealing_pos"][i]) else 0.5,
        "macro": nz(B["macro"][i]), "pdh": nz(B["pdh"][i]), "pdl": nz(B["pdl"][i]),
        "d_atr": float(B["atr"][i]) if not np.isnan(B["atr"][i]) else np.nan,
        "ranges": ranges,
        "d_sweep": nz(B["sweep_low"][i] if long else B["sweep_high"][i]),
        "d_bos": nz(B["bos_bull"][i] if long else B["bos_bear"][i]),
        "d_choch": nz(B["choch"][i]),
        "d_breaker": nz(B["bb_bull"][i] if long else B["bb_bear"][i]),
        "d_ifvg": nz(B["ifvg_bull"][i] if long else B["ifvg_bear"][i]),
    }


def oof_daily_probs(daily_df: pd.DataFrame, feature_columns: list, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold daily probabilities [DOWN, CONS, UP]. The production daily model was fit on
    these same rows, so its in-sample probabilities are far too confident to train H1 on."""
    X = daily_df[feature_columns].to_numpy(float)
    y = daily_df["Target"].to_numpy(int)
    out = np.full((len(X), 3), np.nan)
    for tr, te in TimeSeriesSplit(n_splits=n_splits).split(X):
        sc = StandardScaler().fit(X[tr])
        m = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=20,
                                   class_weight="balanced", random_state=42, n_jobs=-1)
        m.fit(sc.transform(X[tr]), y[tr])
        p = m.predict_proba(sc.transform(X[te]))
        for k, cls in enumerate(m.classes_):
            out[te, int(cls)] = p[:, k]
    print(f"   Out-of-fold daily probs ready ({int((~np.isnan(out[:, 0])).sum())}/{len(X)} rows)")
    return out


# ============================================
# CANDIDATE GENERATION
# ============================================
def _features(P, t, side, z, E, S, T, rr, e_frac, s_style, rank, is_eq, ctx, confl):
    sg = 1.0 if side == "LONG" else -1.0
    c, atr = P["c"], P["atr"][t]
    c_t = c[t]; hr = int(P["hour"][t])
    f = {}
    for k in KINDS:
        f["k_" + k] = float(z["kind"] == k)
    ss = _sess(hr)
    for s in SESSIONS:
        f["s_" + s] = float(ss == s)
    f["kz_london"] = float(7 <= hr < 10); f["kz_nyam"] = float(12 <= hr < 15)
    f["kz_sb"] = float(14 <= hr < 15)
    f["hour_sin"] = float(np.sin(2 * np.pi * hr / 24)); f["hour_cos"] = float(np.cos(2 * np.pi * hr / 24))
    f["zone_age"] = float(t - z["known"]); f["zone_form_age"] = float(t - z["formation"])
    f["zone_width_atr"] = (z["top"] - z["bottom"]) / atr
    f["zone_disp"] = z.get("disp", 0.0)
    f["inside_now"] = float(z["bottom"] <= c_t <= z["top"])
    f["confluence_n"] = float(confl)
    f["e_frac"] = e_frac; f["entry_dist_atr"] = abs(c_t - E) / atr
    f["stop_swing"] = float(s_style == "swing"); f["risk_atr"] = abs(E - S) / atr
    f["target_rank"] = float(rank); f["reward_atr"] = abs(T - E) / atr
    f["rr"] = rr; f["target_eq"] = float(is_eq)
    f["trend_aligned"] = sg * P["trend"][t]; f["bos_aligned"] = sg * P["bos_dir"][t]
    f["bars_since_bos"] = float(min(P["since"][t], 100))
    w0 = max(0, t - 47)
    lo48, hi48 = P["l"][w0:t + 1].min(), P["h"][w0:t + 1].max()
    rng = hi48 - lo48
    rp = float(np.clip((E - lo48) / rng, 0, 1)) if rng > 0 else 0.5
    f["range_pos"] = rp if side == "LONG" else 1.0 - rp      # 0 = favourable (discount for longs / premium for shorts)
    f["mom3"] = sg * (c_t - c[t - 3]) / atr; f["mom12"] = sg * (c_t - c[t - 12]) / atr
    f["recent_range_atr"] = (P["h"][t - 11:t + 1].max() - P["l"][t - 11:t + 1].min()) / atr
    f["atr_pct"] = atr / c_t * 100; f["side_sign"] = sg
    f["prob_up"] = ctx["prob_up"]; f["prob_down"] = ctx["prob_down"]; f["prob_cons"] = ctx["prob_cons"]
    f["macro_aligned"] = sg * ctx["macro"]
    f["dealing_fav"] = ctx["dealing_pos"] if side == "LONG" else 1.0 - ctx["dealing_pos"]
    f["d_pdh"] = ctx["pdh"]; f["d_pdl"] = ctx["pdl"]
    d_atr = ctx["d_atr"]
    best = 3.0
    if d_atr and not np.isnan(d_atr) and d_atr > 0:
        for b, tp in ctx["ranges"]:
            if np.isnan(b) or np.isnan(tp):
                continue
            best = min(best, max(b - E, E - tp, 0.0) / d_atr)
    f["daily_zone_dist_atr"] = best; f["in_daily_zone"] = float(best <= 0.1)
    for k in ("d_sweep", "d_bos", "d_choch", "d_breaker", "d_ifvg"):
        f[k] = ctx[k]
    return np.nan_to_num(np.array([f[k] for k in FEATURE_COLUMNS], dtype=float))


def _candidates(P: dict, t: int, side: str, ctx: dict) -> list:
    long = side == "LONG"
    c_t, atr = P["c"][t], P["atr"][t]
    if not (atr > 0 and c_t > 0):
        return []
    pool = []
    for k in range(max(0, t - H1_ZONE_MAX_AGE), t + 1):
        for z in P["Z"].get(k, ()):
            if (z["dir"] == "bull") != long:
                continue
            if z["inv"] is not None and z["inv"] <= t:
                continue
            if long and z["bottom"] >= c_t:
                continue
            if (not long) and z["top"] <= c_t:
                continue
            prox = (c_t - z["top"]) if long else (z["bottom"] - c_t)
            dist = max(prox, 0.0) / atr
            if dist > H1_MAX_ENTRY_DIST_ATR:
                continue
            pool.append((dist, -z["known"], z))
    if not pool:
        return []
    pool.sort(key=lambda x: (x[0], x[1]))
    lows = _live_swings(P, t, "sl"); highs = _live_swings(P, t, "sh")
    stop_sw, tgt_sw = (lows, highs) if long else (highs, lows)
    out = []
    for _, _, z in pool[:H1_TOP_ZONES]:
        width = z["top"] - z["bottom"]
        seen = set()
        for e_frac in (0.0, 0.5):
            E = z["top"] - e_frac * width if long else z["bottom"] + e_frac * width
            E = min(E, c_t) if long else max(E, c_t)
            key = round(E / atr, 2)
            if key in seen:
                continue
            seen.add(key)
            confl = sum(1 for _, _, z2 in pool if z2 is not z and
                        z2["bottom"] - 0.25 * atr <= E <= z2["top"] + 0.25 * atr)
            stops = [("zone", z["bottom"] - H1_STOP_BUF_ATR * atr if long else z["top"] + H1_STOP_BUF_ATR * atr)]
            sw = [(i, p) for i, p in stop_sw if (p < E - 0.1 * atr if long else p > E + 0.1 * atr)]
            if sw:
                _, p_ = max(sw, key=lambda x: x[0])        # most recent swing beyond entry
                S2 = p_ - H1_STOP_BUF_ATR * atr if long else p_ + H1_STOP_BUF_ATR * atr
                if abs(S2 - stops[0][1]) > 0.1 * atr:
                    stops.append(("swing", S2))
            for s_style, S in stops:
                risk = abs(E - S)
                if not (H1_MIN_RISK_ATR <= risk / atr <= H1_MAX_RISK_ATR):
                    continue
                tl = sorted([p for _, p in tgt_sw if (p > E + 0.1 * atr if long else p < E - 0.1 * atr)],
                            key=lambda p: p if long else -p)
                rank = 0
                for p in tl:
                    T = p - H1_TARGET_BUF_ATR * atr if long else p + H1_TARGET_BUF_ATR * atr
                    rr = abs(T - E) / risk
                    if rr < H1_MIN_RR:
                        continue
                    if rr > H1_MAX_RR:
                        break
                    is_eq = sum(1 for _, q in tgt_sw if abs(q - p) <= 0.15 * atr) > 1
                    x = _features(P, t, side, z, E, S, T, rr, e_frac, s_style, rank, is_eq, ctx, confl)
                    out.append({"zone": z, "entry": float(E), "stop": float(S), "target": float(T),
                                "rr": float(rr), "e_frac": e_frac, "s_style": s_style,
                                "rank": rank, "is_eq": is_eq, "x": x})
                    rank += 1
                    if rank >= 2:
                        break
    return out


# ============================================
# TRADE SIMULATION (limit order -> stop / target / timeout)
# ============================================
def _simulate(P: dict, t: int, side: str, E: float, S: float, T: float):
    """Returns (filled, R) or None if there isn't enough data to resolve."""
    long = side == "LONG"
    hi, lo, c, n = P["h"], P["l"], P["c"], P["n"]
    end_fill = t + H1_VALID_BARS
    if end_fill + H1_HOLD_BARS + 1 >= n:
        return None
    risk = abs(E - S); rr = abs(T - E) / risk
    s_lo, s_hi = lo[t + 1:end_fill + 1], hi[t + 1:end_fill + 1]
    mask = s_lo <= E if long else s_hi >= E
    if not mask.any():
        return 0.0, np.nan
    fj = int(mask.argmax())
    pre = s_hi[:fj] >= T if long else s_lo[:fj] <= T
    if pre.any():                                   # target traded before entry -> setup is gone
        return 0.0, np.nan
    f = t + 1 + fj
    w_lo, w_hi = lo[f:f + H1_HOLD_BARS + 1], hi[f:f + H1_HOLD_BARS + 1]
    stop_hit = w_lo <= S if long else w_hi >= S
    tgt_hit = w_hi >= T if long else w_lo <= T
    tgt_hit[0] = False                              # can't credit the target on the fill bar
    fs = int(stop_hit.argmax()) if stop_hit.any() else 10 ** 9
    ft = int(tgt_hit.argmax()) if tgt_hit.any() else 10 ** 9
    if fs <= ft and fs < 10 ** 9:
        return 1.0, -1.0                            # stop wins ties
    if ft < 10 ** 9:
        return 1.0, float(rr)
    exit_px = c[f + H1_HOLD_BARS]
    r = ((exit_px - E) if long else (E - exit_px)) / risk
    return 1.0, float(np.clip(r, -1.0, rr))


# ============================================
# TRAINING SET + MODEL
# ============================================
def build_training_set(P, B, probs, index_name, sample_every=H1_SAMPLE_EVERY) -> pd.DataFrame:
    t0 = time.time()
    n = P["n"]
    d_idx = np.searchsorted(B["dates"], P["date"], side="left") - 1   # last COMPLETED daily bar
    chunks = []
    for t in range(H1_WARMUP, n - H1_VALID_BARS - H1_HOLD_BARS - 2, sample_every):
        di = d_idx[t]
        if di < 0 or np.isnan(probs[di]).any():
            continue
        p = probs[di]
        side = "LONG" if p[2] >= p[0] else "SHORT"
        ctx = daily_ctx_at(B, di, {"DOWN": p[0], "CONSOLIDATION": p[1], "UP": p[2]}, side)
        local = []
        for cd in _candidates(P, t, side, ctx):
            sim = _simulate(P, t, side, cd["entry"], cd["stop"], cd["target"])
            if sim is None:
                continue
            local.append(np.r_[cd["x"], sim[0], sim[1], t])
        if local:
            chunks.append(np.vstack(local).astype(np.float32))
    if not chunks:
        print(f"   ⚠️  [{index_name}] no H1 candidates generated")
        return pd.DataFrame()
    arr = np.vstack(chunks)
    df = pd.DataFrame(arr, columns=FEATURE_COLUMNS + ["fill", "r", "_at_idx"])
    f = df[df["fill"] == 1]
    print(f"   [{index_name}] {len(df)} simulated candidates over {df['_at_idx'].nunique()} bars "
          f"({time.time() - t0:.1f}s)")
    if len(f):
        print(f"      fill rate {df['fill'].mean():.1%} | filled win rate {(f['r'] > 0).mean():.1%} "
              f"| filled avg R {f['r'].mean():+.3f} | median risk {df['risk_atr'].median():.2f} ATR "
              f"| median RR {df['rr'].median():.2f}")
    return df


def _fit(tr: pd.DataFrame):
    X = tr[FEATURE_COLUMNS]
    clf = RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_leaf=40,
                                 max_features=0.5, random_state=42, n_jobs=-1).fit(X, tr["fill"])
    f = tr[tr["fill"] == 1]
    reg = RandomForestRegressor(n_estimators=250, max_depth=8, min_samples_leaf=40,
                                max_features=0.5, random_state=42, n_jobs=-1).fit(f[FEATURE_COLUMNS], f["r"])
    return clf, reg


def _ev(clf, reg, X) -> np.ndarray:
    pf = clf.predict_proba(X)
    p_fill = pf[:, list(clf.classes_).index(1)] if 1 in clf.classes_ else np.zeros(len(X))
    return p_fill * reg.predict(X), p_fill


def train_model(df: pd.DataFrame, index_name: str) -> dict:
    n = len(df)
    if n < H1_MIN_ROWS:
        print(f"   ⚠️  H1 model skipped — only {n} candidates (need {H1_MIN_ROWS})")
        return {"accepted": False, "n": n, "auc": float("nan"), "reason": f"only {n} rows"}
    df = df.sort_values("_at_idx").reset_index(drop=True)
    at = df["_at_idx"].to_numpy()
    cut = np.quantile(np.unique(at), H1_TRAIN_FRACTION)
    tr = df[at <= cut - (H1_VALID_BARS + H1_HOLD_BARS + 1)]        # embargo so trades can't overlap the split
    va = df[at > cut]
    if va["_at_idx"].nunique() < H1_MIN_VAL_GROUPS or len(tr[tr["fill"] == 1]) < 300:
        return {"accepted": False, "n": n, "auc": float("nan"), "reason": "not enough data per split"}
    clf, reg = _fit(tr)
    ev, p_fill = _ev(clf, reg, va[FEATURE_COLUMNS])
    realized = (va["fill"] * va["r"].fillna(0.0)).to_numpy()
    tmp = pd.DataFrame({"g": va["_at_idx"].to_numpy(), "ev": ev, "pf": p_fill, "real": realized})
    picked = tmp.loc[tmp.groupby("g")["ev"].idxmax()]
    base = float(tmp.groupby("g")["real"].mean().mean())
    sel = float(picked["real"].mean())
    print(f"   H1 model [{index_name}] hold-out: {len(picked)} bars | avg-candidate {base:+.3f}R "
          f"| model-picked {sel:+.3f}R per bar (unfilled = 0R)")
    for thr in (0.0, 0.05, 0.10, 0.20):
        m = picked["ev"] >= thr
        if m.sum():
            print(f"      EV >= {thr:.2f}: {int(m.sum()):>5} bars | realized {picked.loc[m, 'real'].mean():+.3f}R")
    accepted = sel > 0 and sel >= base + H1_MIN_EDGE_R
    if not accepted:
        print(f"   ⚠️  H1 model rejected (edge {sel - base:+.3f}R < {H1_MIN_EDGE_R}) — heuristic fallback")
        return {"accepted": False, "n": n, "auc": float("nan"), "reason": f"edge {sel - base:+.3f}R"}
    clf, reg = _fit(df)
    imp = pd.Series(reg.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    print("      Top 10 features (E[R] model):")
    print(imp.head(10).to_string())
    return {"accepted": True, "fill": clf, "r": reg, "n": n, "auc": float("nan"),
            "val_sel_r": sel, "val_base_r": base}


# ============================================
# LIVE SELECTION
# ============================================
def pick_live_trade(P: dict, t: int, side: str, ctx: dict, model: dict):
    """Returns (pick, reason). pick is None when there is nothing worth trading."""
    cands = _candidates(P, t, side, ctx)
    if not cands:
        return None, "No valid H1 zone near price in the daily-bias direction"
    X = np.vstack([c["x"] for c in cands])
    if model.get("accepted"):
        ev, p_fill = _ev(model["fill"], model["r"], pd.DataFrame(X, columns=FEATURE_COLUMNS))
        k = int(np.argmax(ev))
        if ev[k] < H1_MIN_EV_R:
            return None, f"Best H1 setup has expected {ev[k]:+.2f}R (< {H1_MIN_EV_R:+.2f}R)"
        score, pf, evk = float(ev[k]), float(p_fill[k]), float(ev[k])
    else:
        h = (-X[:, COL["entry_dist_atr"]] + 0.5 * X[:, COL["confluence_n"]]
             + 0.5 * X[:, COL["in_daily_zone"]] + 0.3 * X[:, COL["trend_aligned"]]
             + 0.2 * X[:, COL["bos_aligned"]] - 0.3 * np.abs(X[:, COL["rr"]] - 1.5))
        k = int(np.argmax(h))
        score, pf, evk = float(h[k]), float("nan"), float("nan")
    cd = cands[k]; z = cd["zone"]
    long = side == "LONG"
    atr = float(P["atr"][t]); c_t = float(P["c"][t])
    hr = int(P["hour"][z["known"]])
    kz = "London KZ" if 7 <= hr < 10 else ("Silver Bullet" if 14 <= hr < 15
                                            else ("NY AM KZ" if 12 <= hr < 15 else "none"))
    style = "edge" if cd["e_frac"] == 0 else "mid"
    tag = "model" if model.get("accepted") else "heuristic"
    pick = {
        "entry": cd["entry"], "stop": cd["stop"], "target": cd["target"], "score": score,
        "entry_type": (f"H1 {z['kind'].upper()} {style} limit ("
                       + (f"model; P(fill)={pf:.2f}, EV={evk:+.2f}R" if tag == "model" else "heuristic")
                       + f"; valid {H1_VALID_BARS} H1 bars)"),
        "stop_source": ("beyond zone" if cd["s_style"] == "zone"
                        else f"beyond recent H1 swing {'low' if long else 'high'}"),
        "target_source": (f"H1 swing {'high' if long else 'low'} #{cd['rank'] + 1}"
                          + (" (equal-level liquidity)" if cd["is_eq"] else "")),
        "hourly_meta": {
            "kind": z["kind"], "hour_utc": hr, "session": _sess(hr).title(),
            "zone_top": z["top"], "zone_bottom": z["bottom"], "entry_price": cd["entry"],
            "model_prob": pf, "expected_r": evk, "h1_trend": int(P["trend"][t]),
            "bars_since_zone": int(t - z["known"]),
            "distance_pct": abs(c_t - cd["entry"]) / c_t * 100,
            "stop_reference": cd["stop"], "applied": True, "ote_aligned": False,
            "ote_705": None, "ote_band_low": None, "ote_band_high": None,
            "kill_zone": kz, "hourly_rr": None, "daily_entry": c_t, "daily_rr": None,
        },
    }
    return pick, ""


# ============================================
# TRADE JOURNAL
# ============================================

JOURNAL_COLUMNS = [
    'signal_date', 'target_date', 'source',
    'predicted_state', 'prob_up', 'prob_down', 'prob_cons', 'confidence', 'move_threshold',
    'action', 'entry_type', 'entry_style', 'entry', 'stop_loss', 'take_profit',
    'stop_source', 'target_source', 'close_t', 'atr', 'is_market', 'd_s', 'd_t', 'risk_reward',
    'zone_pos', 'vol20', 'rsi', 'market_structure', 'analog_updown',
    'zone_strength', 'dow', 'entry_depth_atr',
    'adapt_applied', 'adapt_shift', 'adapt_extra', 'adapt_tscale',
    'adapt_entry', 'adapt_stop', 'adapt_target', 'journal_win_prob', 'meta_filtered',
    'hourly_hour_utc', 'hourly_session', 'hourly_zone_kind',
    'hourly_entry_price', 'hourly_distance_pct', 'hourly_bars_since_zone',
    'hourly_stop_reference', 'hourly_applied', 'hourly_risk_reward',
    'hourly_model_prob', 'hourly_h1_trend',
    'stop_anchor_kind', 'stop_anchor_prob',
    'target_anchor_kind', 'target_anchor_prob',
    'daily_struct_entry', 'daily_struct_rr',
    'ote_aligned', 'ote_705', 'ote_band_low', 'ote_band_high',
    'dealing_pos_at_signal', 'dist_to_pdh', 'dist_to_pdl',
    'bias_gate_passed', 'kill_zone',
    'dol_target', 'dol_source', 'dol_rank',
    'resolved', 'win_open', 'win_high', 'win_low', 'win_close', 'next_close',
    'actual_return', 'actual_state', 'dir_correct', 'right_direction',
    'filled', 'outcome', 'pnl_r', 'mae_atr', 'mfe_atr', 'stopped_but_right',
    'adapt_pnl_r', 'adapt_outcome',
]
_DATE_COLS = ['signal_date', 'target_date']
_BOOL_COLS = ['is_market', 'adapt_applied', 'meta_filtered', 'resolved', 'filled',
              'dir_correct', 'right_direction', 'stopped_but_right', 'hourly_applied',
              'ote_aligned', 'bias_gate_passed']
_TEXT_COLS = ['source', 'predicted_state', 'action', 'entry_type', 'entry_style',
              'stop_source', 'target_source', 'actual_state', 'outcome', 'adapt_outcome',
              'hourly_session', 'hourly_zone_kind', 'kill_zone', 'dol_source',
              'stop_anchor_kind', 'target_anchor_kind']
_NUM_COLS = [c for c in JOURNAL_COLUMNS
             if c not in _DATE_COLS and c not in _BOOL_COLS and c not in _TEXT_COLS]


def _journal_path(index_name: str) -> str:
    safe = re.sub(r'[^A-Za-z0-9]+', '_', index_name).strip('_')
    return os.path.join(JOURNAL_DIR, f"{safe}_trade_journal.csv")


def _coerce_journal(j: pd.DataFrame) -> pd.DataFrame:
    j = j.copy()
    for c in JOURNAL_COLUMNS:
        if c not in j.columns:
            j[c] = np.nan
    for c in _DATE_COLS:
        j[c] = pd.to_datetime(j[c], errors='coerce')
    for c in _BOOL_COLS:
        j[c] = j[c].astype(str).str.strip().str.lower().isin(['true', '1', '1.0'])
    for c in _NUM_COLS:
        j[c] = pd.to_numeric(j[c], errors='coerce')
    for c in _TEXT_COLS:
        j[c] = j[c].fillna('').astype(str)
        j.loc[j[c].str.lower() == 'nan', c] = ''
    return j[JOURNAL_COLUMNS].reset_index(drop=True)


def _load_journal(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            j = pd.read_csv(path)
            j = _coerce_journal(j)
            j = j.dropna(subset=['signal_date']).drop_duplicates('signal_date', keep='last')
            j = j.sort_values('signal_date').reset_index(drop=True)
            print(f"\n📓 Loaded existing journal: {path} ({len(j)} rows)")
            return j
        except Exception as e:
            bak = path + ".corrupt.bak"
            print(f"\n⚠️  Could not read journal {path} ({e}) — moved to {bak}, starting fresh")
            try:
                os.replace(path, bak)
            except OSError:
                pass
    else:
        print(f"\n📓 No journal found — a new one will be created at {path}")
    return _coerce_journal(pd.DataFrame(columns=JOURNAL_COLUMNS))


def _save_journal(j: pd.DataFrame, path: str):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    out = j.sort_values('signal_date')[JOURNAL_COLUMNS]
    try:
        out.to_csv(path, index=False)
        print(f"💾 Journal saved: {path} ({len(out)} rows)")
    except OSError as e:
        alt = path.replace('.csv', f"_{int(time.time())}.csv")
        out.to_csv(alt, index=False)
        print(f"⚠️  Could not write {path} ({e}) — saved to {alt}")


def _journal_record(row: pd.DataFrame, probs: dict, trade: dict, next_date, source: str) -> dict:
    r = row.iloc[0]
    close_t = float(r['Close'])
    atr = r['ATR_14']
    atr = float(atr) if pd.notna(atr) and atr > 0 else 0.01 * close_t
    dominant = max(probs, key=probs.get)
    rec = {c: np.nan for c in JOURNAL_COLUMNS}
    for c in _TEXT_COLS:
        rec[c] = ''
    for c in _BOOL_COLS:
        rec[c] = False
    rec.update({
        'signal_date': r['Date'], 'target_date': pd.to_datetime(next_date), 'source': source,
        'predicted_state': dominant,
        'prob_up': probs.get('UP', np.nan), 'prob_down': probs.get('DOWN', np.nan),
        'prob_cons': probs.get('CONSOLIDATION', np.nan), 'confidence': probs[dominant],
        'move_threshold': float(r['Move_Threshold']), 'action': trade['action'],
        'close_t': close_t, 'atr': atr,
        'zone_pos': r['Premium_Discount'], 'vol20': r['Volatility_20'], 'rsi': r['RSI'],
        'market_structure': r['Macro_Structure'], 'analog_updown': r['Analog_Up_Minus_Down'],
        'dow': r['DayOfWeek'],
        'dealing_pos_at_signal': r.get('Dealing_Pos', np.nan),
        'dist_to_pdh': r.get('Dist_To_PDH', np.nan),
        'dist_to_pdl': r.get('Dist_To_PDL', np.nan),
    })
    if trade['action'] in ('LONG', 'SHORT'):
        entry = float(trade['entry'])
        rec.update({
            'entry_type': trade['entry_type'], 'entry_style': trade.get('entry_style', ''),
            'entry': entry, 'stop_loss': float(trade['stop_loss']),
            'take_profit': float(trade['take_profit']),
            'stop_source': trade.get('stop_source', ''), 'target_source': trade.get('target_source', ''),
            'is_market': str(trade['entry_type']).startswith('market'),
            'd_s': abs(entry - float(trade['stop_loss'])),
            'd_t': abs(float(trade['take_profit']) - entry),
            'risk_reward': trade.get('risk_reward') if trade.get('risk_reward') is not None else np.nan,
            'zone_strength': trade.get('zone_strength', np.nan),
            'entry_depth_atr': abs(close_t - entry) / atr,
            'daily_struct_entry': trade.get('daily_entry', entry),
            'daily_struct_rr': trade.get('daily_rr', np.nan),
            'bias_gate_passed': bool(trade.get('bias_gate_passed', False)),
            'dol_target': trade.get('dol_target', np.nan),
            'dol_source': trade.get('dol_source', ''),
            'dol_rank': trade.get('dol_rank', np.nan),
        })
        hm = trade.get('hourly_meta')
        if hm is not None:
            rec.update({
                'hourly_hour_utc': hm.get('hour_utc', np.nan),
                'hourly_session': hm.get('session', ''),
                'hourly_zone_kind': hm.get('kind', ''),
                'hourly_entry_price': hm.get('entry_price', np.nan),
                'hourly_distance_pct': hm.get('distance_pct', np.nan),
                'hourly_bars_since_zone': hm.get('bars_since_zone', np.nan),
                'hourly_stop_reference': hm.get('stop_reference', np.nan),
                'hourly_applied': hm.get('applied', False),
                'hourly_risk_reward': hm.get('hourly_rr', np.nan),
                'hourly_model_prob': hm.get('model_prob', np.nan),
                'hourly_h1_trend': hm.get('h1_trend', np.nan),
                'ote_aligned': bool(hm.get('ote_aligned', False)),
                'ote_705': hm.get('ote_705', np.nan) if hm.get('ote_705') is not None else np.nan,
                'ote_band_low': hm.get('ote_band_low', np.nan) if hm.get('ote_band_low') is not None else np.nan,
                'ote_band_high': hm.get('ote_band_high', np.nan) if hm.get('ote_band_high') is not None else np.nan,
                'kill_zone': hm.get('kill_zone') or '',
            })
        sa = trade.get('stop_anchor')
        if sa and sa.get('zone') is not None:
            rec['stop_anchor_kind'] = sa['zone']['kind']
            rec['stop_anchor_prob'] = sa['zone'].get('model_prob', np.nan)
        ta = trade.get('target_anchor')
        if ta and ta.get('zone') is not None:
            rec['target_anchor_kind'] = ta['zone']['kind']
            rec['target_anchor_prob'] = ta['zone'].get('model_prob', np.nan)
    return rec


# ============================================
# TRADE JOURNAL — REPLAY ENGINE
# ============================================

def _arrays(sub: pd.DataFrame) -> dict:
    return {
        'sign': np.where(sub['action'].values == 'LONG', 1.0, -1.0),
        'e_ref': sub['entry'].values.astype(float),
        'is_mkt': sub['is_market'].values.astype(bool),
        'd_s': sub['d_s'].values.astype(float),
        'd_t': sub['d_t'].values.astype(float),
        'atr': sub['atr'].values.astype(float),
        'op': sub['win_open'].values.astype(float),
        'hi': sub['win_high'].values.astype(float),
        'lo': sub['win_low'].values.astype(float),
        'cl': sub['win_close'].values.astype(float),
        'right': sub['right_direction'].values.astype(bool),
    }


def _slice(a: dict, s) -> dict:
    return {k: v[s] for k, v in a.items()}


def _replay(a: dict, shift: float = 0.0, extra: float = 0.0, tscale: float = 1.0) -> dict:
    sign, atr = a['sign'], a['atr']
    hi_l = np.where(sign > 0, a['hi'], -a['lo'])
    lo_l = np.where(sign > 0, a['lo'], -a['hi'])
    cl_l = sign * a['cl']; op_l = sign * a['op']
    e_l = sign * a['e_ref']; e_lim = e_l - shift * atr
    mkt_fill = a['is_mkt'] if shift == 0 else np.zeros(len(sign), dtype=bool)
    entry = np.where(mkt_fill, op_l, e_lim)
    filled = mkt_fill | (lo_l <= e_lim)
    ds = a['d_s'] + extra * atr
    dt = a['d_t'] * tscale
    stop = entry - ds; tgt = entry + dt
    stop_hit = filled & (lo_l <= stop)
    tgt_hit = filled & (~stop_hit) & (hi_l >= tgt)
    with np.errstate(divide='ignore', invalid='ignore'):
        r_exit = (cl_l - entry) / ds
        r = np.where(stop_hit, -1.0, np.where(tgt_hit, dt / ds, r_exit))
        mae = np.maximum(entry - lo_l, 0) / atr
        mfe = np.maximum(hi_l - entry, 0) / atr
    return {'r': np.where(filled, r, np.nan), 'filled': filled,
            'stop_hit': stop_hit, 'tgt_hit': tgt_hit,
            'mae': np.where(filled, mae, np.nan), 'mfe': np.where(filled, mfe, np.nan)}


def _outcome_labels(res: dict):
    return np.where(~res['filled'], 'no_fill',
                    np.where(res['stop_hit'], 'stopped',
                             np.where(res['tgt_hit'], 'target', 'expired')))


def _expectancy(a: dict, shift, extra, tscale):
    res = _replay(a, shift, extra, tscale)
    n = max(len(res['r']), 1)
    return float(np.nansum(res['r']) / n), res


def _resolve_journal(journal: pd.DataFrame, df: pd.DataFrame, hold: int = JOURNAL_HOLD_BARS) -> int:
    pending_idx = journal.index[~journal['resolved']]
    if len(pending_idx) == 0:
        return 0
    win = pd.DataFrame({
        'w_open': df['Open'].shift(-1).values,
        'w_high': df['High'].rolling(hold).max().shift(-hold).values,
        'w_low': df['Low'].rolling(hold).min().shift(-hold).values,
        'w_close': df['Close'].shift(-hold).values,
        'n_close': df['Close'].shift(-1).values,
    }, index=pd.DatetimeIndex(df['Date'].values))
    win = win[~win.index.duplicated(keep='last')]
    w = win.reindex(pd.DatetimeIndex(journal.loc[pending_idx, 'signal_date']))
    ok = w.notna().all(axis=1).values
    idx = pending_idx[ok]
    if len(idx) == 0:
        return 0
    w = w[ok]
    journal.loc[idx, 'win_open'] = w['w_open'].values
    journal.loc[idx, 'win_high'] = w['w_high'].values
    journal.loc[idx, 'win_low'] = w['w_low'].values
    journal.loc[idx, 'win_close'] = w['w_close'].values
    journal.loc[idx, 'next_close'] = w['n_close'].values
    close_t = journal.loc[idx, 'close_t'].values.astype(float)
    thr = journal.loc[idx, 'move_threshold'].values.astype(float)
    actual_ret = w['n_close'].values / close_t - 1
    actual_state = np.where(actual_ret > thr, 'UP',
                            np.where(actual_ret < -thr, 'DOWN', 'CONSOLIDATION'))
    journal.loc[idx, 'actual_return'] = actual_ret
    journal.loc[idx, 'actual_state'] = actual_state
    journal.loc[idx, 'dir_correct'] = (journal.loc[idx, 'predicted_state'].values == actual_state)
    journal.loc[idx, 'resolved'] = True
    is_trade = journal.loc[idx, 'action'].isin(['LONG', 'SHORT']).values
    journal.loc[idx[~is_trade], 'outcome'] = 'no_trade'
    sub = journal.loc[idx[is_trade]]
    good = (sub['d_s'] > 0) & (sub['atr'] > 0) & sub['entry'].notna()
    tidx = sub.index[good.values]
    if len(tidx):
        sgn = np.where(journal.loc[tidx, 'action'].values == 'LONG', 1.0, -1.0)
        nxt = journal.loc[tidx, 'next_close'].values.astype(float)
        ct = journal.loc[tidx, 'close_t'].values.astype(float)
        journal.loc[tidx, 'right_direction'] = (sgn * (nxt - ct) > 0)
        a = _arrays(journal.loc[tidx])
        res = _replay(a)
        journal.loc[tidx, 'filled'] = res['filled']
        journal.loc[tidx, 'outcome'] = _outcome_labels(res)
        journal.loc[tidx, 'pnl_r'] = res['r']
        journal.loc[tidx, 'mae_atr'] = res['mae']
        journal.loc[tidx, 'mfe_atr'] = res['mfe']
        journal.loc[tidx, 'stopped_but_right'] = res['stop_hit'] & a['right']
        sub2 = journal.loc[tidx]
        sub2 = sub2[sub2['adapt_applied']]
        if len(sub2):
            for (s, e, t), g in sub2.groupby(['adapt_shift', 'adapt_extra', 'adapt_tscale']):
                ag = _arrays(journal.loc[g.index])
                rr = _replay(ag, float(s), float(e), float(t))
                journal.loc[g.index, 'adapt_pnl_r'] = rr['r']
                journal.loc[g.index, 'adapt_outcome'] = _outcome_labels(rr)
    return len(idx)


def _print_journal_summary(journal: pd.DataFrame, index_name: str):
    res = journal[journal['resolved']]
    print(f"\n📓 {index_name} JOURNAL: {len(journal)} rows | resolved {len(res)} | "
          f"pending {int((~journal['resolved']).sum())}")
    if res.empty:
        return
    live = res[res['source'] == 'live']
    print(f"   Direction accuracy (all resolved): {res['dir_correct'].mean():.1%}  "
          f"| live-only: {live['dir_correct'].mean():.1%} (n={len(live)})" if len(live)
          else f"   Direction accuracy (all resolved): {res['dir_correct'].mean():.1%}")
    tr = res[res['filled']]
    if len(tr):
        rd = tr[tr['right_direction']]
        sbr = (rd['outcome'] == 'stopped').mean() if len(rd) else float('nan')
        print(f"   Baseline trades: {len(tr)} filled | win {((tr['pnl_r'] > 0).mean()):.1%} "
              f"| stopped {((tr['outcome'] == 'stopped').mean()):.1%} "
              f"| target {((tr['outcome'] == 'target').mean()):.1%} "
              f"| avg {tr['pnl_r'].mean():+.3f}R")
        print(f"   ↳ Of trades where the DIRECTION was right ({len(rd)}), "
              f"{sbr:.1%} were stopped out first")
        if 'hourly_model_prob' in tr.columns and tr['hourly_model_prob'].notna().any():
            hh = tr[tr['hourly_model_prob'].notna()]
            if len(hh) >= 20:
                hi = hh[hh['hourly_model_prob'] >= 0.6]
                lo = hh[hh['hourly_model_prob'] < 0.6]
                if len(hi):
                    print(f"   ↳ H1 prob ≥ 0.6:  {len(hi):>4} trades | avg {hi['pnl_r'].mean():+.3f}R")
                if len(lo):
                    print(f"   ↳ H1 prob < 0.6:  {len(lo):>4} trades | avg {lo['pnl_r'].mean():+.3f}R")


def _trades_for_learning(journal: pd.DataFrame) -> pd.DataFrame:
    m = (journal['resolved'] & journal['action'].isin(['LONG', 'SHORT'])
         & journal['win_close'].notna() & (journal['d_s'] > 0) & (journal['atr'] > 0)
         & journal['entry'].notna())
    return journal[m].sort_values('signal_date').reset_index(drop=True)


def _grid_search(a: dict) -> dict:
    best = None
    for s in CAL_SHIFT_GRID:
        for e in CAL_EXTRA_STOP_GRID:
            for t in CAL_TARGET_SCALE_GRID:
                exp, _ = _expectancy(a, s, e, t)
                score = exp - CAL_COMPLEXITY_PENALTY * (s + e)
                if best is None or score > best['score']:
                    best = {'score': score, 'exp': exp, 'params': (s, e, t)}
    return best


def _fit_calibration_subset(sub: pd.DataFrame) -> dict:
    n = len(sub)
    if n < CAL_MIN_TRADES:
        return {'accepted': False, 'params': None, 'n': n,
                'reason': f'only {n} resolved trades (need {CAL_MIN_TRADES})'}
    a = _arrays(sub)
    cut = int(n * CAL_TRAIN_FRACTION)
    a_tr, a_ho = _slice(a, slice(0, cut)), _slice(a, slice(cut, n))
    best_tr = _grid_search(a_tr)
    base_ho, _ = _expectancy(a_ho, 0.0, 0.0, 1.0)
    learn_ho, _ = _expectancy(a_ho, *best_tr['params'])
    out = {'n': n, 'hold_base': base_ho, 'hold_learn': learn_ho,
           'accepted': False, 'params': None}
    improved = (learn_ho - base_ho) >= CAL_MIN_IMPROVEMENT and best_tr['params'] != (0.0, 0.0, 1.0)
    if not improved:
        out['reason'] = 'no adjustment beat the baseline on the validation slice'
        return out
    final = _grid_search(a)
    out.update({'accepted': True, 'params': final['params']})
    return out


def _fit_calibrations(trades: pd.DataFrame) -> dict:
    out = {'ALL': _fit_calibration_subset(trades)}
    for side in ('LONG', 'SHORT'):
        out[side] = _fit_calibration_subset(trades[trades['action'] == side])
    return out


def _pick_calibration(cals, side):
    if not cals:
        return None
    c = cals.get(side)
    if c and c.get('n', 0) >= CAL_MIN_TRADES:
        return c['params'] if c.get('accepted') else None
    c = cals.get('ALL')
    if c and c.get('accepted'):
        return c['params']
    return None


def _apply_calibration(trade: dict, close: float, atr: float, params) -> dict:
    shift, extra, ts = params
    sgn = 1.0 if trade['action'] == 'LONG' else -1.0
    e0 = float(trade['entry'])
    d_s = abs(e0 - float(trade['stop_loss']))
    d_t = abs(float(trade['take_profit']) - e0)
    is_mkt = str(trade['entry_type']).startswith('market')
    if is_mkt and shift == 0:
        e1 = close
    elif is_mkt:
        e1 = close - sgn * shift * atr
    else:
        e1 = e0 - sgn * shift * atr
    s1 = e1 - sgn * (d_s + extra * atr)
    t1 = e1 + sgn * d_t * ts
    risk, reward = abs(e1 - s1), abs(t1 - e1)
    new = dict(trade)
    new['entry'] = round(e1, 2)
    new['stop_loss'] = round(s1, 2)
    new['take_profit'] = round(t1, 2)
    new['risk_reward'] = round(reward / risk, 2) if risk > 0 else None
    new['stop_source'] = f"{trade.get('stop_source', '')} +{extra:.2f} ATR (journal)"
    new['target_source'] = f"{trade.get('target_source', '')} x{ts:.2f} (journal)"
    return new


META_FEATURES = [
    'prob_up', 'prob_down', 'prob_cons', 'confidence', 'side_sign',
    'zone_pos', 'vol20', 'rsi', 'market_structure', 'analog_updown',
    'atr_pct', 'd_s_atr', 'd_t_atr', 'risk_reward', 'entry_depth_atr',
    'is_market', 'dow', 'hourly_model_prob', 'h1_trend_num',
]


def _meta_matrix(j: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=j.index)
    atr = pd.to_numeric(j['atr'], errors='coerce').replace(0, np.nan)
    close_t = pd.to_numeric(j['close_t'], errors='coerce')
    for c in ['prob_up', 'prob_down', 'prob_cons', 'confidence', 'zone_pos', 'vol20', 'rsi',
              'market_structure', 'analog_updown', 'risk_reward', 'entry_depth_atr', 'dow',
              'hourly_model_prob', 'hourly_h1_trend']:
        X[c] = pd.to_numeric(j[c], errors='coerce')
    X['h1_trend_num'] = X['hourly_h1_trend'].fillna(0.0)
    X['side_sign'] = np.where(j['action'].values == 'LONG', 1.0, -1.0)
    X['atr_pct'] = atr / close_t
    X['d_s_atr'] = pd.to_numeric(j['d_s'], errors='coerce') / atr
    X['d_t_atr'] = pd.to_numeric(j['d_t'], errors='coerce') / atr
    X['is_market'] = j['is_market'].astype(float)
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)[META_FEATURES]


def _meta_model():
    return RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=15,
                                  class_weight='balanced', random_state=42, n_jobs=-1)


def _fit_meta(trades: pd.DataFrame):
    t = trades[trades['filled'] & trades['pnl_r'].notna()].reset_index(drop=True)
    n = len(t)
    if n < META_MIN_TRADES:
        return {'accepted': False, 'n': n, 'reason': f'only {n} filled trades'}
    X = _meta_matrix(t)
    y = (t['pnl_r'] > 0).astype(int).values
    cut = int(n * CAL_TRAIN_FRACTION)
    if len(np.unique(y[:cut])) < 2 or len(np.unique(y[cut:])) < 2:
        return {'accepted': False, 'n': n, 'reason': 'one outcome class missing'}
    m = _meta_model().fit(X.iloc[:cut], y[:cut])
    auc = roc_auc_score(y[cut:], m.predict_proba(X.iloc[cut:])[:, 1])
    out = {'n': n, 'auc': float(auc), 'base_win_rate': float(y.mean()), 'accepted': False}
    if auc < META_MIN_AUC:
        out['reason'] = f'hold-out AUC {auc:.3f} < {META_MIN_AUC}'
        return out
    out['model'] = _meta_model().fit(X, y)
    out['accepted'] = True
    return out


def _meta_predict(meta: dict, rec: dict) -> float:
    X = _meta_matrix(pd.DataFrame([rec]))
    return float(meta['model'].predict_proba(X)[0][1])


def _summ(label, r, filled, stop, right):
    n = len(r)
    nf = int(filled.sum())
    exp = float(np.nansum(r) / n) if n else float('nan')
    win = float((r[filled] > 0).mean()) if nf else float('nan')
    stp = float(stop[filled].mean()) if nf else float('nan')
    rf = filled & right
    sbr = float((stop & right).sum() / rf.sum()) if rf.sum() else float('nan')
    print(f"   {label:<10}{n:>8}{nf:>8}{win:>9.1%}{stp:>10.1%}{sbr:>15.1%}{exp:>14.3f}")


def _walkforward_report(journal: pd.DataFrame, index_name: str):
    t = _trades_for_learning(journal)
    n = len(t)
    if n < CAL_MIN_TRADES * 2:
        print(f"\n🧪 {index_name} walk-forward check skipped ({n} resolved trades)")
        return
    a_all = _arrays(t)
    start = max(CAL_MIN_TRADES, int(n * 0.5))
    step = max(25, (n - start) // 40)
    store = {k: {'r': [], 'filled': [], 'stop': [], 'right': []} for k in ('base', 'adapt')}
    for i in range(start, n, step):
        j2 = min(i + step, n)
        cals = _fit_calibrations(t.iloc[:i])
        chunk = _slice(a_all, slice(i, j2))
        for side, sgn in (('LONG', 1.0), ('SHORT', -1.0)):
            m = chunk['sign'] == sgn
            if not m.any():
                continue
            c = {k: v[m] for k, v in chunk.items()}
            params = _pick_calibration(cals, side) or (0.0, 0.0, 1.0)
            for key, p in (('base', (0.0, 0.0, 1.0)), ('adapt', params)):
                res = _replay(c, *p)
                store[key]['r'].append(res['r'])
                store[key]['filled'].append(res['filled'])
                store[key]['stop'].append(res['stop_hit'])
                store[key]['right'].append(c['right'])
    print(f"\n🧪 {index_name} WALK-FORWARD CHECK")
    print(f"   {'':<10}{'signals':>8}{'filled':>8}{'win%':>9}{'stopped%':>10}"
          f"{'stopped|right':>15}{'avg R/signal':>14}")
    for label, key in (('baseline', 'base'), ('adjusted', 'adapt')):
        d = store[key]
        if not d['r']:
            continue
        _summ(label, np.concatenate(d['r']), np.concatenate(d['filled']),
              np.concatenate(d['stop']), np.concatenate(d['right']))


# ============================================
# MASTER PIPELINE
# ============================================

def run_single_index(csv_path: str, index_name: str, dxy_df=None):
    print("\n\n" + "#"*80)
    print(f"#  PROCESSING: {index_name}  ({csv_path})")
    print("#"*80)

    if not os.path.exists(csv_path):
        print(f"⚠️  Skipping {index_name}: file not found at {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    df['Date'] = _parse_dates(df['Date'])
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    HAS_VOLUME = 'Volume' in df.columns and df['Volume'].notna().any()
    if HAS_VOLUME:
        df['Volume'] = pd.to_numeric(
            df['Volume'].astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
    else:
        df['Volume'] = 0.0

    print(f"\n📊 Data loaded: {len(df)} rows "
          f"({df['Date'].min().date()} → {df['Date'].max().date()})")
    print(f"🔊 Volume available: {HAS_VOLUME}")

    hourly_csv_path = HOURLY_CONFIGS.get(index_name)
    hourly_df = load_hourly_csv(hourly_csv_path) if hourly_csv_path else None
    HAS_HOURLY = hourly_df is not None and len(hourly_df) > 500

    h1_P = None
    if HAS_HOURLY:
        print(f"\n⏰ Hourly data loaded: {len(hourly_df)} bars "
              f"({hourly_df['timestamp'].min().date()} → {hourly_df['timestamp'].max().date()})")
        print("   Precomputing H1 swings, structure, causal zones...")
        h1_P = prep_hourly(hourly_df)
    else:
        print(f"\nℹ️  No hourly data — H1 learned model unavailable")

    # ---------------- CLASSIC FEATURES ----------------
    print("\n🔧 Building classic technical features...")
    df['Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    for w in [5, 10, 20, 50, 200]:
        df[f'MA_{w}'] = df['Close'].rolling(w).mean()
    df['Price_vs_MA5'] = df['Close'] / df['MA_5'] - 1
    df['Price_vs_MA20'] = df['Close'] / df['MA_20'] - 1
    df['Price_vs_MA50'] = df['Close'] / df['MA_50'] - 1
    df['MA5_vs_MA20'] = df['MA_5'] / df['MA_20'] - 1
    df['MA50_vs_MA200'] = df['MA_50'] / df['MA_200'] - 1
    df['Volatility_5'] = df['Return'].rolling(5).std()
    df['Volatility_20'] = df['Return'].rolling(20).std()
    df['Volatility_50'] = df['Return'].rolling(50).std()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    df['BB_Middle'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    if HAS_VOLUME:
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_vs_MA20'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['Volume_Zscore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
    else:
        df['Volume_Change'] = 0.0
        df['Volume_vs_MA20'] = 1.0
        df['Volume_Zscore'] = 0.0
    df['Body'] = (df['Close'] - df['Open']) / df['Open']
    df['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Open']
    df['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Open']
    df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    for lag in [1, 2, 3, 5, 10]:
        df[f'Return_Lag{lag}'] = df['Return'].shift(lag)
    df['Volume_Price_Divergence'] = (df['Return'].rolling(10).corr(df['Volume'].pct_change())
                                      if HAS_VOLUME else 0.0)
    df['Vol_ROC_5'] = df['Volatility_20'].pct_change(5)
    df['Vol_ROC_10'] = df['Volatility_20'].pct_change(10)

    def _consec(returns, up=True):
        flag = (returns > 0).astype(int) if up else (returns < 0).astype(int)
        group = (returns <= 0).cumsum() if up else (returns >= 0).cumsum()
        return flag.groupby(group).cumsum()
    df['Consec_Up'] = _consec(df['Return'], True)
    df['Consec_Down'] = _consec(df['Return'], False)
    df['Overnight_Return'] = df['Open'] / df['Close'].shift(1) - 1
    df['Intraday_Return'] = df['Close'] / df['Open'] - 1
    df['Overnight_vs_Intraday'] = df['Overnight_Return'] - df['Intraday_Return']
    df['DayOfWeek'] = df['Date'].dt.dayofweek

    HAS_DXY = False
    dxy_features = []
    if dxy_df is not None and len(dxy_df) > 0:
        print("🔧 Merging DXY cross-asset features...")
        _dxy_cols = ['Date', 'DXY_Close', 'DXY_Return', 'DXY_vs_MA20', 'DXY_RSI',
                     'DXY_Return_Lag1', 'DXY_Return_Lag5']
        df = df.merge(dxy_df[_dxy_cols], on='Date', how='left')
        df['Corr_20d_DXY'] = df['Return'].rolling(20).corr(df['DXY_Return'])
        df['Corr_Change_20d'] = df['Corr_20d_DXY'] - df['Corr_20d_DXY'].shift(5)
        df['Relative_Strength_vs_DXY'] = df['Return'] - df['DXY_Return']
        for col, val in {'DXY_Return': 0.0, 'DXY_vs_MA20': 0.0, 'DXY_RSI': 50.0,
                          'DXY_Return_Lag1': 0.0, 'DXY_Return_Lag5': 0.0,
                          'Corr_20d_DXY': 0.0, 'Corr_Change_20d': 0.0,
                          'Relative_Strength_vs_DXY': 0.0}.items():
            df[col] = df[col].fillna(val)
        dxy_features = ['DXY_Return', 'DXY_Return_Lag1', 'DXY_vs_MA20', 'DXY_RSI',
                        'DXY_Return_Lag5', 'Corr_20d_DXY', 'Corr_Change_20d',
                        'Relative_Strength_vs_DXY']
        HAS_DXY = True
        print(f"   ✅ DXY merged")

    print("🔧 Building ICT/SMC features...")

    def _find_swings(frame, left, right):
        highs = frame['High'].values; lows = frame['Low'].values
        n = len(frame)
        sh = np.zeros(n, dtype=bool); sl = np.zeros(n, dtype=bool)
        for i in range(left, n - right):
            if highs[i] == highs[i - left:i + right + 1].max():
                sh[i] = True
            if lows[i] == lows[i - left:i + right + 1].min():
                sl[i] = True
        return sh, sl

    def _build_structure(frame, left, right, label):
        sh_raw, sl_raw = _find_swings(frame, left, right)
        last_sh = pd.Series(np.where(sh_raw, frame['High'].values, np.nan)).shift(right).ffill()
        last_sl = pd.Series(np.where(sl_raw, frame['Low'].values, np.nan)).shift(right).ffill()
        bos_bull = (frame['Close'] > last_sh).astype(int)
        bos_bear = (frame['Close'] < last_sl).astype(int)
        trend = np.zeros(len(frame)); cur = 0
        for i, (b, s) in enumerate(zip(bos_bull.values, bos_bear.values)):
            if b: cur = 1
            elif s: cur = -1
            trend[i] = cur
        structure = pd.Series(trend, index=frame.index)
        choch = ((structure != structure.shift(1)) & (structure != 0)).astype(int)
        return {f'{label}_Swing_High': last_sh, f'{label}_Swing_Low': last_sl,
                f'{label}_BOS_Bull': bos_bull, f'{label}_BOS_Bear': bos_bear,
                f'{label}_Structure': structure, f'{label}_CHoCH': choch}

    macro = _build_structure(df, 10, 10, 'Macro')
    micro = _build_structure(df, 2, 2, 'Micro')
    for k, v in {**macro, **micro}.items():
        df[k] = v
    df['Last_Swing_High'] = df['Macro_Swing_High']
    df['Last_Swing_Low'] = df['Macro_Swing_Low']
    df['BOS_Bull'] = df['Macro_BOS_Bull']; df['BOS_Bear'] = df['Macro_BOS_Bear']
    df['Market_Structure'] = df['Macro_Structure']; df['CHoCH'] = df['Macro_CHoCH']

    counter = np.zeros(len(df)); c = 0
    for i in range(len(df)):
        if (df['Macro_Structure'].iloc[i] != 0 and df['Micro_Structure'].iloc[i] != 0
                and df['Macro_Structure'].iloc[i] != df['Micro_Structure'].iloc[i]):
            c += 1
        else:
            c = 0
        counter[i] = c
    df['Counter_Trend_Bars'] = counter
    df['Pullback_In_Uptrend_MTF'] = ((df['Macro_Structure'] == 1) & (df['Micro_Structure'] == -1)
                                      & (df['Counter_Trend_Bars'] >= 1) & (df['Counter_Trend_Bars'] <= 6)).astype(int)
    df['Bounce_In_Downtrend_MTF'] = ((df['Macro_Structure'] == -1) & (df['Micro_Structure'] == 1)
                                      & (df['Counter_Trend_Bars'] >= 1) & (df['Counter_Trend_Bars'] <= 6)).astype(int)

    _fvg_bull_top, _fvg_bull_bot = [], []
    _fvg_bear_top, _fvg_bear_bot = [], []
    _lb_top = _lb_bot = np.nan; _lr_top = _lr_bot = np.nan
    _fvg_bull_flag = np.zeros(len(df), dtype=int)
    _fvg_bear_flag = np.zeros(len(df), dtype=int)
    _h2 = df['High'].shift(2).values; _l2 = df['Low'].shift(2).values
    _h = df['High'].values; _l = df['Low'].values
    for i in range(len(df)):
        if i >= 2 and not np.isnan(_h2[i]) and _l[i] > _h2[i]:
            _fvg_bull_flag[i] = 1; _lb_bot = _h2[i]; _lb_top = _l[i]
        if i >= 2 and not np.isnan(_l2[i]) and _h[i] < _l2[i]:
            _fvg_bear_flag[i] = 1; _lr_top = _l2[i]; _lr_bot = _h[i]
        _fvg_bull_top.append(_lb_top); _fvg_bull_bot.append(_lb_bot)
        _fvg_bear_top.append(_lr_top); _fvg_bear_bot.append(_lr_bot)
    df['FVG_Bull'] = _fvg_bull_flag; df['FVG_Bear'] = _fvg_bear_flag
    df['FVG_Bull_Top'] = _fvg_bull_top; df['FVG_Bull_Bottom'] = _fvg_bull_bot
    df['FVG_Bear_Top'] = _fvg_bear_top; df['FVG_Bear_Bottom'] = _fvg_bear_bot
    df['FVG_Bull_Mid'] = (df['FVG_Bull_Top'] + df['FVG_Bull_Bottom']) / 2
    df['FVG_Bear_Mid'] = (df['FVG_Bear_Top'] + df['FVG_Bear_Bottom']) / 2
    df['FVG_Bull_Size'] = np.where(df['FVG_Bull'] == 1,
                                    (df['FVG_Bull_Top'] - df['FVG_Bull_Bottom']) / df['Close'], 0.0)
    df['FVG_Bear_Size'] = np.where(df['FVG_Bear'] == 1,
                                    (df['FVG_Bear_Top'] - df['FVG_Bear_Bottom']) / df['Close'], 0.0)
    df['Dist_To_Last_Bull_FVG'] = (df['Close'] - df['FVG_Bull_Bottom']) / df['Close']
    df['Dist_To_Last_Bear_FVG'] = (df['FVG_Bear_Top'] - df['Close']) / df['Close']
    df['Bull_IFVG'] = ((df['FVG_Bull_Bottom'].notna()) & (df['Close'] < df['FVG_Bull_Bottom'])).astype(int)
    df['Bear_IFVG'] = ((df['FVG_Bear_Top'].notna()) & (df['Close'] > df['FVG_Bear_Top'])).astype(int)
    df['Dist_To_Bull_IFVG'] = np.where(df['FVG_Bull_Bottom'].notna(),
                                        (df['Close'] - df['FVG_Bull_Bottom']) / df['Close'], np.nan)
    df['Dist_To_Bear_IFVG'] = np.where(df['FVG_Bear_Top'].notna(),
                                        (df['FVG_Bear_Top'] - df['Close']) / df['Close'], np.nan)

    _body = df['Close'] - df['Open']
    _avg_body = _body.abs().rolling(20).mean()
    df['Bull_OB'] = ((_body.shift(1) < 0) & (_body > 1.5 * _avg_body)).astype(int)
    df['Bear_OB'] = ((_body.shift(1) > 0) & (-_body > 1.5 * _avg_body)).astype(int)
    _o1 = df['Open'].shift(1); _c1 = df['Close'].shift(1)
    _h1s = df['High'].shift(1); _l1s = df['Low'].shift(1)
    _bt1 = pd.concat([_o1, _c1], axis=1).max(axis=1)
    _bb1 = pd.concat([_o1, _c1], axis=1).min(axis=1)
    df['Bull_OB_Body_Top'] = pd.Series(np.where(df['Bull_OB'] == 1, _bt1, np.nan)).ffill()
    df['Bull_OB_Body_Bottom'] = pd.Series(np.where(df['Bull_OB'] == 1, _bb1, np.nan)).ffill()
    df['Bull_OB_Wick_Bottom'] = pd.Series(np.where(df['Bull_OB'] == 1, _l1s, np.nan)).ffill()
    df['Bear_OB_Body_Top'] = pd.Series(np.where(df['Bear_OB'] == 1, _bt1, np.nan)).ffill()
    df['Bear_OB_Body_Bottom'] = pd.Series(np.where(df['Bear_OB'] == 1, _bb1, np.nan)).ffill()
    df['Bear_OB_Wick_Top'] = pd.Series(np.where(df['Bear_OB'] == 1, _h1s, np.nan)).ffill()
    df['Bull_OB_Entry_Aggressive'] = df['Bull_OB_Body_Top']
    df['Bull_OB_Entry_Moderate'] = (df['Bull_OB_Body_Bottom'] + df['Bull_OB_Wick_Bottom']) / 2
    df['Bull_OB_Entry_Conservative'] = df['Bull_OB_Wick_Bottom']
    df['Bear_OB_Entry_Aggressive'] = df['Bear_OB_Body_Bottom']
    df['Bear_OB_Entry_Moderate'] = (df['Bear_OB_Body_Top'] + df['Bear_OB_Wick_Top']) / 2
    df['Bear_OB_Entry_Conservative'] = df['Bear_OB_Wick_Top']
    df['Bull_OB_Level'] = df['Bull_OB_Body_Top']
    df['Bear_OB_Level'] = df['Bear_OB_Body_Bottom']
    df['Dist_To_Bull_OB'] = (df['Close'] - df['Bull_OB_Level']) / df['Close']
    df['Dist_To_Bear_OB'] = (df['Bear_OB_Level'] - df['Close']) / df['Close']
    _obr = (_h1s - _l1s).replace(0, np.nan)
    df['Bull_OB_Body_Ratio'] = pd.Series(np.where(
        df['Bull_OB'] == 1, (_bt1 - _bb1).abs() / _obr, np.nan)).ffill()
    df['Bear_OB_Body_Ratio'] = pd.Series(np.where(
        df['Bear_OB'] == 1, (_bt1 - _bb1).abs() / _obr, np.nan)).ffill()
    df['Bull_BB'] = ((df['Bull_OB_Level'].notna()) & (df['Close'] < df['Bull_OB_Level'])).astype(int)
    df['Bear_BB'] = ((df['Bear_OB_Level'].notna()) & (df['Close'] > df['Bear_OB_Level'])).astype(int)
    df['Dist_To_Bull_BB'] = np.where(df['Bull_OB_Level'].notna(),
                                      (df['Close'] - df['Bull_OB_Level']) / df['Close'], np.nan)
    df['Dist_To_Bear_BB'] = np.where(df['Bear_OB_Level'].notna(),
                                      (df['Bear_OB_Level'] - df['Close']) / df['Close'], np.nan)
    EQ_TOL = 0.0015
    df['Equal_High'] = (abs(df['High'] - df['High'].shift(1)) / df['High'] < EQ_TOL).astype(int)
    df['Equal_Low'] = (abs(df['Low'] - df['Low'].shift(1)) / df['Low'] < EQ_TOL).astype(int)
    df['Liquidity_Sweep_High'] = ((df['High'] > df['Last_Swing_High']) &
                                   (df['Close'] < df['Last_Swing_High'])).astype(int)
    df['Liquidity_Sweep_Low'] = ((df['Low'] < df['Last_Swing_Low']) &
                                  (df['Close'] > df['Last_Swing_Low'])).astype(int)
    df['Premium_Discount'] = (df['Close'] - df['Low'].rolling(20).min()) / (
        df['High'].rolling(20).max() - df['Low'].rolling(20).min())
    df['Displacement'] = (_body.abs() > 1.5 * _avg_body).astype(int)

    prev_close = df['Close'].shift(1)
    tr = pd.concat([df['High'] - df['Low'],
                    (df['High'] - prev_close).abs(),
                    (df['Low'] - prev_close).abs()], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()

    df['Dealing_High'] = df['High'].rolling(50, min_periods=10).max()
    df['Dealing_Low'] = df['Low'].rolling(50, min_periods=10).min()
    df['Dealing_EQ'] = (df['Dealing_High'] + df['Dealing_Low']) / 2
    df['Dealing_Pos'] = (df['Close'] - df['Dealing_Low']) / (df['Dealing_High'] - df['Dealing_Low'])
    df['In_Discount'] = (df['Dealing_Pos'] < 0.50).astype(int)
    df['In_Premium'] = (df['Dealing_Pos'] > 0.50).astype(int)
    df['Dist_To_PDH'] = (df['High'].shift(1) - df['Close']) / df['Close']
    df['Dist_To_PDL'] = (df['Close'] - df['Low'].shift(1)) / df['Close']

    LADDER_SIZE = 5
    def _ladder(flags, levels):
        out = []; cur = []
        for i, (f, lvl) in enumerate(zip(flags, levels)):
            if f and not np.isnan(lvl):
                cur.append(float(lvl))
                if len(cur) > LADDER_SIZE:
                    cur.pop(0)
            out.append(list(cur))
        return out
    sh_raw, sl_raw = _find_swings(df, 10, 10)
    df['Swing_High_Ladder'] = _ladder(sh_raw, df['High'].values)
    df['Swing_Low_Ladder'] = _ladder(sl_raw, df['Low'].values)

    ZW = 0.005
    def _ob_touches(flag, lvl, highs, lows):
        n = len(flag); t = np.zeros(n); last = np.nan; last_i = -1; c = 0
        for i in range(n):
            if flag[i] == 1:
                last = lvl[i]; last_i = i; c = 0
            if not np.isnan(last) and 0 <= last_i < i:
                if lows[i] <= last * (1 + ZW) and highs[i] >= last * (1 - ZW):
                    c += 1
            t[i] = c
        return t
    def _ob_disp(flag, lvl, closes, w=5):
        n = len(closes); d = np.full(n, np.nan); last = np.nan; last_i = -1
        for i in range(n):
            if flag[i] == 1:
                last = lvl[i]; last_i = i
            if last_i >= 0 and (i - last_i) <= w:
                d[i] = max((np.max(closes[last_i:i+1]) - last) / last, 0)
        return d
    def _ob_age(flag):
        n = len(flag); a = np.full(n, np.nan); last_i = -1
        for i in range(n):
            if flag[i] == 1: last_i = i
            if last_i >= 0: a[i] = i - last_i
        return a
    def _ob_volz(flag, vz, lag=2):
        n = len(vz); v = np.full(n, np.nan)
        for i in range(n):
            if flag[i] == 1: v[i] = vz[max(0, i - lag)]
        return pd.Series(v).ffill().bfill().fillna(0.0).values
    def _wick_pen(flag, bb, wb, lows, w=10):
        n = len(lows); p = np.full(n, np.nan); last_b = np.nan; last_w = np.nan; last_i = -1
        for i in range(n):
            if flag[i] == 1: last_b = bb[i]; last_w = wb[i]; last_i = i
            if last_i >= 0 and (i - last_i) <= w:
                rng = last_b - last_w
                if rng > 0:
                    p[i] = max(0.0, min((last_b - np.min(lows[last_i:i+1])) / rng, 1.0))
        return p
    _bof = df['Bull_OB'].values; _bol = df['Bull_OB_Level'].values
    _highs = df['High'].values; _lows = df['Low'].values; _closes = df['Close'].values
    _vz = df['Volume_Zscore'].fillna(0).values
    df['Bull_OB_Touches'] = _ob_touches(_bof, _bol, _highs, _lows)
    df['Bull_OB_Displacement'] = _ob_disp(_bof, _bol, _closes)
    df['Bull_OB_Age'] = _ob_age(_bof)
    df['Bull_OB_HighVol'] = _ob_volz(_bof, _vz)
    df['Bull_OB_Wick_Penetration'] = _wick_pen(
        _bof, df['Bull_OB_Body_Bottom'].fillna(0).values,
        df['Bull_OB_Wick_Bottom'].fillna(0).values, _lows)
    _rof = df['Bear_OB'].values; _rol = df['Bear_OB_Level'].values
    df['Bear_OB_Touches'] = _ob_touches(_rof, _rol, _highs, _lows)
    df['Bear_OB_Displacement'] = _ob_disp(_rof, _rol, _closes)
    df['Bear_OB_Age'] = _ob_age(_rof)
    df['Bear_OB_HighVol'] = _ob_volz(_rof, _vz)
    df['Bear_OB_Wick_Penetration'] = _wick_pen(
        _rof, -df['Bear_OB_Body_Top'].fillna(0).values,
        -df['Bear_OB_Wick_Top'].fillna(0).values, -_highs)
    def _fvg_track(flag, ft, fb, highs, lows, closes, vz):
        n = len(flag)
        fp = np.full(n, np.nan); t = np.zeros(n); a = np.full(n, np.nan)
        d = np.full(n, np.nan); vf = np.full(n, np.nan)
        last_t = np.nan; last_b = np.nan; last_i = -1; last_mc = np.nan; tc = 0
        for i in range(n):
            if flag[i] == 1:
                last_t = ft[i]; last_b = fb[i]; last_i = i
                last_mc = closes[i]; tc = 0; vf[i] = vz[max(0, i - 1)]
            if 0 <= last_i < i:
                gap = last_t - last_b
                if gap > 0:
                    lo_v = np.min(lows[last_i:i+1]); hi_v = np.max(highs[last_i:i+1])
                    if lo_v <= last_b and hi_v >= last_t: fp[i] = 1.0
                    elif lo_v > last_t or hi_v < last_b: fp[i] = 0.0
                    else: fp[i] = min((last_t - max(min(lo_v, last_t), last_b)) / gap, 1.0)
                    if lows[i] <= last_t and highs[i] >= last_b: tc += 1
                a[i] = i - last_i
                mid = (last_t + last_b) / 2
                if last_mc > mid:
                    d[i] = max((np.max(closes[last_i:i+1]) - mid) / mid, 0)
                else:
                    d[i] = max((mid - np.min(closes[last_i:i+1])) / mid, 0)
            t[i] = tc
        return fp, t, a, d, vf
    _bt = _fvg_track(_fvg_bull_flag, df['FVG_Bull_Top'].values, df['FVG_Bull_Bottom'].values,
                     _highs, _lows, _closes, _vz)
    df['Bull_FVG_Fill_Pct'] = _bt[0]; df['Bull_FVG_Touches'] = _bt[1]
    df['Bull_FVG_Age'] = _bt[2]; df['Bull_FVG_Displacement'] = _bt[3]
    df['Bull_FVG_VolAtForm'] = pd.Series(_bt[4]).ffill().bfill().fillna(0.0).values
    _rt = _fvg_track(_fvg_bear_flag, df['FVG_Bear_Top'].values, df['FVG_Bear_Bottom'].values,
                     _highs, _lows, _closes, _vz)
    df['Bear_FVG_Fill_Pct'] = _rt[0]; df['Bear_FVG_Touches'] = _rt[1]
    df['Bear_FVG_Age'] = _rt[2]; df['Bear_FVG_Displacement'] = _rt[3]
    df['Bear_FVG_VolAtForm'] = pd.Series(_rt[4]).ffill().bfill().fillna(0.0).values
    def _s(c): return c.fillna(0.0)
    df['Bull_OB_Strength'] = (-_s(df['Bull_OB_Touches'])*0.5 + _s(df['Bull_OB_Displacement'])*8.0
                              + _s(df['Bull_OB_HighVol'])*0.3 - _s(df['Bull_OB_Age'])/100.0)
    df['Bear_OB_Strength'] = (-_s(df['Bear_OB_Touches'])*0.5 + _s(df['Bear_OB_Displacement'])*8.0
                              + _s(df['Bear_OB_HighVol'])*0.3 - _s(df['Bear_OB_Age'])/100.0)
    df['Bull_FVG_Strength'] = (-_s(df['Bull_FVG_Touches'])*0.5 + _s(df['Bull_FVG_Displacement'])*8.0
                               - _s(df['Bull_FVG_Fill_Pct'])*1.5 - _s(df['Bull_FVG_Age'])/100.0)
    df['Bear_FVG_Strength'] = (-_s(df['Bear_FVG_Touches'])*0.5 + _s(df['Bear_FVG_Displacement'])*8.0
                               - _s(df['Bear_FVG_Fill_Pct'])*1.5 - _s(df['Bear_FVG_Age'])/100.0)
    df = df.replace([np.inf, -np.inf], np.nan)
    zone_fill_zero = [
        'Bull_OB_Touches', 'Bull_OB_Displacement', 'Bull_OB_Age', 'Bull_OB_HighVol',
        'Bear_OB_Touches', 'Bear_OB_Displacement', 'Bear_OB_Age', 'Bear_OB_HighVol',
        'Bull_FVG_Fill_Pct', 'Bull_FVG_Touches', 'Bull_FVG_Age',
        'Bull_FVG_Displacement', 'Bull_FVG_VolAtForm',
        'Bear_FVG_Fill_Pct', 'Bear_FVG_Touches', 'Bear_FVG_Age',
        'Bear_FVG_Displacement', 'Bear_FVG_VolAtForm',
        'Bull_OB_Strength', 'Bear_OB_Strength', 'Bull_FVG_Strength', 'Bear_FVG_Strength',
        'Bull_OB_Wick_Penetration', 'Bear_OB_Wick_Penetration',
        'Bull_OB_Body_Ratio', 'Bear_OB_Body_Ratio',
    ]
    for c in zone_fill_zero:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    classic_features = [
        'Return', 'Log_Return', 'Price_vs_MA5', 'Price_vs_MA20', 'Price_vs_MA50',
        'MA5_vs_MA20', 'MA50_vs_MA200', 'Volatility_5', 'Volatility_20', 'Volatility_50',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Position',
        'Volume_Change', 'Volume_vs_MA20', 'Body', 'Upper_Shadow', 'Lower_Shadow', 'Gap',
        'Return_Lag1', 'Return_Lag2', 'Return_Lag3', 'Return_Lag5', 'Return_Lag10',
        'Volume_Price_Divergence', 'Vol_ROC_5', 'Vol_ROC_10', 'Consec_Up', 'Consec_Down',
        'Overnight_Return', 'Intraday_Return', 'Overnight_vs_Intraday', 'DayOfWeek',
    ]
    ict_smc_features = [
        'BOS_Bull', 'BOS_Bear', 'Market_Structure', 'CHoCH',
        'FVG_Bull', 'FVG_Bear', 'FVG_Bull_Size', 'FVG_Bear_Size',
        'Dist_To_Last_Bull_FVG', 'Dist_To_Last_Bear_FVG',
        'Bull_IFVG', 'Bear_IFVG', 'Dist_To_Bull_IFVG', 'Dist_To_Bear_IFVG',
        'Bull_OB', 'Bear_OB', 'Dist_To_Bull_OB', 'Dist_To_Bear_OB',
        'Bull_BB', 'Bear_BB', 'Dist_To_Bull_BB', 'Dist_To_Bear_BB',
        'Equal_High', 'Equal_Low', 'Liquidity_Sweep_High', 'Liquidity_Sweep_Low',
        'Premium_Discount', 'Displacement', 'Counter_Trend_Bars',
        'Pullback_In_Uptrend_MTF', 'Bounce_In_Downtrend_MTF',
        'Dealing_Pos', 'In_Discount', 'In_Premium',
        'Dist_To_PDH', 'Dist_To_PDL',
    ]
    zone_strength_features = [
        'Bull_OB_Touches', 'Bull_OB_Displacement', 'Bull_OB_Age', 'Bull_OB_HighVol',
        'Bear_OB_Touches', 'Bear_OB_Displacement', 'Bear_OB_Age', 'Bear_OB_HighVol',
        'Bull_FVG_Fill_Pct', 'Bull_FVG_Touches', 'Bull_FVG_Age',
        'Bull_FVG_Displacement', 'Bull_FVG_VolAtForm',
        'Bear_FVG_Fill_Pct', 'Bear_FVG_Touches', 'Bear_FVG_Age',
        'Bear_FVG_Displacement', 'Bear_FVG_VolAtForm',
        'Bull_OB_Strength', 'Bear_OB_Strength', 'Bull_FVG_Strength', 'Bear_FVG_Strength',
        'Bull_OB_Wick_Penetration', 'Bear_OB_Wick_Penetration',
        'Bull_OB_Body_Ratio', 'Bear_OB_Body_Ratio',
    ]
    feature_columns = classic_features + ict_smc_features + zone_strength_features
    if HAS_DXY:
        feature_columns = feature_columns + dxy_features

    df['Future_Return'] = df['Close'].shift(-1) / df['Close'] - 1
    df['Next_Date'] = df['Date'].shift(-1)
    VOL_MULT = 0.5
    MIN_THRESHOLD = 0.001
    df['Move_Threshold'] = (df['Volatility_20'] * VOL_MULT).clip(lower=MIN_THRESHOLD)
    CLASS_NAMES = {0: 'DOWN', 1: 'CONSOLIDATION', 2: 'UP'}
    CLASS_ORDER = ['DOWN', 'CONSOLIDATION', 'UP']
    _target = np.select(
        [df['Future_Return'] > df['Move_Threshold'],
         df['Future_Return'] < -df['Move_Threshold']],
        [2, 0], default=1).astype(float)
    _target[df['Future_Return'].isna()] = np.nan
    df['Target'] = _target

    print("🔧 Historical analog features...")
    ANALOG_K = 25
    ANALOG_MIN_HISTORY = 500
    ANALOG_RECOMPUTE_EVERY = 20
    SIG_FEATURES = [
        'Return', 'Price_vs_MA5', 'Price_vs_MA20', 'MA5_vs_MA20',
        'MA50_vs_MA200', 'Volatility_20', 'RSI', 'BB_Position',
        'Volume_vs_MA20', 'Body', 'MACD_Hist', 'Premium_Discount',
    ]
    _sig_df = df[SIG_FEATURES].copy()
    _sig_norm = (_sig_df - _sig_df.mean()) / (_sig_df.std().replace(0, np.nan))
    _sig_matrix = _sig_norm.fillna(0).values
    n = len(df)
    _target_arr = df['Target'].values
    _future_ret_arr = df['Future_Return'].values
    def _analog_stats(idx, history_end):
        if history_end < ANALOG_MIN_HISTORY:
            return (np.nan,) * 6
        diffs = _sig_matrix[:history_end] - _sig_matrix[idx]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        if len(dists) <= ANALOG_K:
            nearest_idx = np.arange(len(dists))
        else:
            nearest_idx = np.argpartition(dists, ANALOG_K)[:ANALOG_K]
        tgt = _target_arr[nearest_idx]; ret = _future_ret_arr[nearest_idx]
        valid = ~np.isnan(tgt) & ~np.isnan(ret)
        if not valid.any():
            return (np.nan,) * 6
        tgt = tgt[valid].astype(int); ret = ret[valid]
        return (float((tgt == 2).mean()), float((tgt == 0).mean()),
                float((tgt == 1).mean()), float(np.mean(ret)),
                float(np.std(ret)), float(np.mean(dists[nearest_idx][valid])))
    analog_up = np.full(n, np.nan); analog_down = np.full(n, np.nan)
    analog_flat = np.full(n, np.nan); analog_mean = np.full(n, np.nan)
    analog_std = np.full(n, np.nan)
    for chunk_start in range(0, n, ANALOG_RECOMPUTE_EVERY):
        chunk_end = min(chunk_start + ANALOG_RECOMPUTE_EVERY, n)
        if chunk_start < ANALOG_MIN_HISTORY:
            continue
        for j in range(chunk_start, chunk_end):
            up, dn, fl, mu, sd, _ = _analog_stats(j, chunk_start)
            analog_up[j], analog_down[j], analog_flat[j] = up, dn, fl
            analog_mean[j], analog_std[j] = mu, sd
    df['Analog_Up_Rate'] = analog_up; df['Analog_Down_Rate'] = analog_down
    df['Analog_Flat_Rate'] = analog_flat; df['Analog_Mean_Return'] = analog_mean
    df['Analog_Std_Return'] = analog_std
    df['Analog_Up_Minus_Down'] = df['Analog_Up_Rate'] - df['Analog_Down_Rate']
    for c, v in {'Analog_Up_Rate': 0.5, 'Analog_Down_Rate': 0.5, 'Analog_Flat_Rate': 0.0,
                  'Analog_Mean_Return': 0.0, 'Analog_Std_Return': 0.0,
                  'Analog_Up_Minus_Down': 0.0}.items():
        df[c] = df[c].fillna(v)
    pattern_features = [
        'Analog_Up_Rate', 'Analog_Down_Rate', 'Analog_Flat_Rate',
        'Analog_Mean_Return', 'Analog_Std_Return', 'Analog_Up_Minus_Down',
    ]
    feature_columns = feature_columns + pattern_features

    all_nan_cols = [c for c in feature_columns if df[c].isna().all()]
    if all_nan_cols:
        print(f"\n⚠️  Dropping all-NaN: {all_nan_cols}")
        feature_columns = [c for c in feature_columns if c not in all_nan_cols]

    df_all = df.dropna(subset=feature_columns).reset_index(drop=True)
    df_model = df_all.dropna(subset=['Target']).reset_index(drop=True)
    df_model['Target'] = df_model['Target'].astype(int)
    df_live = df_all[df_all['Target'].isna()].reset_index(drop=True)

    print(f"\n✅ Features created ({len(feature_columns)} total) | "
          f"Training rows: {len(df_model)} | Live rows: {len(df_live)}")

    split_idx = int(len(df_model) * 0.8)
    train = df_model.iloc[:split_idx]
    test = df_model.iloc[split_idx:]
    X_train = train[feature_columns]; y_train = train['Target']
    X_test = test[feature_columns]; y_test = test['Target']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n🤖 Training Random Forest (Model 1 — daily bias)...")
    model1 = RandomForestClassifier(
        n_estimators=400, max_depth=14,
        min_samples_split=50, min_samples_leaf=20,
        class_weight='balanced', random_state=42, n_jobs=-1)
    model1.fit(X_train_scaled, y_train)
    y_pred = model1.predict(X_test_scaled)
    y_proba = model1.predict_proba(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Test Accuracy: {accuracy:.2%}")

    importance_df = pd.DataFrame({
        'Feature': feature_columns, 'Importance': model1.feature_importances_
    }).sort_values('Importance', ascending=False)

    # ---------------- H1 INTRADAY TRADE MODEL ----------------
    h1_model = {"accepted": False, "n": 0, "auc": float('nan'), "reason": "H1 not attempted"}
    if HAS_HOURLY and H1_MODEL_ENABLED:
        print("\n🎯 Training intraday H1 trade model (simulated entries)...")
        try:
            daily_base = daily_base_ctx(df_model)
            oof_probs = oof_daily_probs(df_model, feature_columns)
            training_df = build_training_set(h1_P, daily_base, oof_probs, index_name)
            if len(training_df):
                h1_model = train_model(training_df, index_name)
        except Exception as e:
            print(f"   ⚠️  H1 training failed: {e}")
            import traceback; traceback.print_exc()
            h1_model = {"accepted": False, "n": 0, "auc": float('nan'), "reason": str(e)}

    journal_state = {'cals': None, 'meta': None}

    def _find_row_by_date(frame, LATEST_DATE):
        input_date = pd.to_datetime(LATEST_DATE)
        matches = frame[frame['Date'] == input_date]
        if len(matches) > 0:
            return matches, False
        closest_idx = (frame['Date'] - input_date).abs().idxmin()
        return frame.loc[[closest_idx]], True

    def _next_trading_date(after_date):
        future = df_all[df_all['Date'] > after_date]
        if len(future):
            return future['Date'].iloc[0]
        return after_date + pd.offsets.BDay(1)

    _DIR_ICON = {'UP': '🟢 UP', 'DOWN': '🔴 DOWN', 'CONSOLIDATION': '🟡 CONSOLIDATION'}

    def _describe_mtf(macro, micro, pullback, bounce):
        if pullback == 1:
            return 'Bullish macro, bearish micro — PULLBACK IN UPTREND'
        if bounce == 1:
            return 'Bearish macro, bullish micro — BOUNCE IN DOWNTREND'
        if macro == 1 and micro == 1:
            return 'All timeframes bullish'
        if macro == -1 and micro == -1:
            return 'All timeframes bearish'
        return 'Mixed / unresolved'

    def _bias_gate(row, probs):
        dominant = max(probs, key=probs.get)
        if dominant == 'UP':
            return 'LONG'
        if dominant == 'DOWN':
            return 'SHORT'
        return 'LONG' if probs['UP'] >= probs['DOWN'] else 'SHORT'

    def _dol_candidates(side, entry, stop_loss, row):
        out = []
        pdh = row['High'].shift(1).iloc[0]
        pdl = row['Low'].shift(1).iloc[0]
        if side == 'LONG':
            swing_highs = row['Swing_High_Ladder'].iloc[0] or []
            for lvl in swing_highs:
                if pd.notna(lvl) and lvl > entry:
                    out.append((float(lvl), f'Swing High {lvl:.2f}', 3))
            if pd.notna(pdh) and pdh > entry:
                out.append((float(pdh), f'PDH {pdh:.2f}', 2))
        else:
            swing_lows = row['Swing_Low_Ladder'].iloc[0] or []
            for lvl in swing_lows:
                if pd.notna(lvl) and lvl < entry:
                    out.append((float(lvl), f'Swing Low {lvl:.2f}', 3))
            if pd.notna(pdl) and pdl < entry:
                out.append((float(pdl), f'PDL {pdl:.2f}', 2))
        return out

    def recommend_trade_entry(row, probs):
        close = float(row['Close'].iloc[0])
        atr = row['ATR_14'].iloc[0]
        if pd.isna(atr):
            atr = 0.01 * close
        zone_pos = row['Premium_Discount'].iloc[0]
        if pd.isna(zone_pos):
            zone_pos = 0.5
        dominant = max(probs, key=probs.get)
        confidence = probs[dominant]
        conviction = 'High' if confidence >= 0.55 else ('Medium' if confidence >= 0.38 else 'Low')
        side = _bias_gate(row, probs)
        hourly_meta = None; stop_anchor = None; target_anchor = None
        entry = close
        entry_type = 'market at close (no H1 zone available)'
        stop_loss = close - 1.5 * atr if side == 'LONG' else close + 1.5 * atr
        stop_src = 'ATR 1.5x fallback'
        take_profit = close + 3.0 * atr if side == 'LONG' else close - 3.0 * atr
        target_src = '3 ATR fallback'
        ob_style = 'auto'; zone_strength = np.nan
        if HAS_HOURLY and HOURLY_OPT_ENABLED:
            ctx = daily_ctx_at(daily_base_ctx(row), 0, probs, side)
            pick, why = pick_live_trade(h1_P, len(hourly_df) - 1, side, ctx, h1_model)
            if pick is None:
                return {'action': 'NO TRADE', 'reason': why,
                        'confidence': confidence, 'conviction': conviction}
            entry, stop_loss, take_profit = pick['entry'], pick['stop'], pick['target']
            entry_type = pick['entry_type']
            stop_src = pick['stop_source']
            target_src = pick['target_source']
            zone_strength = pick['score']
            hourly_meta = pick['hourly_meta']
            hourly_meta['daily_entry'] = float(close)
            hourly_meta['daily_entry_type'] = 'market at close'
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        actual_rr = round(reward / risk, 2) if risk > 0 else None
        if hourly_meta is not None:
            hourly_meta['hourly_rr'] = actual_rr
            hourly_meta['daily_rr'] = actual_rr
        return {
            'action': side, 'entry_type': entry_type, 'entry_style': ob_style,
            'entry': round(entry, 2), 'stop_loss': round(stop_loss, 2),
            'stop_source': stop_src, 'take_profit': round(take_profit, 2),
            'target_source': target_src, 'risk_reward': actual_rr,
            'confidence': confidence, 'conviction': conviction,
            'zone': 'Discount' if zone_pos < 0.5 else 'Premium',
            'zone_strength': zone_strength,
            'reason': f"{side} — {dominant} at {confidence:.1%}",
            'hourly_meta': hourly_meta, 'daily_entry': round(close, 2),
            'daily_rr': actual_rr, 'bias_gate_passed': True,
            'dol_target': np.nan, 'dol_source': '', 'dol_rank': np.nan,
            'stop_anchor': stop_anchor, 'target_anchor': target_anchor,
        }

    def adapt_trade_with_journal(baseline, row, probs, next_date):
        info = {'params': None, 'meta_p': None, 'meta_exp_r': None, 'filtered': False}
        rec = _journal_record(row, probs, baseline, next_date, 'live')
        info['record'] = rec
        if baseline['action'] == 'NO TRADE' or not USE_JOURNAL:
            return baseline, info
        # Intraday H1 trades are sized in H1 ATR; the journal calibration / meta filter
        # replay on daily bars and daily ATR, so they are skipped for these trades.
        if baseline.get('hourly_meta') is not None:
            return baseline, info
        trade = dict(baseline)
        close = float(row['Close'].iloc[0])
        atr = rec['atr']
        params = _pick_calibration(journal_state['cals'], baseline['action'])
        if params is not None:
            trade = _apply_calibration(baseline, close, atr, params)
            info['params'] = params
        meta = journal_state['meta']
        if meta is not None and meta.get('accepted'):
            p = _meta_predict(meta, rec)
            rr = baseline.get('risk_reward') or 1.0
            exp_r = p * rr - (1 - p)
            info['meta_p'] = p; info['meta_exp_r'] = exp_r
            if exp_r < META_MIN_EXPECTED_R:
                info['filtered'] = True
                trade = {'action': 'NO TRADE',
                          'reason': f"Journal filter: p_win={p:.1%}, expected {exp_r:+.2f}R"}
        return trade, info

    
    
    def predict_for_date(LATEST_DATE):

        input_date = pd.to_datetime(LATEST_DATE)

        use_live = len(df_live) and (df_live['Date'] == input_date).any()
        source = df_live if use_live else df_model

        row, was_fuzzy = _find_row_by_date(source, input_date)

        if was_fuzzy:
            print(f"⚠️  Date not found. Using closest: {row['Date'].iloc[0].date()}")

        X_input = row[feature_columns]
        X_scaled = scaler.transform(X_input)

        prediction = int(model1.predict(X_scaled)[0])
        probability = model1.predict_proba(X_scaled)[0]

        probs = {
            CLASS_NAMES[int(c)]: float(p)
            for c, p in zip(model1.classes_, probability)
        }

        predicted_state = CLASS_NAMES[prediction]
        confidence = probs[predicted_state]

        input_date_actual = row['Date'].iloc[0]
        close_t = row['Close'].iloc[0]
        next_date = _next_trading_date(input_date_actual)

        macro_s = int(row['Macro_Structure'].iloc[0])
        micro_s = int(row['Micro_Structure'].iloc[0])

        mtf = _describe_mtf(
            macro_s,
            micro_s,
            int(row['Pullback_In_Uptrend_MTF'].iloc[0]),
            int(row['Bounce_In_Downtrend_MTF'].iloc[0])
        )

        # Detect active SMC signals on the prediction date
        signals = []

        if row['Bull_BB'].iloc[0] == 1:
            signals.append('Bull Breaker')

        if row['Bear_BB'].iloc[0] == 1:
            signals.append('Bear Breaker')

        if row['Bull_IFVG'].iloc[0] == 1:
            signals.append('Bull IFVG')

        if row['Bear_IFVG'].iloc[0] == 1:
            signals.append('Bear IFVG')

        if row['BOS_Bull'].iloc[0] == 1:
            signals.append('BOS Bull')

        if row['BOS_Bear'].iloc[0] == 1:
            signals.append('BOS Bear')

        if row['CHoCH'].iloc[0] == 1:
            signals.append('CHoCH')

        if row['Liquidity_Sweep_High'].iloc[0] == 1:
            signals.append('Sweep High')

        if row['Liquidity_Sweep_Low'].iloc[0] == 1:
            signals.append('Sweep Low')

        # Prediction output
        print("\n" + "=" * 60)
        print(f"🔮 {index_name} PREDICTION")
        print("=" * 60)

        print(f"📅 Data date (T):        {input_date_actual.date()}")
        print(f"💰 Close price (T):      {close_t:.2f}")
        print(f"📅 Predicting move for:  {next_date.date()}")

        print(f"\n🎯 Predicted state:      {_DIR_ICON[predicted_state]}")
        print(f"📊 Confidence:           {confidence:.2%}")

        print(
            f"   UP {probs['UP']:.1%} | "
            f"DOWN {probs['DOWN']:.1%} | "
            f"CONS {probs['CONSOLIDATION']:.1%}"
        )

        print(f"\n🧭 MTF: {mtf}")
        print(f"   Dealing pos: {row['Dealing_Pos'].iloc[0]:.2f}")

        print(
            f"\n🔔 Active SMC signals: "
            f"{', '.join(signals) if signals else 'None'}"
        )

        # Trade recommendation
        baseline = recommend_trade_entry(row, probs)

        trade, adapt_info = adapt_trade_with_journal(
            baseline, row, probs, next_date
        )

        print(f"\n💼 TRADE ENTRY RECOMMENDATION")
        print("-" * 60)

        if trade['action'] == 'NO TRADE':
            print(f"   NO TRADE — {trade['reason']}")

        else:
            hm = trade.get('hourly_meta')

            print(f"   Side:         {trade['action']}")
            print(f"   Entry:        {trade['entry']:.2f} ({trade['entry_type']})")
            print(f"   Stop:         {trade['stop_loss']:.2f} ({trade['stop_source']})")
            print(f"   Target:       {trade['take_profit']:.2f} ({trade['target_source']})")
            print(f"   R:R:          {trade['risk_reward']}")

            if hm is not None:
                print(f"\n   H1 ZONE (model):")
                print(f"     Kind:          {hm['kind'].upper()}")
                print(f"     Probability:   {hm['model_prob']:.1%}")
                print(f"     H1 trend:      {hm['h1_trend']:+d}")
                print(f"     Age:           {hm['bars_since_zone']} bars")
                print(f"     Session:       {hm['session']} {hm['hour_utc']:02d}:00 UTC")
                print(f"     OTE aligned:   {'YES' if hm.get('ote_aligned') else 'NO'}")

        # Journal output
        if USE_JOURNAL and baseline['action'] != 'NO TRADE':

            if adapt_info['params'] is not None:
                s_, e_, t_ = adapt_info['params']

                print(
                    f"\n📓 Journal: entry +{s_:.2f} ATR | "
                    f"stop +{e_:.2f} ATR | target x{t_:.2f}"
                )

            if adapt_info['meta_p'] is not None:
                print(
                    f"📓 Win prob: {adapt_info['meta_p']:.1%} "
                    f"(exp {adapt_info['meta_exp_r']:+.2f}R)"
                    f"{'  → FILTERED' if adapt_info['filtered'] else ''}"
                )

        print("=" * 60)

        # Return all prediction data, including SMC signals
        return (
            predicted_state,
            probs,
            row,
            trade,
            baseline,
            adapt_info,
            signals
        )

    # ---------------- RUN ----------------
    LATEST_DATE = df['Date'].max().date()
    print(f"\n🔎 Latest date: {LATEST_DATE}")

    journal = None
    journal_file = _journal_path(index_name)
    if USE_JOURNAL:
        print("\n" + "-"*60)
        print(f"📓 TRADE JOURNAL — {index_name}")
        print("-"*60)
        journal = _load_journal(journal_file)
        n_resolved = _resolve_journal(journal, df)
        print(f"✅ Resolved {n_resolved} pending signal(s)")
        _print_journal_summary(journal, index_name)
        learn_trades = _trades_for_learning(journal)
        journal_state['cals'] = _fit_calibrations(learn_trades)
        journal_state['meta'] = _fit_meta(learn_trades)
        print(f"\n🧠 LEARNING FROM JOURNAL ({len(learn_trades)} resolved trades)")
        for key in ('ALL', 'LONG', 'SHORT'):
            c = journal_state['cals'][key]
            if c.get('accepted'):
                s_, e_, t_ = c['params']
                print(f"   Cal [{key}]: entry+{s_:.2f}ATR, stop+{e_:.2f}ATR, target×{t_:.2f}")
            elif 'reason' in c:
                print(f"   Cal [{key}]: not applied ({c['reason']})")
        if journal_state['meta'].get('accepted'):
            print(f"   Meta filter ACTIVE (AUC {journal_state['meta']['auc']:.3f})")
        _walkforward_report(journal, index_name)


    predicted_state, probs, row, trade, baseline, adapt_info, signals = predict_for_date(LATEST_DATE)

    if USE_JOURNAL:
        rec = adapt_info['record']
        rec['adapt_applied'] = bool(adapt_info['params'] is not None and trade['action'] != 'NO TRADE')
        if rec['adapt_applied']:
            rec['adapt_shift'], rec['adapt_extra'], rec['adapt_tscale'] = adapt_info['params']
            rec['adapt_entry'] = trade['entry']
            rec['adapt_stop'] = trade['stop_loss']
            rec['adapt_target'] = trade['take_profit']
        rec['journal_win_prob'] = adapt_info['meta_p'] if adapt_info['meta_p'] is not None else np.nan
        rec['meta_filtered'] = bool(adapt_info['filtered'])
        if trade.get('hourly_meta') is not None:
            hm = trade['hourly_meta']
            rec['hourly_hour_utc'] = hm.get('hour_utc', np.nan)
            rec['hourly_session'] = hm.get('session', '')
            rec['hourly_zone_kind'] = hm.get('kind', '')
            rec['hourly_entry_price'] = hm.get('entry_price', np.nan)
            rec['hourly_distance_pct'] = hm.get('distance_pct', np.nan)
            rec['hourly_bars_since_zone'] = hm.get('bars_since_zone', np.nan)
            rec['hourly_stop_reference'] = hm.get('stop_reference', np.nan)
            rec['hourly_applied'] = hm.get('applied', False)
            rec['hourly_risk_reward'] = hm.get('hourly_rr', np.nan)
            rec['hourly_model_prob'] = hm.get('model_prob', np.nan)
            rec['hourly_h1_trend'] = hm.get('h1_trend', np.nan)
            rec['ote_aligned'] = bool(hm.get('ote_aligned', False))
            rec['ote_705'] = hm.get('ote_705', np.nan) if hm.get('ote_705') is not None else np.nan
            rec['ote_band_low'] = hm.get('ote_band_low', np.nan) if hm.get('ote_band_low') is not None else np.nan
            rec['ote_band_high'] = hm.get('ote_band_high', np.nan) if hm.get('ote_band_high') is not None else np.nan
            rec['kill_zone'] = hm.get('kill_zone') or ''
        sa = trade.get('stop_anchor')
        if sa and sa.get('zone') is not None:
            rec['stop_anchor_kind'] = sa['zone']['kind']
            rec['stop_anchor_prob'] = sa['zone'].get('model_prob', np.nan)
        ta = trade.get('target_anchor')
        if ta and ta.get('zone') is not None:
            rec['target_anchor_kind'] = ta['zone']['kind']
            rec['target_anchor_prob'] = ta['zone'].get('model_prob', np.nan)
        keep = ~((journal['signal_date'] == rec['signal_date']) & (~journal['resolved']))
        journal = journal[keep]
        journal = pd.concat([journal, pd.DataFrame([rec])], ignore_index=True) if len(journal) \
            else pd.DataFrame([rec])
        journal = _coerce_journal(journal)
        _save_journal(journal, journal_file)

    return {
        'index_name': index_name,
        'accuracy': accuracy,
        'predicted_state': predicted_state,
        'probs': probs,
        'close': float(row['Close'].iloc[0]),
        'latest_date': str(LATEST_DATE),
        'trade': trade,
        'signals': signals,
        'h1_model_accepted': h1_model.get("accepted", False),
        'h1_model_auc': h1_model.get('auc', float('nan')),
        'h1_samples': h1_model.get('n', 0),
        'journal': journal if USE_JOURNAL else None,
    }


# ============================================
# RUN ALL INDEXES
# ============================================

print("\n" + "="*80)
print("🚀 MULTI-INDEX PIPELINE STARTING")
print("="*80)

def _json_num(v):
    """float for JSON, or None for None/NaN (json.dump would emit invalid NaN)."""
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else v


def _trade_for_json(trade: dict) -> dict:
    """Entry-position data for predictions.json."""
    if trade is None or trade.get('action') not in ('LONG', 'SHORT'):
        return {"action": "NO TRADE", "reason": (trade or {}).get('reason', '')}
    entry = _json_num(trade.get('entry'))
    stop = _json_num(trade.get('stop_loss'))
    target = _json_num(trade.get('take_profit'))
    return {
        "action":       trade['action'],                    # "LONG" | "SHORT"
        "entry":        entry,
        "stop_loss":    stop,
        "take_profit":  target,
        "risk":         round(abs(entry - stop), 2) if entry is not None and stop is not None else None,
        "reward":       round(abs(target - entry), 2) if entry is not None and target is not None else None,
        "risk_reward":  _json_num(trade.get('risk_reward')),
        "conviction":   trade.get('conviction'),
        "entry_type":   trade.get('entry_type'),
        "stop_source":  trade.get('stop_source'),
        "target_source": trade.get('target_source'),
    }


dxy_df = load_dxy_series(DXY_CSV_PATH)
results_list = []   # ← renamed from `results` so the new dict can use that name
predictions = {}    # ← NEW: per-market prediction dict

for csv_path, index_name in INDEX_CONFIGS:
    try:
        res = run_single_index(csv_path, index_name, dxy_df=dxy_df)
        if res is not None:
            results_list.append(res)

            # : collect per-market prediction for predictions.json
            market_key = MARKET_KEY_MAP.get(index_name, index_name)
            predictions[market_key] = {
                "bias":       res['predicted_state'],

                # Probability of the predicted direction
                "confidence": float(res['probs'][res['predicted_state']]),

                # Probability of every possible market direction
                "probabilities": {
                    "UP": float(res['probs']['UP']),
                    "DOWN": float(res['probs']['DOWN']),
                    "CONSOLIDATION": float(res['probs']['CONSOLIDATION']),
                },
                "signals": res['signals'],

                "close": float(res['close']),
                "date": str(res['latest_date']),
                "trade": _trade_for_json(res['trade']),
            }
    except Exception as e:
        print(f"\n❌ Error processing {index_name}: {e}")
        import traceback
        traceback.print_exc()

print("\n\n" + "="*80)
print("📊 CROSS-INDEX SUMMARY")
print("="*80)
if not results_list:
    print("\n⚠️  No indexes were processed successfully.")
else:
    print(f"\n{'Index':<12}{'Date':<12}{'Close':>10}{'Prediction':>14}{'Conf':>8}{'H1 AUC':>10}{'H1 n':>8}")
    print("-"*80)
    for r in results_list:
        pred_short = {'UP': '🟢 UP', 'DOWN': '🔴 DOWN', 'CONSOLIDATION': '🟡 FLAT'}[r['predicted_state']]
        conf = r['probs'][r['predicted_state']]
        auc_str = f"{r['h1_model_auc']:.3f}" if not np.isnan(r['h1_model_auc']) else "—"
        print(f"{r['index_name']:<12}{r['latest_date']:<12}{r['close']:>10.2f}"
              f"{pred_short:>14}{conf:>7.1%}{auc_str:>10}{r['h1_samples']:>8}")

    print("\n" + "-"*80)
    print("Trade recommendations:")
    for r in results_list:
        trade = r['trade']
        print(f"\n  {r['index_name']} ({r['latest_date']}):")
        if trade['action'] == 'NO TRADE':
            print(f"    → NO TRADE ({trade['reason']})")
        else:
            hm = trade.get('hourly_meta')
            tag = ""
            if hm is not None:
                tag = f"  [H1 {hm['kind'].upper()} P={hm['model_prob']:.2f}, {hm['session']}]"
            print(f"    → {trade['action']}  entry {trade['entry']:.2f}  "
                  f"stop {trade['stop_loss']:.2f} ({trade['stop_source']})  "
                  f"target {trade['take_profit']:.2f} ({trade['target_source']})  "
                  f"R:R {trade['risk_reward']}{tag}")

    print("\n" + "="*80)
    print("✅ ALL INDEXES PROCESSED")
    print("="*80)

# ============================================
# WRITE predictions.json  ← NEW
# ============================================
with open("predictions.json", "w") as f:
    json.dump(predictions, f, indent=2)
print(f"\n✅ Wrote predictions.json with {len(predictions)} markets")
print(json.dumps(predictions, indent=2))

print("\n⚠️  Research tool — not financial advice.")