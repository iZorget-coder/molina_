import yfinance as yf
import pandas as pd
import os
import asyncio
import threading
import time
from datetime import datetime, timezone, timedelta

# Map daily CSV -> (Yahoo ticker, hourly CSV, TradingView symbol)
#
# The TradingView symbols are chosen to match the instrument each daily CSV
# was built from (cash indices for ^GSPC/^IXIC/^DJI, front-month gold for GC=F),
# so a TradingView bar doesn't introduce a futures-vs-cash price jump.
# NOTE: these symbol strings are untested with tvkit — check the first run's log.
TICKER_MAP = {
    "SP500_1d_data.csv":  ("^GSPC", "SP500_index_H1.csv",     "SP:SPX"),
    "NASDAQ_1d_data.csv": ("^IXIC", "NASDAQ100_index_H1.csv", "NASDAQ:IXIC"),
    "DOW_1d_data.csv":    ("^DJI",  "DOW_index_H1.csv",       "DJ:DJI"),
    "XAUUSD_1d_data.csv": ("GC=F",  "XAUUSD_H1.csv",          "COMEX:GC1!"),
    # FX pairs and Bitcoin: same TradingView symbols the hourly / 15-minute fetchers use
    "EURUSD_1d_data.csv": ("EURUSD=X", "EURUSD_H1.csv", "OANDA:EURUSD"),
    "USDCAD_1d_data.csv": ("CAD=X",    "USDCAD_H1.csv", "OANDA:USDCAD"),
    "USDCHF_1d_data.csv": ("CHF=X",    "USDCHF_H1.csv", "OANDA:USDCHF"),
    "BTCUSD_1d_data.csv": ("BTC-USD",  "BTCUSD_H1.csv", "BINANCE:BTCUSDT"),
}

# Map daily CSV -> 15-minute CSV (same market, different granularity)
FIFTEEN_MAP = {
    "SP500_1d_data.csv":  ("^GSPC", "SP500_15m.csv",     "SP:SPX"),
    "NASDAQ_1d_data.csv": ("^IXIC", "NASDAQ100_15m.csv", "NASDAQ:IXIC"),
    "DOW_1d_data.csv":    ("^DJI",  "DOW_15m.csv",       "DJ:DJI"),
    "XAUUSD_1d_data.csv": ("GC=F",  "XAUUSD_15m.csv",    "COMEX:GC1!"),
    "EURUSD_1d_data.csv": ("EURUSD=X", "EURUSD_15m.csv", "OANDA:EURUSD"),
    "USDCAD_1d_data.csv": ("CAD=X",    "USDCAD_15m.csv", "OANDA:USDCAD"),
    "USDCHF_1d_data.csv": ("CHF=X",    "USDCHF_15m.csv", "OANDA:USDCHF"),
    "BTCUSD_1d_data.csv": ("BTC-USD",  "BTCUSD_15m.csv", "BINANCE:BTCUSDT"),
}

# ── 15-minute data replaces the OHLC of each newly appended daily candle ──────────────
# The 15-minute feed is the more accurate one, so when a completed day is added to a daily CSV its
# Open / High / Low / Close are rebuilt from that day's 15-minute bars:
#   Open  = open of the first bar of the session     High = highest high of the session
#   Close = close of the last bar of the session     Low  = lowest low of the session
# The day is only replaced when the 15-minute data fully covers it (see MIN_15M_BARS and the gap /
# price checks in daily_from_15m); otherwise the Yahoo / TradingView / hourly candle is kept.
USE_15M_FOR_DAILY = True
USE_15M_VOLUME    = False   # True = Volume becomes the sum of the 15-minute volumes (can differ in scale)
MIN_15M_BARS = {            # a normal session has ~26 bars for the indices and ~92 for gold
    "SP500_15m.csv": 20, "NASDAQ100_15m.csv": 20, "DOW_15m.csv": 20, "XAUUSD_15m.csv": 60,
    "EURUSD_15m.csv": 60, "USDCAD_15m.csv": 60, "USDCHF_15m.csv": 60, "BTCUSD_15m.csv": 80,
}
# Hours added to a bar's UTC time to find its trading day. Indices / gold / FX reopen ~22:00 UTC, so +2h
# rolls that open into the next date. Bitcoin trades 24/7 and its daily candle is a plain UTC day.
DAY_OFFSET_HOURS = {"BTCUSD_15m.csv": 0}
DEFAULT_DAY_OFFSET_HOURS = 2
MAX_15M_VS_SOURCE_PCT = 2.0  # skip the replacement if the 15m close is further than this from the source close
MAX_15M_GAP_HOURS     = 3    # skip if the day's bars have a hole bigger than this


def today_utc():
    """Today's date in UTC (GitHub Actions runs in UTC). Any bar dated today or later is still in progress."""
    return datetime.now(timezone.utc).date()


# ─────────────────────────────────────────────────────────────
# 1. TRADINGVIEW DAILY (PRIMARY)
# ─────────────────────────────────────────────────────────────

def _tv_trading_date(ts):
    """
    Map a TradingView daily-bar timestamp (UTC) to the trading date.
    Cash-index bars are stamped at the session open (same day), but futures and
    forex sessions open the evening before, so a stamp at/after 18:00 UTC belongs
    to the next day. The raw stamps are printed so this can be verified.
    """
    d = ts.date()
    if ts.hour >= 18:
        d = d + timedelta(days=1)
    return d


def get_tradingview_daily_bars(tv_symbol, after_date, today):
    """Completed daily bars from TradingView with after_date < Date < today, oldest first."""
    try:
        from tvkit.api.chart.ohlcv import OHLCV
    except ImportError:
        print("  tvkit not installed — skipping TradingView (pip install tvkit)")
        return []

    box = {"bars": None, "error": None}

    async def _inner():
        async with OHLCV() as client:
            return await client.get_historical_ohlcv(
                exchange_symbol=tv_symbol,
                interval="1D",
                start=after_date.isoformat(),
                end=today.isoformat(),
            )

    def _runner():
        try:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                box["bars"] = loop.run_until_complete(_inner())
            finally:
                try:
                    loop.close()
                finally:
                    asyncio.set_event_loop(None)
        except Exception as e:
            box["error"] = e

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=120)

    if t.is_alive():
        print(f"  TradingView daily fetch timed out ({tv_symbol})")
        return []
    if box["error"] is not None:
        e = box["error"]
        print(f"  TradingView daily fetch failed ({tv_symbol}): {e.__class__.__name__}: {e}")
        return []
    if not box["bars"]:
        print(f"  TradingView returned 0 daily bars for {tv_symbol}")
        return []

    bars = []
    for b in sorted(box["bars"], key=lambda x: x.timestamp):
        ts = pd.to_datetime(b.timestamp, unit="s", utc=True)
        bar_date = _tv_trading_date(ts)
        print(f"    TV raw stamp {ts} → trading date {bar_date}")

        if bar_date >= today:
            continue  # today's bar is still forming
        if bar_date <= after_date:
            continue  # already in the CSV
        if any(pd.isna(v) for v in (b.open, b.high, b.low, b.close)):
            print(f"  TradingView returned a NaN/incomplete bar for {bar_date}")
            continue

        vol = getattr(b, "volume", None)
        bars.append({
            "Date":   str(bar_date),
            "Open":   float(b.open),
            "High":   float(b.high),
            "Low":    float(b.low),
            "Close":  float(b.close),
            "Volume": float(vol) if vol is not None and not pd.isna(vol) else 0,
        })
    return bars


# ─────────────────────────────────────────────────────────────
# 2. YAHOO DAILY (FALLBACK)
# ─────────────────────────────────────────────────────────────

def get_yahoo_daily_bars(ticker, after_date, today):
    """Completed daily bars from Yahoo with after_date < Date < today, oldest first."""
    try:
        # A small window is enough to catch up after a weekend, a holiday or a missed run
        df = yf.download(ticker, period="10d", interval="1d", progress=False, auto_adjust=False)
        if df.empty:
            return []

        # yfinance may return a MultiIndex column; flatten if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        bars = []
        for ts, row in df.iterrows():
            bar_date = ts.date()

            if bar_date >= today:
                continue  # today's bar is still forming
            if bar_date <= after_date:
                continue  # already in the CSV

            # Reject Yahoo's "corrupt row" case (NaN OHLC)
            if any(pd.isna(row[c]) for c in ("Open", "High", "Low", "Close")):
                print(f"  Yahoo returned a NaN/incomplete bar for {bar_date}")
                continue

            bars.append({
                "Date":   str(bar_date),
                "Open":   float(row["Open"]),
                "High":   float(row["High"]),
                "Low":    float(row["Low"]),
                "Close":  float(row["Close"]),
                "Volume": float(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
            })
        return bars
    except Exception as e:
        print(f"  Yahoo fetch failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# 3. HOURLY AGGREGATION (LAST RESORT)
# ─────────────────────────────────────────────────────────────

def aggregate_hourly_bars(hourly_path, after_date, today):
    """Daily bars built from the hourly CSV for after_date < UTC date < today, oldest first."""
    if not os.path.exists(hourly_path):
        return []
    try:
        hourly = pd.read_csv(hourly_path)
        if hourly.empty:
            return []

        hourly["timestamp"] = pd.to_datetime(hourly["timestamp"], utc=True)
        hourly["date"] = hourly["timestamp"].dt.date
        hourly = hourly[(hourly["date"] > after_date) & (hourly["date"] < today)]
        if hourly.empty:
            return []

        bars = []
        for bar_date, day in hourly.sort_values("timestamp").groupby("date"):
            has_volume = "volume" in day and day["volume"].notna().any()
            bars.append({
                "Date":   str(bar_date),
                "Open":   float(day["open"].iloc[0]),
                "High":   float(day["high"].max()),
                "Low":    float(day["low"].min()),
                "Close":  float(day["close"].iloc[-1]),
                "Volume": float(day["volume"].sum()) if has_volume else 0,
            })
        return bars
    except Exception as e:
        print(f"  Hourly aggregation failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# 3b. DAILY CANDLE FROM 15-MINUTE BARS
# ─────────────────────────────────────────────────────────────

def daily_from_15m(csv_15m, date_str, ref_close=None):
    """
    OHLC (and summed volume) of one trading day rebuilt from the 15-minute CSV, or None when the
    15-minute data can't be trusted for that day.

    A bar belongs to the trading day of (timestamp + 2h), the same rule the web app uses. Cash indices
    trade ~13:30-20:00 UTC, so they land on their own date; gold and other futures reopen the evening
    before (22:00-23:00 UTC), which rolls those bars into the next trading date.
    """
    if not csv_15m or not os.path.exists(csv_15m):
        return None
    try:
        m = pd.read_csv(csv_15m)
        if m.empty:
            return None
        m["timestamp"] = pd.to_datetime(m["timestamp"], utc=True)
        day = pd.to_datetime(date_str).date()
        off = DAY_OFFSET_HOURS.get(os.path.basename(csv_15m), DEFAULT_DAY_OFFSET_HOURS)
        m = m[(m["timestamp"] + pd.Timedelta(hours=off)).dt.date == day].sort_values("timestamp")
        m = m.dropna(subset=["open", "high", "low", "close"])

        need = MIN_15M_BARS.get(os.path.basename(csv_15m), 20)
        if len(m) < need:
            print(f"    15m: only {len(m)} bar(s) for {date_str} (need {need}) — keeping the source candle")
            return None
        gaps = m["timestamp"].diff().dropna()
        if len(gaps) and gaps.max() > pd.Timedelta(hours=MAX_15M_GAP_HOURS):
            print(f"    15m: a {gaps.max()} hole in the bars for {date_str} — keeping the source candle")
            return None

        out = {
            "Open":   float(m["open"].iloc[0]),
            "High":   float(m["high"].max()),
            "Low":    float(m["low"].min()),
            "Close":  float(m["close"].iloc[-1]),
            "Volume": float(m["volume"].fillna(0).sum()) if "volume" in m else 0.0,
            "bars":   len(m),
        }
        if ref_close and abs(out["Close"] / ref_close - 1) * 100 > MAX_15M_VS_SOURCE_PCT:
            print(f"    15m: close {out['Close']:.2f} is more than {MAX_15M_VS_SOURCE_PCT}% from the "
                  f"source close {ref_close:.2f} — keeping the source candle (check the symbol)")
            return None
        return out
    except Exception as e:
        print(f"    15m: could not build {date_str} from {csv_15m}: {e}")
        return None


def apply_15m_to_bar(bar, csv_15m):
    """Return `bar` with its OHLC (and optionally Volume) replaced by the 15-minute values, if available."""
    if not USE_15M_FOR_DAILY:
        return bar, False
    f = daily_from_15m(csv_15m, bar["Date"], ref_close=bar.get("Close"))
    if f is None:
        return bar, False
    new = dict(bar)
    for k in ("Open", "High", "Low", "Close"):
        new[k] = f[k]
    if USE_15M_VOLUME and f["Volume"] > 0:
        new["Volume"] = f["Volume"]
    print(f"    15m ({f['bars']} bars): O {bar['Open']:.2f}→{f['Open']:.2f}  H {bar['High']:.2f}→{f['High']:.2f}  "
          f"L {bar['Low']:.2f}→{f['Low']:.2f}  C {bar['Close']:.2f}→{f['Close']:.2f}")
    return new, True


# ─────────────────────────────────────────────────────────────
# UPDATE ONE DAILY CSV
# ─────────────────────────────────────────────────────────────

def update_daily_csv(daily_path, yahoo_ticker, hourly_path, tv_symbol, fifteen_path=None):
    if not os.path.exists(daily_path):
        print(f"Skipping {daily_path} (file not found)")
        return

    today = today_utc()
    daily = pd.read_csv(daily_path)
    changed = False

    # 0. Drop any row dated today or later (an unfinished candle added by an earlier run)
    row_dates = pd.to_datetime(daily["Date"]).dt.date
    keep = row_dates < today
    if not keep.all():
        dropped = ", ".join(str(d) for d in row_dates[~keep])
        print(f"  Removing in-progress row(s): {dropped}")
        daily = daily[keep].reset_index(drop=True)
        changed = True

    last_daily_date = pd.to_datetime(daily["Date"].iloc[-1]).date()
    last_close = float(daily["Close"].iloc[-1])
    print(f"{daily_path}: last date = {last_daily_date} "
          f"(today UTC = {today}, newest allowed = {today - timedelta(days=1)})")

    # Nothing can be added if the CSV already holds yesterday's bar
    if last_daily_date >= today - timedelta(days=1):
        print(f"  CSV is already up to date through {last_daily_date}.")
        if changed:
            daily.to_csv(daily_path, index=False)
        return

    # Source priority: TradingView → Yahoo → hourly aggregation
    sources = [
        ("TradingView", lambda: get_tradingview_daily_bars(tv_symbol, last_daily_date, today)),
        ("Yahoo",       lambda: get_yahoo_daily_bars(yahoo_ticker, last_daily_date, today)),
        ("Hourly",      lambda: aggregate_hourly_bars(hourly_path, last_daily_date, today)),
    ]

    bars, source = [], None
    for name, fetch in sources:
        bars = fetch()
        if bars:
            source = name
            break
        print(f"  {name} had no new complete bar.")

    # Append every completed day we're missing (never today)
    if bars:
        rows = []
        for latest in bars:
            # the 15-minute candle (open of the session, close of the session, session high / low)
            # replaces the source candle's OHLC whenever the 15m data fully covers that day
            latest, from_15m = apply_15m_to_bar(latest, fifteen_path)
            row = {col: latest.get(col, None) for col in daily.columns}
            if "Adj Close" in daily.columns:
                row["Adj Close"] = row["Close"]
            rows.append(row)
            jump = (row["Close"] / last_close - 1) * 100
            last_close = row["Close"]
            print(f"  ✅ Appended {latest['Date']} from {source}{' + 15m OHLC' if from_15m else ''} → "
                  f"Close: {row['Close']:.2f} ({jump:+.2f}% vs previous close)")
        daily = pd.concat([daily, pd.DataFrame(rows)], ignore_index=True)
        changed = True
    else:
        print(f"  No new completed days to add. CSV is up to date through {last_daily_date}.")

    if changed:
        daily.to_csv(daily_path, index=False)


# ─────────────────────────────────────────────────────────────
# 4. 15-MINUTE UPDATER
# ─────────────────────────────────────────────────────────────

# Chunked 15-minute fetch (same approach as the fetch_15m_75d notebook): TradingView only serves ~80 days of
# 15m history per request window, so the range is split into 25-day chunks, each fetched on its own.
TV_15M_LOOKBACK_DAYS = 75
TV_15M_CHUNK_DAYS    = 25
TV_15M_TIMEOUT_S     = 90


def _tv_fetch_chunk(tv_symbol, c_start, c_end):
    """One TradingView 15m request for [c_start, c_end) (ISO dates). Returns the bar list or []."""
    from tvkit.api.chart.ohlcv import OHLCV
    box = {"bars": None, "error": None}

    async def _inner():
        async with OHLCV() as client:
            return await client.get_historical_ohlcv(
                exchange_symbol=tv_symbol, interval="15", start=c_start, end=c_end)

    def _runner():
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            box["bars"] = loop.run_until_complete(_inner())
        except Exception as e:
            box["error"] = e
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=TV_15M_TIMEOUT_S)
    if t.is_alive():
        print(f"    chunk {c_start} -> {c_end}: timed out")
        return []
    if box["error"] is not None:
        e = box["error"]
        print(f"    chunk {c_start} -> {c_end}: {e.__class__.__name__}: {e}")
        return []
    if not box["bars"]:
        print(f"    chunk {c_start} -> {c_end}: 0 bars")
        return []
    print(f"    chunk {c_start} -> {c_end}: {len(box['bars'])} bars")
    return list(box["bars"])


def _tv_15m_bars(tv_symbol, after_ts, now_utc):
    """Completed 15-minute bars from TradingView with timestamp > after_ts and bar fully closed.
    Starts one day before the newest bar we have (so it self-heals) and never further back than the
    75-day window; a missing file therefore gets a full 75-day backfill in one run."""
    try:
        import tvkit  # noqa: F401
    except ImportError:
        print("  tvkit not installed — skipping TradingView 15m")
        return []

    floor = now_utc.date() - timedelta(days=TV_15M_LOOKBACK_DAYS)
    start_d = max((after_ts - timedelta(days=1)).date(), floor)
    end_d = now_utc.date() + timedelta(days=1)          # include today's still-forming bar (dropped below)

    raw, cur = [], start_d
    while cur < end_d:
        nxt = min(cur + timedelta(days=TV_15M_CHUNK_DAYS), end_d)
        raw += _tv_fetch_chunk(tv_symbol, cur.isoformat(), nxt.isoformat())
        cur = nxt
        time.sleep(1)                                    # be kind to TradingView
    if not raw:
        print(f"  TradingView returned 0 15m bars for {tv_symbol}")
        return []

    # The current 15m window is not complete yet; drop anything stamped >= the last closed slot
    last_closed = now_utc.replace(minute=(now_utc.minute // 15) * 15, second=0, microsecond=0)

    bars = []
    for b in sorted(raw, key=lambda x: x.timestamp):
        ts = pd.to_datetime(b.timestamp, unit="s", utc=True)
        if ts <= after_ts or ts >= last_closed:
            continue
        o, h, l, c = b.open, b.high, b.low, b.close
        if any(pd.isna(v) for v in (o, h, l, c)) or min(o, h, l, c) <= 0 or h < l:
            continue
        vol = getattr(b, "volume", None)
        bars.append({
            "timestamp": ts.isoformat(),
            "open":   float(o),
            "high":   float(max(h, o, c)),
            "low":    float(min(l, o, c)),
            "close":  float(c),
            "volume": float(vol) if vol is not None and not pd.isna(vol) else 0,
        })

    # one bad print must not wreck the axis: everything within +/-50% of the median close
    if bars:
        med = float(pd.Series([x["close"] for x in bars]).median())
        bars = [x for x in bars if all(med * 0.5 <= x[k] <= med * 1.5 for k in ("open", "high", "low", "close"))]
    return bars


def _yahoo_15m_bars(ticker, after_ts, now_utc):
    """Completed 15-minute bars from Yahoo with timestamp > after_ts."""
    try:
        df = yf.download(ticker, period="5d", interval="15m", progress=False, auto_adjust=False)
        if df.empty:
            return []
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        last_closed = now_utc.replace(minute=(now_utc.minute // 15) * 15, second=0, microsecond=0)

        bars = []
        for ts, row in df.iterrows():
            ts_utc = ts.tz_convert("UTC") if ts.tz is not None else ts.tz_localize("UTC")
            if ts_utc <= after_ts:
                continue
            if ts_utc >= last_closed:
                continue
            if any(pd.isna(row[c]) for c in ("Open", "High", "Low", "Close")):
                continue
            bars.append({
                "timestamp": ts_utc.isoformat(),
                "open":   float(row["Open"]),
                "high":   float(row["High"]),
                "low":    float(row["Low"]),
                "close":  float(row["Close"]),
                "volume": float(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
            })
        return bars
    except Exception as e:
        print(f"  Yahoo 15m fetch failed: {e}")
        return []


def update_15m_csv(csv_path, yahoo_ticker, tv_symbol):
    """Append new completed 15-minute bars to csv_path. Creates the file if missing."""
    now_utc = datetime.now(timezone.utc)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if df.empty:
            after_ts = now_utc - timedelta(days=TV_15M_LOOKBACK_DAYS)
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            after_ts = df["timestamp"].max()
    else:
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        after_ts = now_utc - timedelta(days=TV_15M_LOOKBACK_DAYS)

    print(f"{csv_path}: last 15m bar = {after_ts} (now UTC = {now_utc})")

    # Source priority: TradingView → Yahoo
    bars, source = [], None
    for name, fetch in [
        ("TradingView", lambda: _tv_15m_bars(tv_symbol, after_ts, now_utc)),
        ("Yahoo",       lambda: _yahoo_15m_bars(yahoo_ticker, after_ts, now_utc)),
    ]:
        bars = fetch()
        if bars:
            source = name
            break
        print(f"  {name} had no new 15m bar.")

    if not bars:
        print(f"  No new completed 15m bars. CSV unchanged.")
        return

    new_df = pd.DataFrame(bars)
    combined = pd.concat([df, new_df], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    combined = (
        combined.drop_duplicates(subset=["timestamp"], keep="last")
                .sort_values("timestamp")
                .reset_index(drop=True)
    )
    combined.to_csv(csv_path, index=False)
    print(f"  ✅ Appended {len(bars)} bar(s) from {source} → {csv_path} "
          f"(now {len(combined)} rows, last = {combined['timestamp'].iloc[-1]})")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # The 15-minute files are updated FIRST so the daily update below can rebuild each new daily candle from them.
    print("=== 15-minute ===")
    for daily_file, (yahoo_ticker, csv_15m, tv_symbol) in FIFTEEN_MAP.items():
        update_15m_csv(csv_15m, yahoo_ticker, tv_symbol)

    print("\n=== Daily ===")
    for daily_file, (yahoo_ticker, hourly_file, tv_symbol) in TICKER_MAP.items():
        update_daily_csv(daily_file, yahoo_ticker, hourly_file, tv_symbol,
                         fifteen_path=FIFTEEN_MAP[daily_file][1])