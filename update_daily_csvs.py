import yfinance as yf
import pandas as pd
import os
import asyncio
import threading
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
}

# Map daily CSV -> 15-minute CSV (same market, different granularity)
FIFTEEN_MAP = {
    "SP500_1d_data.csv":  ("^GSPC", "SP500_15m.csv",     "SP:SPX"),
    "NASDAQ_1d_data.csv": ("^IXIC", "NASDAQ100_15m.csv", "NASDAQ:IXIC"),
    "DOW_1d_data.csv":    ("^DJI",  "DOW_15m.csv",       "DJ:DJI"),
    "XAUUSD_1d_data.csv": ("GC=F",  "XAUUSD_15m.csv",    "COMEX:GC1!"),
}


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
# UPDATE ONE DAILY CSV
# ─────────────────────────────────────────────────────────────

def update_daily_csv(daily_path, yahoo_ticker, hourly_path, tv_symbol):
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
            row = {col: latest.get(col, None) for col in daily.columns}
            if "Adj Close" in daily.columns:
                row["Adj Close"] = row["Close"]
            rows.append(row)
            jump = (row["Close"] / last_close - 1) * 100
            last_close = row["Close"]
            print(f"  ✅ Appended {latest['Date']} from {source} → "
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

def _tv_15m_bars(tv_symbol, after_ts, now_utc):
    """Completed 15-minute bars from TradingView with timestamp > after_ts and bar fully closed."""
    try:
        from tvkit.api.chart.ohlcv import OHLCV
    except ImportError:
        print("  tvkit not installed — skipping TradingView 15m")
        return []

    start = (after_ts - timedelta(days=1)).date().isoformat()
    end   = (now_utc.date() + timedelta(days=1)).isoformat()
    box = {"bars": None, "error": None}

    async def _inner():
        async with OHLCV() as client:
            return await client.get_historical_ohlcv(
                exchange_symbol=tv_symbol,
                interval="15",
                start=start,
                end=end,
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
    t.join(timeout=90)

    if t.is_alive():
        print(f"  TradingView 15m fetch timed out ({tv_symbol})")
        return []
    if box["error"] is not None:
        e = box["error"]
        print(f"  TradingView 15m fetch failed ({tv_symbol}): {e.__class__.__name__}: {e}")
        return []
    if not box["bars"]:
        print(f"  TradingView returned 0 15m bars for {tv_symbol}")
        return []

    # The current 15m window is not complete yet; drop anything stamped >= the last closed slot
    last_closed = now_utc.replace(minute=(now_utc.minute // 15) * 15, second=0, microsecond=0)

    bars = []
    for b in sorted(box["bars"], key=lambda x: x.timestamp):
        ts = pd.to_datetime(b.timestamp, unit="s", utc=True)
        if ts <= after_ts:
            continue
        if ts >= last_closed:
            continue  # still forming
        if any(pd.isna(v) for v in (b.open, b.high, b.low, b.close)):
            continue
        vol = getattr(b, "volume", None)
        bars.append({
            "timestamp": ts.isoformat(),
            "open":   float(b.open),
            "high":   float(b.high),
            "low":    float(b.low),
            "close":  float(b.close),
            "volume": float(vol) if vol is not None and not pd.isna(vol) else 0,
        })
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
            after_ts = now_utc - timedelta(days=5)
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            after_ts = df["timestamp"].max()
    else:
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        after_ts = now_utc - timedelta(days=5)

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
    print("=== Daily ===")
    for daily_file, (yahoo_ticker, hourly_file, tv_symbol) in TICKER_MAP.items():
        update_daily_csv(daily_file, yahoo_ticker, hourly_file, tv_symbol)

    print("\n=== 15-minute ===")
    for daily_file, (yahoo_ticker, csv_15m, tv_symbol) in FIFTEEN_MAP.items():
        update_15m_csv(csv_15m, yahoo_ticker, tv_symbol)