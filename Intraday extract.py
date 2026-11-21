import sys
import pandas as pd
from datetime import date, timedelta, datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import os
import asyncio

import yfinance as yf

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

# Fields: (yahoo_ticker, outfile, tv_symbol, csv_symbol_tag)
INSTRUMENTS = {
    "sp500_idx":   ("^GSPC",  "SP500_index_H1.csv",     "CME_MINI:ES1!",  "USA500"),
    "nasdaq_idx":  ("^IXIC",  "NASDAQ100_index_H1.csv", "CME_MINI:NQ1!",  "USATECH"),
    "dow_idx":     ("^DJI",   "DOW_index_H1.csv",       "CBOT_MINI:YM1!", "USA30"),
    "gold":        ("GC=F",   "XAUUSD_H1.csv",          "OANDA:XAUUSD",   "XAUUSD"),
}

TO_DOWNLOAD = ["sp500_idx", "nasdaq_idx", "dow_idx", "gold"]

START = date(2021, 11, 11)
END   = date.today()

# Yahoo 1h data only goes back ~730 days.
YAHOO_LOOKBACK_DAYS = 729

# Re-fetch the last N days from TradingView so the newest bars overwrite
# anything that was written while a bar was still forming.
TV_OVERLAP_DAYS = 2

# Serial writes; the CSV merge is simpler with 1 worker.
MAX_WORKERS = 1

OUTPUT_COLUMNS = [
    "symbol", "timestamp", "symbol_id", "open", "high", "low", "close",
    "vwap", "volume", "trade_count", "is_gap", "gap_fill_method",
    "hours_since_prev", "is_contiguous", "source",
]


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _to_datetime_series(series):
    return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")


def _ensure_datetime(df, col="timestamp"):
    if df is None or df.empty or col not in df.columns:
        return df
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = _to_datetime_series(df[col])
    return df


def safe_end() -> date:
    """Latest completed daily close in UTC."""
    today_utc = datetime.now(timezone.utc).date()
    return min(END, today_utc - timedelta(days=1))


def last_saved_timestamp(csv_path: str):
    if not os.path.exists(csv_path):
        return None
    try:
        existing = pd.read_csv(csv_path, usecols=["timestamp"])
        if existing.empty:
            return None
        return _to_datetime_series(existing["timestamp"]).max()
    except Exception as e:
        print(f"  ⚠️  could not read {csv_path}: {e} — will re-download full history")
        return None


def _latest_ts(*dfs):
    vals = [d["timestamp"].max() for d in dfs
            if d is not None and not d.empty]
    return max(vals) if vals else None


def _finalize_schema(df: pd.DataFrame, csv_symbol_tag: str, source: str) -> pd.DataFrame:
    """Add the constant columns and order everything like the CSV schema."""
    df["symbol"]           = csv_symbol_tag
    df["symbol_id"]        = 0
    df["vwap"]             = None
    df["trade_count"]      = 0
    df["is_gap"]           = False
    df["gap_fill_method"]  = 0
    df["hours_since_prev"] = df["timestamp"].diff().dt.total_seconds() / 3600
    df["is_contiguous"]    = df["hours_since_prev"] == 1
    df["source"]           = source
    return df[[c for c in OUTPUT_COLUMNS if c in df.columns]]


# ─────────────────────────────────────────────────────────────
# TRADINGVIEW FETCHER (PRIMARY)
# ─────────────────────────────────────────────────────────────

def fetch_from_tradingview(symbol: str, start: date, end: date,
                            csv_symbol_tag: str) -> pd.DataFrame:
    try:
        from tvkit.api.chart.ohlcv import OHLCV
    except ImportError:
        print(f"  ⚠️  tvkit not installed — pip install tvkit", flush=True)
        return pd.DataFrame()

    from_iso = start.isoformat()
    to_iso   = end.isoformat()
    print(f"  📺 TradingView fetch: {symbol}  1h  {from_iso} → {to_iso}", flush=True)

    result_box = {"bars": None, "error": None}

    async def _inner():
        async with OHLCV() as client:
            return await client.get_historical_ohlcv(
                exchange_symbol=symbol,
                interval="1H",
                start=from_iso,
                end=to_iso,
            )

    def _runner():
        try:
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                result_box["bars"] = new_loop.run_until_complete(_inner())
            finally:
                try:
                    new_loop.close()
                finally:
                    asyncio.set_event_loop(None)
        except Exception as e:
            result_box["error"] = e

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=120)

    if t.is_alive():
        print(f"  ❌ TradingView fetch timed out for {symbol}", flush=True)
        return pd.DataFrame()

    if result_box["error"] is not None:
        e = result_box["error"]
        print(f"  ❌ TradingView fetch failed ({symbol}): "
              f"{e.__class__.__name__}: {e}", flush=True)
        return pd.DataFrame()

    bars = result_box["bars"]
    if not bars:
        print(f"  ℹ️  TradingView returned 0 bars for {symbol} "
              f"{from_iso} → {to_iso}", flush=True)
        return pd.DataFrame()

    rows = [{
        "timestamp": pd.to_datetime(b.timestamp, unit="s", utc=True),
        "open":   float(b.open),
        "high":   float(b.high),
        "low":    float(b.low),
        "close":  float(b.close),
        "volume": float(b.volume),
    } for b in bars]

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df = _finalize_schema(df, csv_symbol_tag, "tv")

    print(f"  ✅ TradingView returned {len(df)} bars "
          f"({df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]})", flush=True)
    return df


# ─────────────────────────────────────────────────────────────
# YAHOO FETCHER (FALLBACK)
# ─────────────────────────────────────────────────────────────

def fetch_yahoo_hourly(yahoo_ticker: str, start: date, end: date,
                        csv_symbol_tag: str) -> pd.DataFrame:
    """
    Fetch hourly bars from Yahoo for [start, end), clamped to the 730-day
    lookback. Returns a DataFrame in the same schema as the TradingView source.
    """
    today_utc = datetime.now(timezone.utc).date()
    yahoo_earliest = today_utc - timedelta(days=YAHOO_LOOKBACK_DAYS)
    effective_start = max(start, yahoo_earliest)

    if effective_start >= end:
        print(f"  ⚠️  [{yahoo_ticker}] Yahoo 1h only goes back to "
              f"{yahoo_earliest}; nothing to fetch for {start}→{end}", flush=True)
        return pd.DataFrame()

    print(f"  🌐 Yahoo fetch: {yahoo_ticker}  1h  "
          f"{effective_start} → {end}", flush=True)

    try:
        raw = yf.download(
            yahoo_ticker,
            start=effective_start.isoformat(),
            end=end.isoformat(),
            interval="1h",
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception as e:
        print(f"  ❌ Yahoo fetch failed ({yahoo_ticker}): "
              f"{e.__class__.__name__}: {e}", flush=True)
        return pd.DataFrame()

    if raw is None or raw.empty:
        print(f"  ℹ️  Yahoo returned 0 rows for {yahoo_ticker}", flush=True)
        return pd.DataFrame()

    # Flatten MultiIndex columns if present (yfinance sometimes returns them)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Ensure the index is a DatetimeIndex and is tz-aware in UTC.
    if not isinstance(raw.index, pd.DatetimeIndex):
        raw.index = pd.to_datetime(raw.index, utc=True, errors="coerce")
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC")
    else:
        raw.index = raw.index.tz_convert("UTC")

    rows = []
    for ts, row in raw.iterrows():
        # Skip rows with any missing OHLC
        if any(pd.isna(row.get(c)) for c in ("Open", "High", "Low", "Close")):
            continue
        vol = row.get("Volume")
        rows.append({
            "timestamp": ts,
            "open":   float(row["Open"]),
            "high":   float(row["High"]),
            "low":    float(row["Low"]),
            "close":  float(row["Close"]),
            "volume": float(vol) if vol is not None and not pd.isna(vol) else 0.0,
        })

    if not rows:
        print(f"  ℹ️  [{yahoo_ticker}] all Yahoo rows had NaN OHLC", flush=True)
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df = _finalize_schema(df, csv_symbol_tag, "yahoo")

    print(f"  ✅ Yahoo returned {len(df)} bars "
          f"({df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]})", flush=True)
    return df


# ─────────────────────────────────────────────────────────────
# DOWNLOAD ONE INSTRUMENT (TradingView first, Yahoo fallback)
# ─────────────────────────────────────────────────────────────

def download(yahoo_ticker: str, outfile: str,
             tv_symbol: str = None, csv_symbol_tag: str = None,
             start: date = START, end: date = END) -> pd.DataFrame:

    tag    = csv_symbol_tag or yahoo_ticker
    target = safe_end()

    existing_df = None
    resume_from = start

    if os.path.exists(outfile):
        latest = last_saved_timestamp(outfile)
        if latest is not None:
            if latest.date() >= target:
                print(f"  ✅ [{yahoo_ticker}] already up to date "
                      f"(last bar {latest}), skipping", flush=True)
                return _ensure_datetime(pd.read_csv(outfile))
            resume_from = max(start, latest.date() - timedelta(days=TV_OVERLAP_DAYS))
            print(f"  📂 [{yahoo_ticker}] last bar = {latest} "
                  f"→ resuming from {resume_from}", flush=True)
            if latest.date() > datetime.now(timezone.utc).date():
                print(f"  ⚠️  [{yahoo_ticker}] CSV claims a future timestamp "
                      f"({latest}); verify before trusting", flush=True)
            try:
                existing_df = _ensure_datetime(pd.read_csv(outfile))
            except Exception as e:
                print(f"  ⚠️  could not read {outfile}: {e} — treating as missing",
                      flush=True)
                existing_df = None
        else:
            print(f"  📂 {outfile} empty/unreadable → full download", flush=True)
    else:
        print(f"  📂 no {outfile} → full download from {start}", flush=True)

    # ── 1. PRIMARY: TradingView ──────────────────────────────
    new_df = pd.DataFrame()
    if tv_symbol:
        new_df = _ensure_datetime(
            fetch_from_tradingview(tv_symbol, resume_from, target, tag))

    # ── 2. FALLBACK: Yahoo, only for what TV didn't cover ────
    known_ts   = _latest_ts(existing_df, new_df)
    known_date = known_ts.date() if known_ts is not None else resume_from

    if known_date < target:
        why = ("TradingView returned nothing" if new_df.empty
               else f"TradingView ended at {known_ts}")
        print(f"  🌐 [{yahoo_ticker}] {why} → Yahoo fallback "
              f"{known_date} → {target}", flush=True)
        print(f"  ⚠️  [{yahoo_ticker}] Yahoo bars are a different session/"
              f"price basis than TV — expect a seam in the CSV", flush=True)
        y_df = _ensure_datetime(
            fetch_yahoo_hourly(yahoo_ticker, known_date, target, tag))
        if y_df is not None and not y_df.empty:
            if known_ts is not None:
                y_df = y_df[y_df["timestamp"] > known_ts]
            if not y_df.empty:
                print(f"  ✅ [{yahoo_ticker}] Yahoo supplied {len(y_df)} bars",
                      flush=True)
                new_df = pd.concat([new_df, y_df], ignore_index=True)

    # ── 3. MERGE (new rows win on duplicate timestamps) ──────
    if existing_df is not None and not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    if combined is None or combined.empty:
        print(f"  ❌ [{yahoo_ticker}] no data at all", flush=True)
        return pd.DataFrame()

    n_before = len(existing_df) if existing_df is not None and not existing_df.empty else 0
    combined = _ensure_datetime(combined).dropna(subset=["timestamp"])
    combined = (combined.drop_duplicates(subset=["timestamp"], keep="last")
                        .sort_values("timestamp").reset_index(drop=True))

    # Rows written before the `source` column existed
    if "source" not in combined.columns:
        combined["source"] = "unknown"
    combined["source"] = combined["source"].fillna("unknown")

    combined["hours_since_prev"] = combined["timestamp"].diff().dt.total_seconds() / 3600
    combined["is_contiguous"]    = combined["hours_since_prev"] == 1

    n_new = len(combined) - n_before
    if n_new != 0:
        print(f"  📈 [{yahoo_ticker}] {n_new:+d} bar(s) "
              f"(total {len(combined)})", flush=True)
    else:
        print(f"  ℹ️  [{yahoo_ticker}] no change (total {len(combined)})", flush=True)
    return combined


# ─────────────────────────────────────────────────────────────
# WRAPPER
# ─────────────────────────────────────────────────────────────

def fetch_one(key: str):
    yahoo_ticker, outfile, tv_symbol, csv_symbol_tag = INSTRUMENTS[key]
    print(f"[start] {key} ({yahoo_ticker} / {tv_symbol})")
    df = download(yahoo_ticker, outfile,
                  tv_symbol=tv_symbol, csv_symbol_tag=csv_symbol_tag)

    if df is None or df.empty:
        return key, yahoo_ticker, outfile, pd.DataFrame()

    df = _ensure_datetime(df)
    df["hours_since_prev"] = df["timestamp"].diff().dt.total_seconds() / 3600
    df["is_contiguous"]    = df["hours_since_prev"] == 1

    cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    df[cols].to_csv(outfile, index=False)
    print(f"[done ] {key} ({yahoo_ticker}): {len(df)} bars → {outfile}", flush=True)
    return key, yahoo_ticker, outfile, df


# ─────────────────────────────────────────────────────────────
# OPTIONAL ONE-TIME REBUILD
# ─────────────────────────────────────────────────────────────

def backup_existing_csvs():
    """Move existing H1 CSVs aside so TradingView repopulates them from scratch."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for key in TO_DOWNLOAD:
        outfile = INSTRUMENTS[key][1]
        if os.path.exists(outfile):
            bak = f"{outfile}.{stamp}.bak"
            os.replace(outfile, bak)
            print(f"🗄️  {outfile} → {bak}", flush=True)


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────

def main():
    if "--rebuild" in sys.argv:
        print("♻️  --rebuild: backing up existing CSVs and re-downloading "
              "from TradingView", flush=True)
        backup_existing_csvs()

    t0 = time.time()
    print(f"Target end date (latest completed daily close UTC): "
          f"{safe_end()}", flush=True)
    print(f"Yahoo 1h lookback floor (fallback only): "
          f"{datetime.now(timezone.utc).date() - timedelta(days=YAHOO_LOOKBACK_DAYS)}",
          flush=True)

    results = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_one, k): k for k in TO_DOWNLOAD}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                k, ticker, outfile, df = fut.result()
                results[k] = df
            except Exception as e:
                print(f"[FAIL ] {key}: {e}")

    failed = [k for k in TO_DOWNLOAD if results.get(k) is None or results[k].empty]
    if failed:
        print(f"\n🔁 retrying failed instruments: {failed}", flush=True)
        for key in failed:
            try:
                k, ticker, outfile, df = fetch_one(key)
                results[k] = df
            except Exception as e:
                print(f"[FAIL ] {key} (2nd pass): {e}")

    print(f"\n========== SUMMARY ({time.time()-t0:.1f}s) ==========")
    for k in TO_DOWNLOAD:
        df = results.get(k)
        if df is None or df.empty:
            print(f"{k:12s} ❌ no data")
        else:
            src = ""
            if "source" in df.columns:
                counts = df["source"].value_counts().to_dict()
                src = " | " + ", ".join(f"{s}:{n}" for s, n in counts.items())
            print(f"{k:12s} ✅ {len(df):>6} bars | "
                  f"{df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}{src}")


if __name__ == "__main__":
    main()
