"""
live_hourly.py -- fetch the latest hourly candles and publish them to Firestore.

Runs every hour from its own workflow (hourly-live.yml). It does NOT train anything, does NOT
touch the CSVs and does NOT touch daily_predictions: it only writes live_hourly/latest, which
the markets page listens to and draws over the model's entry / stop / target.

Source order per market:
  1. TradingView (tvkit) -- same futures symbols the daily pipeline uses, so prices line up
     with the levels the model produced.
  2. Yahoo futures (ES=F, NQ=F, YM=F, GC=F, EURUSD=X, CAD=X, CHF=X, BTC-USD) -- only if TradingView failed or is stale.
     Futures, not the cash indices, so the price basis matches TradingView's ES1!/NQ1!/YM1!.
The document records which source each market used.
"""
import asyncio
import json
import os
import sys
import threading
from datetime import datetime, timedelta, timezone

import pandas as pd

KEEP_BARS = 120            # candles stored per market (~5 trading days)
LOOKBACK_DAYS = 7          # how far back to ask the data providers
STALE_AFTER_HOURS = 3      # TradingView newer than this is "fresh"; otherwise try Yahoo too
TV_TIMEOUT_S = 90

# market key (as used by predictions.json / the page) -> data sources
MARKETS = {
    "SPX":    {"tv": "CME_MINI:ES1!",  "yahoo": "ES=F"},
    "NDX":    {"tv": "CME_MINI:NQ1!",  "yahoo": "NQ=F"},
    "DJI":    {"tv": "CBOT_MINI:YM1!", "yahoo": "YM=F"},
    "XAUUSD": {"tv": "OANDA:XAUUSD",   "yahoo": "GC=F"},
    "EURUSD": {"tv": "OANDA:EURUSD",   "yahoo": "EURUSD=X"},
    "USDCAD": {"tv": "OANDA:USDCAD",   "yahoo": "CAD=X"},
    "USDCHF": {"tv": "OANDA:USDCHF",   "yahoo": "CHF=X"},
    "BTCUSD": {"tv": "BINANCE:BTCUSDT", "yahoo": "BTC-USD"},
}


# ---------------------------------------------------------------- fetchers
def fetch_tradingview(symbol: str) -> pd.DataFrame:
    try:
        from tvkit.api.chart.ohlcv import OHLCV
    except ImportError:
        print("  tvkit not installed", flush=True)
        return pd.DataFrame()

    today = datetime.now(timezone.utc).date()
    start = (today - timedelta(days=LOOKBACK_DAYS)).isoformat()
    end = (today + timedelta(days=1)).isoformat()      # include today's still-forming bar
    box = {"bars": None, "error": None}

    async def _inner():
        async with OHLCV() as client:
            return await client.get_historical_ohlcv(
                exchange_symbol=symbol, interval="1H", start=start, end=end)

    def _runner():
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            box["bars"] = loop.run_until_complete(_inner())
        except Exception as e:                              # noqa: BLE001
            box["error"] = e
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=TV_TIMEOUT_S)
    if t.is_alive():
        print(f"  TradingView timed out ({symbol})", flush=True)
        return pd.DataFrame()
    if box["error"] is not None:
        print(f"  TradingView failed ({symbol}): {box['error'].__class__.__name__}: {box['error']}", flush=True)
        return pd.DataFrame()
    if not box["bars"]:
        print(f"  TradingView returned nothing ({symbol})", flush=True)
        return pd.DataFrame()

    return pd.DataFrame([{
        "timestamp": pd.to_datetime(b.timestamp, unit="s", utc=True),
        "open": float(b.open), "high": float(b.high), "low": float(b.low), "close": float(b.close),
    } for b in box["bars"]])


def fetch_yahoo(ticker: str) -> pd.DataFrame:
    try:
        import yfinance as yf
        raw = yf.download(ticker, period="7d", interval="1h", progress=False,
                          auto_adjust=False, threads=False)
    except Exception as e:                                  # noqa: BLE001
        print(f"  Yahoo failed ({ticker}): {e.__class__.__name__}: {e}", flush=True)
        return pd.DataFrame()
    if raw is None or raw.empty:
        print(f"  Yahoo returned nothing ({ticker})", flush=True)
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    idx = pd.to_datetime(raw.index, utc=True) if raw.index.tz is None else raw.index.tz_convert("UTC")
    df = pd.DataFrame({
        "timestamp": idx,
        "open": raw["Open"].to_numpy(float), "high": raw["High"].to_numpy(float),
        "low": raw["Low"].to_numpy(float), "close": raw["Close"].to_numpy(float),
    })
    return df


# ---------------------------------------------------------------- cleaning
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop impossible bars, de-duplicate, sort, keep the newest KEEP_BARS."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).copy()
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1) & (df["high"] >= df["low"])]
    if df.empty:
        return df
    med = df["close"].median()                              # one bad print must not wreck the axis
    ok = lambda s: (s >= med * 0.5) & (s <= med * 1.5)       # noqa: E731
    df = df[ok(df["open"]) & ok(df["high"]) & ok(df["low"]) & ok(df["close"])]
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    df = df.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp")
    return df.tail(KEEP_BARS).reset_index(drop=True)


def age_hours(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return float("inf")
    return (datetime.now(timezone.utc) - df["timestamp"].iloc[-1].to_pydatetime()).total_seconds() / 3600


def build_market(key: str, cfg: dict):
    """Returns the Firestore map for one market, or None if nothing usable came back."""
    print(f"[{key}]", flush=True)
    tv = clean(fetch_tradingview(cfg["tv"]))
    best, source = tv, "tv"
    if tv.empty or age_hours(tv) > STALE_AFTER_HOURS:
        yh = clean(fetch_yahoo(cfg["yahoo"]))
        # prefer Yahoo only if it is clearly newer (more than an hour fresher)
        if not yh.empty and (tv.empty or yh["timestamp"].iloc[-1] - tv["timestamp"].iloc[-1] > pd.Timedelta(hours=1)):
            best, source = yh, "yahoo"
    if best is None or best.empty:
        print("  no usable data", flush=True)
        return None

    now = datetime.now(timezone.utc)
    candles = []
    for r in best.itertuples():
        ts = r.timestamp
        candles.append({
            # same text format as the CSV-based candles the page already has
            "time": str(ts),
            "open": round(float(r.open), 6), "high": round(float(r.high), 6),
            "low": round(float(r.low), 6), "close": round(float(r.close), 6),
            "partial": bool(ts.to_pydatetime() + timedelta(hours=1) > now),   # bar still forming
        })
    print(f"  {source}: {len(candles)} bars, last {candles[-1]['time']} close {candles[-1]['close']}", flush=True)
    return {"source": source, "fetched_at": now.isoformat(), "candles": candles}


# ---------------------------------------------------------------- firestore
def publish(symbols: dict):
    import firebase_admin
    from firebase_admin import credentials, firestore

    cred = credentials.Certificate(json.loads(os.environ["FIREBASE_SERVICE_ACCOUNT"]))
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    # merge=True: a market that failed this hour keeps its previous candles instead of being wiped
    db.collection("live_hourly").document("latest").set(
        {"symbols": symbols, "updated_at": firestore.SERVER_TIMESTAMP}, merge=True)


def main():
    symbols = {}
    for key, cfg in MARKETS.items():
        try:
            m = build_market(key, cfg)
        except Exception as e:                              # noqa: BLE001
            print(f"[{key}] failed: {e.__class__.__name__}: {e}", flush=True)
            m = None
        if m:
            symbols[key] = m

    if not symbols:
        print("No market returned data -- nothing written.", flush=True)
        sys.exit(1)

    publish(symbols)
    print(f"Wrote live_hourly/latest for: {', '.join(symbols)}", flush=True)


if __name__ == "__main__":
    main()