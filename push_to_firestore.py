# push_to_firestore.py
import firebase_admin
from firebase_admin import credentials, firestore
import json
import os
import pandas as pd

# Load the service account from the environment variable
service_account_info = json.loads(os.environ["FIREBASE_SERVICE_ACCOUNT"])
cred = credentials.Certificate(service_account_info)
firebase_admin.initialize_app(cred)

db = firestore.client()

# Load the predictions produced by market data.py
with open("predictions.json", "r") as f:
    predictions = json.load(f)

DAILY_FILES = {
    "SPX":    "SP500_1d_data.csv",
    "NDX":    "NASDAQ_1d_data.csv",
    "DJI":    "DOW_1d_data.csv",
    "XAUUSD": "XAUUSD_1d_data.csv",
    "EURUSD": "EURUSD_1d_data.csv",
    "USDCAD": "USDCAD_1d_data.csv",
    "USDCHF": "USDCHF_1d_data.csv",
    "BTCUSD": "BTCUSD_1d_data.csv",
}

HOURLY_FILES = {
    "SPX":    "SP500_index_H1.csv",
    "NDX":    "NASDAQ100_index_H1.csv",
    "DJI":    "DOW_index_H1.csv",
    "XAUUSD": "XAUUSD_H1.csv",
    "EURUSD": "EURUSD_H1.csv",
    "USDCAD": "USDCAD_H1.csv",
    "USDCHF": "USDCHF_H1.csv",
    "BTCUSD": "BTCUSD_H1.csv",
}


def _load_sorted(path, time_col, n, max_gap_days):
    """Return the newest n rows of a CSV, ordered earliest -> latest.

    The row order inside a CSV is NOT trusted. Some files are stored
    newest-first (DOW, S&P 500) and have extra rows appended at the bottom,
    others are oldest-first (NASDAQ, Gold). Whatever the layout, this:
      1. parses the time column,
      2. drops rows with an unreadable time,
      3. removes duplicate timestamps (keeps the last copy in the file),
      4. sorts from earliest to latest,
      5. keeps the last n rows (= the n most recent candles).
    """
    df = pd.read_csv(path)
    df["_t"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = (
        df.dropna(subset=["_t"])
          .drop_duplicates(subset="_t", keep="last")
          .sort_values("_t", kind="stable")
          .tail(n)
          .reset_index(drop=True)
    )

    # Sanity check: warn if the window is not one continuous recent run
    if len(df) > 1:
        biggest_gap = df["_t"].diff().dropna().max()
        if biggest_gap > pd.Timedelta(days=max_gap_days):
            print(
                f"[warn] {path}: gap of {biggest_gap} inside the last {n} rows "
                f"({df['_t'].iloc[0]} -> {df['_t'].iloc[-1]}). "
                "The CSV is probably missing recent data."
            )
    return df


def daily_candles(symbol, n=100):
    path = DAILY_FILES.get(symbol)
    if not path or not os.path.exists(path):
        print(f"[warn] No daily CSV for {symbol} ({path})")
        return []
    df = _load_sorted(path, "Date", n, max_gap_days=14)
    out = []
    for _, row in df.iterrows():
        try:
            out.append({
                "date":  row["_t"].strftime("%Y-%m-%d"),
                "open":  float(row["Open"]),
                "high":  float(row["High"]),
                "low":   float(row["Low"]),
                "close": float(row["Close"]),
            })
        except (KeyError, ValueError) as e:
            print(f"[warn] {path} bad row: {e}")
    return out


def hourly_candles(symbol, n=200):
    path = HOURLY_FILES.get(symbol)
    if not path or not os.path.exists(path):
        print(f"[warn] No hourly CSV for {symbol} ({path})")
        return []
    df = _load_sorted(path, "timestamp", n, max_gap_days=5)
    out = []
    for _, row in df.iterrows():
        try:
            out.append({
                "time":  str(row["timestamp"]),
                "open":  float(row["open"]),
                "high":  float(row["high"]),
                "low":   float(row["low"]),
                "close": float(row["close"]),
            })
        except (KeyError, ValueError) as e:
            print(f"[warn] {path} bad row: {e}")
    return out


def _check_order(symbol, name, candles, key):
    """Make sure a candle list really runs earliest -> latest."""
    if not candles:
        return
    stamps = [pd.Timestamp(c[key]) for c in candles]
    if stamps != sorted(stamps):
        raise SystemExit(f"[error] {symbol} {name} candles are not in ascending order")
    print(f"[ok] {symbol} {name}: {len(candles)} candles, {candles[0][key]} -> {candles[-1][key]}")


for symbol in predictions:
    predictions[symbol]["candles"] = daily_candles(symbol, 100)
    predictions[symbol]["candlesH1"] = hourly_candles(symbol, 50)
    _check_order(symbol, "daily",  predictions[symbol]["candles"],   "date")
    _check_order(symbol, "hourly", predictions[symbol]["candlesH1"], "time")

if not predictions:
    raise SystemExit("[error] predictions.json is empty - the model produced no markets, nothing to push")

# Bitcoin trades on weekends, so its latest daily date can be later than the other markets'.
# The document id follows the weekday markets so a weekend doesn't shift it.
_weekday_dates = [p["date"] for k, p in predictions.items() if k != "BTCUSD" and p.get("date")]
data_date = max(_weekday_dates) if _weekday_dates else next(iter(predictions.values()))["date"]

doc_ref = db.collection("daily_predictions").document(data_date)
doc_ref.set({
    "predictions": predictions,
    "generated_at": firestore.SERVER_TIMESTAMP,
})

print(f"Wrote daily_predictions/{data_date}")
print(f"Daily candles: {[len(v.get('candles', [])) for v in predictions.values()]}")
print(f"Hourly candles: {[len(v.get('candlesH1', [])) for v in predictions.values()]}")