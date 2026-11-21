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
}

HOURLY_FILES = {
    "SPX":    "SP500_index_H1.csv",
    "NDX":    "NASDAQ100_index_H1.csv",
    "DJI":    "DOW_index_H1.csv",
    "XAUUSD": "XAUUSD_H1.csv",
}


def _load_sorted(path, time_col, n, max_gap_days):
    """Read a CSV, sort it by time (oldest -> newest) and return the newest n rows.

    CSV row order is not trusted: some files have old history mixed in, so a
    plain .tail(n) can return decades-old candles.
    """
    df = pd.read_csv(path)
    df["_t"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = (
        df.dropna(subset=["_t"])
          .drop_duplicates(subset="_t", keep="last")
          .sort_values("_t")
          .tail(n)
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


for symbol in predictions:
    predictions[symbol]["candles"] = daily_candles(symbol, 100)
    predictions[symbol]["candlesH1"] = hourly_candles(symbol, 50)

data_date = next(iter(predictions.values()))["date"]

doc_ref = db.collection("daily_predictions").document(data_date)
doc_ref.set({
    "predictions": predictions,
    "generated_at": firestore.SERVER_TIMESTAMP,
})

print(f"Wrote daily_predictions/{data_date}")
print(f"Daily candles: {[len(v.get('candles', [])) for v in predictions.values()]}")
print(f"Hourly candles: {[len(v.get('candlesH1', [])) for v in predictions.values()]}")