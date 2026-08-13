"""Download monthly ETHUSDT klines from Binance Vision and merge them."""

from __future__ import annotations

import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


def month_range(start: str, end: str) -> list[tuple[int, int]]:
    """Inclusive YYYY-MM range -> list of (year, month)."""
    y0, m0 = map(int, start.split("-"))
    y1, m1 = map(int, end.split("-"))
    out: list[tuple[int, int]] = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        out.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def download_month(
    symbol: str,
    interval: str,
    year: int,
    month: int,
    raw_dir: Path,
    session: requests.Session,
) -> Path | None:
    name = f"{symbol}-{interval}-{year}-{month:02d}"
    zip_name = f"{name}.zip"
    url = f"{BASE_URL}/{symbol}/{interval}/{zip_name}"
    dest_zip = raw_dir / zip_name
    dest_csv = raw_dir / f"{name}.csv"

    if dest_csv.exists() and dest_csv.stat().st_size > 0:
        return dest_csv

    print(f"  GET {url}")
    try:
        r = session.get(url, timeout=120)
    except requests.RequestException as exc:
        print(f"  skip {name}: {exc}")
        return None

    if r.status_code == 404:
        print(f"  skip {name}: not found")
        return None
    if r.status_code != 200:
        print(f"  skip {name}: HTTP {r.status_code}")
        return None

    dest_zip.write_bytes(r.content)
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        members = [m for m in zf.namelist() if m.endswith(".csv")]
        if not members:
            print(f"  skip {name}: empty zip")
            return None
        with zf.open(members[0]) as src, dest_csv.open("wb") as out:
            out.write(src.read())
    return dest_csv


def load_month_csv(path: Path) -> pd.DataFrame:
    # Some months have a header row; detect it.
    peek = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:1]
    header = 0 if peek and peek[0].startswith("open_time") else None
    # Keep open_time as string first to avoid float64 precision loss on µs timestamps.
    df = pd.read_csv(path, header=header, names=COLUMNS, dtype={"open_time": "string"})
    for col in ("open", "high", "low", "close", "volume", "quote_volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open_time", "open", "high", "low", "close"])
    # Parse integer timestamps safely
    ot = df["open_time"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["open_time"] = ot.apply(int)
    sample = int(df["open_time"].iloc[0])
    if sample < 10_000_000_000:
        df["open_time"] = df["open_time"] * 1000
    elif sample > 100_000_000_000_000:  # microseconds
        df["open_time"] = df["open_time"] // 1000
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[
        ["datetime", "open_time", "open", "high", "low", "close", "volume", "quote_volume"]
    ]


def download_and_merge(
    symbol: str = "ETHUSDT",
    interval: str = "1h",
    start: str = "2021-01",
    end: str | None = None,
    project_root: Path | None = None,
) -> Path:
    root = project_root or Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw" / f"{symbol}_{interval}"
    merged_dir = root / "data" / "merged"
    raw_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    if end is None:
        now = datetime.now(timezone.utc)
        # Prefer last fully completed month.
        if now.month == 1:
            end = f"{now.year - 1}-12"
        else:
            end = f"{now.year}-{now.month - 1:02d}"

    print(f"Downloading {symbol} {interval} from {start} to {end}")
    frames: list[pd.DataFrame] = []
    with requests.Session() as session:
        for year, month in month_range(start, end):
            path = download_month(symbol, interval, year, month, raw_dir, session)
            if path is None:
                continue
            frames.append(load_month_csv(path))

    if not frames:
        raise RuntimeError("No monthly files downloaded")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    out = merged_dir / f"{symbol}_{interval}.csv"
    df.to_csv(out, index=False)
    print(f"Merged {len(df):,} candles -> {out}")
    print(f"Range: {df['datetime'].iloc[0]} .. {df['datetime'].iloc[-1]}")
    return out


if __name__ == "__main__":
    download_and_merge()
