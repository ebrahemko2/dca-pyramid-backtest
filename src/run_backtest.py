"""Run baseline backtest + sequential parameter optimization for ETHUSDT."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from download_data import download_and_merge  # noqa: E402
from optimize import optimize_sequential, save_results, score  # noqa: E402
from strategy import StrategyParams, print_report, run_backtest  # noqa: E402


def load_ohlc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    return df.sort_values("datetime").reset_index(drop=True)


def main() -> None:
    symbol = "ETHUSDT"
    interval = "1h"
    start = "2021-01"

    merged = ROOT / "data" / "merged" / f"{symbol}_{interval}.csv"
    if not merged.exists():
        merged = download_and_merge(symbol=symbol, interval=interval, start=start, project_root=ROOT)
    else:
        print(f"Using cached merged data: {merged}")

    df = load_ohlc(merged)
    print(f"Loaded {len(df):,} bars  {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")

    baseline_params = StrategyParams()
    baseline = run_backtest(df, baseline_params)
    print_report(baseline, "BASELINE — original DCA Pyramid")

    best_params, history, final = optimize_sequential(df, base=baseline_params)

    out = ROOT / "results"
    save_results(out, baseline, final, history, best_params)

    print("\n" + "=" * 64)
    print("COMPARISON")
    print("=" * 64)
    print(f"Baseline return : {baseline.total_return_pct:+.2f}%  dd={baseline.max_drawdown_pct:.2f}%")
    print(f"Optimized return: {final.total_return_pct:+.2f}%  dd={final.max_drawdown_pct:.2f}%")
    print(f"Improvement     : {final.total_return_pct - baseline.total_return_pct:+.2f} percentage points")
    print(f"Best params     : {best_params.to_dict()}")
    print(f"Score baseline/final: {score(baseline):+.2f} / {score(final):+.2f}")


if __name__ == "__main__":
    main()
