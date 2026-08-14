"""Backtest final inverted+SL strategy on ETHUSDT 1m candles."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from download_data import download_and_merge  # noqa: E402
from strategy import StrategyParams, print_report, run_backtest  # noqa: E402


def final_params(interval: str) -> StrategyParams:
    """
    Final recommended config.
    max_hold is 72 hours wall-clock — convert to bars by interval.
    """
    minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}.get(interval)
    if minutes is None:
        raise ValueError(f"unsupported interval: {interval}")
    max_hold_bars = int(72 * 60 / minutes)  # 72 hours
    return StrategyParams(
        initial_pct=0.05,
        dca_level_weights=(8, 4, 2, 1),
        sub_order_weights=(2, 3, 4),
        dca_depths=(0.02, 0.08, 0.2, 0.4),
        take_profit_pct=0.012,
        fee_rate=0.001,
        capital=10_000.0,
        reentry_delay_bars=0,
        stop_loss_pct=0.05,
        stop_loss_ref="p0",
        max_hold_bars=max_hold_bars,
    )


def no_sl_params(interval: str) -> StrategyParams:
    p = final_params(interval)
    p.stop_loss_pct = 0.0
    p.max_hold_bars = 0
    return p


def load_ohlc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    return df.sort_values("datetime").reset_index(drop=True)


def cycles_to_df(res) -> pd.DataFrame:
    rows = []
    for c in res.cycles:
        rows.append(
            {
                "entry_time": str(c.entry_time),
                "exit_time": str(c.exit_time),
                "p0": c.p0,
                "avg_entry": c.avg_entry,
                "exit_price": c.exit_price,
                "qty": c.qty,
                "cost_quote": c.cost_quote,
                "proceeds": c.proceeds,
                "pnl": c.pnl,
                "pnl_pct": c.pnl_pct,
                "fills": c.fills,
                "hold_bars": c.hold_bars,
                "max_dd_from_p0": c.max_dd_from_p0,
                "reason": c.reason,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    symbol = "ETHUSDT"
    interval = "1m"
    start = "2021-01"

    merged = ROOT / "data" / "merged" / f"{symbol}_{interval}.csv"
    if not merged.exists():
        print("Downloading 1m monthly data (this can take a while)...")
        t0 = time.time()
        merged = download_and_merge(
            symbol=symbol,
            interval=interval,
            start=start,
            project_root=ROOT,
        )
        print(f"Download+merge done in {time.time() - t0:.1f}s")
    else:
        print(f"Using cached: {merged}")

    print("Loading OHLC...")
    t0 = time.time()
    df = load_ohlc(merged)
    print(
        f"Loaded {len(df):,} bars in {time.time() - t0:.1f}s  "
        f"{df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}"
    )

    out = ROOT / "results" / "tf_1m"
    out.mkdir(parents=True, exist_ok=True)

    # Final config with SL
    p_final = final_params(interval)
    print(f"\nmax_hold_bars for 1m (72h) = {p_final.max_hold_bars}")
    print("Running FINAL (SL 5% P0 + max hold 72h) on 1m...")
    t0 = time.time()
    res_final = run_backtest(df, p_final)
    print(f"Done in {time.time() - t0:.1f}s")
    print_report(res_final, "FINAL inverted+SL — ETHUSDT 1m")

    # No-SL inverted for comparison on same 1m data
    p_nosl = no_sl_params(interval)
    print("\nRunning inverted NO-SL on 1m for comparison...")
    t0 = time.time()
    res_nosl = run_backtest(df, p_nosl)
    print(f"Done in {time.time() - t0:.1f}s")
    print_report(res_nosl, "Inverted NO-SL — ETHUSDT 1m")

    # Also reload 1h final for side-by-side if available
    h1_path = ROOT / "data" / "merged" / "ETHUSDT_1h.csv"
    res_1h = None
    if h1_path.exists():
        df1h = load_ohlc(h1_path)
        p_1h = final_params("1h")
        print("\nRe-running FINAL on 1h for apples-to-apples...")
        t0 = time.time()
        res_1h = run_backtest(df1h, p_1h)
        print(f"Done in {time.time() - t0:.1f}s")
        print_report(res_1h, "FINAL inverted+SL — ETHUSDT 1h")

    summary = {
        "interval": interval,
        "bars": len(df),
        "range": [str(df["datetime"].iloc[0]), str(df["datetime"].iloc[-1])],
        "final_1m": res_final.summary(),
        "nosl_1m": res_nosl.summary(),
        "final_1h": None if res_1h is None else res_1h.summary(),
        "note": (
            "max_hold_bars scaled to 72 wall-clock hours "
            f"(1m={p_final.max_hold_bars} bars, 1h=72 bars)"
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    cycles_to_df(res_final).to_csv(out / "cycles_final_1m.csv", index=False)
    cycles_to_df(res_nosl).to_csv(out / "cycles_nosl_1m.csv", index=False)
    if res_final.equity_curve is not None:
        # Downsample equity for smaller file (every 60 bars ~= 1h points)
        eq = res_final.equity_curve.iloc[::60]
        eq.to_csv(out / "equity_final_1m_downsampled_1h.csv", header=True)
        res_final.equity_curve.iloc[::1].iloc[::1440].to_csv(
            out / "equity_final_1m_daily.csv", header=True
        )

    # Plot comparison if matplotlib available
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))
        eq1m = res_final.equity_curve.iloc[::60]
        ax.plot(eq1m.index, eq1m.values, label="Final SL on 1m (shown @1h sample)", lw=1.2)
        if res_1h is not None and res_1h.equity_curve is not None:
            ax.plot(
                res_1h.equity_curve.index,
                res_1h.equity_curve.values,
                label="Final SL on 1h",
                lw=1.2,
                alpha=0.85,
            )
        ax.set_title("Inverted Pyramid + SL 5% P0 — 1m vs 1h")
        ax.set_ylabel("Equity (USDT)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "equity_1m_vs_1h.png", dpi=140)
        print(f"Saved plot -> {out / 'equity_1m_vs_1h.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"plot skipped: {exc}")

    print("\n" + "=" * 64)
    print("COMPARISON")
    print("=" * 64)
    print(
        f"1m FINAL: ret={res_final.total_return_pct:+.2f}% dd={res_final.max_drawdown_pct:.2f}% "
        f"cycles={res_final.num_cycles} avgHoldBars={res_final.avg_hold_bars:.1f} "
        f"avgHoldHours={res_final.avg_hold_bars/60:.2f} "
        f"maxHoldHours={res_final.max_hold_bars_seen/60:.1f}"
    )
    print(
        f"1m NO-SL: ret={res_nosl.total_return_pct:+.2f}% dd={res_nosl.max_drawdown_pct:.2f}% "
        f"cycles={res_nosl.num_cycles} avgHoldBars={res_nosl.avg_hold_bars:.1f}"
    )
    if res_1h is not None:
        print(
            f"1h FINAL: ret={res_1h.total_return_pct:+.2f}% dd={res_1h.max_drawdown_pct:.2f}% "
            f"cycles={res_1h.num_cycles} avgHoldHours={res_1h.avg_hold_bars:.2f} "
            f"maxHoldHours={res_1h.max_hold_bars_seen:.1f}"
        )
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
