"""Original DCA Pyramid: baseline + one-at-a-time optimization on ETHUSDT 1m."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optimize import SEARCH_SPACE, optimize_sequential, save_results, score  # noqa: E402
from strategy import StrategyParams, print_report, run_backtest  # noqa: E402


def load_ohlc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    return df.sort_values("datetime").reset_index(drop=True)


def main() -> None:
    t0 = time.time()
    merged = ROOT / "data" / "merged" / "ETHUSDT_1m.csv"
    if not merged.exists():
        from download_data import download_and_merge

        print("Downloading 1m data...", flush=True)
        merged = download_and_merge(
            symbol="ETHUSDT", interval="1m", start="2021-01", project_root=ROOT
        )

    print("Loading 1m OHLC...", flush=True)
    df = load_ohlc(merged)
    print(
        f"Loaded {len(df):,} bars  {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}",
        flush=True,
    )

    # Exact original settings (classic pyramid, TP 1%, no SL)
    baseline_params = StrategyParams(
        initial_pct=0.05,
        dca_level_weights=(1, 2, 4, 8),
        sub_order_weights=(1, 2, 4),
        dca_depths=(0.03, 0.10, 0.25, 0.50),
        take_profit_pct=0.01,
        fee_rate=0.001,
        capital=10_000.0,
        reentry_delay_bars=0,
        stop_loss_pct=0.0,
        max_hold_bars=0,
    )

    print("\nRunning ORIGINAL baseline on 1m...", flush=True)
    t1 = time.time()
    baseline = run_backtest(df, baseline_params)
    print(f"Baseline done in {time.time() - t1:.1f}s", flush=True)
    print_report(baseline, "ORIGINAL DCA Pyramid — ETHUSDT 1m")

    # Sequential optimize (same axes as before). Keep SL disabled in search
    # by not including stop_loss in SEARCH_SPACE (already the case).
    space = [item for item in SEARCH_SPACE if item[0] != "reentry_delay_bars"]
    # Also try a few reentry delays at the end
    space.append(("reentry_delay_bars", [0, 60, 180, 360, 720]))  # minutes on 1m

    print("\nStarting one-at-a-time optimization on 1m...", flush=True)
    best_params, history, final = optimize_sequential(df, base=baseline_params, space=space)

    out = ROOT / "results" / "pyramid_original_1m"
    save_results(out, baseline, final, history, best_params)

    # Extra: pyramid-preserving pass (freeze level weights 1:2:4:8)
    print("\n##### Pyramid-preserving optimization (keep 1:2:4:8) #####", flush=True)
    from copy import deepcopy

    from optimize import apply_override

    current = deepcopy(baseline_params)
    preserve_space = [
        ("take_profit_pct", [0.005, 0.008, 0.01, 0.012, 0.015, 0.02, 0.025, 0.03]),
        ("initial_pct", [0.03, 0.05, 0.08, 0.10, 0.15]),
        (
            "dca_depths",
            [
                (0.02, 0.06, 0.15, 0.30),
                (0.03, 0.10, 0.25, 0.50),
                (0.02, 0.08, 0.20, 0.40),
                (0.03, 0.08, 0.18, 0.35),
                (0.05, 0.12, 0.25, 0.45),
                (0.04, 0.10, 0.22, 0.45),
            ],
        ),
        ("sub_order_weights", [(1, 2, 4), (1, 1, 1), (1, 2, 3), (2, 3, 4)]),
        ("reentry_delay_bars", [0, 60, 180, 360]),
    ]
    hist_p = []
    for name, candidates in preserve_space:
        print(f"\n### preserve optimize {name}", flush=True)
        best_local = current
        best_sc = score(run_backtest(df, current))
        for val in candidates:
            trial = apply_override(current, name, val)
            trial.dca_level_weights = (1, 2, 4, 8)
            trial.stop_loss_pct = 0.0
            res = run_backtest(df, trial)
            sc = score(res)
            marker = " <-- best" if sc > best_sc else ""
            print(
                f"  {name}={val!r:40s} ret={res.total_return_pct:+8.2f}% "
                f"dd={res.max_drawdown_pct:6.2f}% cycles={res.num_cycles:5d} "
                f"score={sc:+8.2f}{marker}",
                flush=True,
            )
            hist_p.append(
                {
                    "stage": f"preserve:{name}",
                    "param": name,
                    "value": val if not isinstance(val, tuple) else list(val),
                    "score": sc,
                    **res.summary(),
                }
            )
            if sc > best_sc:
                best_sc = sc
                best_local = trial
        current = best_local
        print(f"  => keep {name}={getattr(current, name)!r}", flush=True)

    current.dca_level_weights = (1, 2, 4, 8)
    current.stop_loss_pct = 0.0
    preserved = run_backtest(df, current)
    print_report(preserved, "PYRAMID-PRESERVING BEST — ETHUSDT 1m")

    preserve_dir = out / "pyramid_preserving"
    preserve_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hist_p).to_csv(preserve_dir / "optimization_history.csv", index=False)
    (preserve_dir / "summary.json").write_text(
        json.dumps(
            {
                "baseline": baseline.summary(),
                "final": preserved.summary(),
                "best_params": current.to_dict(),
                "improvement_return_pp": preserved.total_return_pct - baseline.total_return_pct,
            },
            indent=2,
            default=str,
        )
    )

    # Comparison plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))
        b = baseline.equity_curve.iloc[::60]
        f = final.equity_curve.iloc[::60]
        p = preserved.equity_curve.iloc[::60]
        ax.plot(b.index, b.values, label="Original baseline", lw=1.2)
        ax.plot(f.index, f.values, label="Unconstrained optimized", lw=1.2)
        ax.plot(p.index, p.values, label="Pyramid-preserving optimized", lw=1.2)
        ax.set_title("DCA Pyramid original — ETHUSDT 1m")
        ax.set_ylabel("Equity (USDT)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "equity_comparison.png", dpi=140)
    except Exception as exc:  # noqa: BLE001
        print(f"plot skipped: {exc}", flush=True)

    print("\n" + "=" * 64)
    print("COMPARISON (1m)")
    print("=" * 64)
    print(
        f"Baseline          : {baseline.total_return_pct:+.2f}%  dd={baseline.max_drawdown_pct:.2f}%  "
        f"cycles={baseline.num_cycles}"
    )
    print(
        f"Unconstrained opt : {final.total_return_pct:+.2f}%  dd={final.max_drawdown_pct:.2f}%  "
        f"cycles={final.num_cycles}"
    )
    print(
        f"Pyramid-preserving: {preserved.total_return_pct:+.2f}%  dd={preserved.max_drawdown_pct:.2f}%  "
        f"cycles={preserved.num_cycles}"
    )
    print(f"Buy & hold        : {baseline.buy_hold_return_pct:+.2f}%")
    print(f"Elapsed           : {time.time() - t0:.1f}s")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
