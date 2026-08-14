"""Re-optimize stop-loss / TP for inverted config on ETHUSDT 1m (path-accurate)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from strategy import StrategyParams, print_report, run_backtest  # noqa: E402


def base(**kw) -> StrategyParams:
    p = StrategyParams(
        initial_pct=0.05,
        dca_level_weights=(8, 4, 2, 1),
        sub_order_weights=(2, 3, 4),
        dca_depths=(0.02, 0.08, 0.2, 0.4),
        take_profit_pct=0.012,
        fee_rate=0.001,
        capital=10_000.0,
        reentry_delay_bars=0,
        stop_loss_pct=0.0,
        stop_loss_ref="p0",
        max_hold_bars=0,
    )
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def load_ohlc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    return df.sort_values("datetime").reset_index(drop=True)


def score(res) -> float:
    # Favor survival + return; punish wipeouts and extreme DD
    if res.equity_final < 1000:  # lost >90% of 10k
        return res.total_return_pct - 200.0
    return res.total_return_pct - 0.4 * res.max_drawdown_pct - 0.002 * (res.avg_hold_bars / 60.0)


def main() -> None:
    df = load_ohlc(ROOT / "data" / "merged" / "ETHUSDT_1m.csv")
    print(f"1m bars={len(df):,}  {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")

    # Crash slices
    c22 = df[(df["datetime"] >= "2021-11-01") & (df["datetime"] <= "2022-12-31")].reset_index(drop=True)
    c25 = df[(df["datetime"] >= "2024-12-01") & (df["datetime"] <= "2026-07-31")].reset_index(drop=True)

    history = []
    # Hold in minutes-bars: 0, 6h, 12h, 24h, 48h, 72h
    hold_hours = [0, 6, 12, 24, 48, 72]
    sl_grid = [0.0, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40]
    tp_grid = [0.012, 0.015, 0.02, 0.025]
    refs = ["p0", "wap"]

    print("\n=== Phase A: SL × ref on FULL 1m (max_hold=0, tp=1.2%) ===")
    best = None
    for ref in refs:
        for sl in sl_grid:
            p = base(stop_loss_pct=sl, stop_loss_ref=ref, max_hold_bars=0, take_profit_pct=0.012)
            t0 = time.time()
            res = run_backtest(df, p)
            sc = score(res)
            row = {
                "phase": "sl_full",
                "sl": sl,
                "ref": ref,
                "tp": 0.012,
                "hold_h": 0,
                "score": sc,
                "seconds": time.time() - t0,
                **res.summary(),
            }
            history.append(row)
            marker = ""
            if best is None or sc > best["score"]:
                best = {"score": sc, "params": p, "res": res, "meta": row}
                marker = " <-- best"
            print(
                f"  SL={sl:.0%} {ref:3s} ret={res.total_return_pct:+8.1f}% dd={res.max_drawdown_pct:5.1f}% "
                f"eq={res.equity_final:10.1f} cycles={res.num_cycles:5d} "
                f"SLex={res.sl_exits:4d} avgHmin={res.avg_hold_bars:7.1f} score={sc:+8.1f}{marker}"
            )

    print("\n=== Phase B: around best SL, try TP and max-hold ===")
    # Take top SL candidates from history (full phase)
    ranked = sorted([h for h in history if h["phase"] == "sl_full"], key=lambda x: x["score"], reverse=True)
    top_sl = []
    seen = set()
    for h in ranked:
        key = (h["sl"], h["ref"])
        if key in seen:
            continue
        seen.add(key)
        top_sl.append(key)
        if len(top_sl) >= 4:
            break

    for sl, ref in top_sl:
        for tp in tp_grid:
            for hh in hold_hours:
                # skip exact already-run default
                if tp == 0.012 and hh == 0:
                    continue
                p = base(
                    stop_loss_pct=sl,
                    stop_loss_ref=ref,
                    take_profit_pct=tp,
                    max_hold_bars=hh * 60,  # 1m bars
                )
                res = run_backtest(df, p)
                sc = score(res)
                row = {
                    "phase": "refine",
                    "sl": sl,
                    "ref": ref,
                    "tp": tp,
                    "hold_h": hh,
                    "score": sc,
                    **res.summary(),
                }
                history.append(row)
                marker = ""
                if sc > best["score"]:
                    best = {"score": sc, "params": p, "res": res, "meta": row}
                    marker = " <-- best"
                print(
                    f"  SL={sl:.0%} {ref} TP={tp:.1%} hold={hh:2d}h "
                    f"ret={res.total_return_pct:+8.1f}% dd={res.max_drawdown_pct:5.1f}% "
                    f"eq={res.equity_final:10.1f} score={sc:+8.1f}{marker}"
                )

    # Evaluate best on crash windows too
    print("\n=== Best on crash windows ===")
    for name, w in ("crash_2022", c22), ("crash_2025_26", c25), ("full", df):
        res = run_backtest(w, best["params"])
        print_report(res, f"BEST 1m — {name}")
        history.append({"phase": f"best:{name}", "score": score(res), **res.summary()})

    # Also report prior 1h-tuned config already known bad
    prior = base(stop_loss_pct=0.05, stop_loss_ref="p0", max_hold_bars=72 * 60, take_profit_pct=0.012)
    res_prior = run_backtest(df, prior)

    out = ROOT / "results" / "tf_1m"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out / "optimize_1m_history.csv", index=False)
    summary = {
        "best_params": best["params"].to_dict(),
        "best_full": best["res"].summary(),
        "prior_1h_tuned_on_1m": res_prior.summary(),
        "note": (
            "1h OHLC backtests overstate edge (same-bar low-fill then high-TP path bias). "
            "1m is closer to real path; params retuned here."
        ),
    }
    (out / "optimize_1m_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Save cycles / equity downsample for best
    rows = []
    for c in best["res"].cycles:
        rows.append(
            {
                "entry_time": str(c.entry_time),
                "exit_time": str(c.exit_time),
                "p0": c.p0,
                "avg_entry": c.avg_entry,
                "exit_price": c.exit_price,
                "pnl": c.pnl,
                "pnl_pct": c.pnl_pct,
                "fills": c.fills,
                "hold_bars": c.hold_bars,
                "reason": c.reason,
            }
        )
    pd.DataFrame(rows).to_csv(out / "cycles_best_1m.csv", index=False)
    if best["res"].equity_curve is not None:
        best["res"].equity_curve.iloc[::60].to_csv(out / "equity_best_1m_1h_sample.csv", header=True)

    print("\nSELECTED:", best["params"].to_dict())
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
