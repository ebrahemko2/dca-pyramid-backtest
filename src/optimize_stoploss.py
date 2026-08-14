"""Optimize stop-loss / max-hold for the inverted-weights (profit-max) config.

1) Sweep SL on crash windows (preserve capital + shorten holds).
2) Validate the best candidate(s) on the full history.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from strategy import StrategyParams, print_report, run_backtest  # noqa: E402


def profit_max_base(**overrides: Any) -> StrategyParams:
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
        stop_loss_ref="wap",
        max_hold_bars=0,
    )
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


def load_ohlc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    return df.sort_values("datetime").reset_index(drop=True)


def slice_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    m = (df["datetime"] >= start) & (df["datetime"] <= end)
    out = df.loc[m].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"empty slice {start}..{end}")
    return out


def crash_score(res) -> float:
    """
    Prefer: higher return, lower DD, shorter average holds, not too many tiny SL chops.
    Tuned for crash windows where no-SL often sits underwater for months.
    """
    hold_pen = 0.02 * res.avg_hold_bars + 0.001 * res.max_hold_bars_seen
    return (
        res.total_return_pct
        - 0.55 * res.max_drawdown_pct
        - hold_pen
        + 3.0 * min(res.profit_factor, 2.5)
    )


def full_score(res) -> float:
    """Full-history score: keep profit edge, punish DD and long holds."""
    return (
        res.total_return_pct
        - 0.35 * res.max_drawdown_pct
        - 0.01 * res.avg_hold_bars
        - 0.0003 * res.max_hold_bars_seen
    )


def row_from(res, stage: str, extra: dict | None = None) -> dict:
    d = {"stage": stage, "score_crash": crash_score(res), "score_full": full_score(res)}
    d.update(res.summary())
    if extra:
        d.update(extra)
    return d


def main() -> None:
    merged = ROOT / "data" / "merged" / "ETHUSDT_1h.csv"
    df = load_ohlc(merged)
    print(f"Full data: {len(df):,} bars  {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")

    # Crash / bear windows for SL design
    windows = {
        "crash_2022": slice_range(df, "2021-11-01", "2022-12-31"),
        "crash_2025_26": slice_range(df, "2024-12-01", "2026-07-31"),
        "full": df,
    }
    for name, w in windows.items():
        print(f"  window {name}: {len(w):,} bars  {w['datetime'].iloc[0]} -> {w['datetime'].iloc[-1]}")

    baseline = profit_max_base()
    history: list[dict] = []

    # --- Baseline no SL on each window ---
    print("\n### Baseline inverted (no SL)")
    base_by_window = {}
    for name, w in windows.items():
        res = run_backtest(w, baseline)
        base_by_window[name] = res
        history.append(row_from(res, f"baseline:{name}", {"window": name}))
        print_report(res, f"BASELINE no-SL — {name}")

    # --- Sweep stop loss ---
    sl_grid = [0.0, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
    refs = ["wap", "p0"]
    hold_grid = [0, 24, 48, 72, 120, 168, 336]  # hours on 1h bars

    print("\n### Phase 1: SL-only sweep on crash windows")
    candidates: list[dict] = []

    for ref in refs:
        for sl in sl_grid:
            p = profit_max_base(stop_loss_pct=sl, stop_loss_ref=ref, max_hold_bars=0)
            crash_scores = []
            detail = {}
            for wname in ("crash_2022", "crash_2025_26"):
                res = run_backtest(windows[wname], p)
                sc = crash_score(res)
                crash_scores.append(sc)
                detail[wname] = res
                history.append(
                    row_from(
                        res,
                        f"sl_sweep:{wname}",
                        {"window": wname, "sl": sl, "ref": ref, "max_hold": 0},
                    )
                )
                print(
                    f"  SL={sl:.0%} ref={ref:3s} [{wname}] "
                    f"ret={res.total_return_pct:+7.1f}% dd={res.max_drawdown_pct:5.1f}% "
                    f"avgHold={res.avg_hold_bars:6.1f} maxHold={res.max_hold_bars_seen:6.0f} "
                    f"SL_ex={res.sl_exits:4d} score={sc:+7.1f}"
                )
            # Combined crash score (equal weight)
            combo = float(sum(crash_scores) / len(crash_scores))
            # Quick full peek for ranking
            full_res = run_backtest(df, p)
            history.append(
                row_from(
                    full_res,
                    "sl_sweep:full",
                    {"window": "full", "sl": sl, "ref": ref, "max_hold": 0},
                )
            )
            candidates.append(
                {
                    "kind": "sl_only",
                    "sl": sl,
                    "ref": ref,
                    "max_hold": 0,
                    "crash_combo": combo,
                    "full_score": full_score(full_res),
                    "full_res": full_res,
                    "crash_2022": detail["crash_2022"],
                    "crash_2025_26": detail["crash_2025_26"],
                }
            )
            print(
                f"  >> combo_crash={combo:+7.1f}  full_ret={full_res.total_return_pct:+7.1f}% "
                f"full_dd={full_res.max_drawdown_pct:5.1f}% full_score={full_score(full_res):+.1f}"
            )

    # Rank: must beat no-SL on crash combo OR clearly cut max hold while keeping decent full return
    no_sl_crash = float(
        (
            crash_score(base_by_window["crash_2022"])
            + crash_score(base_by_window["crash_2025_26"])
        )
        / 2
    )
    no_sl_full = base_by_window["full"]

    print(f"\nNo-SL crash combo score: {no_sl_crash:+.1f}")
    print(
        f"No-SL full: ret={no_sl_full.total_return_pct:+.1f}% dd={no_sl_full.max_drawdown_pct:.1f}% "
        f"avgHold={no_sl_full.avg_hold_bars:.1f} maxHold={no_sl_full.max_hold_bars_seen:.0f}"
    )

    # Prefer candidates that improve crash score and don't destroy full return too badly
    # (< 40% of no-SL full return is rejected)
    viable = [
        c
        for c in candidates
        if c["full_res"].total_return_pct >= 0.4 * no_sl_full.total_return_pct
        or c["crash_combo"] > no_sl_crash
    ]
    if not viable:
        viable = candidates

    viable.sort(key=lambda c: (c["crash_combo"], c["full_score"]), reverse=True)
    best_sl = viable[0]
    print("\n### Best SL-only by crash combo (viable):")
    print(
        f"  SL={best_sl['sl']:.0%} ref={best_sl['ref']} "
        f"crash_combo={best_sl['crash_combo']:+.1f} "
        f"full_ret={best_sl['full_res'].total_return_pct:+.1f}%"
    )

    # --- Phase 2: around best SL, try max_hold combos ---
    print("\n### Phase 2: max-hold around best SL (+ no-SL time-stop only)")
    phase2: list[dict] = []
    base_sl_refs = [
        (best_sl["sl"], best_sl["ref"]),
        (0.0, "wap"),  # time-stop only
    ]
    # Also try a couple nearby SL values
    neighbors = sorted(
        {
            best_sl["sl"],
            max(0.0, best_sl["sl"] - 0.02),
            best_sl["sl"] + 0.02,
            best_sl["sl"] + 0.05,
        }
    )
    for sl in neighbors:
        base_sl_refs.append((sl, best_sl["ref"]))
    # unique
    seen = set()
    uniq_refs = []
    for sl, ref in base_sl_refs:
        key = (round(sl, 4), ref)
        if key in seen:
            continue
        seen.add(key)
        uniq_refs.append((sl, ref))

    for sl, ref in uniq_refs:
        for mh in hold_grid:
            if sl <= 0 and mh <= 0:
                continue  # baseline already recorded
            p = profit_max_base(stop_loss_pct=sl, stop_loss_ref=ref, max_hold_bars=mh)
            crash_scores = []
            detail = {}
            for wname in ("crash_2022", "crash_2025_26"):
                res = run_backtest(windows[wname], p)
                crash_scores.append(crash_score(res))
                detail[wname] = res
                history.append(
                    row_from(
                        res,
                        f"combo_sweep:{wname}",
                        {"window": wname, "sl": sl, "ref": ref, "max_hold": mh},
                    )
                )
            combo = float(sum(crash_scores) / len(crash_scores))
            full_res = run_backtest(df, p)
            history.append(
                row_from(
                    full_res,
                    "combo_sweep:full",
                    {"window": "full", "sl": sl, "ref": ref, "max_hold": mh},
                )
            )
            item = {
                "kind": "combo",
                "sl": sl,
                "ref": ref,
                "max_hold": mh,
                "crash_combo": combo,
                "full_score": full_score(full_res),
                "full_res": full_res,
                "crash_2022": detail["crash_2022"],
                "crash_2025_26": detail["crash_2025_26"],
            }
            phase2.append(item)
            print(
                f"  SL={sl:.0%} ref={ref} hold={mh:3d}h  "
                f"crash={combo:+7.1f} full_ret={full_res.total_return_pct:+7.1f}% "
                f"dd={full_res.max_drawdown_pct:5.1f}% avgH={full_res.avg_hold_bars:5.1f} "
                f"maxH={full_res.max_hold_bars_seen:5.0f} SLex={full_res.sl_exits}"
            )

    all_cand = candidates + phase2
    # Final pick: maximize crash_combo among those that keep full return >= 50% of no-SL
    # and reduce max_hold vs no-SL
    finalists = [
        c
        for c in all_cand
        if c["full_res"].total_return_pct >= 0.5 * no_sl_full.total_return_pct
        and c["full_res"].max_hold_bars_seen <= no_sl_full.max_hold_bars_seen
    ]
    if not finalists:
        finalists = [
            c
            for c in all_cand
            if c["full_res"].total_return_pct >= 0.3 * no_sl_full.total_return_pct
        ]
    finalists.sort(
        key=lambda c: (
            c["crash_combo"],
            c["full_score"],
            -c["full_res"].max_drawdown_pct,
        ),
        reverse=True,
    )
    best = finalists[0]

    print("\n### SELECTED")
    print(
        f"  SL={best['sl']:.0%} ref={best['ref']} max_hold={best['max_hold']} "
        f"crash_combo={best['crash_combo']:+.1f}"
    )
    print_report(best["crash_2022"], "SELECTED — crash_2022")
    print_report(best["crash_2025_26"], "SELECTED — crash_2025_26")
    print_report(best["full_res"], "SELECTED — FULL HISTORY")
    print_report(no_sl_full, "COMPARE — FULL HISTORY no SL")

    # Also report best by full_score among short-hold candidates
    short = [
        c
        for c in all_cand
        if c["full_res"].avg_hold_bars <= max(12.0, 0.5 * no_sl_full.avg_hold_bars)
        and c["full_res"].total_return_pct > 0
    ]
    best_short = None
    if short:
        short.sort(key=lambda c: c["full_score"], reverse=True)
        best_short = short[0]
        print_report(best_short["full_res"], "ALT — best full-score with shorter avg hold")

    # Save
    out = ROOT / "results" / "stoploss_inverted"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out / "optimization_history.csv", index=False)

    def pack(res):
        return res.summary()

    summary = {
        "base_config": "inverted_profit_max",
        "no_sl_full": pack(no_sl_full),
        "no_sl_crash_2022": pack(base_by_window["crash_2022"]),
        "no_sl_crash_2025_26": pack(base_by_window["crash_2025_26"]),
        "selected": {
            "stop_loss_pct": best["sl"],
            "stop_loss_ref": best["ref"],
            "max_hold_bars": best["max_hold"],
            "crash_combo_score": best["crash_combo"],
            "full": pack(best["full_res"]),
            "crash_2022": pack(best["crash_2022"]),
            "crash_2025_26": pack(best["crash_2025_26"]),
        },
        "alt_short_hold": None
        if best_short is None
        else {
            "stop_loss_pct": best_short["sl"],
            "stop_loss_ref": best_short["ref"],
            "max_hold_bars": best_short["max_hold"],
            "full": pack(best_short["full_res"]),
        },
        "improvement_vs_nosl": {
            "full_return_pp": best["full_res"].total_return_pct - no_sl_full.total_return_pct,
            "full_dd_pp": best["full_res"].max_drawdown_pct - no_sl_full.max_drawdown_pct,
            "avg_hold_delta": best["full_res"].avg_hold_bars - no_sl_full.avg_hold_bars,
            "max_hold_delta": best["full_res"].max_hold_bars_seen - no_sl_full.max_hold_bars_seen,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Cycle logs + equity
    for tag, res in (
        ("nosl_full", no_sl_full),
        ("selected_full", best["full_res"]),
        ("selected_crash_2022", best["crash_2022"]),
    ):
        rows = []
        for c in res.cycles:
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
                    "max_dd_from_p0": c.max_dd_from_p0,
                    "reason": c.reason,
                }
            )
        pd.DataFrame(rows).to_csv(out / f"cycles_{tag}.csv", index=False)
        if res.equity_curve is not None:
            res.equity_curve.to_csv(out / f"equity_{tag}.csv", header=True)

    # Equity comparison plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(no_sl_full.equity_curve.index, no_sl_full.equity_curve.values, label="No SL", lw=1.2)
        ax.plot(
            best["full_res"].equity_curve.index,
            best["full_res"].equity_curve.values,
            label=f"SL {best['sl']:.0%} ({best['ref']}) hold={best['max_hold']}",
            lw=1.2,
        )
        if best_short is not None and best_short is not best:
            ax.plot(
                best_short["full_res"].equity_curve.index,
                best_short["full_res"].equity_curve.values,
                label=f"Alt SL {best_short['sl']:.0%} hold={best_short['max_hold']}",
                lw=1.0,
                alpha=0.85,
            )
        ax.set_title("Inverted Pyramid — Stop Loss optimization (full history)")
        ax.set_ylabel("Equity (USDT)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "equity_comparison.png", dpi=140)
        print(f"Saved plot -> {out / 'equity_comparison.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"plot skipped: {exc}")

    print(f"\nSaved results -> {out}")
    print(json.dumps(summary["selected"], indent=2, default=str)[:1200])


if __name__ == "__main__":
    main()
