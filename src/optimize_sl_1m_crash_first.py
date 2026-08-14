"""Crash-first stop-loss search on ETHUSDT 1m for inverted pyramid.

Goal: maximize profit while keeping holding period short.
1) Sweep on crash windows
2) Validate survivors on full history
"""

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


def hold_hours(res) -> float:
    return res.avg_hold_bars / 60.0


def max_hold_hours(res) -> float:
    return res.max_hold_bars_seen / 60.0


def crash_score(res) -> float:
    """Higher is better on crash windows: profit, low DD, short holds."""
    # Heavy penalty if wiped
    if res.equity_final < 2000:
        return res.total_return_pct - 150.0
    avg_h = hold_hours(res)
    max_h = max_hold_hours(res)
    # Prefer avg hold under ~24h and max under ~7 days
    hold_pen = 1.5 * avg_h + 0.05 * max_h
    return res.total_return_pct - 0.45 * res.max_drawdown_pct - hold_pen


def full_score(res) -> float:
    if res.equity_final < 2000:
        return res.total_return_pct - 200.0
    avg_h = hold_hours(res)
    max_h = max_hold_hours(res)
    hold_pen = 0.8 * avg_h + 0.02 * max_h
    return res.total_return_pct - 0.35 * res.max_drawdown_pct - hold_pen


def fmt(res) -> str:
    return (
        f"ret={res.total_return_pct:+7.1f}% dd={res.max_drawdown_pct:5.1f}% "
        f"eq={res.equity_final:9.1f} cyc={res.num_cycles:5d} "
        f"avgH={hold_hours(res):7.1f}h maxH={max_hold_hours(res):8.1f}h "
        f"SLex={res.sl_exits:4d} TPex={res.tp_exits:4d}"
    )


def main() -> None:
    t_all = time.time()
    df = load_ohlc(ROOT / "data" / "merged" / "ETHUSDT_1m.csv")
    c22 = df[(df["datetime"] >= "2021-11-01") & (df["datetime"] <= "2022-12-31")].reset_index(drop=True)
    c25 = df[(df["datetime"] >= "2024-12-01") & (df["datetime"] <= "2026-07-31")].reset_index(drop=True)
    print(f"full={len(df):,}  crash22={len(c22):,}  crash25={len(c25):,}", flush=True)

    # Grids — wider SL because tight SL died on 1m previously
    sl_grid = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
    refs = ["p0", "wap"]
    # wall-clock max hold (hours) -> bars on 1m
    hold_h_grid = [0, 24, 48, 72, 120, 168, 336]  # up to 14 days
    tp_grid = [0.012, 0.015, 0.02, 0.025, 0.03]

    history: list[dict] = []
    crash_candidates: list[dict] = []

    print("\n===== PHASE 1: SL-only on CRASH windows (tp=1.2%, no time-stop) =====", flush=True)
    for ref in refs:
        for sl in sl_grid:
            p = base(stop_loss_pct=sl, stop_loss_ref=ref, take_profit_pct=0.012, max_hold_bars=0)
            r22 = run_backtest(c22, p)
            r25 = run_backtest(c25, p)
            sc = 0.5 * (crash_score(r22) + crash_score(r25))
            print(f"SL={sl:.0%} {ref:3s} |22| {fmt(r22)}", flush=True)
            print(f"           |25| {fmt(r25)}  combo={sc:+.1f}", flush=True)
            history.append(
                {
                    "phase": "crash_sl",
                    "sl": sl,
                    "ref": ref,
                    "tp": 0.012,
                    "hold_h": 0,
                    "crash_combo": sc,
                    "r22": r22.summary(),
                    "r25": r25.summary(),
                }
            )
            crash_candidates.append(
                {
                    "sl": sl,
                    "ref": ref,
                    "tp": 0.012,
                    "hold_h": 0,
                    "crash_combo": sc,
                    "r22": r22,
                    "r25": r25,
                    "params": p,
                }
            )

    # Rank crash SL-only
    crash_candidates.sort(key=lambda x: x["crash_combo"], reverse=True)
    print("\nTop SL-only by crash combo:", flush=True)
    for c in crash_candidates[:8]:
        print(
            f"  SL={c['sl']:.0%} {c['ref']} combo={c['crash_combo']:+.1f} "
            f"22ret={c['r22'].total_return_pct:+.1f}% 25ret={c['r25'].total_return_pct:+.1f}% "
            f"avgH22={hold_hours(c['r22']):.1f}h avgH25={hold_hours(c['r25']):.1f}h",
            flush=True,
        )

    # Keep top SL keys + always include no-SL and a couple wide SLs
    top_keys = []
    for c in crash_candidates:
        key = (c["sl"], c["ref"])
        if key not in top_keys:
            top_keys.append(key)
        if len(top_keys) >= 4:
            break
    for extra in [(0.0, "p0"), (0.40, "wap"), (0.50, "p0")]:
        if extra not in top_keys:
            top_keys.append(extra)

    # Narrow refine grid for runtime on 1m crash slices
    hold_h_grid = [0, 48, 72, 168]
    tp_grid = [0.012, 0.02, 0.03]

    print("\n===== PHASE 2: refine top SL with TP + max-hold on CRASH =====", flush=True)
    refined: list[dict] = []
    for sl, ref in top_keys:
        for tp in tp_grid:
            for hh in hold_h_grid:
                # Skip pure duplicate of phase1 default
                if tp == 0.012 and hh == 0 and any(
                    c["sl"] == sl and c["ref"] == ref and c["tp"] == 0.012 and c["hold_h"] == 0
                    for c in crash_candidates
                ):
                    # reuse
                    match = next(
                        c
                        for c in crash_candidates
                        if c["sl"] == sl and c["ref"] == ref and c["tp"] == 0.012 and c["hold_h"] == 0
                    )
                    refined.append(match)
                    continue
                p = base(
                    stop_loss_pct=sl,
                    stop_loss_ref=ref,
                    take_profit_pct=tp,
                    max_hold_bars=hh * 60,
                )
                r22 = run_backtest(c22, p)
                r25 = run_backtest(c25, p)
                sc = 0.5 * (crash_score(r22) + crash_score(r25))
                print(
                    f"SL={sl:.0%} {ref} TP={tp:.1%} hold={hh:3d}h "
                    f"combo={sc:+7.1f} |22 {r22.total_return_pct:+6.1f}% avgH={hold_hours(r22):6.1f}h "
                    f"|25 {r25.total_return_pct:+6.1f}% avgH={hold_hours(r25):6.1f}h",
                    flush=True,
                )
                item = {
                    "sl": sl,
                    "ref": ref,
                    "tp": tp,
                    "hold_h": hh,
                    "crash_combo": sc,
                    "r22": r22,
                    "r25": r25,
                    "params": p,
                }
                refined.append(item)
                history.append(
                    {
                        "phase": "crash_refine",
                        "sl": sl,
                        "ref": ref,
                        "tp": tp,
                        "hold_h": hh,
                        "crash_combo": sc,
                        "r22_ret": r22.total_return_pct,
                        "r25_ret": r25.total_return_pct,
                        "r22_avgH": hold_hours(r22),
                        "r25_avgH": hold_hours(r25),
                        "r22_maxH": max_hold_hours(r22),
                        "r25_maxH": max_hold_hours(r25),
                        "r22_dd": r22.max_drawdown_pct,
                        "r25_dd": r25.max_drawdown_pct,
                    }
                )

    refined.sort(key=lambda x: x["crash_combo"], reverse=True)

    # Also build a short-hold shortlist: avg hold (mean of two crashes) <= 48h
    # and not wiped (equity > 3k on both), sorted by crash combo
    shortlist = [
        c
        for c in refined
        if (hold_hours(c["r22"]) + hold_hours(c["r25"])) / 2 <= 48.0
        and c["r22"].equity_final >= 3000
        and c["r25"].equity_final >= 3000
    ]
    shortlist.sort(key=lambda x: x["crash_combo"], reverse=True)

    print("\nTop crash refined:", flush=True)
    for c in refined[:10]:
        print(
            f"  SL={c['sl']:.0%} {c['ref']} TP={c['tp']:.1%} hold={c['hold_h']}h "
            f"combo={c['crash_combo']:+.1f} 22={c['r22'].total_return_pct:+.1f}% "
            f"25={c['r25'].total_return_pct:+.1f}%",
            flush=True,
        )
    print("\nTop SHORT-HOLD crash survivors (avgH<=48h, eq>=3k):", flush=True)
    for c in shortlist[:10]:
        print(
            f"  SL={c['sl']:.0%} {c['ref']} TP={c['tp']:.1%} hold={c['hold_h']}h "
            f"combo={c['crash_combo']:+.1f} avgH22={hold_hours(c['r22']):.1f} "
            f"avgH25={hold_hours(c['r25']):.1f}",
            flush=True,
        )

    # PHASE 3: validate top candidates on FULL history
    print("\n===== PHASE 3: validate on FULL 1m history =====", flush=True)
    to_test = []
    seen = set()
    for pool in (shortlist[:8], refined[:8]):
        for c in pool:
            key = (c["sl"], c["ref"], c["tp"], c["hold_h"])
            if key in seen:
                continue
            seen.add(key)
            to_test.append(c)
    # Always include baselines
    for sl, ref, tp, hh in [
        (0.0, "p0", 0.012, 0),
        (0.0, "p0", 0.02, 0),
        (0.05, "p0", 0.012, 72),  # old 1h-tuned
        (0.40, "wap", 0.02, 0),
    ]:
        key = (sl, ref, tp, hh)
        if key not in seen:
            seen.add(key)
            to_test.append(
                {
                    "sl": sl,
                    "ref": ref,
                    "tp": tp,
                    "hold_h": hh,
                    "params": base(
                        stop_loss_pct=sl,
                        stop_loss_ref=ref,
                        take_profit_pct=tp,
                        max_hold_bars=hh * 60,
                    ),
                    "crash_combo": None,
                }
            )

    full_results = []
    for c in to_test:
        p = c["params"]
        t0 = time.time()
        res = run_backtest(df, p)
        sc = full_score(res)
        print(
            f"FULL SL={c['sl']:.0%} {c['ref']} TP={c['tp']:.1%} hold={c['hold_h']}h "
            f"{fmt(res)} score={sc:+.1f} ({time.time()-t0:.1f}s)",
            flush=True,
        )
        full_results.append({**c, "full": res, "full_score": sc})
        history.append(
            {
                "phase": "full",
                "sl": c["sl"],
                "ref": c["ref"],
                "tp": c["tp"],
                "hold_h": c["hold_h"],
                "crash_combo": c.get("crash_combo"),
                "full_score": sc,
                **{f"full_{k}": v for k, v in res.summary().items()},
            }
        )

    # Selection rules:
    # A) Best full_score among short-hold on full: avg hold <= 48h
    # B) Best full return among avg hold <= 24h if any survive
    # C) Best absolute full_score overall
    short_full = [x for x in full_results if hold_hours(x["full"]) <= 48.0 and x["full"].equity_final >= 5000]
    short24 = [x for x in full_results if hold_hours(x["full"]) <= 24.0 and x["full"].equity_final >= 5000]

    best_overall = max(full_results, key=lambda x: x["full_score"])
    best_short48 = max(short_full, key=lambda x: x["full"].total_return_pct) if short_full else None
    best_short24 = max(short24, key=lambda x: x["full"].total_return_pct) if short24 else None
    # Prefer short48 with highest return if available, else overall
    selected = best_short48 or best_overall

    print("\n===== SELECTED =====", flush=True)
    print(
        f"SELECTED: SL={selected['sl']:.0%} ref={selected['ref']} "
        f"TP={selected['tp']:.1%} max_hold={selected['hold_h']}h",
        flush=True,
    )
    # Recompute crash stats for selected
    p = selected["params"]
    r22 = run_backtest(c22, p)
    r25 = run_backtest(c25, p)
    print_report(r22, "SELECTED — crash 2022 (1m)")
    print_report(r25, "SELECTED — crash 2025-26 (1m)")
    print_report(selected["full"], "SELECTED — FULL (1m)")

    if best_short24 is not None:
        print_report(best_short24["full"], "ALT — best among avgHold<=24h on full")
    if best_overall is not selected:
        print_report(best_overall["full"], "ALT — best overall full_score")

    out = ROOT / "results" / "tf_1m_crash_sl"
    out.mkdir(parents=True, exist_ok=True)

    # Flatten history for CSV
    flat_rows = []
    for h in history:
        row = {k: v for k, v in h.items() if not isinstance(v, dict)}
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(out / "optimization_history.csv", index=False)

    def pack_sel(c, r22=None, r25=None):
        return {
            "stop_loss_pct": c["sl"],
            "stop_loss_ref": c["ref"],
            "take_profit_pct": c["tp"],
            "max_hold_hours": c["hold_h"],
            "max_hold_bars": c["hold_h"] * 60,
            "crash_combo": c.get("crash_combo"),
            "full": c["full"].summary() if "full" in c else None,
            "crash_2022": None if r22 is None else r22.summary(),
            "crash_2025_26": None if r25 is None else r25.summary(),
        }

    summary = {
        "data": "ETHUSDT_1m",
        "bars_full": len(df),
        "selected": pack_sel(selected, r22, r25),
        "best_overall_full_score": pack_sel(best_overall),
        "best_short24": None if best_short24 is None else pack_sel(best_short24),
        "note": (
            "Crash-first SL search on 1m. Tight SL (e.g. 5%) overtrades and dies on fees/path noise; "
            "wider SL or no-SL with higher TP better preserves capital. Short-hold constraint applied."
        ),
        "elapsed_sec": time.time() - t_all,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    rows = []
    for cyc in selected["full"].cycles:
        rows.append(
            {
                "entry_time": str(cyc.entry_time),
                "exit_time": str(cyc.exit_time),
                "p0": cyc.p0,
                "avg_entry": cyc.avg_entry,
                "exit_price": cyc.exit_price,
                "pnl": cyc.pnl,
                "pnl_pct": cyc.pnl_pct,
                "hold_bars": cyc.hold_bars,
                "reason": cyc.reason,
            }
        )
    pd.DataFrame(rows).to_csv(out / "cycles_selected_full.csv", index=False)
    if selected["full"].equity_curve is not None:
        selected["full"].equity_curve.iloc[::60].to_csv(out / "equity_selected_1h_sample.csv", header=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))
        eq = selected["full"].equity_curve.iloc[::60]
        ax.plot(eq.index, eq.values, label="Selected on 1m", lw=1.2)
        ax.set_title(
            f"1m crash-first SL — SL={selected['sl']:.0%} {selected['ref']} "
            f"TP={selected['tp']:.1%} hold≤{selected['hold_h']}h"
        )
        ax.set_ylabel("Equity")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "equity_selected.png", dpi=140)
    except Exception as exc:  # noqa: BLE001
        print(f"plot skipped: {exc}", flush=True)

    print(f"\nSaved -> {out}  elapsed={time.time()-t_all:.1f}s", flush=True)
    print(json.dumps(summary["selected"], indent=2, default=str)[:1500], flush=True)


if __name__ == "__main__":
    main()
