"""Lean search: cut hold time while raising return on ETHUSDT 1m (classic pyramid).

Preserves level weights 1:2:4:8. Focuses on wide WAP stop-loss + modest TP/depth
tweaks; optional soft breakeven / max-hold around survivors only.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from strategy import StrategyParams, print_report, run_backtest  # noqa: E402

DATA = ROOT / "data" / "merged" / "ETHUSDT_1m.csv"
OUT = ROOT / "results" / "pyramid_short_hold_1m"


def hours(bars: float) -> float:
    return bars / 60.0


def score(res, base_ret: float, base_avg_h: float) -> float:
    """Reward return & hold cut; penalize DD and extreme max hold."""
    avg_h = hours(res.avg_hold_bars)
    max_h = hours(res.max_hold_bars_seen)
    hold_cut = max(0.0, base_avg_h - avg_h)  # hours saved vs baseline avg
    hold_penalty = max(0.0, avg_h - base_avg_h) * 1.5
    max_penalty = max(0.0, max_h - 2000.0) * 0.02  # soft; crash legs still long
    dd_pen = max(0.0, res.max_drawdown_pct - 40.0) * 2.0
    ret_bonus = res.total_return_pct - base_ret
    return ret_bonus + hold_cut * 0.8 - hold_penalty - max_penalty - dd_pen


def fmt(res) -> str:
    return (
        f"ret={res.total_return_pct:+7.1f}% dd={res.max_drawdown_pct:5.1f}% "
        f"avgH={hours(res.avg_hold_bars):7.1f}h maxH={hours(res.max_hold_bars_seen):8.1f}h "
        f"cyc={res.num_cycles:5d} TP/SL/BE/T="
        f"{res.tp_exits}/{res.sl_exits}/{res.breakeven_exits}/{res.time_exits}"
    )


def load_df() -> pd.DataFrame:
    if not DATA.exists():
        raise FileNotFoundError(f"Missing {DATA}; download 1m first")
    df = pd.read_csv(DATA)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df


def downsample_equity(eq: pd.Series, rule: str = "1h") -> pd.Series:
    return eq.resample(rule).last().dropna()


def main() -> None:
    t0 = time.time()
    df = load_df()
    print(f"1m bars={len(df):,}", flush=True)

    original = StrategyParams(
        initial_pct=0.05,
        dca_level_weights=(1, 2, 4, 8),
        sub_order_weights=(1, 2, 4),
        dca_depths=(0.03, 0.10, 0.25, 0.50),
        take_profit_pct=0.01,
    )
    # Prior pyramid-preserving high-profit (long holds)
    long_hold = StrategyParams(
        initial_pct=0.08,
        dca_level_weights=(1, 2, 4, 8),
        sub_order_weights=(1, 2, 4),
        dca_depths=(0.03, 0.08, 0.18, 0.35),
        take_profit_pct=0.03,
    )

    print("Running ORIGINAL baseline...", flush=True)
    base_res = run_backtest(df, original)
    print_report(base_res, "ORIGINAL baseline 1m")
    base_ret = base_res.total_return_pct
    base_avg_h = hours(base_res.avg_hold_bars)
    print(f"baseline_ret={base_ret:.2f}% baseline_avgH={base_avg_h:.1f}h", flush=True)

    print("Running prior long-hold...", flush=True)
    prev = run_backtest(df, long_hold)
    print_report(prev, "Prior pyramid-preserving (long holds)")

    history: list[dict] = []
    best_overall = {"params": original, "res": base_res, "score": score(base_res, base_ret, base_avg_h)}
    best_dual = None  # ret >= baseline AND avgH <= 80% baseline

    def consider(tag: str, p: StrategyParams, res=None):
        nonlocal best_overall, best_dual
        if res is None:
            res = run_backtest(df, p)
        sc = score(res, base_ret, base_avg_h)
        avg_h = hours(res.avg_hold_bars)
        dual_ok = res.total_return_pct >= base_ret and avg_h <= 0.8 * base_avg_h
        history.append({"tag": tag, "score": sc, "dual_ok": dual_ok, **res.summary()})
        marker = ""
        if sc > best_overall["score"]:
            best_overall = {"params": p, "res": res, "score": sc}
            marker += " <-- best score"
        if dual_ok and (
            best_dual is None
            or (res.total_return_pct, -avg_h)
            > (best_dual["res"].total_return_pct, -hours(best_dual["res"].avg_hold_bars))
        ):
            best_dual = {"params": p, "res": res, "score": sc}
            marker += " [DUAL]"
        print(f"  {tag:48s} {fmt(res)} score={sc:+7.1f}{marker}", flush=True)
        return res

    # Seed: known survivor from prior interrupted search (SL 35% WAP)
    print("\n### seeds", flush=True)
    seed_sl35 = replace(
        long_hold, stop_loss_pct=0.35, stop_loss_ref="wap", take_profit_pct=0.03
    )
    consider("seed_sl35_wap", seed_sl35)

    # Focused SL sweep on long_hold skeleton
    print("\n### SL sweep (WAP) on pyramid-preserving skeleton", flush=True)
    for sl in [0.28, 0.30, 0.32, 0.35, 0.38, 0.40, 0.45]:
        consider(f"sl_wap={sl}", replace(long_hold, stop_loss_pct=sl, stop_loss_ref="wap"))

    print("\n### SL sweep (P0) on same skeleton", flush=True)
    for sl in [0.30, 0.35, 0.40, 0.50]:
        consider(f"sl_p0={sl}", replace(long_hold, stop_loss_pct=sl, stop_loss_ref="p0"))

    # TP around best SL so far
    best_p = best_overall["params"]
    print("\n### TP around best", flush=True)
    for tp in [0.012, 0.015, 0.02, 0.025, 0.03, 0.035]:
        consider(f"tp={tp}", replace(best_p, take_profit_pct=tp))

    # Soft hold cutters on best score params
    print("\n### soft hold cutters on best score", flush=True)
    bp = best_overall["params"]
    for be_h in [48, 72, 120, 168]:  # hours
        consider(
            f"be_after={be_h}h",
            replace(bp, breakeven_after_bars=be_h * 60, breakeven_buffer_pct=0.001),
        )
    for days in [7, 14, 30]:
        consider(f"max_hold={days}d", replace(bp, max_hold_bars=days * 24 * 60))

    # Compact combo: best SL ± nearby TP/SL with optional 14d max hold
    print("\n### compact refine", flush=True)
    for tp in [0.02, 0.025, 0.03]:
        for sl in [0.32, 0.35, 0.38]:
            for mh_d in [0, 30]:
                p = replace(
                    long_hold,
                    take_profit_pct=tp,
                    stop_loss_pct=sl,
                    stop_loss_ref="wap",
                    max_hold_bars=mh_d * 24 * 60,
                )
                consider(f"refine_tp{tp}_sl{sl}_mh{mh_d}d", p)

    selected = best_dual or best_overall
    print("\n===== SELECTED =====", flush=True)
    print_report(base_res, "BASELINE original")
    print_report(selected["res"], "SELECTED short-hold/profit")
    if best_dual and best_dual is not selected:
        print_report(best_dual["res"], "DUAL (ret>=baseline & avgH<=80% baseline)")
    if best_overall is not selected:
        print_report(best_overall["res"], "BEST score overall")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(OUT / "optimization_history.csv", index=False)

    def pack(res, params=None, label=""):
        return {
            "label": label,
            "params": (params or res.params).to_dict(),
            "metrics": {
                k: v
                for k, v in res.summary().items()
                if not k.startswith("param_")
            },
            "score": score(res, base_ret, base_avg_h),
            "avg_hold_hours": hours(res.avg_hold_bars),
            "max_hold_hours": hours(res.max_hold_bars_seen),
        }

    summary = {
        "baseline": pack(base_res, original, "original"),
        "prior_long_hold": pack(prev, long_hold, "prior_long_hold"),
        "selected": pack(selected["res"], selected["params"], "selected"),
        "best_score": pack(best_overall["res"], best_overall["params"], "best_score"),
        "best_dual": pack(best_dual["res"], best_dual["params"], "best_dual")
        if best_dual
        else None,
        "elapsed_s": time.time() - t0,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # Equity samples (1h) to keep GitHub-friendly
    for name, res in [
        ("baseline", base_res),
        ("selected", selected["res"]),
        ("best_score", best_overall["res"]),
    ]:
        if res.equity_curve is not None:
            downsample_equity(res.equity_curve).to_csv(OUT / f"equity_{name}_1h_sample.csv")

    # Cycle dump for selected
    sel_cycles = selected["res"].cycles
    if sel_cycles:
        pd.DataFrame(
            [
                {
                    "entry_time": c.entry_time,
                    "exit_time": c.exit_time,
                    "p0": c.p0,
                    "avg_entry": c.avg_entry,
                    "pnl": c.pnl,
                    "pnl_pct": c.pnl_pct,
                    "hold_bars": c.hold_bars,
                    "reason": c.reason,
                    "fills": c.fills,
                }
                for c in sel_cycles
            ]
        ).to_csv(OUT / "cycles_selected.csv", index=False)

    try:
        fig, ax = plt.subplots(figsize=(11, 5))
        for label, res, style in [
            ("baseline", base_res, "-"),
            ("selected", selected["res"], "-"),
            ("prior long-hold", prev, "--"),
        ]:
            if res.equity_curve is None:
                continue
            eq = downsample_equity(res.equity_curve)
            ax.plot(eq.index, eq.values, style, label=label, linewidth=1.2)
        ax.set_title("ETHUSDT 1m — short-hold vs baseline (1h equity sample)")
        ax.set_ylabel("Equity")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "equity_comparison.png", dpi=120)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"plot skipped: {exc}", flush=True)

    print(f"\nSaved -> {OUT} elapsed={time.time()-t0:.1f}s", flush=True)
    print("SELECTED PARAMS:", json.dumps(selected["params"].to_dict(), indent=2), flush=True)
    print(
        f"selected ret={selected['res'].total_return_pct:+.2f}% "
        f"avgH={hours(selected['res'].avg_hold_bars):.1f}h "
        f"vs baseline {base_ret:+.2f}% / {base_avg_h:.1f}h",
        flush=True,
    )


if __name__ == "__main__":
    main()
