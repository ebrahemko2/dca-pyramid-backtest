"""Backtest + sequential optimize Staged Recovery strategy on ETHUSDT 1m/1h."""

from __future__ import annotations

import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from staged_recovery import StagedParams, print_staged, run_staged  # noqa: E402


def load_ohlc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    return df.sort_values("datetime").reset_index(drop=True)


def score(res) -> float:
    if res.equity_final < 2000:
        return res.total_return_pct - 150.0
    # Prefer return, punish DD and very long holds (bars assumed 1m when used on 1m)
    hold_h = res.avg_hold_bars / 60.0
    return res.total_return_pct - 0.4 * res.max_drawdown_pct - 0.3 * hold_h


def crash_score(res) -> float:
    if res.equity_final < 2000:
        return res.total_return_pct - 120.0
    hold_h = res.avg_hold_bars / 60.0
    return res.total_return_pct - 0.5 * res.max_drawdown_pct - 0.8 * hold_h


def apply(base: StagedParams, name: str, value: Any) -> StagedParams:
    p = deepcopy(base)
    setattr(p, name, value)
    return p


SEARCH: list[tuple[str, list[Any]]] = [
    ("normal_sl_pct", [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]),
    ("recovery_sl_pct", [0.06, 0.08, 0.10, 0.12, 0.15, 0.20]),
    ("normal_tp_pct", [0.006, 0.008, 0.01, 0.012, 0.015, 0.02]),
    ("recovery_tp_pct", [0.008, 0.01, 0.012, 0.015, 0.02, 0.025]),
    ("normal_dca_drop", [0.02, 0.03, 0.04, 0.05]),
    ("recovery_dca2_drop", [0.04, 0.05, 0.06, 0.08]),
    ("sl_ref", ["p0", "wap"]),
    (
        "normal_initial_pct",
        [0.15, 0.20, 0.25, 0.30],
    ),
    (
        "recovery_initial_pct",
        [0.40, 0.50, 0.60],
    ),
]


def optimize_on(df: pd.DataFrame, base: StagedParams, score_fn) -> tuple[StagedParams, list[dict]]:
    current = deepcopy(base)
    history: list[dict] = []
    base_res = run_staged(df, current)
    history.append({"stage": "baseline", "param": None, "value": None, "score": score_fn(base_res), **base_res.summary()})
    print_staged(base_res, "BASELINE")

    for name, values in SEARCH:
        print(f"\n### optimize {name}", flush=True)
        best_p = current
        best_sc = score_fn(run_staged(df, current))
        vals = list(values)
        cur = getattr(current, name)
        if cur not in vals:
            vals.insert(0, cur)
        for val in vals:
            trial = apply(current, name, val)
            # Keep ladder sums <= 1
            if name == "normal_initial_pct":
                trial.normal_dca_pct = min(trial.normal_dca_pct, 1.0 - trial.normal_initial_pct)
            if name == "recovery_initial_pct":
                rem = 1.0 - trial.recovery_initial_pct
                # keep equal split of remainder for two DCAs if needed
                trial.recovery_dca1_pct = rem / 2
                trial.recovery_dca2_pct = rem / 2
            try:
                res = run_staged(df, trial)
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {name}={val}: {exc}", flush=True)
                continue
            sc = score_fn(res)
            marker = ""
            if sc > best_sc:
                best_sc = sc
                best_p = trial
                marker = " <-- best"
            print(
                f"  {name}={val!r:12s} ret={res.total_return_pct:+7.1f}% dd={res.max_drawdown_pct:5.1f}% "
                f"wr={res.win_rate:5.1f}% avgHbars={res.avg_hold_bars:7.1f} "
                f"N/R={res.normal_cycles}/{res.recovery_cycles} score={sc:+7.1f}{marker}",
                flush=True,
            )
            history.append(
                {
                    "stage": f"search:{name}",
                    "param": name,
                    "value": val,
                    "score": sc,
                    **res.summary(),
                }
            )
        current = best_p
        print(f"  => keep {name}={getattr(current, name)!r}", flush=True)

    return current, history


def cycles_df(res) -> pd.DataFrame:
    rows = []
    for c in res.cycles:
        rows.append(
            {
                "mode": c.mode,
                "entry_time": str(c.entry_time),
                "exit_time": str(c.exit_time),
                "p0": c.p0,
                "avg_entry": c.avg_entry,
                "exit_price": c.exit_price,
                "budget": c.budget,
                "spent": c.spent,
                "pnl": c.pnl,
                "pnl_pct": c.pnl_pct,
                "fills": c.fills,
                "hold_bars": c.hold_bars,
                "reason": c.reason,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    df1m = load_ohlc(ROOT / "data" / "merged" / "ETHUSDT_1m.csv")
    df1h = load_ohlc(ROOT / "data" / "merged" / "ETHUSDT_1h.csv")
    c22 = df1m[(df1m["datetime"] >= "2021-11-01") & (df1m["datetime"] <= "2022-12-31")].reset_index(drop=True)
    c25 = df1m[(df1m["datetime"] >= "2024-12-01") & (df1m["datetime"] <= "2026-07-31")].reset_index(drop=True)
    print(f"1m={len(df1m):,} 1h={len(df1h):,} c22={len(c22):,} c25={len(c25):,}", flush=True)

    user = StagedParams()  # as described by user
    out = ROOT / "results" / "staged_recovery"
    out.mkdir(parents=True, exist_ok=True)

    print("\n===== USER BASELINE on 1m crashes + full + 1h =====", flush=True)
    r22 = run_staged(c22, user)
    r25 = run_staged(c25, user)
    r1m = run_staged(df1m, user)
    r1h = run_staged(df1h, user)
    print_staged(r22, "USER — crash 2022 1m")
    print_staged(r25, "USER — crash 2025-26 1m")
    print_staged(r1m, "USER — FULL 1m")
    print_staged(r1h, "USER — FULL 1h")

    # Optimize on crash combo first (average of two crash windows via concatenated approach:
    # optimize sequentially on crash_2022, then fine-tune on crash_2025, then validate full)
    print("\n===== OPTIMIZE on crash_2022 (1m) =====", flush=True)
    best_c22, hist1 = optimize_on(c22, user, crash_score)

    print("\n===== CONTINUE optimize on crash_2025-26 from best_c22 =====", flush=True)
    best_crash, hist2 = optimize_on(c25, best_c22, crash_score)

    print("\n===== VALIDATE + light refine on FULL 1m =====", flush=True)
    # One more pass on full with milder score
    best_full, hist3 = optimize_on(df1m, best_crash, score)

    r_best_1m = run_staged(df1m, best_full)
    r_best_1h = run_staged(df1h, best_full)
    r_best_22 = run_staged(c22, best_full)
    r_best_25 = run_staged(c25, best_full)
    print_staged(r_best_22, "BEST — crash 2022 1m")
    print_staged(r_best_25, "BEST — crash 2025-26 1m")
    print_staged(r_best_1m, "BEST — FULL 1m")
    print_staged(r_best_1h, "BEST — FULL 1h")

    history = hist1 + hist2 + hist3
    pd.DataFrame(history).to_csv(out / "optimization_history.csv", index=False)
    cycles_df(r1m).to_csv(out / "cycles_user_1m.csv", index=False)
    cycles_df(r_best_1m).to_csv(out / "cycles_best_1m.csv", index=False)
    if r_best_1m.equity_curve is not None:
        r_best_1m.equity_curve.iloc[::60].to_csv(out / "equity_best_1m_1h_sample.csv", header=True)
    if r1m.equity_curve is not None:
        r1m.equity_curve.iloc[::60].to_csv(out / "equity_user_1m_1h_sample.csv", header=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(r1m.equity_curve.iloc[::60].index, r1m.equity_curve.iloc[::60].values, label="User baseline", lw=1.2)
        ax.plot(
            r_best_1m.equity_curve.iloc[::60].index,
            r_best_1m.equity_curve.iloc[::60].values,
            label="Optimized staged",
            lw=1.2,
        )
        ax.set_title("Staged Recovery DCA — ETHUSDT 1m")
        ax.set_ylabel("Equity")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "equity_comparison.png", dpi=140)
    except Exception as exc:  # noqa: BLE001
        print(f"plot skipped: {exc}", flush=True)

    summary = {
        "idea": (
            "Normal: 25% now + 25% at -3% with tight SL; "
            "after loss Recovery: 50% + 25%@-3% + 25%@-5% with wider SL; "
            "after win return to Normal."
        ),
        "user_params": user.to_dict(),
        "best_params": best_full.to_dict(),
        "user_1m": r1m.summary(),
        "user_1h": r1h.summary(),
        "user_crash_2022": r22.summary(),
        "user_crash_2025_26": r25.summary(),
        "best_1m": r_best_1m.summary(),
        "best_1h": r_best_1h.summary(),
        "best_crash_2022": r_best_22.summary(),
        "best_crash_2025_26": r_best_25.summary(),
        "elapsed_sec": time.time() - t0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved -> {out}  elapsed={time.time()-t0:.1f}s", flush=True)
    print("BEST PARAMS:", json.dumps(best_full.to_dict(), indent=2), flush=True)


if __name__ == "__main__":
    main()
