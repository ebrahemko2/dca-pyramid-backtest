"""Crash-first search: >=5% flat (non-compounded) monthly return in ALL crash windows.

Definition of flat monthly rate for a window:
  flat_mo = total_return_pct / n_calendar_months
i.e. total profit / initial capital, spread evenly across months (non-compounded).

Also reports calendar-month returns vs fixed capital (pnl_month / capital0).

Pipeline:
  1) Detect major crash windows (+ known bear legs)
  2) Random + coordinate search on crash slices (1m)
  3) Keep candidates closest to / meeting 5%/mo on every crash
  4) Validate survivors on full ETHUSDT 1m history
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from strategy import StrategyParams, print_report, run_backtest  # noqa: E402

DATA = ROOT / "data" / "merged" / "ETHUSDT_1m.csv"
OUT = ROOT / "results" / "crash_monthly_5pct"
TARGET_FLAT_MO = 5.0  # % of capital per month, non-compounded
RNG = random.Random(42)


def load_df() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values("datetime").reset_index(drop=True)


def detect_crash_windows(df: pd.DataFrame, min_dd: float = 0.45, min_days: int = 60) -> list[dict]:
    """Peak-to-trough windows on daily closes with DD >= min_dd lasting >= min_days."""
    daily = (
        df.set_index("datetime")["close"]
        .resample("1D")
        .last()
        .dropna()
    )
    peak = daily.cummax()
    dd = daily / peak - 1.0
    windows: list[dict] = []
    in_crash = False
    start = None
    peak_px = None
    trough_i = None
    trough_px = None
    for ts, px in daily.items():
        if not in_crash:
            if dd.loc[ts] <= -0.15:  # start watching
                in_crash = True
                # walk back to last peak
                peak_ts = peak[:ts][peak[:ts] == peak.loc[ts]].index[-1]
                start = peak_ts
                peak_px = float(peak.loc[ts])
                trough_i = ts
                trough_px = float(px)
        else:
            if float(px) < trough_px:
                trough_i = ts
                trough_px = float(px)
            # recovery: back above 70% of peak from trough, or new ATH
            recovered = float(px) >= trough_px * 1.35 or float(px) >= peak_px * 0.95
            ended = recovered and (ts - trough_i).days >= 14
            if ended or ts == daily.index[-1]:
                end = ts
                depth = trough_px / peak_px - 1.0
                days = (end - start).days
                if depth <= -min_dd and days >= min_days:
                    windows.append(
                        {
                            "name": f"auto_{start.date()}_{end.date()}",
                            "start": str(start.date()),
                            "end": str(end.date()),
                            "dd": float(depth),
                            "days": int(days),
                        }
                    )
                in_crash = False
    # Deduplicate overlapping by keeping deeper/longer
    windows.sort(key=lambda w: (w["start"], w["dd"]))
    merged: list[dict] = []
    for w in windows:
        if not merged:
            merged.append(w)
            continue
        prev = merged[-1]
        if w["start"] <= prev["end"]:
            # overlap: extend / keep worse dd
            prev["end"] = max(prev["end"], w["end"])
            prev["dd"] = min(prev["dd"], w["dd"])
            prev["name"] = f"auto_{prev['start']}_{prev['end']}"
        else:
            merged.append(w)
    return merged


KNOWN_CRASHES = [
    {"name": "crash_2021_05", "start": "2021-05-01", "end": "2021-07-31"},
    {"name": "crash_2021_22", "start": "2021-11-01", "end": "2022-12-31"},
    {"name": "crash_2024_26", "start": "2024-12-01", "end": "2026-07-31"},
]


def slice_df(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    m = (df["datetime"] >= start) & (df["datetime"] <= end + " 23:59:59+00:00")
    return df.loc[m].reset_index(drop=True)


def n_months(start: str, end: str) -> int:
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC")
    return max(1, (e.year - s.year) * 12 + (e.month - s.month) + 1)


def monthly_stats(res, capital: float) -> dict[str, Any]:
    """Calendar-month PnL / capital0 (non-compounded base) + equity simple returns."""
    if res.equity_curve is None or res.equity_curve.empty:
        return {
            "months": 0,
            "flat_from_total": 0.0,
            "pct_months_ge_5_capital": 0.0,
            "pct_months_ge_5_equity": 0.0,
            "min_month_vs_capital": 0.0,
            "avg_month_vs_capital": 0.0,
            "month_rows": [],
        }
    eq = res.equity_curve.copy()
    eq.index = pd.DatetimeIndex(eq.index)
    if eq.index.tz is None:
        eq.index = eq.index.tz_localize("UTC")
    # month-end and month-start
    monthly = eq.resample("ME").last().dropna()
    rows = []
    prev = float(eq.iloc[0])
    for ts, val in monthly.items():
        end_eq = float(val)
        # approximate month start as previous month end (or first equity)
        pnl = end_eq - prev
        vs_cap = pnl / capital * 100.0
        vs_eq = (end_eq / prev - 1.0) * 100.0 if prev > 1e-9 else 0.0
        rows.append(
            {
                "month": str(ts.date())[:7],
                "start_eq": prev,
                "end_eq": end_eq,
                "pnl": pnl,
                "pct_vs_capital": vs_cap,
                "pct_vs_start_eq": vs_eq,
            }
        )
        prev = end_eq
    if not rows:
        return {
            "months": 0,
            "flat_from_total": 0.0,
            "pct_months_ge_5_capital": 0.0,
            "pct_months_ge_5_equity": 0.0,
            "min_month_vs_capital": 0.0,
            "avg_month_vs_capital": 0.0,
            "month_rows": [],
        }
    ge5_cap = sum(1 for r in rows if r["pct_vs_capital"] >= TARGET_FLAT_MO) / len(rows) * 100.0
    ge5_eq = sum(1 for r in rows if r["pct_vs_start_eq"] >= TARGET_FLAT_MO) / len(rows) * 100.0
    return {
        "months": len(rows),
        "flat_from_total": res.total_return_pct / len(rows),
        "pct_months_ge_5_capital": ge5_cap,
        "pct_months_ge_5_equity": ge5_eq,
        "min_month_vs_capital": min(r["pct_vs_capital"] for r in rows),
        "avg_month_vs_capital": float(np.mean([r["pct_vs_capital"] for r in rows])),
        "month_rows": rows,
    }


def eval_window(df_w: pd.DataFrame, params: StrategyParams, start: str, end: str) -> dict:
    res = run_backtest(df_w, params)
    months = n_months(start, end)
    flat = res.total_return_pct / months
    ms = monthly_stats(res, params.capital)
    return {
        "res": res,
        "months": months,
        "flat_mo": flat,
        "monthly": ms,
        "hit_flat5": flat >= TARGET_FLAT_MO,
    }


def score_candidate(crash_evals: dict[str, dict]) -> float:
    """Higher better. Hard reward for hitting 5% flat on all crashes; else minimize shortfall."""
    flats = [v["flat_mo"] for v in crash_evals.values()]
    min_flat = min(flats)
    mean_flat = float(np.mean(flats))
    hits = sum(1 for v in crash_evals.values() if v["hit_flat5"])
    all_hit = hits == len(crash_evals)
    # shortfall vs 5%
    short = sum(max(0.0, TARGET_FLAT_MO - f) for f in flats)
    # also prefer more months that individually clear 5% vs capital
    month_hit = float(
        np.mean([v["monthly"]["pct_months_ge_5_capital"] for v in crash_evals.values()])
    )
    dd_pen = float(np.mean([v["res"].max_drawdown_pct for v in crash_evals.values()])) * 0.15
    if all_hit:
        return 1000.0 + mean_flat * 10.0 + month_hit - dd_pen
    return mean_flat * 5.0 + hits * 50.0 + month_hit * 0.5 - short * 20.0 - dd_pen


def sample_params() -> StrategyParams:
    """Random pyramid-family params (classic + inverted + SL/TP/hold)."""
    classic = RNG.random() < 0.55
    lvl = (1, 2, 4, 8) if classic else (8, 4, 2, 1)
    depths = RNG.choice(
        [
            (0.03, 0.10, 0.25, 0.50),
            (0.03, 0.08, 0.18, 0.35),
            (0.02, 0.08, 0.20, 0.40),
            (0.02, 0.06, 0.15, 0.30),
            (0.04, 0.12, 0.25, 0.45),
            (0.05, 0.12, 0.22, 0.40),
        ]
    )
    subs = RNG.choice([(1, 2, 4), (1, 3, 5), (2, 3, 4), (1, 2, 3)])
    init = RNG.choice([0.05, 0.08, 0.10, 0.12, 0.15, 0.20])
    tp = RNG.choice([0.008, 0.01, 0.012, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05])
    sl = RNG.choice([0.0, 0.15, 0.20, 0.25, 0.28, 0.30, 0.35, 0.40, 0.50, 0.60])
    ref = RNG.choice(["wap", "p0"])
    mh = RNG.choice([0, 7 * 1440, 14 * 1440, 30 * 1440, 45 * 1440])
    be = RNG.choice([0, 48 * 60, 72 * 60, 120 * 60, 168 * 60])
    return StrategyParams(
        initial_pct=init,
        dca_level_weights=lvl,
        sub_order_weights=subs,
        dca_depths=depths,
        take_profit_pct=tp,
        stop_loss_pct=sl,
        stop_loss_ref=ref,
        max_hold_bars=mh,
        breakeven_after_bars=be,
        breakeven_buffer_pct=0.001,
    )


def fmt_crash(name: str, ev: dict) -> str:
    r = ev["res"]
    return (
        f"{name}: flat={ev['flat_mo']:+5.2f}%/mo ret={r.total_return_pct:+7.1f}% "
        f"dd={r.max_drawdown_pct:5.1f}% mo>=5%cap={ev['monthly']['pct_months_ge_5_capital']:5.1f}%"
    )


def params_key(p: StrategyParams) -> tuple:
    return (
        p.initial_pct,
        p.take_profit_pct,
        p.stop_loss_pct,
        p.stop_loss_ref,
        p.max_hold_bars,
        p.breakeven_after_bars,
        p.dca_depths,
        p.dca_level_weights,
        p.sub_order_weights,
    )


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_df()
    print(f"1m bars={len(df):,}  {df['datetime'].iloc[0]} .. {df['datetime'].iloc[-1]}", flush=True)

    auto = detect_crash_windows(df)
    print("Auto-detected crash windows (info):", flush=True)
    for w in auto:
        print(f"  {w['name']} dd={w['dd']*100:.1f}% days={w['days']}", flush=True)

    # Use major non-overlapping crash legs only (user goal: all major crashes).
    # Auto legs are logged for transparency but not mixed in (they overlap known bears).
    crash_list = list(KNOWN_CRASHES)
    print("Using crash set (major only):", flush=True)
    slices: dict[str, pd.DataFrame] = {}
    for c in crash_list:
        sl = slice_df(df, c["start"], c["end"])
        slices[c["name"]] = sl
        print(
            f"  {c['name']} {c['start']}..{c['end']} bars={len(sl):,} months~{n_months(c['start'], c['end'])}",
            flush=True,
        )

    history: list[dict] = []
    best = None
    seen: set[tuple] = set()

    def evaluate(tag: str, p: StrategyParams) -> float | None:
        nonlocal best
        key = params_key(p)
        if key in seen:
            return None
        seen.add(key)
        crash_evals = {}
        for c in crash_list:
            crash_evals[c["name"]] = eval_window(slices[c["name"]], p, c["start"], c["end"])
        sc = score_candidate(crash_evals)
        min_flat = min(v["flat_mo"] for v in crash_evals.values())
        hits = sum(1 for v in crash_evals.values() if v["hit_flat5"])
        row = {
            "tag": tag,
            "score": sc,
            "min_flat_mo": min_flat,
            "hits": hits,
            "n_crashes": len(crash_evals),
            "all_hit": hits == len(crash_evals),
            "params": p.to_dict(),
            "crashes": {
                k: {
                    "flat_mo": v["flat_mo"],
                    "ret": v["res"].total_return_pct,
                    "dd": v["res"].max_drawdown_pct,
                    "months": v["months"],
                    "pct_months_ge_5_capital": v["monthly"]["pct_months_ge_5_capital"],
                    "min_month_vs_capital": v["monthly"]["min_month_vs_capital"],
                }
                for k, v in crash_evals.items()
            },
        }
        history.append(row)
        marker = ""
        if best is None or sc > best["score"]:
            best = {
                "score": sc,
                "params": p,
                "crash_evals": crash_evals,
                "row": row,
            }
            marker = " <-- BEST"
        print(
            f"[{tag}] score={sc:+8.1f} min_flat={min_flat:+5.2f}% hits={hits}/{len(crash_evals)}{marker}",
            flush=True,
        )
        for name, ev in crash_evals.items():
            print(f"    {fmt_crash(name, ev)}", flush=True)
        return sc

    # Seeds: prior good configs from project history
    print("\n### seeds", flush=True)
    seeds = [
        StrategyParams(),  # original
        StrategyParams(
            initial_pct=0.08,
            dca_depths=(0.03, 0.08, 0.18, 0.35),
            take_profit_pct=0.025,
            stop_loss_pct=0.28,
            stop_loss_ref="wap",
        ),
        StrategyParams(
            initial_pct=0.08,
            dca_depths=(0.03, 0.08, 0.18, 0.35),
            take_profit_pct=0.03,
        ),
        StrategyParams(
            initial_pct=0.05,
            dca_level_weights=(8, 4, 2, 1),
            sub_order_weights=(2, 3, 4),
            dca_depths=(0.02, 0.08, 0.2, 0.4),
            take_profit_pct=0.012,
            stop_loss_pct=0.05,
            stop_loss_ref="p0",
            max_hold_bars=72 * 60,
        ),
        StrategyParams(
            initial_pct=0.08,
            dca_depths=(0.03, 0.08, 0.18, 0.35),
            take_profit_pct=0.02,
            stop_loss_pct=0.35,
            stop_loss_ref="wap",
        ),
        StrategyParams(
            initial_pct=0.10,
            dca_level_weights=(8, 4, 2, 1),
            dca_depths=(0.02, 0.08, 0.2, 0.4),
            take_profit_pct=0.015,
            stop_loss_pct=0.40,
            stop_loss_ref="wap",
        ),
    ]
    for i, p in enumerate(seeds):
        evaluate(f"seed{i}", p)

    # Random search
    N_RANDOM = 36
    print(f"\n### random search n={N_RANDOM}", flush=True)
    for i in range(N_RANDOM):
        evaluate(f"rand{i}", sample_params())

    # Coordinate descent from best
    print("\n### coordinate descent from best", flush=True)
    assert best is not None
    cur = best["params"]
    axes = [
        ("take_profit_pct", [0.01, 0.012, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]),
        ("stop_loss_pct", [0.0, 0.2, 0.25, 0.28, 0.3, 0.35, 0.4, 0.5, 0.6]),
        ("stop_loss_ref", ["wap", "p0"]),
        ("initial_pct", [0.05, 0.08, 0.1, 0.12, 0.15, 0.2]),
        (
            "dca_depths",
            [
                (0.03, 0.1, 0.25, 0.5),
                (0.03, 0.08, 0.18, 0.35),
                (0.02, 0.08, 0.2, 0.4),
                (0.02, 0.06, 0.15, 0.3),
                (0.05, 0.12, 0.22, 0.4),
            ],
        ),
        ("dca_level_weights", [(1, 2, 4, 8), (8, 4, 2, 1), (1, 2, 3, 6), (2, 3, 4, 8)]),
        ("sub_order_weights", [(1, 2, 4), (1, 3, 5), (2, 3, 4)]),
        ("max_hold_bars", [0, 7 * 1440, 14 * 1440, 30 * 1440]),
        ("breakeven_after_bars", [0, 72 * 60, 120 * 60, 168 * 60]),
    ]
    for name, values in axes:
        print(f"  axis {name}", flush=True)
        local_best_p = cur
        local_best_sc = best["score"]
        for val in values:
            p = replace(cur, **{name: val})
            sc = evaluate(f"cd_{name}={val}", p)
            if sc is not None and sc > local_best_sc:
                local_best_sc = sc
                local_best_p = p
        cur = local_best_p

    # Full-history validation for top candidates by min_flat then score
    print("\n### full-history validation (top candidates)", flush=True)
    ranked = sorted(history, key=lambda r: (r["min_flat_mo"], r["score"]), reverse=True)
    # unique top
    top: list[dict] = []
    top_keys = set()
    for r in ranked:
        k = json.dumps(r["params"], sort_keys=True)
        if k in top_keys:
            continue
        top_keys.add(k)
        top.append(r)
        if len(top) >= 5:
            break

    full_results = []
    for i, r in enumerate(top):
        p = StrategyParams(
            initial_pct=r["params"]["initial_pct"],
            dca_level_weights=tuple(r["params"]["dca_level_weights"]),
            sub_order_weights=tuple(r["params"]["sub_order_weights"]),
            dca_depths=tuple(r["params"]["dca_depths"]),
            take_profit_pct=r["params"]["take_profit_pct"],
            fee_rate=r["params"].get("fee_rate", 0.001),
            capital=r["params"].get("capital", 10_000.0),
            reentry_delay_bars=r["params"].get("reentry_delay_bars", 0),
            stop_loss_pct=r["params"].get("stop_loss_pct", 0.0),
            stop_loss_ref=r["params"].get("stop_loss_ref", "wap"),
            max_hold_bars=r["params"].get("max_hold_bars", 0),
            breakeven_after_bars=r["params"].get("breakeven_after_bars", 0),
            breakeven_buffer_pct=r["params"].get("breakeven_buffer_pct", 0.001),
        )
        print(f"\n--- full validate top#{i+1} min_flat={r['min_flat_mo']:.2f} ---", flush=True)
        fres = run_backtest(df, p)
        print_report(fres, f"FULL HISTORY top#{i+1}")
        fmonths = n_months(str(df["datetime"].iloc[0].date()), str(df["datetime"].iloc[-1].date()))
        fflat = fres.total_return_pct / fmonths
        fms = monthly_stats(fres, p.capital)
        print(
            f"FULL flat_mo={fflat:+.2f}%/mo  months_ge_5%cap={fms['pct_months_ge_5_capital']:.1f}%  "
            f"min_month_vs_cap={fms['min_month_vs_capital']:+.2f}%",
            flush=True,
        )
        full_results.append(
            {
                "rank": i + 1,
                "crash_min_flat_mo": r["min_flat_mo"],
                "crash_hits": r["hits"],
                "crash_all_hit": r["all_hit"],
                "crash_score": r["score"],
                "params": p.to_dict(),
                "full": {
                    "ret": fres.total_return_pct,
                    "dd": fres.max_drawdown_pct,
                    "flat_mo": fflat,
                    "months": fmonths,
                    "pct_months_ge_5_capital": fms["pct_months_ge_5_capital"],
                    "min_month_vs_capital": fms["min_month_vs_capital"],
                    "avg_month_vs_capital": fms["avg_month_vs_capital"],
                    "cycles": fres.num_cycles,
                    "avg_hold_hours": fres.avg_hold_bars / 60.0,
                },
                "crash_detail": r["crashes"],
                "equity": fres,
                "params_obj": p,
                "monthly_full": fms,
            }
        )

    # Select: prefer all_hit on crashes; else highest min_flat; tie-break full flat
    selected = None
    for fr in full_results:
        if fr["crash_all_hit"]:
            selected = fr
            break
    if selected is None:
        selected = max(full_results, key=lambda x: (x["crash_min_flat_mo"], x["full"]["flat_mo"]))

    # Save
    hist_df = pd.DataFrame(
        [
            {
                "tag": h["tag"],
                "score": h["score"],
                "min_flat_mo": h["min_flat_mo"],
                "hits": h["hits"],
                "all_hit": h["all_hit"],
                **{f"param_{k}": v for k, v in h["params"].items()},
                **{
                    f"{ck}_flat": cv["flat_mo"]
                    for ck, cv in h["crashes"].items()
                },
            }
            for h in history
        ]
    )
    hist_df.to_csv(OUT / "optimization_history.csv", index=False)

    # monthly detail for selected
    pd.DataFrame(selected["monthly_full"]["month_rows"]).to_csv(
        OUT / "selected_monthly_full.csv", index=False
    )

    summary = {
        "target": {
            "flat_monthly_pct": TARGET_FLAT_MO,
            "definition": "flat_mo = total_return_pct / n_months (non-compounded vs initial capital)",
            "also_tracked": "calendar month pnl / capital0 >= 5%",
        },
        "data": {
            "bars": len(df),
            "start": str(df["datetime"].iloc[0]),
            "end": str(df["datetime"].iloc[-1]),
        },
        "crashes_used": crash_list,
        "target_met_all_crashes": bool(selected["crash_all_hit"]),
        "selected": {
            "params": selected["params"],
            "crash_min_flat_mo": selected["crash_min_flat_mo"],
            "crash_hits": selected["crash_hits"],
            "crash_all_hit": selected["crash_all_hit"],
            "crash_detail": selected["crash_detail"],
            "full": selected["full"],
        },
        "top_validated": [
            {k: v for k, v in fr.items() if k not in ("equity", "params_obj", "monthly_full")}
            for fr in full_results
        ],
        "elapsed_s": time.time() - t0,
        "trials": len(history),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    eq = selected["equity"].equity_curve
    if eq is not None:
        eq.resample("1h").last().dropna().to_csv(OUT / "equity_selected_1h_sample.csv")

    try:
        fig, ax = plt.subplots(figsize=(11, 5))
        eqh = eq.resample("1D").last().dropna()
        ax.plot(eqh.index, eqh.values, linewidth=1.2, label="selected")
        ax.set_title("Crash-monthly search — selected on full 1m (daily equity)")
        ax.set_ylabel("Equity")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "equity_selected.png", dpi=120)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"plot skipped: {exc}", flush=True)

    print("\n===== SELECTED =====", flush=True)
    print_report(selected["equity"], "SELECTED (crash-monthly objective)")
    print(
        f"crash min flat={selected['crash_min_flat_mo']:.2f}%/mo  "
        f"all_hit={selected['crash_all_hit']}  "
        f"full flat={selected['full']['flat_mo']:.2f}%/mo  "
        f"full ret={selected['full']['ret']:+.1f}%",
        flush=True,
    )
    print("SELECTED PARAMS:", json.dumps(selected["params"], indent=2), flush=True)
    print(f"Saved -> {OUT} elapsed={time.time()-t0:.1f}s trials={len(history)}", flush=True)

    if not selected["crash_all_hit"]:
        print(
            "\nNOTE: No config reached >=5% flat monthly on ALL crash windows. "
            "Selected is the closest by min_flat across crashes.",
            flush=True,
        )


if __name__ == "__main__":
    main()
