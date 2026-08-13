"""One-parameter-at-a-time optimization for DCA Pyramid."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from strategy import BacktestResult, StrategyParams, print_report, run_backtest


def score(result: BacktestResult) -> float:
    """
    Primary objective: total return, with mild penalties for severe drawdown.
    Higher is better.
    """
    return result.total_return_pct - 0.15 * max(0.0, result.max_drawdown_pct - 20.0)


def apply_override(base: StrategyParams, name: str, value: Any) -> StrategyParams:
    p = deepcopy(base)
    if name == "initial_pct":
        p.initial_pct = float(value)
    elif name == "take_profit_pct":
        p.take_profit_pct = float(value)
    elif name == "fee_rate":
        p.fee_rate = float(value)
    elif name == "dca_depths":
        p.dca_depths = tuple(float(x) for x in value)
    elif name == "dca_level_weights":
        p.dca_level_weights = tuple(float(x) for x in value)
    elif name == "sub_order_weights":
        p.sub_order_weights = tuple(float(x) for x in value)
    elif name == "reentry_delay_bars":
        p.reentry_delay_bars = int(value)
    else:
        raise KeyError(name)
    return p


# Candidate grids — one axis at a time from the original baseline.
SEARCH_SPACE: list[tuple[str, list[Any]]] = [
    (
        "take_profit_pct",
        [0.005, 0.008, 0.01, 0.012, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05],
    ),
    (
        "initial_pct",
        [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30],
    ),
    (
        "dca_depths",
        [
            (0.02, 0.06, 0.15, 0.30),
            (0.03, 0.10, 0.25, 0.50),  # original
            (0.04, 0.12, 0.28, 0.55),
            (0.05, 0.15, 0.30, 0.60),
            (0.02, 0.08, 0.20, 0.40),
            (0.03, 0.08, 0.18, 0.35),
            (0.05, 0.12, 0.25, 0.45),
            (0.03, 0.10, 0.20, 0.40),
            (0.04, 0.10, 0.22, 0.45),
        ],
    ),
    (
        "dca_level_weights",
        [
            (1, 2, 4, 8),  # original pyramid
            (1, 1, 1, 1),
            (1, 2, 3, 4),
            (8, 4, 2, 1),  # inverted
            (1, 3, 5, 7),
            (2, 3, 4, 5),
            (1, 2, 4, 6),
            (1, 2, 3, 6),
        ],
    ),
    (
        "sub_order_weights",
        [
            (1, 2, 4),  # original
            (1, 1, 1),
            (4, 2, 1),
            (1, 2, 3),
            (1, 3, 5),
            (2, 3, 4),
        ],
    ),
    (
        "reentry_delay_bars",
        [0, 1, 3, 6, 12, 24],
    ),
]


def optimize_sequential(
    df: pd.DataFrame,
    base: StrategyParams | None = None,
    space: list[tuple[str, list[Any]]] | None = None,
) -> tuple[StrategyParams, list[dict[str, Any]], BacktestResult]:
    """
    Coordinate descent: for each parameter, try candidates while freezing others,
    keep the best value, then move to the next parameter.
    """
    base = base or StrategyParams()
    space = space or SEARCH_SPACE
    current = deepcopy(base)
    history: list[dict[str, Any]] = []

    baseline = run_backtest(df, current)
    history.append(
        {
            "stage": "baseline",
            "param": None,
            "value": None,
            "score": score(baseline),
            **baseline.summary(),
        }
    )
    print_report(baseline, "BASELINE (original settings)")

    for param_name, candidates in space:
        print("\n" + "#" * 64)
        print(f"Optimizing parameter: {param_name}")
        print("#" * 64)
        best_local = current
        best_result = run_backtest(df, current)
        best_score = score(best_result)
        values = list(candidates)
        cur_val = getattr(current, param_name)
        if isinstance(cur_val, tuple):
            if cur_val not in values:
                values.insert(0, cur_val)
        elif cur_val not in values:
            values.insert(0, cur_val)
        seen: set[str] = set()
        for val in values:
            key = repr(val)
            if key in seen:
                continue
            seen.add(key)
            trial = apply_override(current, param_name, val)
            try:
                res = run_backtest(df, trial)
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {param_name}={val}: {exc}")
                continue
            sc = score(res)
            row = {
                "stage": f"search:{param_name}",
                "param": param_name,
                "value": val if not isinstance(val, tuple) else list(val),
                "score": sc,
                **res.summary(),
            }
            history.append(row)
            marker = ""
            if sc > best_score:
                best_score = sc
                best_local = trial
                best_result = res
                marker = " <-- new best"
            print(
                f"  {param_name}={val!r:40s}  ret={res.total_return_pct:+8.2f}%  "
                f"dd={res.max_drawdown_pct:6.2f}%  cycles={res.num_cycles:5d}  "
                f"score={sc:+8.2f}{marker}"
            )

        if best_local is not current:
            print(f"  => keep {param_name} = {getattr(best_local, param_name)!r}")
            current = best_local
        else:
            print(f"  => keep original {param_name} = {getattr(current, param_name)!r}")

    final = run_backtest(df, current)
    history.append(
        {
            "stage": "final",
            "param": None,
            "value": None,
            "score": score(final),
            **final.summary(),
        }
    )
    print_report(final, "FINAL (after sequential optimization)")
    return current, history, final


def save_results(
    out_dir: Path,
    baseline: BacktestResult,
    final: BacktestResult,
    history: list[dict[str, Any]],
    best_params: StrategyParams,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out_dir / "optimization_history.csv", index=False)

    summary = {
        "baseline": baseline.summary(),
        "final": final.summary(),
        "best_params": best_params.to_dict(),
        "improvement_return_pp": final.total_return_pct - baseline.total_return_pct,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Cycle logs
    for name, res in ("baseline", baseline), ("final", final):
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
                    "max_dd_from_p0": c.max_dd_from_p0,
                    "reason": c.reason,
                }
            )
        pd.DataFrame(rows).to_csv(out_dir / f"cycles_{name}.csv", index=False)

    if final.equity_curve is not None:
        final.equity_curve.to_csv(out_dir / "equity_curve_final.csv", header=True)
    if baseline.equity_curve is not None:
        baseline.equity_curve.to_csv(out_dir / "equity_curve_baseline.csv", header=True)

    print(f"Saved results -> {out_dir}")
