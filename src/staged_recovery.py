"""Staged Recovery DCA strategy.

Normal mode (after wins / start):
  - Buy initial_pct of equity immediately at P0
  - Buy dca_pct at P0*(1-dca_drop)
  - Exit: TP at WAP*(1+tp) OR SL at ref*(1-sl)
  - Unused equity stays cash

Recovery mode (after a losing cycle):
  - Larger initial + two DCA legs + wider SL
  - On win -> back to Normal; on loss -> stay in Recovery
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class StagedParams:
    # --- Normal regime ---
    normal_initial_pct: float = 0.25
    normal_dca_pct: float = 0.25
    normal_dca_drop: float = 0.03
    normal_sl_pct: float = 0.03
    normal_tp_pct: float = 0.01

    # --- Recovery regime (after a loss) ---
    recovery_initial_pct: float = 0.50
    recovery_dca1_pct: float = 0.25
    recovery_dca1_drop: float = 0.03
    recovery_dca2_pct: float = 0.25
    recovery_dca2_drop: float = 0.05
    recovery_sl_pct: float = 0.10
    recovery_tp_pct: float = 0.01

    sl_ref: str = "p0"  # "p0" or "wap"
    fee_rate: float = 0.001
    capital: float = 10_000.0
    # Optional: don't start recovery if equity below this fraction of peak
    # (0 disables)
    min_equity_frac_of_peak: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def ladder_for(self, mode: str) -> list[dict[str, float]]:
        if mode == "normal":
            return [
                {"drop": 0.0, "alloc": self.normal_initial_pct, "kind": "n_init"},
                {
                    "drop": self.normal_dca_drop,
                    "alloc": self.normal_dca_pct,
                    "kind": "n_dca1",
                },
            ]
        if mode == "recovery":
            return [
                {"drop": 0.0, "alloc": self.recovery_initial_pct, "kind": "r_init"},
                {
                    "drop": self.recovery_dca1_drop,
                    "alloc": self.recovery_dca1_pct,
                    "kind": "r_dca1",
                },
                {
                    "drop": self.recovery_dca2_drop,
                    "alloc": self.recovery_dca2_pct,
                    "kind": "r_dca2",
                },
            ]
        raise ValueError(mode)

    def tp_for(self, mode: str) -> float:
        return self.normal_tp_pct if mode == "normal" else self.recovery_tp_pct

    def sl_for(self, mode: str) -> float:
        return self.normal_sl_pct if mode == "normal" else self.recovery_sl_pct


@dataclass
class StagedCycle:
    mode: str
    entry_time: Any
    exit_time: Any = None
    p0: float = 0.0
    avg_entry: float = 0.0
    exit_price: float = 0.0
    budget: float = 0.0
    spent: float = 0.0
    proceeds: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fills: int = 0
    hold_bars: int = 0
    reason: str = ""


@dataclass
class StagedResult:
    params: StagedParams
    equity_final: float
    total_return_pct: float
    buy_hold_return_pct: float
    num_cycles: int
    win_rate: float
    avg_pnl_pct: float
    max_drawdown_pct: float
    profit_factor: float
    total_fees: float
    avg_hold_bars: float
    median_hold_bars: float
    max_hold_bars_seen: float
    normal_cycles: int
    recovery_cycles: int
    normal_wins: int
    recovery_wins: int
    tp_exits: int
    sl_exits: int
    cycles: list[StagedCycle] = field(default_factory=list)
    equity_curve: pd.Series | None = None

    def summary(self) -> dict[str, Any]:
        return {
            **{f"param_{k}": v for k, v in self.params.to_dict().items()},
            "equity_final": self.equity_final,
            "total_return_pct": self.total_return_pct,
            "buy_hold_return_pct": self.buy_hold_return_pct,
            "num_cycles": self.num_cycles,
            "win_rate": self.win_rate,
            "avg_pnl_pct": self.avg_pnl_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "profit_factor": self.profit_factor,
            "total_fees": self.total_fees,
            "avg_hold_bars": self.avg_hold_bars,
            "median_hold_bars": self.median_hold_bars,
            "max_hold_bars_seen": self.max_hold_bars_seen,
            "normal_cycles": self.normal_cycles,
            "recovery_cycles": self.recovery_cycles,
            "normal_wins": self.normal_wins,
            "recovery_wins": self.recovery_wins,
            "tp_exits": self.tp_exits,
            "sl_exits": self.sl_exits,
        }


def run_staged(df: pd.DataFrame, params: StagedParams | None = None) -> StagedResult:
    params = params or StagedParams()
    if params.sl_ref not in ("p0", "wap"):
        raise ValueError("sl_ref must be p0 or wap")

    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    times = df["datetime"].to_numpy()
    n = len(df)
    fee = params.fee_rate

    cash = float(params.capital)
    total_fees = 0.0
    cycles: list[StagedCycle] = []
    equity_hist = np.empty(n, dtype=float)

    mode = "normal"
    in_cycle = False
    p0 = 0.0
    qty = 0.0
    spent_for_avg = 0.0
    cost_quote = 0.0
    equity_budget = 0.0  # equity snapshot at cycle start
    filled: list[bool] = []
    ladder: list[dict[str, float]] = []
    cycle: StagedCycle | None = None
    cycle_start_i = 0
    peak_equity = cash
    max_dd = 0.0

    def wap() -> float:
        return spent_for_avg / qty if qty > 0 else 0.0

    def mark(i: int) -> float:
        return cash + qty * closes[i]

    def close_cycle(i: int, exit_px: float, reason: str) -> None:
        nonlocal cash, total_fees, qty, spent_for_avg, cost_quote, in_cycle, cycle, mode
        assert cycle is not None
        gross = qty * exit_px
        fee_paid = gross * fee
        net = gross - fee_paid
        cash += net
        total_fees += fee_paid
        cycle.exit_time = times[i]
        cycle.exit_price = exit_px
        cycle.spent = cost_quote
        cycle.proceeds = net
        cycle.pnl = net - cost_quote
        cycle.pnl_pct = cycle.pnl / cost_quote if cost_quote else 0.0
        cycle.avg_entry = wap()
        cycle.hold_bars = i - cycle_start_i + 1
        cycle.reason = reason
        cycles.append(cycle)

        lost = cycle.pnl < 0 or reason == "stop_loss"
        mode = "recovery" if lost else "normal"

        qty = 0.0
        spent_for_avg = 0.0
        cost_quote = 0.0
        in_cycle = False
        cycle = None

    i = 0
    while i < n:
        eq = mark(i)
        peak_equity = max(peak_equity, eq)

        if not in_cycle:
            if cash <= 1e-8:
                equity_hist[i] = eq
                if peak_equity > 0:
                    max_dd = max(max_dd, (peak_equity - eq) / peak_equity)
                i += 1
                continue

            # Optional: refuse recovery sizing if too deep underwater vs peak
            use_mode = mode
            if (
                mode == "recovery"
                and params.min_equity_frac_of_peak > 0
                and peak_equity > 0
                and cash / peak_equity < params.min_equity_frac_of_peak
            ):
                use_mode = "normal"

            ladder = params.ladder_for(use_mode)
            # Budget = current cash (flat). Allocations are fractions of this equity.
            equity_budget = cash
            # Guard: total alloc cannot exceed 1
            total_alloc = sum(x["alloc"] for x in ladder)
            if total_alloc > 1.0 + 1e-9:
                raise ValueError(f"ladder alloc sum {total_alloc} > 1 for mode {use_mode}")

            p0 = opens[i]
            filled = [False] * len(ladder)
            qty = 0.0
            spent_for_avg = 0.0
            cost_quote = 0.0
            in_cycle = True
            cycle_start_i = i
            cycle = StagedCycle(
                mode=use_mode, entry_time=times[i], p0=p0, budget=equity_budget
            )

            # Immediate initial buy
            init = ladder[0]
            alloc_quote = min(equity_budget * init["alloc"], cash)
            fee_paid = alloc_quote * fee
            buy_quote = alloc_quote - fee_paid
            buy_qty = buy_quote / p0 if p0 > 0 else 0.0
            cash -= alloc_quote
            total_fees += fee_paid
            qty += buy_qty
            spent_for_avg += buy_quote
            cost_quote += alloc_quote
            filled[0] = True
            cycle.fills = 1
            cycle.avg_entry = wap()
            mode = use_mode  # lock displayed mode

        assert cycle is not None

        # Fill pending DCA
        for j in range(1, len(ladder)):
            if filled[j]:
                continue
            limit_px = p0 * (1.0 - ladder[j]["drop"])
            if lows[i] <= limit_px:
                alloc_quote = equity_budget * ladder[j]["alloc"]
                if alloc_quote > cash:
                    alloc_quote = cash
                if alloc_quote <= 1e-12:
                    filled[j] = True
                    continue
                fee_paid = alloc_quote * fee
                buy_quote = alloc_quote - fee_paid
                buy_qty = buy_quote / limit_px
                cash -= alloc_quote
                total_fees += fee_paid
                qty += buy_qty
                spent_for_avg += buy_quote
                cost_quote += alloc_quote
                filled[j] = True
                cycle.fills += 1
                cycle.avg_entry = wap()

        if qty > 0:
            avg = wap()
            tp = avg * (1.0 + params.tp_for(cycle.mode))
            sl_ref_px = avg if params.sl_ref == "wap" else p0
            sl = sl_ref_px * (1.0 - params.sl_for(cycle.mode))
            hit_tp = highs[i] >= tp
            hit_sl = lows[i] <= sl
            # Adverse: SL first if both possible
            if hit_sl:
                close_cycle(i, sl, "stop_loss")
            elif hit_tp:
                close_cycle(i, tp, "take_profit")

        eq = mark(i)
        equity_hist[i] = eq
        peak_equity = max(peak_equity, eq)
        if peak_equity > 0:
            max_dd = max(max_dd, (peak_equity - eq) / peak_equity)
        i += 1

    if in_cycle and qty > 0 and cycle is not None:
        close_cycle(n - 1, closes[-1], "eod_close")

    equity_final = cash
    holds = [float(c.hold_bars) for c in cycles]
    wins = [c for c in cycles if c.pnl > 0]
    normal_c = [c for c in cycles if c.mode == "normal"]
    recovery_c = [c for c in cycles if c.mode == "recovery"]
    gp = sum(c.pnl for c in cycles if c.pnl > 0)
    gl = abs(sum(c.pnl for c in cycles if c.pnl < 0))
    pf = (gp / gl) if gl > 1e-12 else 999.0

    return StagedResult(
        params=params,
        equity_final=equity_final,
        total_return_pct=(equity_final / params.capital - 1.0) * 100.0,
        buy_hold_return_pct=(closes[-1] / closes[0] - 1.0) * 100.0,
        num_cycles=len(cycles),
        win_rate=(len(wins) / len(cycles) * 100.0) if cycles else 0.0,
        avg_pnl_pct=float(np.mean([c.pnl_pct for c in cycles]) * 100.0) if cycles else 0.0,
        max_drawdown_pct=max_dd * 100.0,
        profit_factor=pf,
        total_fees=total_fees,
        avg_hold_bars=float(np.mean(holds)) if holds else 0.0,
        median_hold_bars=float(np.median(holds)) if holds else 0.0,
        max_hold_bars_seen=float(np.max(holds)) if holds else 0.0,
        normal_cycles=len(normal_c),
        recovery_cycles=len(recovery_c),
        normal_wins=sum(1 for c in normal_c if c.pnl > 0),
        recovery_wins=sum(1 for c in recovery_c if c.pnl > 0),
        tp_exits=sum(1 for c in cycles if c.reason == "take_profit"),
        sl_exits=sum(1 for c in cycles if c.reason == "stop_loss"),
        cycles=cycles,
        equity_curve=pd.Series(equity_hist, index=pd.DatetimeIndex(df["datetime"]), name="equity"),
    )


def print_staged(res: StagedResult, title: str = "Staged Recovery") -> None:
    p = res.params
    print("=" * 64)
    print(title)
    print("=" * 64)
    print(f"Final equity        : {res.equity_final:,.2f}")
    print(f"Strategy return     : {res.total_return_pct:+.2f}%")
    print(f"Buy & hold          : {res.buy_hold_return_pct:+.2f}%")
    print(f"Cycles              : {res.num_cycles}  (N={res.normal_cycles} R={res.recovery_cycles})")
    print(f"Win rate            : {res.win_rate:.1f}%  (Nw={res.normal_wins} Rw={res.recovery_wins})")
    print(f"Max drawdown        : {res.max_drawdown_pct:.2f}%")
    print(f"Profit factor       : {res.profit_factor:.3f}")
    print(f"Fees                : {res.total_fees:,.2f}")
    print(
        f"Hold bars avg/med/max: {res.avg_hold_bars:.1f} / {res.median_hold_bars:.1f} / {res.max_hold_bars_seen:.0f}"
    )
    print(f"Exits TP/SL         : {res.tp_exits} / {res.sl_exits}")
    print(
        f"Normal: init={p.normal_initial_pct:.0%} dca={p.normal_dca_pct:.0%}@{p.normal_dca_drop:.0%} "
        f"SL={p.normal_sl_pct:.0%} TP={p.normal_tp_pct:.0%} ref={p.sl_ref}"
    )
    print(
        f"Recovery: init={p.recovery_initial_pct:.0%} "
        f"dca={p.recovery_dca1_pct:.0%}@{p.recovery_dca1_drop:.0%} + "
        f"{p.recovery_dca2_pct:.0%}@{p.recovery_dca2_drop:.0%} "
        f"SL={p.recovery_sl_pct:.0%} TP={p.recovery_tp_pct:.0%}"
    )
