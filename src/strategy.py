"""DCA Pyramid Spot strategy — core engine and backtester."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd


def _ratio_parts(weights: list[float]) -> list[float]:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must sum > 0")
    return [w / total for w in weights]


def build_ladder(
    initial_pct: float,
    dca_level_weights: list[float],
    sub_order_weights: list[float],
    dca_depths: list[float],
) -> list[dict[str, float]]:
    """
    Build buy ladder relative to entry P0.

    initial_pct: fraction of capital for immediate market buy (e.g. 0.05).
    dca_level_weights: e.g. [1,2,4,8] across DCA levels (remaining capital).
    sub_order_weights: e.g. [1,2,4] inside each DCA level.
    dca_depths: max drawdown fraction per DCA level, e.g. [0.03, 0.10, 0.25, 0.50].
                 Sub-order drops are spaced evenly to that depth (1/3, 2/3, 3/3)
                 matching the original -1/-2/-3, -3.33/-6.67/-10, etc.
    """
    if len(dca_level_weights) != len(dca_depths):
        raise ValueError("dca_level_weights and dca_depths length mismatch")
    if not (0 < initial_pct < 1):
        raise ValueError("initial_pct must be in (0,1)")

    remaining = 1.0 - initial_pct
    level_fracs = _ratio_parts(dca_level_weights)
    sub_fracs = _ratio_parts(sub_order_weights)
    n_sub = len(sub_order_weights)

    ladder: list[dict[str, float]] = [
        {"drop": 0.0, "alloc": initial_pct, "kind": "initial"}
    ]

    for level_i, (level_w, depth) in enumerate(zip(level_fracs, dca_depths)):
        level_alloc = remaining * level_w
        for sub_i, sub_w in enumerate(sub_fracs):
            # Even spacing to depth: (1/n)*depth, (2/n)*depth, ...
            drop = depth * (sub_i + 1) / n_sub
            ladder.append(
                {
                    "drop": drop,
                    "alloc": level_alloc * sub_w,
                    "kind": f"dca{level_i + 1}_s{sub_i + 1}",
                }
            )

    # Numerical safety: renormalize tiny float drift.
    s = sum(x["alloc"] for x in ladder)
    for x in ladder:
        x["alloc"] *= 1.0 / s
    return ladder


@dataclass
class StrategyParams:
    initial_pct: float = 0.05
    dca_level_weights: tuple[float, ...] = (1, 2, 4, 8)
    sub_order_weights: tuple[float, ...] = (1, 2, 4)
    dca_depths: tuple[float, ...] = (0.03, 0.10, 0.25, 0.50)
    take_profit_pct: float = 0.01  # sell when price >= WAP * (1 + tp)
    fee_rate: float = 0.001  # Binance spot taker-ish default 0.1%
    capital: float = 10_000.0
    # Re-entry: after TP, wait this many bars before starting a new cycle at market.
    reentry_delay_bars: int = 0

    def ladder(self) -> list[dict[str, float]]:
        return build_ladder(
            self.initial_pct,
            list(self.dca_level_weights),
            list(self.sub_order_weights),
            list(self.dca_depths),
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dca_level_weights"] = list(self.dca_level_weights)
        d["sub_order_weights"] = list(self.sub_order_weights)
        d["dca_depths"] = list(self.dca_depths)
        return d


@dataclass
class TradeCycle:
    entry_time: Any
    exit_time: Any = None
    p0: float = 0.0
    avg_entry: float = 0.0
    qty: float = 0.0
    cost_quote: float = 0.0  # quote spent incl. fees on buys
    exit_price: float = 0.0
    proceeds: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fills: int = 0
    max_dd_from_p0: float = 0.0
    reason: str = ""


@dataclass
class BacktestResult:
    params: StrategyParams
    equity_final: float
    total_return_pct: float
    buy_hold_return_pct: float
    num_cycles: int
    win_rate: float
    avg_pnl_pct: float
    max_drawdown_pct: float
    profit_factor: float
    total_fees: float
    cycles: list[TradeCycle] = field(default_factory=list)
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
        }


def run_backtest(df: pd.DataFrame, params: StrategyParams | None = None) -> BacktestResult:
    """
    Event-driven spot backtest on OHLC bars.

    Cycle logic:
      1. At cycle start, market-buy initial allocation at open (or first close).
      2. Remaining ladder levels are limit buys at P0 * (1 - drop).
      3. A limit fills if bar low <= limit price (fill at limit).
      4. After every fill, update WAP and TP = WAP * (1 + take_profit_pct).
      5. If bar high >= TP, exit entire position at TP (conservative fill).
      6. After exit, start a new cycle (optionally after delay) with remaining cash.

    Capital is recycled cycle-to-cycle (compounding). Unfilled ladder cash stays idle
    until filled or cycle ends (on TP, unfilled cash remains cash).
    """
    params = params or StrategyParams()
    ladder = params.ladder()
    fee = params.fee_rate

    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    times = df["datetime"].to_numpy()
    n = len(df)

    cash = float(params.capital)
    total_fees = 0.0
    cycles: list[TradeCycle] = []
    equity_hist = np.empty(n, dtype=float)

    # Position state
    in_cycle = False
    p0 = 0.0
    qty = 0.0
    cost_quote = 0.0  # net quote paid for holdings (excl. fee accounted separately via cash)
    spent_for_avg = 0.0  # quote incl. for WAP (price * qty before fee)
    filled_flags: list[bool] = []
    cycle_cash_budget = 0.0
    cycle_cash_left = 0.0
    pending_reentry = 0
    cycle: TradeCycle | None = None
    peak_equity = cash
    max_dd = 0.0

    def wap() -> float:
        return spent_for_avg / qty if qty > 0 else 0.0

    def mark_equity(i: int) -> float:
        # Mark-to-market at close
        eq = cash + qty * float(df["close"].iloc[i])
        return eq

    i = 0
    while i < n:
        # Start new cycle
        if not in_cycle:
            if pending_reentry > 0:
                pending_reentry -= 1
                equity_hist[i] = mark_equity(i)
                peak_equity = max(peak_equity, equity_hist[i])
                max_dd = max(max_dd, (peak_equity - equity_hist[i]) / peak_equity if peak_equity else 0.0)
                i += 1
                continue

            if cash <= 1e-8:
                equity_hist[i] = mark_equity(i)
                i += 1
                continue

            # Begin cycle at this bar's open
            p0 = opens[i]
            cycle_cash_budget = cash
            cycle_cash_left = cash
            filled_flags = [False] * len(ladder)
            qty = 0.0
            cost_quote = 0.0
            spent_for_avg = 0.0
            in_cycle = True
            cycle = TradeCycle(entry_time=times[i], p0=p0)

            # Immediate initial buy at open
            init = ladder[0]
            alloc_quote = cycle_cash_budget * init["alloc"]
            if alloc_quote > cycle_cash_left:
                alloc_quote = cycle_cash_left
            fee_paid = alloc_quote * fee
            buy_quote = alloc_quote - fee_paid
            buy_qty = buy_quote / p0 if p0 > 0 else 0.0
            cash -= alloc_quote
            cycle_cash_left -= alloc_quote
            total_fees += fee_paid
            qty += buy_qty
            spent_for_avg += buy_quote  # effective base cost at fill price
            cost_quote += alloc_quote
            filled_flags[0] = True
            cycle.fills = 1
            cycle.avg_entry = wap()

        assert cycle is not None

        # Track drawdown from P0 within cycle
        bar_dd = (p0 - lows[i]) / p0 if p0 > 0 else 0.0
        cycle.max_dd_from_p0 = max(cycle.max_dd_from_p0, bar_dd)

        # Fill pending DCA limits (skip initial which is index 0)
        # Process in order of drop (already sorted)
        tp_price = wap() * (1.0 + params.take_profit_pct) if qty > 0 else 0.0

        for j in range(1, len(ladder)):
            if filled_flags[j]:
                continue
            limit_px = p0 * (1.0 - ladder[j]["drop"])
            if lows[i] <= limit_px:
                alloc_quote = cycle_cash_budget * ladder[j]["alloc"]
                if alloc_quote > cycle_cash_left + 1e-12:
                    alloc_quote = cycle_cash_left
                if alloc_quote <= 1e-12:
                    filled_flags[j] = True
                    continue
                fee_paid = alloc_quote * fee
                buy_quote = alloc_quote - fee_paid
                buy_qty = buy_quote / limit_px
                cash -= alloc_quote
                cycle_cash_left -= alloc_quote
                total_fees += fee_paid
                qty += buy_qty
                spent_for_avg += buy_quote
                cost_quote += alloc_quote
                filled_flags[j] = True
                cycle.fills += 1
                cycle.avg_entry = wap()
                tp_price = cycle.avg_entry * (1.0 + params.take_profit_pct)

        # Check take-profit against bar high (after buys on this bar)
        if qty > 0:
            tp_price = wap() * (1.0 + params.take_profit_pct)
            # Conservative: if both fill and TP possible same bar, assume buys first
            # then TP if high still reaches TP.
            if highs[i] >= tp_price:
                # Exit at TP
                exit_px = tp_price
                gross = qty * exit_px
                fee_paid = gross * fee
                net = gross - fee_paid
                cash += net
                total_fees += fee_paid
                cycle.exit_time = times[i]
                cycle.exit_price = exit_px
                cycle.qty = qty
                cycle.cost_quote = cost_quote
                cycle.proceeds = net
                cycle.pnl = net - cost_quote
                cycle.pnl_pct = cycle.pnl / cost_quote if cost_quote else 0.0
                cycle.avg_entry = wap()
                cycle.reason = "take_profit"
                cycles.append(cycle)

                qty = 0.0
                spent_for_avg = 0.0
                cost_quote = 0.0
                in_cycle = False
                cycle = None
                pending_reentry = params.reentry_delay_bars

        equity_hist[i] = cash + qty * float(df["close"].iloc[i])
        peak_equity = max(peak_equity, equity_hist[i])
        if peak_equity > 0:
            max_dd = max(max_dd, (peak_equity - equity_hist[i]) / peak_equity)
        i += 1

    # Force-close open cycle at last close
    if in_cycle and qty > 0 and cycle is not None:
        exit_px = float(df["close"].iloc[-1])
        gross = qty * exit_px
        fee_paid = gross * fee
        net = gross - fee_paid
        cash += net
        total_fees += fee_paid
        cycle.exit_time = times[-1]
        cycle.exit_price = exit_px
        cycle.qty = qty
        cycle.cost_quote = cost_quote
        cycle.proceeds = net
        cycle.pnl = net - cost_quote
        cycle.pnl_pct = cycle.pnl / cost_quote if cost_quote else 0.0
        cycle.avg_entry = wap()
        cycle.reason = "eod_close"
        cycles.append(cycle)
        qty = 0.0
        in_cycle = False

    equity_final = cash
    total_return_pct = (equity_final / params.capital - 1.0) * 100.0
    bh = (float(df["close"].iloc[-1]) / float(df["close"].iloc[0]) - 1.0) * 100.0

    closed = [c for c in cycles if c.reason == "take_profit"]
    wins = [c for c in closed if c.pnl > 0]
    losses = [c for c in cycles if c.pnl <= 0]
    win_rate = (len(wins) / len(closed) * 100.0) if closed else 0.0
    avg_pnl = float(np.mean([c.pnl_pct for c in closed]) * 100.0) if closed else 0.0
    gross_profit = sum(c.pnl for c in cycles if c.pnl > 0)
    gross_loss = abs(sum(c.pnl for c in cycles if c.pnl < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-12 else float("inf")

    equity_curve = pd.Series(equity_hist, index=pd.DatetimeIndex(df["datetime"]), name="equity")

    return BacktestResult(
        params=params,
        equity_final=equity_final,
        total_return_pct=total_return_pct,
        buy_hold_return_pct=bh,
        num_cycles=len(cycles),
        win_rate=win_rate,
        avg_pnl_pct=avg_pnl,
        max_drawdown_pct=max_dd * 100.0,
        profit_factor=profit_factor if profit_factor != float("inf") else 999.0,
        total_fees=total_fees,
        cycles=cycles,
        equity_curve=equity_curve,
    )


def print_report(result: BacktestResult, title: str = "Backtest") -> None:
    p = result.params
    print("=" * 64)
    print(title)
    print("=" * 64)
    print(f"Initial capital     : {p.capital:,.2f}")
    print(f"Final equity        : {result.equity_final:,.2f}")
    print(f"Strategy return     : {result.total_return_pct:+.2f}%")
    print(f"Buy & hold return   : {result.buy_hold_return_pct:+.2f}%")
    print(f"Cycles              : {result.num_cycles}")
    print(f"Win rate (TP exits) : {result.win_rate:.1f}%")
    print(f"Avg TP PnL %        : {result.avg_pnl_pct:+.3f}%")
    print(f"Max drawdown        : {result.max_drawdown_pct:.2f}%")
    print(f"Profit factor       : {result.profit_factor:.3f}")
    print(f"Total fees          : {result.total_fees:,.2f}")
    print(
        f"Params: init={p.initial_pct:.3f} tp={p.take_profit_pct:.4f} "
        f"depths={list(p.dca_depths)} lvl_w={list(p.dca_level_weights)} "
        f"sub_w={list(p.sub_order_weights)} fee={p.fee_rate}"
    )
    ladder = p.ladder()
    print("Ladder (drop% -> alloc%):")
    for row in ladder:
        print(f"  {row['kind']:12s}  drop={row['drop']*100:6.2f}%  alloc={row['alloc']*100:6.3f}%")
