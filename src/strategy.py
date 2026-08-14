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
    # Stop loss: 0 disables. Exit when price <= ref * (1 - stop_loss_pct).
    stop_loss_pct: float = 0.0
    # Reference for SL: "wap" (avg entry) or "p0" (cycle entry).
    stop_loss_ref: str = "wap"
    # Time stop: 0 disables. Force-close after this many bars in cycle.
    max_hold_bars: int = 0

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
    hold_bars: int = 0
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
    avg_hold_bars: float = 0.0
    median_hold_bars: float = 0.0
    max_hold_bars_seen: float = 0.0
    tp_exits: int = 0
    sl_exits: int = 0
    time_exits: int = 0
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
            "avg_hold_bars": self.avg_hold_bars,
            "median_hold_bars": self.median_hold_bars,
            "max_hold_bars_seen": self.max_hold_bars_seen,
            "tp_exits": self.tp_exits,
            "sl_exits": self.sl_exits,
            "time_exits": self.time_exits,
        }


def run_backtest(df: pd.DataFrame, params: StrategyParams | None = None) -> BacktestResult:
    """
    Event-driven spot backtest on OHLC bars.

    Cycle logic:
      1. At cycle start, market-buy initial allocation at open.
      2. Remaining ladder levels are limit buys at P0 * (1 - drop).
      3. A limit fills if bar low <= limit price (fill at limit).
      4. After every fill, update WAP and TP = WAP * (1 + take_profit_pct).
      5. Optional SL: if bar low <= ref*(1-stop_loss_pct), exit at SL
         (adverse fill if TP and SL both possible same bar).
      6. Else if bar high >= TP, exit entire position at TP.
      7. Optional time-stop: force close at bar close after max_hold_bars.
      8. After exit, start a new cycle (optionally after delay).

    Capital is recycled cycle-to-cycle (compounding). Unfilled ladder cash stays idle
    until filled or cycle ends (unfilled cash remains cash).
    """
    params = params or StrategyParams()
    if params.stop_loss_ref not in ("wap", "p0"):
        raise ValueError("stop_loss_ref must be 'wap' or 'p0'")
    ladder = params.ladder()
    fee = params.fee_rate

    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
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
    cost_quote = 0.0
    spent_for_avg = 0.0
    filled_flags: list[bool] = []
    cycle_cash_budget = 0.0
    cycle_cash_left = 0.0
    pending_reentry = 0
    cycle: TradeCycle | None = None
    cycle_start_i = 0
    peak_equity = cash
    max_dd = 0.0

    def wap() -> float:
        return spent_for_avg / qty if qty > 0 else 0.0

    def mark_equity(i: int) -> float:
        return cash + qty * closes[i]

    def close_cycle(i: int, exit_px: float, reason: str) -> None:
        nonlocal cash, total_fees, qty, spent_for_avg, cost_quote, in_cycle, cycle
        nonlocal pending_reentry, cycle_start_i
        assert cycle is not None
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
        cycle.hold_bars = i - cycle_start_i + 1
        cycle.reason = reason
        cycles.append(cycle)
        qty = 0.0
        spent_for_avg = 0.0
        cost_quote = 0.0
        in_cycle = False
        cycle = None
        pending_reentry = params.reentry_delay_bars

    i = 0
    while i < n:
        # Start new cycle
        if not in_cycle:
            if pending_reentry > 0:
                pending_reentry -= 1
                equity_hist[i] = mark_equity(i)
                peak_equity = max(peak_equity, equity_hist[i])
                max_dd = max(
                    max_dd,
                    (peak_equity - equity_hist[i]) / peak_equity if peak_equity else 0.0,
                )
                i += 1
                continue

            if cash <= 1e-8:
                equity_hist[i] = mark_equity(i)
                i += 1
                continue

            p0 = opens[i]
            cycle_cash_budget = cash
            cycle_cash_left = cash
            filled_flags = [False] * len(ladder)
            qty = 0.0
            cost_quote = 0.0
            spent_for_avg = 0.0
            in_cycle = True
            cycle_start_i = i
            cycle = TradeCycle(entry_time=times[i], p0=p0)

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
            spent_for_avg += buy_quote
            cost_quote += alloc_quote
            filled_flags[0] = True
            cycle.fills = 1
            cycle.avg_entry = wap()

        assert cycle is not None

        bar_dd = (p0 - lows[i]) / p0 if p0 > 0 else 0.0
        cycle.max_dd_from_p0 = max(cycle.max_dd_from_p0, bar_dd)

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

        exited = False
        if qty > 0:
            avg = wap()
            tp_price = avg * (1.0 + params.take_profit_pct)
            sl_price = 0.0
            if params.stop_loss_pct and params.stop_loss_pct > 0:
                ref = avg if params.stop_loss_ref == "wap" else p0
                sl_price = ref * (1.0 - params.stop_loss_pct)

            hit_tp = highs[i] >= tp_price
            hit_sl = sl_price > 0 and lows[i] <= sl_price

            # Adverse assumption: if both TP and SL possible in same bar, take SL.
            if hit_sl:
                close_cycle(i, sl_price, "stop_loss")
                exited = True
            elif hit_tp:
                close_cycle(i, tp_price, "take_profit")
                exited = True
            elif (
                params.max_hold_bars > 0
                and (i - cycle_start_i + 1) >= params.max_hold_bars
            ):
                close_cycle(i, closes[i], "time_stop")
                exited = True

        equity_hist[i] = mark_equity(i)
        peak_equity = max(peak_equity, equity_hist[i])
        if peak_equity > 0:
            max_dd = max(max_dd, (peak_equity - equity_hist[i]) / peak_equity)
        i += 1

    if in_cycle and qty > 0 and cycle is not None:
        close_cycle(n - 1, closes[-1], "eod_close")

    equity_final = cash
    total_return_pct = (equity_final / params.capital - 1.0) * 100.0
    bh = (closes[-1] / closes[0] - 1.0) * 100.0

    holds = [float(c.hold_bars) for c in cycles]
    tp_exits = sum(1 for c in cycles if c.reason == "take_profit")
    sl_exits = sum(1 for c in cycles if c.reason == "stop_loss")
    time_exits = sum(1 for c in cycles if c.reason == "time_stop")

    # Win rate over all closed cycles (not only TP)
    wins = [c for c in cycles if c.pnl > 0]
    win_rate = (len(wins) / len(cycles) * 100.0) if cycles else 0.0
    avg_pnl = float(np.mean([c.pnl_pct for c in cycles]) * 100.0) if cycles else 0.0
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
        avg_hold_bars=float(np.mean(holds)) if holds else 0.0,
        median_hold_bars=float(np.median(holds)) if holds else 0.0,
        max_hold_bars_seen=float(np.max(holds)) if holds else 0.0,
        tp_exits=tp_exits,
        sl_exits=sl_exits,
        time_exits=time_exits,
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
    print(f"Win rate (all)      : {result.win_rate:.1f}%")
    print(f"Avg cycle PnL %     : {result.avg_pnl_pct:+.3f}%")
    print(f"Max drawdown        : {result.max_drawdown_pct:.2f}%")
    print(f"Profit factor       : {result.profit_factor:.3f}")
    print(f"Total fees          : {result.total_fees:,.2f}")
    print(
        f"Hold bars avg/med/max: {result.avg_hold_bars:.1f} / "
        f"{result.median_hold_bars:.1f} / {result.max_hold_bars_seen:.0f}"
    )
    print(
        f"Exits TP/SL/Time    : {result.tp_exits} / {result.sl_exits} / {result.time_exits}"
    )
    print(
        f"Params: init={p.initial_pct:.3f} tp={p.take_profit_pct:.4f} "
        f"sl={p.stop_loss_pct:.4f}({p.stop_loss_ref}) max_hold={p.max_hold_bars} "
        f"depths={list(p.dca_depths)} lvl_w={list(p.dca_level_weights)} "
        f"sub_w={list(p.sub_order_weights)} fee={p.fee_rate}"
    )
    ladder = p.ladder()
    print("Ladder (drop% -> alloc%):")
    for row in ladder:
        print(f"  {row['kind']:12s}  drop={row['drop']*100:6.2f}%  alloc={row['alloc']*100:6.3f}%")
