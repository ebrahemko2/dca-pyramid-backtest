# DCA Pyramid Backtest (ETHUSDT)

Spot DCA Pyramid strategy backtester with sequential (one-parameter-at-a-time) optimization.

## Strategy (baseline)

- **Initial buy:** 5% of cycle capital at entry \(P_0\)
- **Remaining 95%:** 4 DCA levels weighted `1 : 2 : 4 : 8`
- **Each DCA level:** 3 sub-orders weighted `1 : 2 : 4`
- **Depths from \(P_0\):**
  - DCA1 to -3% → -1% / -2% / -3%
  - DCA2 to -10% → -3.33% / -6.67% / -10%
  - DCA3 to -25% → -8.33% / -16.67% / -25%
  - DCA4 to -50% → -16.67% / -33.33% / -50%
- **Take profit:** sell full position at `WeightedAveragePrice × 1.01`
- **No stop loss** (original design)
- After TP, capital is recycled into a new cycle (compounding)
- Fee model: `0.1%` per fill (buy/sell)

## Backtest window

| Item | Value |
|------|-------|
| Symbol | ETHUSDT |
| Timeframe | 1h |
| Source | [Binance Vision](https://data.binance.vision/) monthly klines (merged) |
| Period | 2021-01-01 → 2026-07-31 |
| Bars | 48,898 |
| Starting capital | 10,000 USDT |

## Baseline results (original settings)

| Metric | Value |
|--------|-------|
| Final equity | 26,614 USDT |
| Strategy return | **+166.14%** |
| Buy & hold | +153.74% |
| Cycles | 2,167 |
| Max drawdown | 62.73% |
| Profit factor | 2.11 |

## Sequential optimization (maximize profit)

One parameter changed at a time; best value kept before moving to the next.

| Step | Parameter | Chosen value | Notes |
|------|-----------|--------------|-------|
| 1 | `take_profit_pct` | **1.2%** (was 1.0%) | Best among 0.5%–5% |
| 2 | `initial_pct` | **5%** (unchanged) | Still best |
| 3 | `dca_depths` | **2% / 8% / 20% / 40%** | Shallower than original |
| 4 | `dca_level_weights` | **8 : 4 : 2 : 1** | Inverted pyramid (more size on shallow dips) |
| 5 | `sub_order_weights` | **2 : 3 : 4** | Slightly flatter than 1:2:4 |
| 6 | `reentry_delay_bars` | **0** (unchanged) | Immediate re-entry best |

### Optimized vs baseline

| Metric | Baseline | Optimized |
|--------|----------|-----------|
| Final equity | 26,614 | **214,228** |
| Return | +166% | **+2,042%** |
| Max drawdown | 62.7% | 79.5% |
| Cycles | 2,167 | 1,338 |
| Profit factor | 2.11 | 1.69 |

> The profit-max set puts most capital near shallow dips (inverted weights). Returns rise a lot, but drawdown also rises. A risk-aware alternative is saved under `results/risk_aware/`.

Equity curve: `results/equity_curve.png`

## Setup

```bash
cd dca-pyramid-backtest
pip install -r requirements.txt
```

## Run

```bash
# Download (if needed) + baseline + optimization
python src/run_backtest.py

# Download only
python src/download_data.py
```

Results are written to `results/`:

- `summary.json` — baseline vs optimized
- `optimization_history.csv` — every trial
- `cycles_baseline.csv` / `cycles_final.csv`
- `equity_curve_*.csv` / `equity_curve.png`
- `risk_aware/` — stronger drawdown-penalized search

## Optimization method

Coordinate descent over:
`take_profit_pct` → `initial_pct` → `dca_depths` → `dca_level_weights` → `sub_order_weights` → `reentry_delay_bars`.

Primary score = total return − mild penalty for drawdown above 20%.

---

Built with [BrainDaemon](https://braindaemon.com)
