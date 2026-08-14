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

### Results comparison

| Metric | Baseline | Pyramid-preserving | Profit-max (unconstrained) |
|--------|----------|--------------------|----------------------------|
| Final equity | 26,614 | **41,052** | 214,228 |
| Return | +166% | **+311%** | +2,042% |
| Max drawdown | 62.7% | **67.1%** | 79.5% |
| Cycles | 2,167 | 1,629 | 1,338 |
| Profit factor | 2.11 | **2.83** | 1.69 |
| Level weights | 1:2:4:8 | **1:2:4:8** | 8:4:2:1 (inverted) |

**Recommended (keeps pyramid shape):** TP `1.2%`, initial `5%`, depths `2/8/20/40%`, level weights `1:2:4:8`, sub-weights `2:3:4`, re-entry `0`.  
Saved under `results/pyramid_preserving/`.

**Profit-max:** same but inverted level weights `8:4:2:1` — much higher return, much higher drawdown. Saved as main `results/summary.json`.

Equity curve: `results/equity_curve.png`

## Stop-loss upgrade (inverted / profit-max)

The inverted ladder is strong in ranging markets but sits underwater for months in crashes (no SL). We swept hard stop-loss from \(P_0\) and from WAP, plus optional time-stops, on crash windows then validated on full history.

**Crash windows used for design**
- 2021-11 → 2022-12
- 2024-12 → 2026-07

**Best setting found**

| Parameter | Value |
|-----------|-------|
| Stop loss | **5% below \(P_0\)** (`stop_loss_ref=p0`) |
| Max hold | **72 hours** (optional; caps the longest stuck cycle) |
| Rest | same profit-max inverted config (TP 1.2%, weights 8:4:2:1, …) |

### Crash windows

| Window | No SL | SL 5% \(P_0\) | SL 5% \(P_0\) + max hold 72h |
|--------|-------|--------------|------------------------------|
| 2021-11→2022-12 return | −70% | **+358%** | **+366%** |
| Max DD | 79.5% | **14.1%** | **14.1%** |
| Max hold (hours) | 9,466 | 105 | **72** |
| 2024-12→2026-07 return | −43% | **+135%** | **+132%** |
| Max DD | 65.6% | **14.8%** | **14.8%** |

### Full history (2021-01 → 2026-07)

| Metric | No SL | **SL 5% \(P_0\) + hold≤72h** |
|--------|-------|-----------------------------|
| Return | +2,042% | **+73,339%** |
| Max drawdown | 79.5% | **21.1%** |
| Avg hold (hours) | 36.5 | **5.6** |
| Max hold (hours) | 32,400 (~3.7y) | **72** |
| Win rate | 99.9% | 92.8% |
| SL exits | 0 | 586 |

Reference \(P_0\) beat WAP-based SL for this inverted config (capital is concentrated near entry, so a fixed −5% from entry cuts crashes cleanly).

> Note: turnover and fees rise a lot with SL (many more closed cycles). Numbers assume 0.1% fee; live slippage/fees can shrink the edge.

Artifacts: `results/stoploss_inverted/`  
Re-run: `python src/optimize_stoploss.py`

## 1-minute backtest (path-accurate) + crash-first SL search

1h results were optimistic (same-bar Low→fill then High→TP bias). On **ETHUSDT 1m** (2,933,647 bars, 2021-01→2026-07):

| Config | Full return | Max DD | Avg hold | Max hold |
|--------|-------------|--------|----------|----------|
| Old 1h-tuned SL 5% \(P_0\) + hold≤72h | **−99%** | 99% | ~5h | 72h |
| No SL, TP 1.2% | +72% | 80% | ~102h | ~3.7y |
| No SL, TP 2.0% | **+83.5%** | 80% | ~221h | ~3.7y |
| **Best SL on 1m: SL 40% from WAP, TP 2%** | **+80.5%** | **75%** | **~56h** | **~207d** |
| Any short-hold cap (≤48h) with/without SL | −61% to −88% | high | ≤48h | 48h |

**Crash windows (1m):** no SL setting turned 2022 / 2025–26 into profit while keeping short holds. Tight SL shortens holds but dies on fees/noise; forced time-stops also lose on full history.

**Honest tradeoff on 1m:** highest profit needs long underwater holds; forcing short holds destroys expectancy. Best compromise found: `SL=40%` from WAP + `TP=2%` (no time-stop).

Artifacts: `results/tf_1m/`, `results/tf_1m_crash_sl/`  
Re-run: `python src/run_backtest_1m.py` · `python src/optimize_sl_1m_crash_first.py`

## Staged Recovery (partial size + escalate after loss)

Idea you proposed:

- **Normal:** buy 25% now + 25% at −3%, tight SL (~3%), TP; unused cash stays idle  
- **After a loss → Recovery:** 50% now + 25% at −3% + 25% at −5%, wider SL (~10%)  
- **After a win → back to Normal**

### User baseline (as stated) on 1m

| Window | Return | Max DD |
|--------|--------|--------|
| Full 1m | **−99.8%** | 99.8% |
| Full 1h | −99.2% | 99.2% |
| Crash 2022 | −89.7% | 90.5% |

Tight normal SL (3%) + martingale-style recovery after losses blows up on fees/noise.

### Optimized staged params (crash-first then full 1m)

| | Normal | Recovery |
|--|--|--|
| Initial | **15%** | **50%** |
| DCA | 25% @ **−5%** | 25% @ −3% + 25% @ **−8%** |
| SL (from \(P_0\)) | **8%** | **20%** |
| TP (from WAP) | **2%** | **1.2%** |

| Window | User | Optimized |
|--------|------|-----------|
| Full 1m | −99.8% | **+23.4%** (DD 43%) |
| Full 1h | −99.2% | −3.4% |
| Crash 2022 | −89.7% | −21.6% |
| Crash 2025–26 | −83.1% | −30.3% |

Still underperforms buy&hold on full 1m (+153%), but the staging idea survives after widening SL/TP and cutting normal size.

Artifacts: `results/staged_recovery/` · `python src/run_staged_recovery.py`

## Setup

```bash
cd dca-pyramid-backtest
pip install -r requirements.txt
```

## Run

```bash
# Download (if needed) + baseline + optimization
python src/run_backtest.py

# Stop-loss optimization for inverted config
python src/optimize_stoploss.py

# Download only
python src/download_data.py
```

Results are written to `results/`:

- `summary.json` — baseline vs optimized
- `optimization_history.csv` — every trial
- `cycles_baseline.csv` / `cycles_final.csv`
- `equity_curve_*.csv` / `equity_curve.png`
- `pyramid_preserving/` — best settings while keeping classic `1:2:4:8` level weights
- `risk_aware/` — stronger drawdown-penalized search
- `stoploss_inverted/` — SL / max-hold search for inverted weights

## Optimization method

Coordinate descent over:
`take_profit_pct` → `initial_pct` → `dca_depths` → `dca_level_weights` → `sub_order_weights` → `reentry_delay_bars`.

Primary score = total return − mild penalty for drawdown above 20%.

Stop-loss pass: sweep SL% × {`p0`,`wap`} on crash windows (score penalizes DD + long holds), then validate on full history; refine with `max_hold_bars`.

---

Built with [BrainDaemon](https://braindaemon.com)
