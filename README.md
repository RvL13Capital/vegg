# vegg

## Elasticity Inversion: Phase Transition Detection in Market Microstructure

This module implements the **Elasticity Inversion theory (Theta)** for detecting liquidity phase transitions using Orthogonal Distance Regression (Total Least Squares).

### Core Insight

In normal markets, liquidity **RISES** when prices fall (mean reversion, "buying the dip"). In unstable markets, liquidity **FALLS** when prices fall (panic, reflexivity).

**Theta** measures this elasticity:
- **Theta < 0**: STABLE regime (negative feedback, mean-reverting)
- **Theta > 0**: UNSTABLE regime (positive feedback, destabilizing)

The equation: `Δln(L) ≈ Θ · Δln(P)`

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```python
import pandas as pd
from elasticity_inversion import LiquidityThermodynamics

# 1. Load L3 Tick Data
# Required columns: ['timestamp', 'bid_vol', 'ask_vol', 'bid_p', 'ask_p']
df = pd.read_csv('market_data.csv', parse_dates=['timestamp'])

# 2. Initialize Engine
engine = LiquidityThermodynamics(df, position_size_q=100)

# 3. Calculate Liquidity State
engine.calculate_effective_liquidity()

# 4. Compute Theta (using 60s window for balanced signal)
results = engine.compute_elasticity_theta(window_seconds=60)

# 5. Find Phase Transitions (the "Smoking Gun")
critical_points = engine.detect_phase_transitions()
print(critical_points.head())

# 6. Get Summary Statistics
stats = engine.get_regime_statistics()
print(f"Unstable {stats['pct_unstable']:.1f}% of time")
print(f"Regime transitions: {stats['transition_count']}")

# 7. Plot Results
fig = engine.plot_theta_series()
fig.savefig('theta_analysis.png')
```

### Sensitivity Analysis

For benchmarking, run with different window sizes:

```python
# Fast Signal (noisy but responsive)
results_30s = engine.compute_elasticity_theta(window_seconds=30)

# Stable Signal (confirms structural breaks)
results_5m = engine.compute_elasticity_theta(window_seconds=300)
```

### Data Requirements

- **Timestamp precision**: At least millisecond precision
- **Aggregation**: If multiple trades occur in one millisecond, aggregate them (VWAP price, Sum volume) before processing to avoid inf/NaN artifacts
- **Columns**: `timestamp`, `bid_vol`, `ask_vol`, `bid_p`, `ask_p`

### Methodology Notes

1. **Total Least Squares (ODR)**: Unlike OLS, TLS accounts for measurement error in both price and liquidity variables.

2. **Differencing**: We regress percentage changes (returns) rather than raw levels because Log-Price and Log-Liquidity are non-stationary.

3. **Liquidity Measure**: `L(t) = Depth / Spread` (Inverse Amihud approximation)

---

## ABM Simulation: Thermodynamics of Liquidity

Agent-Based Model for testing phase transition prevention through adaptive market viscosity.

### The Experiment

This simulation tests: **Can we stop a deterministic phase transition (crash) by engineering the market's viscosity?**

Two controlled experiments using identical random seeds:

1. **Newtonian Market (Control)**: Fixed, low latency. Expects flash crash from Predator → Momentum → MM withdrawal cascade.

2. **Shear-Thickening Market (Intervention)**: Adaptive latency. As Phi (systemic strain) rises, aggressive orders slow down, allowing liquidity to replenish.

### Quick Start

```bash
python run_simulation.py
```

Or with options:

```bash
python run_simulation.py --steps 3000 --seed 123 --save results.png
```

### Programmatic Usage

```python
from abm_simulation import (
    run_comparison,
    plot_comparison,
    print_comparison_summary,
    RegulatorMode,
    run_simulation
)

# Run comparison experiment
result_newt, result_shear = run_comparison(steps=2000, seed=42)

# Print summary
print_comparison_summary(result_newt, result_shear)

# Visualize
fig = plot_comparison(result_newt, result_shear)
fig.savefig('phase_transition_test.png')

# Or run single mode
result = run_simulation(mode=RegulatorMode.SHEAR_THICKENING)
```

### Agent Types

| Agent | Behavior | Role in Cascade |
|-------|----------|-----------------|
| **MarketMaker** | Quotes both sides, widens spreads with volatility | Liquidity provider (withdraws under stress) |
| **MomentumTrader** | Sells when price trends down | Amplifies moves (panic selling) |
| **Predator** | Large periodic sell orders after step 500 | Catalyst (exogenous shock) |

### The Order Parameter (Phi)

```
Phi = (Aggressive Selling Pressure) / (Available Depth)
```

- **Phi < 0.5**: STABLE - Normal market function
- **Phi > 0.5**: CRITICAL - Phase transition risk

### Interpreting Results

**Price Chart:**
- Red (Newtonian): Catastrophic failure around step 500-700
- Blue (Shear-Thickening): Orderly decline, no crash

**Phi Chart:**
- Red: Spikes dramatically during crash
- Blue: Rises to threshold, then controlled by viscosity

### Architecture

```
abm_simulation/
├── config.py          # Constants and parameters
├── market.py          # Order, OrderBook matching engine
├── regulator.py       # Phi calculation, viscosity functions
├── agents.py          # MarketMaker, MomentumTrader, Predator
├── engine.py          # Simulation loop
└── visualization.py   # Plotting utilities
```
