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
