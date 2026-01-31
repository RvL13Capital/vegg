"""
ABM Simulation - Thermodynamics of Liquidity

Agent-Based Model for testing phase transition prevention through
adaptive market viscosity (shear-thickening regulation).

This module creates a synthetic financial market to test the hypothesis:
Can we stop a deterministic phase transition (crash) by engineering
the market's viscosity?
"""

from .config import (
    STEPS,
    SEED,
    PREDATOR_START,
    PHI_CRITICAL,
    MM_COUNT,
    MT_COUNT
)
from .market import Order, OrderBook, Side, OrderType
from .regulator import Regulator, RegulatorMode
from .agents import Agent, MarketMaker, MomentumTrader, Predator, NoiseTrader
from .engine import (
    SimulationResult,
    run_simulation,
    run_comparison,
    calculate_max_drawdown
)
from .visualization import (
    plot_comparison,
    plot_single_result,
    print_comparison_summary
)

__version__ = "0.1.0"
__all__ = [
    # Config
    "STEPS",
    "SEED",
    "PREDATOR_START",
    "PHI_CRITICAL",
    "MM_COUNT",
    "MT_COUNT",
    # Market
    "Order",
    "OrderBook",
    "Side",
    "OrderType",
    # Regulator
    "Regulator",
    "RegulatorMode",
    # Agents
    "Agent",
    "MarketMaker",
    "MomentumTrader",
    "Predator",
    "NoiseTrader",
    # Engine
    "SimulationResult",
    "run_simulation",
    "run_comparison",
    "calculate_max_drawdown",
    # Visualization
    "plot_comparison",
    "plot_single_result",
    "print_comparison_summary",
]
