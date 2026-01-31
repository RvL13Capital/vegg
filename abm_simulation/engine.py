"""
Simulation Engine: Core execution loop for the ABM.

Orchestrates agent actions, order processing, and regulator intervention
to simulate market dynamics under different regulatory modes.
"""

import collections
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from .market import Order, OrderBook
from .regulator import Regulator, RegulatorMode
from .agents import Agent, MarketMaker, MomentumTrader, Predator
from .config import STEPS, SEED, MM_COUNT, MT_COUNT


@dataclass
class SimulationResult:
    """
    Container for simulation results.

    Attributes
    ----------
    mode : str
        Regulator mode name.
    price_history : List[float]
        Price at each simulation step.
    phi_history : List[float]
        Order parameter at each step.
    final_price : float
        Price at end of simulation.
    max_drawdown : float
        Maximum peak-to-trough decline.
    regulator_stats : dict
        Phi statistics from regulator.
    """
    mode: str
    price_history: List[float]
    phi_history: List[float]
    final_price: float
    max_drawdown: float
    regulator_stats: dict


def calculate_max_drawdown(prices: List[float]) -> float:
    """
    Calculate maximum drawdown (peak-to-trough decline).

    Parameters
    ----------
    prices : List[float]
        Price time series.

    Returns
    -------
    float
        Maximum drawdown as a positive percentage.
    """
    if not prices:
        return 0.0

    peak = prices[0]
    max_dd = 0.0

    for price in prices:
        if price > peak:
            peak = price
        drawdown = (peak - price) / peak
        if drawdown > max_dd:
            max_dd = drawdown

    return max_dd * 100


def create_default_agents() -> Tuple[List[Agent], Predator]:
    """
    Create the default agent population.

    Returns
    -------
    Tuple[List[Agent], Predator]
        List of all agents and the predator separately.
    """
    agents = []

    # Market Makers
    for i in range(MM_COUNT):
        agents.append(MarketMaker(f"MM_{i}"))

    # Momentum Traders
    for i in range(MT_COUNT):
        agents.append(MomentumTrader(f"MT_{i}"))

    # Predator
    predator = Predator("PREDATOR")
    agents.append(predator)

    return agents, predator


def run_simulation(
    mode: RegulatorMode,
    steps: int = STEPS,
    seed: int = SEED,
    agents: Optional[List[Agent]] = None,
    verbose: bool = True
) -> SimulationResult:
    """
    Execute a full simulation run.

    Parameters
    ----------
    mode : RegulatorMode
        NEWTONIAN (control) or SHEAR_THICKENING (intervention).
    steps : int
        Number of simulation steps.
    seed : int
        Random seed for reproducibility.
    agents : Optional[List[Agent]]
        Custom agent list. If None, uses default population.
    verbose : bool
        Print progress updates.

    Returns
    -------
    SimulationResult
        Complete simulation results including price and Phi histories.
    """
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Initialize components
    book = OrderBook()
    regulator = Regulator(mode=mode)

    if agents is None:
        agents, _ = create_default_agents()

    # Separate market makers for initial book seeding
    market_makers = [a for a in agents if isinstance(a, MarketMaker)]

    # Latency Pipeline: Stores (release_time, order)
    pending_orders: collections.deque = collections.deque()

    price_history = []

    if verbose:
        print(f"--- Running Simulation: Mode = {mode.value} ---")

    for step in range(1, steps + 1):
        # 1. Pre-fill book initially so it's not empty
        if step < 50:
            for mm in market_makers:
                orders = mm.act(step, book, book.history)
                if orders:
                    for o in orders:
                        book.process_order(o)

        # 2. Calculate Systemic State (Phi)
        current_phi = regulator.calculate_phi(book)

        # 3. Agents Generate Orders
        new_orders = []
        if step >= 50:
            random.shuffle(agents)  # Randomize arrival order
            for agent in agents:
                agent_orders = agent.act(step, book, book.history)
                if agent_orders:
                    new_orders.extend(agent_orders)

        # 4. Regulator Assigns Latency & Enqueues
        for order in new_orders:
            latency = regulator.assign_latency(order, current_phi)
            release_time = step + latency
            pending_orders.append((release_time, order))

        # 5. Process Matured Orders from Pipeline
        while pending_orders and pending_orders[0][0] <= step:
            _, matured_order = pending_orders.popleft()
            book.process_order(matured_order)

        price_history.append(book.last_price)

        if verbose and step % 200 == 0:
            regime = regulator.get_regime(current_phi)
            print(f"Step {step}: Price={book.last_price:.2f}, "
                  f"Phi={current_phi:.2f}, Regime={regime}")

    # Calculate statistics
    max_dd = calculate_max_drawdown(price_history)
    reg_stats = regulator.get_statistics()

    if verbose:
        print(f"Completed. Final Price: {price_history[-1]:.2f}, "
              f"Max Drawdown: {max_dd:.1f}%")

    return SimulationResult(
        mode=mode.value,
        price_history=price_history,
        phi_history=regulator.phi_history,
        final_price=price_history[-1],
        max_drawdown=max_dd,
        regulator_stats=reg_stats
    )


def run_comparison(
    steps: int = STEPS,
    seed: int = SEED,
    verbose: bool = True
) -> Tuple[SimulationResult, SimulationResult]:
    """
    Run both Newtonian and Shear-Thickening simulations for comparison.

    Uses identical seeds to ensure the same random agent behavior,
    isolating the effect of the regulatory mode.

    Parameters
    ----------
    steps : int
        Number of simulation steps.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Print progress updates.

    Returns
    -------
    Tuple[SimulationResult, SimulationResult]
        Results from (Newtonian, Shear-Thickening) simulations.
    """
    # Run Control (Newtonian)
    result_newt = run_simulation(
        mode=RegulatorMode.NEWTONIAN,
        steps=steps,
        seed=seed,
        verbose=verbose
    )

    # Run Intervention (Shear-Thickening)
    result_shear = run_simulation(
        mode=RegulatorMode.SHEAR_THICKENING,
        steps=steps,
        seed=seed,
        verbose=verbose
    )

    return result_newt, result_shear
