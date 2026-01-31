"""
Regulator: The Thermodynamics Engine.

Implements the shear-thickening viscosity function that adapts market
latency based on systemic strain (Phi - the Order Parameter).
"""

from enum import Enum
from typing import List

from .market import Order, OrderBook, OrderType
from .config import (
    PHI_CRITICAL,
    BASE_LATENCY,
    MAX_PENALTY_DELAY,
    BOOK_DEPTH_LEVELS
)


class RegulatorMode(Enum):
    """Operating mode for the regulator."""
    NEWTONIAN = "NEWTONIAN"              # Fixed latency (control)
    SHEAR_THICKENING = "SHEAR_THICKENING"  # Adaptive latency (intervention)


class Regulator:
    """
    Implements the Shear-Thickening Viscosity Function.

    The regulator monitors systemic strain (Phi) and adjusts latency
    for aggressive orders when the system approaches critical thresholds.

    In NEWTONIAN mode: Fixed, low latency regardless of market state.
    In SHEAR_THICKENING mode: Latency increases non-linearly as Phi
    exceeds the critical threshold, slowing aggressive orders to
    allow liquidity replenishment.

    Attributes
    ----------
    mode : RegulatorMode
        Operating mode (NEWTONIAN or SHEAR_THICKENING).
    phi_history : List[float]
        Time series of Phi values for analysis.
    phi_critical : float
        The phase transition threshold.
    """

    def __init__(self, mode: RegulatorMode):
        """
        Initialize the regulator.

        Parameters
        ----------
        mode : RegulatorMode
            Operating mode for latency assignment.
        """
        self.mode = mode
        self.phi_history: List[float] = []
        self.phi_critical = PHI_CRITICAL

    def calculate_phi(
        self,
        book: OrderBook,
        depth_levels: int = BOOK_DEPTH_LEVELS
    ) -> float:
        """
        Calculate the Order Parameter (Systemic Strain).

        Phi = (Aggressive Selling Pressure) / (Available Buy Depth)

        This measures the ratio of "kinetic energy" (selling pressure)
        to "structural capacity" (liquidity depth). When Phi exceeds
        critical threshold, the market is in danger of phase transition.

        Parameters
        ----------
        book : OrderBook
            The current order book state.
        depth_levels : int
            Number of price levels to consider for depth.

        Returns
        -------
        float
            The current Phi value (0 = calm, >1 = extreme stress).
        """
        pressure = book.aggregated_sell_pressure
        depth = book.get_depth(levels=depth_levels)

        # Calculate strain ratio
        phi = pressure / depth if depth > 0 else 1.0  # Max strain if no depth

        # Decay pressure for next step
        book.decay_pressure(factor=0.5)

        self.phi_history.append(phi)
        return phi

    def assign_latency(self, order: Order, current_phi: float) -> int:
        """
        The Viscosity Function: Assign latency based on order type and Phi.

        In NEWTONIAN mode, all orders receive base latency.
        In SHEAR_THICKENING mode, aggressive (MARKET) orders receive
        additional delay when Phi exceeds the critical threshold.

        The penalty follows a quadratic relationship (Gamma function concept):
        penalty = (phi / phi_critical)^2

        This creates a non-linear response that becomes increasingly
        resistive as systemic strain increases.

        Parameters
        ----------
        order : Order
            The order to assign latency to.
        current_phi : float
            Current systemic strain value.

        Returns
        -------
        int
            Total latency (base + penalty) in simulation steps.
        """
        if self.mode == RegulatorMode.NEWTONIAN:
            return BASE_LATENCY

        elif self.mode == RegulatorMode.SHEAR_THICKENING:
            # Only penalize aggressive (MARKET) orders when Phi is critical
            if order.order_type == OrderType.MARKET and current_phi > self.phi_critical:
                # Non-linear penalty (quadratic viscosity)
                penalty_factor = (current_phi / self.phi_critical) ** 2
                added_delay = int(MAX_PENALTY_DELAY * penalty_factor)
                return BASE_LATENCY + added_delay
            else:
                return BASE_LATENCY

        return BASE_LATENCY

    def get_regime(self, phi: float) -> str:
        """
        Classify the current market regime based on Phi.

        Parameters
        ----------
        phi : float
            Current Phi value.

        Returns
        -------
        str
            Regime classification: 'STABLE', 'WARNING', or 'CRITICAL'.
        """
        if phi < self.phi_critical * 0.5:
            return 'STABLE'
        elif phi < self.phi_critical:
            return 'WARNING'
        else:
            return 'CRITICAL'

    def get_statistics(self) -> dict:
        """
        Compute summary statistics for the Phi time series.

        Returns
        -------
        dict
            Statistics including mean, max, time in critical regime.
        """
        if not self.phi_history:
            return {}

        import numpy as np
        phi_array = np.array(self.phi_history)

        return {
            'mean_phi': float(np.mean(phi_array)),
            'max_phi': float(np.max(phi_array)),
            'std_phi': float(np.std(phi_array)),
            'pct_critical': float(np.mean(phi_array > self.phi_critical) * 100),
            'pct_warning': float(np.mean(
                (phi_array > self.phi_critical * 0.5) &
                (phi_array <= self.phi_critical)
            ) * 100)
        }
