"""
Market Agents: Different trader archetypes in the ABM simulation.

Each agent type exhibits different behavior patterns that collectively
create emergent market dynamics and potential phase transitions.
"""

import random
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from .market import Order, OrderBook, Side, OrderType
from .config import (
    PREDATOR_START,
    PREDATOR_SIZE,
    PREDATOR_INTERVAL,
    MM_BASE_SPREAD,
    MM_VOL_MULTIPLIER
)


class Agent(ABC):
    """
    Abstract base class for all market agents.

    Attributes
    ----------
    agent_id : str
        Unique identifier for this agent.
    """

    def __init__(self, agent_id: str):
        """
        Initialize an agent.

        Parameters
        ----------
        agent_id : str
            Unique identifier for this agent.
        """
        self.agent_id = agent_id

    @abstractmethod
    def act(
        self,
        step: int,
        book: OrderBook,
        history: List[float]
    ) -> Optional[List[Order]]:
        """
        Generate orders based on current market state.

        Parameters
        ----------
        step : int
            Current simulation step.
        book : OrderBook
            Current order book state.
        history : List[float]
            Price history up to current step.

        Returns
        -------
        Optional[List[Order]]
            List of orders to submit, or None if no action.
        """
        pass


class MarketMaker(Agent):
    """
    Provides liquidity by quoting both sides of the market.

    Market makers widen spreads when volatility is high, exhibiting
    reflexive behavior that can amplify price dislocations. This is
    the "liquidity withdrawal" phenomenon observed in flash crashes.

    The spread formula: spread = base_spread + (volatility * multiplier)
    """

    def __init__(
        self,
        agent_id: str,
        base_spread: float = MM_BASE_SPREAD,
        vol_multiplier: float = MM_VOL_MULTIPLIER
    ):
        """
        Initialize a market maker.

        Parameters
        ----------
        agent_id : str
            Unique identifier.
        base_spread : float
            Minimum spread width in normal conditions.
        vol_multiplier : float
            How much volatility widens spreads.
        """
        super().__init__(agent_id)
        self.base_spread = base_spread
        self.vol_multiplier = vol_multiplier

    def act(
        self,
        step: int,
        book: OrderBook,
        history: List[float]
    ) -> Optional[List[Order]]:
        """
        Quote both sides of the market with volatility-adjusted spreads.

        The reflexive spread mechanism creates a feedback loop:
        High volatility -> wider spreads -> lower liquidity ->
        easier to move price -> higher volatility

        Returns
        -------
        List[Order]
            A buy and sell limit order pair.
        """
        # Calculate recent volatility proxy
        if len(history) > 10:
            recent_prices = history[-10:]
            vol = np.std(recent_prices)
        else:
            vol = 0.1

        mid = book.get_mid()

        # Reflexive Spreads: Higher vol -> wider spreads -> lower liquidity
        spread = self.base_spread + (vol * self.vol_multiplier)

        buy_price = round(mid - spread / 2, 2)
        sell_price = round(mid + spread / 2, 2)
        size = random.randint(1, 3)

        # Submit both sides
        buy_order = Order(
            agent_id=self.agent_id,
            side=Side.BUY,
            price=buy_price,
            size=size,
            order_type=OrderType.LIMIT,
            timestamp=step
        )
        sell_order = Order(
            agent_id=self.agent_id,
            side=Side.SELL,
            price=sell_price,
            size=size,
            order_type=OrderType.LIMIT,
            timestamp=step
        )

        return [buy_order, sell_order]


class MomentumTrader(Agent):
    """
    Takes liquidity based on recent price trends.

    Momentum traders chase trends, amplifying directional moves.
    When prices fall, they sell (panic), creating positive feedback
    that can cascade into flash crashes.

    This models the "pile on" effect observed in real markets.
    """

    def __init__(self, agent_id: str, lookback: int = 5):
        """
        Initialize a momentum trader.

        Parameters
        ----------
        agent_id : str
            Unique identifier.
        lookback : int
            Number of steps to look back for trend detection.
        """
        super().__init__(agent_id)
        self.lookback = lookback

    def act(
        self,
        step: int,
        book: OrderBook,
        history: List[float]
    ) -> Optional[List[Order]]:
        """
        Sell if price is trending down (panic selling).

        Returns
        -------
        Optional[List[Order]]
            Market sell order if downtrend detected, None otherwise.
        """
        if len(history) < self.lookback:
            return None

        # Simple trend detection: is price lower than N steps ago?
        if history[-1] < history[-self.lookback]:
            # Trend down -> panic sell (Market Order)
            size = random.randint(1, 2)
            return [Order(
                agent_id=self.agent_id,
                side=Side.SELL,
                price=None,
                size=size,
                order_type=OrderType.MARKET,
                timestamp=step
            )]

        return None


class Predator(Agent):
    """
    The Catalyst: Executes a large sell program over time.

    The predator represents institutional selling pressure - a large
    participant liquidating a position. This triggers the cascade:
    1. Predator sells
    2. Price drops
    3. Momentum traders detect trend and sell
    4. Market makers widen spreads
    5. Liquidity evaporates
    6. Price crashes

    This is the "exogenous shock" that tips the system into instability.
    """

    def __init__(
        self,
        agent_id: str,
        start_step: int = PREDATOR_START,
        order_size: int = PREDATOR_SIZE,
        interval: int = PREDATOR_INTERVAL
    ):
        """
        Initialize a predator.

        Parameters
        ----------
        agent_id : str
            Unique identifier.
        start_step : int
            Simulation step when selling begins.
        order_size : int
            Size of each sell order.
        interval : int
            Steps between sell orders.
        """
        super().__init__(agent_id)
        self.start_step = start_step
        self.order_size = order_size
        self.interval = interval

    def act(
        self,
        step: int,
        book: OrderBook,
        history: List[float]
    ) -> Optional[List[Order]]:
        """
        Execute periodic large sell orders after start time.

        Returns
        -------
        Optional[List[Order]]
            Large market sell order at intervals, None otherwise.
        """
        # Start selling at configured step, then every interval steps
        if step >= self.start_step and step % self.interval == 0:
            return [Order(
                agent_id=self.agent_id,
                side=Side.SELL,
                price=None,
                size=self.order_size,
                order_type=OrderType.MARKET,
                timestamp=step
            )]
        return None


class NoiseTrader(Agent):
    """
    Random trader providing background noise.

    Adds realistic randomness to the simulation without
    directional bias. Useful for testing robustness.
    """

    def __init__(self, agent_id: str, activity_prob: float = 0.3):
        """
        Initialize a noise trader.

        Parameters
        ----------
        agent_id : str
            Unique identifier.
        activity_prob : float
            Probability of trading each step.
        """
        super().__init__(agent_id)
        self.activity_prob = activity_prob

    def act(
        self,
        step: int,
        book: OrderBook,
        history: List[float]
    ) -> Optional[List[Order]]:
        """
        Randomly buy or sell with small size.

        Returns
        -------
        Optional[List[Order]]
            Random order with probability activity_prob, None otherwise.
        """
        if random.random() > self.activity_prob:
            return None

        side = random.choice([Side.BUY, Side.SELL])
        order_type = random.choice([OrderType.LIMIT, OrderType.MARKET])

        if order_type == OrderType.LIMIT:
            mid = book.get_mid()
            offset = random.uniform(-1, 1)
            price = round(mid + offset, 2)
        else:
            price = None

        size = random.randint(1, 2)

        return [Order(
            agent_id=self.agent_id,
            side=side,
            price=price,
            size=size,
            order_type=order_type,
            timestamp=step
        )]
