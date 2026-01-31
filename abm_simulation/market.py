"""
Market infrastructure: Order and OrderBook implementations.

Implements a standard Price/Time Priority Matching Engine for the ABM simulation.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum

from .config import INITIAL_PRICE, BOOK_DEPTH_LEVELS


class Side(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration."""
    LIMIT = "LIMIT"
    MARKET = "MARKET"


@dataclass
class Order:
    """
    Represents a single order in the market.

    Attributes
    ----------
    agent_id : str
        Unique identifier of the agent submitting the order.
    side : Side
        BUY or SELL.
    price : Optional[float]
        Limit price. None for MARKET orders.
    size : int
        Order quantity.
    order_type : OrderType
        LIMIT or MARKET.
    timestamp : int
        Simulation step when order was created.
    """
    agent_id: str
    side: Side
    price: Optional[float]
    size: int
    order_type: OrderType
    timestamp: int


class OrderBook:
    """
    Standard Price/Time Priority Matching Engine.

    Maintains bid and ask queues, processes orders, and tracks
    trade history and sell pressure for systemic strain calculation.

    Attributes
    ----------
    bids : List
        Buy orders sorted by price (desc), then time (asc).
    asks : List
        Sell orders sorted by price (asc), then time (asc).
    last_price : float
        Most recent trade price.
    history : List[float]
        Full trade price history.
    aggregated_sell_pressure : float
        Cumulative aggressive selling for Phi calculation.
    """

    def __init__(self):
        """Initialize an empty order book."""
        # Each entry: [price, size, timestamp, agent_id]
        self.bids: List[List] = []
        self.asks: List[List] = []
        self.last_price: float = INITIAL_PRICE
        self.history: List[float] = []
        self.aggregated_sell_pressure: float = 0.0

    def get_best_bid(self) -> float:
        """Return best (highest) bid price, or default if empty."""
        return self.bids[0][0] if self.bids else self.last_price - 0.1

    def get_best_ask(self) -> float:
        """Return best (lowest) ask price, or default if empty."""
        return self.asks[0][0] if self.asks else self.last_price + 0.1

    def get_mid(self) -> float:
        """Return mid-price between best bid and ask."""
        return (self.get_best_bid() + self.get_best_ask()) / 2.0

    def get_spread(self) -> float:
        """Return current bid-ask spread."""
        return self.get_best_ask() - self.get_best_bid()

    def get_depth(self, levels: int = BOOK_DEPTH_LEVELS) -> float:
        """
        Calculate currently available 'Hard' liquidity nearby.

        Parameters
        ----------
        levels : int
            Number of price levels to consider on each side.

        Returns
        -------
        float
            Total depth (bid + ask volume) plus epsilon to avoid division by zero.
        """
        bid_depth = sum(b[1] for b in self.bids[:levels])
        ask_depth = sum(a[1] for a in self.asks[:levels])
        return bid_depth + ask_depth + 1e-9

    def _insert_bid(self, order: Order) -> None:
        """Insert a limit buy order into the bid queue."""
        self.bids.append([order.price, order.size, order.timestamp, order.agent_id])
        # Sort by price descending, then time ascending
        self.bids.sort(key=lambda x: (-x[0], x[2]))

    def _insert_ask(self, order: Order) -> None:
        """Insert a limit sell order into the ask queue."""
        self.asks.append([order.price, order.size, order.timestamp, order.agent_id])
        # Sort by price ascending, then time ascending
        self.asks.sort(key=lambda x: (x[0], x[2]))

    def _match_buy_market(self, order: Order) -> List[float]:
        """Match a market buy order against the ask queue."""
        trades = []
        remaining_size = order.size

        while remaining_size > 0 and self.asks:
            best_ask = self.asks[0]
            trade_size = min(remaining_size, best_ask[1])
            trade_price = best_ask[0]

            self.last_price = trade_price
            self.history.append(trade_price)
            trades.append(trade_price)

            remaining_size -= trade_size
            best_ask[1] -= trade_size

            if best_ask[1] <= 0:
                self.asks.pop(0)

        return trades

    def _match_sell_market(self, order: Order) -> List[float]:
        """Match a market sell order against the bid queue."""
        trades = []
        remaining_size = order.size

        while remaining_size > 0 and self.bids:
            best_bid = self.bids[0]
            trade_size = min(remaining_size, best_bid[1])
            trade_price = best_bid[0]

            self.last_price = trade_price
            self.history.append(trade_price)
            trades.append(trade_price)

            remaining_size -= trade_size
            best_bid[1] -= trade_size

            if best_bid[1] <= 0:
                self.bids.pop(0)

        return trades

    def process_order(self, order: Order) -> List[float]:
        """
        Process an incoming order through the matching engine.

        Parameters
        ----------
        order : Order
            The order to process.

        Returns
        -------
        List[float]
            List of trade prices from any fills.
        """
        trades = []

        if order.side == Side.BUY:
            if order.order_type == OrderType.LIMIT:
                self._insert_bid(order)
            elif order.order_type == OrderType.MARKET:
                trades = self._match_buy_market(order)

        elif order.side == Side.SELL:
            # Track sell pressure for Phi calculation
            self.aggregated_sell_pressure += order.size

            if order.order_type == OrderType.LIMIT:
                self._insert_ask(order)
            elif order.order_type == OrderType.MARKET:
                trades = self._match_sell_market(order)

        return trades

    def decay_pressure(self, factor: float = 0.5) -> None:
        """
        Decay the aggregated sell pressure for next step.

        Parameters
        ----------
        factor : float
            Multiplicative decay factor (0-1).
        """
        self.aggregated_sell_pressure *= factor

    def get_book_state(self) -> dict:
        """
        Return current order book state for analysis.

        Returns
        -------
        dict
            Dictionary with bid/ask counts, depth, spread, and last price.
        """
        return {
            'bid_count': len(self.bids),
            'ask_count': len(self.asks),
            'best_bid': self.get_best_bid(),
            'best_ask': self.get_best_ask(),
            'mid': self.get_mid(),
            'spread': self.get_spread(),
            'depth': self.get_depth(),
            'last_price': self.last_price,
            'sell_pressure': self.aggregated_sell_pressure
        }
