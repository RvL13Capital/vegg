"""
Configuration constants for the Thermodynamics of Liquidity ABM simulation.
"""

# Simulation Duration
STEPS = 2000

# Random Seed for Reproducibility
SEED = 42

# Predator Configuration
PREDATOR_START = 500    # When the large seller starts
PREDATOR_SIZE = 5       # Size of predator sell orders
PREDATOR_INTERVAL = 20  # Steps between predator orders

# Agent Population
MM_COUNT = 10           # Number of Market Makers
MT_COUNT = 15           # Number of Momentum Traders

# Regulator Configuration
PHI_CRITICAL = 0.5      # Phase transition threshold
BASE_LATENCY = 1        # Base latency for all orders
MAX_PENALTY_DELAY = 10  # Maximum added delay for market orders

# Market Maker Configuration
MM_BASE_SPREAD = 0.5    # Base spread width
MM_VOL_MULTIPLIER = 5   # How much volatility widens spreads

# Order Book Configuration
INITIAL_PRICE = 100.0   # Starting price
BOOK_DEPTH_LEVELS = 5   # Levels to consider for depth calculation
