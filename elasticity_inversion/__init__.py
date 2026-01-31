"""
Elasticity Inversion - Phase Transition Detection in Market Microstructure

This module implements the Elasticity Inversion theory (Theta) for detecting
liquidity phase transitions using Orthogonal Distance Regression (Total Least Squares).
"""

from .thermodynamics import LiquidityThermodynamics

__version__ = "0.1.0"
__all__ = ["LiquidityThermodynamics"]
