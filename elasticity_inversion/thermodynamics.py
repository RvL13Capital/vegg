"""
Liquidity Thermodynamics Engine for Phase Transition Detection.

Implements the Elasticity Inversion theory using Orthogonal Distance Regression
(Total Least Squares) to measure the relationship between price changes and
liquidity changes in market microstructure.
"""

import numpy as np
import pandas as pd
from scipy import odr
from typing import Optional, Tuple


class LiquidityThermodynamics:
    """
    Engine for detecting Phase Transitions in market microstructure
    using the Elasticity Inversion theory (Theta).

    The core insight: In normal markets, liquidity RISES when prices fall
    (mean reversion, buying the dip). In unstable markets, liquidity FALLS
    when prices fall (panic, reflexivity). Theta measures this elasticity:

    - Theta < 0: STABLE regime (negative feedback, mean-reverting)
    - Theta > 0: UNSTABLE regime (positive feedback, destabilizing)

    Parameters
    ----------
    data : pd.DataFrame
        L3 Tick Data DataFrame with columns:
        ['timestamp', 'price', 'bid_vol', 'ask_vol', 'bid_p', 'ask_p']
    position_size_q : float, default=10.0
        The reference institutional size to measure liquidity cost against.

    Attributes
    ----------
    data : pd.DataFrame
        Processed tick data with computed liquidity metrics.
    Q : float
        Reference position size.
    results : pd.DataFrame or None
        Rolling Theta computation results after running compute_elasticity_theta().

    Examples
    --------
    >>> df = pd.read_csv('market_data.csv', parse_dates=['timestamp'])
    >>> engine = LiquidityThermodynamics(df, position_size_q=100)
    >>> engine.calculate_effective_liquidity()
    >>> results = engine.compute_elasticity_theta(window_seconds=60)
    >>> critical_points = engine.detect_phase_transitions()
    """

    def __init__(self, data: pd.DataFrame, position_size_q: float = 10.0):
        """
        Initialize the Liquidity Thermodynamics engine.

        Parameters
        ----------
        data : pd.DataFrame
            L3 Tick Data DataFrame with columns:
            ['timestamp', 'price', 'bid_vol', 'ask_vol', 'bid_p', 'ask_p']
        position_size_q : float, default=10.0
            The reference institutional size to measure liquidity cost against.
        """
        required_cols = {'timestamp', 'bid_vol', 'ask_vol', 'bid_p', 'ask_p'}
        missing = required_cols - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.data = data.sort_values('timestamp').reset_index(drop=True)
        self.Q = position_size_q
        self.results: Optional[pd.DataFrame] = None

    def calculate_effective_liquidity(self) -> pd.DataFrame:
        """
        Calculate Cost-to-Trade (Implementation Shortfall) for size Q.

        Computes L(t) = Q / (Price Impact Cost) using a depth proxy
        based on Top of Book (L1) spread and volume.

        In a real L3 book, we would walk the limit levels. Here we
        approximate using Top of Book + Depth proxy for demonstration.

        Returns
        -------
        pd.DataFrame
            The data DataFrame with additional columns:
            - mid_price: Mid price between bid and ask
            - liquidity_L: Effective liquidity measure
            - log_P: Log of mid price
            - log_L: Log of liquidity

        Notes
        -----
        Realized Depth = Volume / Spread (Inverse Amihud measure).
        Higher L(t) is better. If spread widens or depth drops, L drops.
        """
        # Calculate Mid Price
        self.data['mid_price'] = (self.data['bid_p'] + self.data['ask_p']) / 2

        # Calculate spread
        spread = self.data['ask_p'] - self.data['bid_p']

        # Calculate "Hard" Liquidity Depth (approximated)
        depth_proxy = (self.data['bid_vol'] + self.data['ask_vol']) / 2

        # Avoid division by zero in spread - forward fill zeros
        spread = spread.replace(0, np.nan).ffill()

        # L(t): Higher is better. If spread widens or depth drops, L drops.
        self.data['liquidity_L'] = depth_proxy / spread

        # Handle edge cases
        self.data['liquidity_L'] = self.data['liquidity_L'].replace(
            [np.inf, -np.inf], np.nan
        ).ffill().bfill()

        # Log Transformations for Elasticity (Percent change vs Percent change)
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        self.data['log_P'] = np.log(self.data['mid_price'].clip(lower=eps))
        self.data['log_L'] = np.log(self.data['liquidity_L'].clip(lower=eps))

        return self.data

    def _run_odr_tls(self, x_chunk: np.ndarray, y_chunk: np.ndarray) -> float:
        """
        Perform Orthogonal Distance Regression (Total Least Squares).

        Handles errors in both Price (X) and Liquidity (Y) variables,
        which is more appropriate than OLS when both variables have
        measurement error.

        Parameters
        ----------
        x_chunk : np.ndarray
            Log price changes (differenced).
        y_chunk : np.ndarray
            Log liquidity changes (differenced).

        Returns
        -------
        float
            The slope (Theta) from TLS regression, or np.nan if
            insufficient data or computation fails.
        """
        # Filter out NaN/Inf values
        valid_mask = np.isfinite(x_chunk) & np.isfinite(y_chunk)
        x_valid = x_chunk[valid_mask]
        y_valid = y_chunk[valid_mask]

        if len(x_valid) < 5:
            return np.nan  # Not enough data

        # Check for zero variance (would cause ODR to fail)
        if np.std(x_valid) < 1e-12 or np.std(y_valid) < 1e-12:
            return np.nan

        # Define linear model: y = mx + b
        def linear_func(p, x):
            return p[0] * x + p[1]

        try:
            # Model instantiation
            linear_model = odr.Model(linear_func)
            data_chunk = odr.Data(x_valid, y_valid)

            # Run ODR
            # beta0=[0, 0] are initial guesses for slope and intercept
            my_odr = odr.ODR(data_chunk, linear_model, beta0=[0., 0.])
            output = my_odr.run()

            # Return the slope (Theta)
            return output.beta[0]
        except Exception:
            return np.nan

    def compute_elasticity_theta(
        self,
        window_seconds: int = 60,
        stride_seconds: int = 1,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Compute Rolling Theta using Total Least Squares over a time window.

        Theta measures the elasticity of liquidity with respect to price:
        Delta ln(L) ~ Theta * Delta ln(P)

        Parameters
        ----------
        window_seconds : int, default=60
            Rolling window size in seconds for TLS regression.
            Recommended: 30s for fast signal, 300s for stable signal.
        stride_seconds : int, default=1
            Resampling frequency in seconds for computational efficiency.
        verbose : bool, default=True
            Print progress information.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['timestamp', 'theta'] containing
            the rolling Theta estimates.

        Raises
        ------
        ValueError
            If calculate_effective_liquidity() has not been called first.

        Notes
        -----
        We use differenced (returns) data rather than raw levels because
        raw Log-Price and Log-Liquidity are non-stationary (they drift).
        Regressing levels gives spurious correlations.

        The equation is: Delta ln(L) ~ Theta * Delta ln(P)
        """
        if 'log_P' not in self.data.columns or 'log_L' not in self.data.columns:
            raise ValueError("Run calculate_effective_liquidity() first.")

        if verbose:
            print(f"Starting ODR/TLS Rolling Regression. Window: {window_seconds}s...")

        # Set timestamp as index for rolling
        df = self.data.set_index('timestamp')

        # STRIDING for efficiency (resample to grid)
        stride_df = df.resample(f'{stride_seconds}s').last().dropna()

        if len(stride_df) < window_seconds + 1:
            raise ValueError(
                f"Insufficient data: need at least {window_seconds + 1} seconds, "
                f"got {len(stride_df)}"
            )

        window_size = window_seconds // stride_seconds

        log_P = stride_df['log_P'].values
        log_L = stride_df['log_L'].values
        ts = stride_df.index

        # Pre-allocate arrays for results
        thetas = []
        timestamps = []

        total_iterations = len(stride_df) - window_size

        for i in range(window_size, len(stride_df)):
            # Window slice
            y_window = log_L[i - window_size : i]  # Liquidity
            x_window = log_P[i - window_size : i]  # Price

            # Differencing for stationarity
            # dL/L = Theta * dP/P (in log space, diff = returns)
            dy = np.diff(y_window)
            dx = np.diff(x_window)

            # Calculate Theta via TLS
            theta = self._run_odr_tls(dx, dy)

            thetas.append(theta)
            timestamps.append(ts[i])

            # Progress indicator
            if verbose and (i - window_size) % 1000 == 0:
                progress = (i - window_size) / total_iterations * 100
                print(f"  Progress: {progress:.1f}%")

        if verbose:
            print("  Progress: 100.0%")
            print(f"Completed. Computed {len(thetas)} Theta estimates.")

        # Create Result DataFrame
        result_df = pd.DataFrame({'timestamp': timestamps, 'theta': thetas})
        self.results = result_df
        return result_df

    def detect_phase_transitions(
        self,
        threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Flag moments where Theta flips positive (Destabilizing regime).

        Parameters
        ----------
        threshold : float, default=0.0
            The boundary value for regime classification.
            Theta > threshold indicates UNSTABLE regime.

        Returns
        -------
        pd.DataFrame
            Subset of results where regime is UNSTABLE (potential phase
            transition points).

        Raises
        ------
        ValueError
            If compute_elasticity_theta() has not been called first.

        Notes
        -----
        The phase boundary at Theta = 0 represents the critical point
        where market dynamics shift from mean-reverting (stable) to
        reflexive (unstable) behavior.
        """
        if self.results is None:
            raise ValueError("Run compute_elasticity_theta() first.")

        self.results['regime'] = np.where(
            self.results['theta'] > threshold,
            'UNSTABLE',
            'STABLE'
        )

        # Filter for Critical Points
        critical_points = self.results[self.results['regime'] == 'UNSTABLE'].copy()
        return critical_points

    def get_regime_statistics(self) -> dict:
        """
        Compute summary statistics for the computed Theta series.

        Returns
        -------
        dict
            Dictionary containing:
            - mean_theta: Mean Theta value
            - std_theta: Standard deviation of Theta
            - pct_unstable: Percentage of time in unstable regime
            - max_theta: Maximum Theta observed
            - min_theta: Minimum Theta observed
            - transition_count: Number of regime transitions

        Raises
        ------
        ValueError
            If compute_elasticity_theta() has not been called first.
        """
        if self.results is None:
            raise ValueError("Run compute_elasticity_theta() first.")

        if 'regime' not in self.results.columns:
            self.detect_phase_transitions()

        theta_clean = self.results['theta'].dropna()

        # Count regime transitions
        regime_shifts = (self.results['regime'] != self.results['regime'].shift()).sum()

        return {
            'mean_theta': theta_clean.mean(),
            'std_theta': theta_clean.std(),
            'pct_unstable': (self.results['regime'] == 'UNSTABLE').mean() * 100,
            'max_theta': theta_clean.max(),
            'min_theta': theta_clean.min(),
            'transition_count': regime_shifts
        }

    def plot_theta_series(
        self,
        figsize: Tuple[int, int] = (14, 6),
        show_regimes: bool = True
    ):
        """
        Plot the Theta time series with regime highlighting.

        Parameters
        ----------
        figsize : tuple, default=(14, 6)
            Figure size (width, height) in inches.
        show_regimes : bool, default=True
            Whether to shade unstable regime periods.

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure object.

        Raises
        ------
        ValueError
            If compute_elasticity_theta() has not been called first.
        ImportError
            If matplotlib is not installed.
        """
        if self.results is None:
            raise ValueError("Run compute_elasticity_theta() first.")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. "
                            "Install with: pip install matplotlib")

        fig, ax = plt.subplots(figsize=figsize)

        # Plot Theta
        ax.plot(
            self.results['timestamp'],
            self.results['theta'],
            color='blue',
            linewidth=0.8,
            alpha=0.8,
            label='Theta (Elasticity)'
        )

        # Phase Boundary
        ax.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Phase Boundary')

        # Shade unstable regions
        if show_regimes and 'regime' in self.results.columns:
            unstable_mask = self.results['regime'] == 'UNSTABLE'
            ax.fill_between(
                self.results['timestamp'],
                self.results['theta'].min(),
                self.results['theta'].max(),
                where=unstable_mask,
                alpha=0.2,
                color='red',
                label='Unstable Regime'
            )

        ax.set_xlabel('Time')
        ax.set_ylabel('Theta (Liquidity-Price Elasticity)')
        ax.set_title('Liquidity Phase Transition Detection: Elasticity Inversion')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
