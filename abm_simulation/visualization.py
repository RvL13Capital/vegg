"""
Visualization tools for ABM simulation results.

Provides comparative charts for analyzing the effect of different
regulatory modes on market dynamics and phase transitions.
"""

from typing import Tuple, Optional

from .engine import SimulationResult
from .config import PREDATOR_START, PHI_CRITICAL


def plot_comparison(
    result_newt: SimulationResult,
    result_shear: SimulationResult,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
):
    """
    Create comparative visualization of Newtonian vs Shear-Thickening results.

    Produces a two-panel chart:
    - Top: Price paths for both modes
    - Bottom: Order Parameter (Phi) for both modes

    Parameters
    ----------
    result_newt : SimulationResult
        Results from Newtonian (control) simulation.
    result_shear : SimulationResult
        Results from Shear-Thickening (intervention) simulation.
    figsize : Tuple[int, int]
        Figure size (width, height) in inches.
    save_path : Optional[str]
        If provided, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization. "
                         "Install with: pip install matplotlib")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Price Chart
    ax1.set_title("Thermodynamics of Liquidity: Phase Transition Test", fontsize=14)
    ax1.plot(
        result_newt.price_history,
        label='Newtonian Market (Control)',
        color='red',
        linewidth=2,
        alpha=0.7
    )
    ax1.plot(
        result_shear.price_history,
        label='Shear-Thickening Market (Intervention)',
        color='blue',
        linewidth=2
    )
    ax1.axvline(
        x=PREDATOR_START,
        color='grey',
        linestyle='--',
        label='Predator Starts Selling'
    )
    ax1.set_ylabel("Asset Price")
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)

    # Add annotations for key metrics
    ax1.annotate(
        f'Newtonian: {result_newt.max_drawdown:.1f}% drawdown',
        xy=(0.98, 0.95),
        xycoords='axes fraction',
        ha='right',
        va='top',
        fontsize=10,
        color='red'
    )
    ax1.annotate(
        f'Shear-Thick: {result_shear.max_drawdown:.1f}% drawdown',
        xy=(0.98, 0.88),
        xycoords='axes fraction',
        ha='right',
        va='top',
        fontsize=10,
        color='blue'
    )

    # Order Parameter (Phi) Chart
    ax2.set_title(r"Systemic Strain: Order Parameter ($\Phi$)", fontsize=14)
    ax2.plot(
        result_newt.phi_history,
        label=r'$\Phi$ Newtonian (Divergence)',
        color='red',
        alpha=0.6
    )
    ax2.plot(
        result_shear.phi_history,
        label=r'$\Phi$ Shear-Thickening (Controlled)',
        color='blue',
        alpha=0.6
    )
    ax2.axhline(
        y=PHI_CRITICAL,
        color='black',
        linestyle=':',
        label=r'Critical Threshold ($\Phi_c$)'
    )
    ax2.set_ylabel(r"$\Phi$ (Selling Pressure / Depth)")
    ax2.set_xlabel("Simulation Step (Time)")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Add annotations for Phi statistics
    newt_stats = result_newt.regulator_stats
    shear_stats = result_shear.regulator_stats

    if newt_stats and shear_stats:
        ax2.annotate(
            f'Newtonian: {newt_stats.get("pct_critical", 0):.1f}% critical',
            xy=(0.98, 0.95),
            xycoords='axes fraction',
            ha='right',
            va='top',
            fontsize=10,
            color='red'
        )
        ax2.annotate(
            f'Shear-Thick: {shear_stats.get("pct_critical", 0):.1f}% critical',
            xy=(0.98, 0.88),
            xycoords='axes fraction',
            ha='right',
            va='top',
            fontsize=10,
            color='blue'
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_single_result(
    result: SimulationResult,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
):
    """
    Plot results from a single simulation run.

    Parameters
    ----------
    result : SimulationResult
        Simulation results to plot.
    figsize : Tuple[int, int]
        Figure size (width, height) in inches.
    save_path : Optional[str]
        If provided, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization.")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Price Chart
    ax1.set_title(f"Simulation Result: {result.mode} Mode", fontsize=14)
    ax1.plot(result.price_history, color='blue', linewidth=1.5)
    ax1.axvline(x=PREDATOR_START, color='grey', linestyle='--', alpha=0.7)
    ax1.set_ylabel("Asset Price")
    ax1.grid(True, alpha=0.3)

    # Phi Chart
    ax2.plot(result.phi_history, color='orange', linewidth=1)
    ax2.axhline(y=PHI_CRITICAL, color='red', linestyle=':', linewidth=2)
    ax2.fill_between(
        range(len(result.phi_history)),
        result.phi_history,
        PHI_CRITICAL,
        where=[p > PHI_CRITICAL for p in result.phi_history],
        alpha=0.3,
        color='red'
    )
    ax2.set_ylabel(r"$\Phi$ (Order Parameter)")
    ax2.set_xlabel("Simulation Step")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def print_comparison_summary(
    result_newt: SimulationResult,
    result_shear: SimulationResult
) -> None:
    """
    Print a text summary comparing the two simulation results.

    Parameters
    ----------
    result_newt : SimulationResult
        Newtonian (control) results.
    result_shear : SimulationResult
        Shear-Thickening (intervention) results.
    """
    print("\n" + "=" * 60)
    print("SIMULATION COMPARISON SUMMARY")
    print("=" * 60)

    print(f"\n{'Metric':<30} {'Newtonian':>12} {'Shear-Thick':>12}")
    print("-" * 60)

    print(f"{'Final Price':<30} {result_newt.final_price:>12.2f} "
          f"{result_shear.final_price:>12.2f}")

    print(f"{'Max Drawdown (%)':<30} {result_newt.max_drawdown:>12.1f} "
          f"{result_shear.max_drawdown:>12.1f}")

    # Phi statistics
    newt_stats = result_newt.regulator_stats
    shear_stats = result_shear.regulator_stats

    if newt_stats and shear_stats:
        print(f"{'Mean Phi':<30} {newt_stats.get('mean_phi', 0):>12.3f} "
              f"{shear_stats.get('mean_phi', 0):>12.3f}")

        print(f"{'Max Phi':<30} {newt_stats.get('max_phi', 0):>12.3f} "
              f"{shear_stats.get('max_phi', 0):>12.3f}")

        print(f"{'% Time Critical':<30} {newt_stats.get('pct_critical', 0):>12.1f} "
              f"{shear_stats.get('pct_critical', 0):>12.1f}")

    # Calculate improvement
    dd_reduction = result_newt.max_drawdown - result_shear.max_drawdown
    if result_newt.max_drawdown > 0:
        dd_reduction_pct = (dd_reduction / result_newt.max_drawdown) * 100
    else:
        dd_reduction_pct = 0

    print("-" * 60)
    print(f"\nDrawdown Reduction: {dd_reduction:.1f}pp ({dd_reduction_pct:.1f}%)")

    if dd_reduction > 0:
        print("RESULT: Shear-Thickening regulation REDUCED crash severity")
    else:
        print("RESULT: No improvement from shear-thickening regulation")

    print("=" * 60 + "\n")
