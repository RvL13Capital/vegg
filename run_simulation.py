#!/usr/bin/env python3
"""
Thermodynamics of Liquidity - ABM Simulation Runner

This script executes the controlled experiment comparing:
1. Newtonian Market (Control): Fixed, low latency
2. Shear-Thickening Market (Intervention): Adaptive viscosity

Both simulations use the same random seed to ensure identical agent
behavior, isolating the effect of the regulatory mechanism.

Usage:
    python run_simulation.py [--no-plot] [--save PATH]

Options:
    --no-plot    Skip visualization (for headless environments)
    --save PATH  Save figure to specified path
"""

import sys
import argparse

from abm_simulation import (
    run_comparison,
    plot_comparison,
    print_comparison_summary,
    STEPS,
    SEED,
    PREDATOR_START
)


def main():
    parser = argparse.ArgumentParser(
        description="Run Thermodynamics of Liquidity simulation"
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip visualization'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Save figure to specified path'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=STEPS,
        help=f'Number of simulation steps (default: {STEPS})'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=SEED,
        help=f'Random seed (default: {SEED})'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("THERMODYNAMICS OF LIQUIDITY")
    print("Phase Transition Prevention via Adaptive Market Viscosity")
    print("=" * 60)
    print(f"\nSimulation Parameters:")
    print(f"  Steps: {args.steps}")
    print(f"  Random Seed: {args.seed}")
    print(f"  Predator Start: Step {PREDATOR_START}")
    print()

    # Run both simulations
    result_newt, result_shear = run_comparison(
        steps=args.steps,
        seed=args.seed,
        verbose=True
    )

    # Print comparison summary
    print_comparison_summary(result_newt, result_shear)

    # Visualization
    if not args.no_plot:
        print("Rendering analysis...")
        try:
            fig = plot_comparison(result_newt, result_shear, save_path=args.save)

            if args.save is None:
                import matplotlib.pyplot as plt
                plt.show()

        except ImportError as e:
            print(f"Visualization skipped: {e}")
            print("Install matplotlib for charts: pip install matplotlib")

    print("\nSimulation Complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
