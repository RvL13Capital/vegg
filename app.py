"""
Thermodynamics of Liquidity - Streamlit Web Dashboard

Interactive web interface for:
1. ABM Simulation: Test phase transition prevention via shear-thickening regulation
2. Elasticity Forensics: Detect liquidity phase transitions in uploaded market data
"""

import streamlit as st
import pandas as pd
import numpy as np

# Import project modules
from abm_simulation import run_comparison, plot_comparison, print_comparison_summary
from elasticity_inversion import LiquidityThermodynamics

# Page configuration
st.set_page_config(
    page_title="Thermodynamics of Liquidity",
    page_icon="🌡️",
    layout="wide"
)

st.title("Thermodynamics of Liquidity")
st.markdown("""
**A Unified Field Theory of Market Fragility**

This dashboard implements two complementary tools for understanding market phase transitions:
- **ABM Simulation**: Agent-Based Model testing shear-thickening market regulation
- **Elasticity Forensics**: ODR-based Theta detector for identifying regime shifts in real data
""")

tab_sim, tab_forensics, tab_theory = st.tabs([
    "🔬 ABM Simulation",
    "🕵️ Elasticity Forensics",
    "📖 Theory"
])

# =============================================================================
# TAB 1: ABM SIMULATION
# =============================================================================
with tab_sim:
    st.header("Phase Transition Prevention Test")
    st.markdown("""
    This simulation tests whether **adaptive market viscosity** can prevent flash crashes.

    Two experiments run with identical random seeds:
    - **Newtonian (Control)**: Fixed latency - expect crash cascade
    - **Shear-Thickening (Intervention)**: Adaptive latency based on systemic strain (Phi)
    """)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Parameters")
        steps = st.slider(
            "Simulation Steps",
            min_value=1000,
            max_value=5000,
            value=2000,
            step=100,
            help="Duration of the simulation"
        )
        seed = st.number_input(
            "Random Seed",
            min_value=1,
            max_value=9999,
            value=42,
            help="Same seed ensures identical agent behavior across modes"
        )

        st.markdown("---")
        st.markdown("**Agent Population:**")
        st.markdown("- 10 Market Makers")
        st.markdown("- 15 Momentum Traders")
        st.markdown("- 1 Predator (starts step 500)")

        st.markdown("---")
        run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

    with col2:
        if run_button:
            with st.spinner("Running Agent-Based Model..."):
                # Run the comparison using the engine
                res_newt, res_shear = run_comparison(
                    steps=steps,
                    seed=int(seed),
                    verbose=False
                )

                # Display metrics
                st.subheader("Results Summary")
                m1, m2, m3, m4 = st.columns(4)

                m1.metric(
                    "Newtonian Drawdown",
                    f"{res_newt.max_drawdown:.1f}%",
                    delta=None
                )
                m2.metric(
                    "Shear-Thick Drawdown",
                    f"{res_shear.max_drawdown:.1f}%",
                    delta=f"-{res_newt.max_drawdown - res_shear.max_drawdown:.1f}pp",
                    delta_color="inverse"
                )

                newt_critical = res_newt.regulator_stats.get('pct_critical', 0)
                shear_critical = res_shear.regulator_stats.get('pct_critical', 0)
                m3.metric(
                    "Newtonian % Critical",
                    f"{newt_critical:.1f}%"
                )
                m4.metric(
                    "Shear-Thick % Critical",
                    f"{shear_critical:.1f}%",
                    delta=f"-{newt_critical - shear_critical:.1f}pp",
                    delta_color="inverse"
                )

                # Plot comparison
                st.subheader("Comparative Analysis")
                fig = plot_comparison(res_newt, res_shear, figsize=(14, 10))
                st.pyplot(fig)

                # Interpretation
                st.subheader("Interpretation")
                delta_dd = res_newt.max_drawdown - res_shear.max_drawdown
                if delta_dd > 5:
                    st.success(f"""
                    **Shear-thickening regulation reduced crash severity by {delta_dd:.1f} percentage points.**

                    The adaptive viscosity mechanism successfully "diffused" the cascade by slowing
                    aggressive orders when systemic strain (Phi) exceeded the critical threshold.
                    """)
                elif delta_dd > 0:
                    st.info(f"""
                    **Modest improvement of {delta_dd:.1f} percentage points.**

                    The intervention showed some effect. Try different seeds or longer simulations
                    to explore the parameter space.
                    """)
                else:
                    st.warning("""
                    **No improvement detected in this run.**

                    This may indicate the crash dynamics were too fast for the viscosity
                    mechanism to engage, or the random seed produced unusual agent behavior.
                    """)
        else:
            st.info("👈 Configure parameters and click **Run Simulation** to begin.")

# =============================================================================
# TAB 2: ELASTICITY FORENSICS
# =============================================================================
with tab_forensics:
    st.header("Elasticity Inversion (Theta) Detector")
    st.markdown("""
    Upload L3 tick data to detect liquidity phase transitions using
    **Orthogonal Distance Regression (Total Least Squares)**.

    **Required CSV columns:** `timestamp`, `bid_vol`, `ask_vol`, `bid_p`, `ask_p`
    """)

    # File upload
    uploaded_file = st.file_uploader(
        "Upload Market Data CSV",
        type=['csv'],
        help="CSV with columns: timestamp, bid_vol, ask_vol, bid_p, ask_p"
    )

    # Demo data option
    use_demo = st.checkbox("Use synthetic demo data instead")

    if use_demo:
        # Generate synthetic data for demonstration
        st.info("Generating synthetic market data with an embedded crash event...")

        np.random.seed(42)
        n_points = 3600  # 1 hour of 1-second data

        timestamps = pd.date_range('2024-01-01 09:30:00', periods=n_points, freq='1s')

        # Base price with crash event
        base_price = 100.0
        prices = [base_price]
        for i in range(1, n_points):
            if 1800 <= i <= 2000:  # Crash period
                drift = -0.02
                vol = 0.3
            else:
                drift = 0.0
                vol = 0.05
            ret = drift + vol * np.random.randn()
            prices.append(prices[-1] * (1 + ret / 100))

        prices = np.array(prices)
        spreads = np.where(
            (np.arange(n_points) >= 1800) & (np.arange(n_points) <= 2000),
            0.5 + np.random.rand(n_points) * 0.5,  # Wide spreads during crash
            0.1 + np.random.rand(n_points) * 0.1   # Tight spreads normally
        )

        df = pd.DataFrame({
            'timestamp': timestamps,
            'bid_p': prices - spreads / 2,
            'ask_p': prices + spreads / 2,
            'bid_vol': np.where(
                (np.arange(n_points) >= 1800) & (np.arange(n_points) <= 2000),
                np.random.randint(1, 5, n_points),   # Low volume during crash
                np.random.randint(10, 50, n_points)  # Normal volume
            ),
            'ask_vol': np.where(
                (np.arange(n_points) >= 1800) & (np.arange(n_points) <= 2000),
                np.random.randint(1, 5, n_points),
                np.random.randint(10, 50, n_points)
            )
        })

        st.dataframe(df.head(10))
        uploaded_file = True  # Flag to proceed

    elif uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
            st.success(f"Loaded {len(df):,} rows")
            st.dataframe(df.head(10))
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            df = None
    else:
        df = None

    if df is not None or (use_demo and 'df' in dir()):
        # Validate columns
        required = {'timestamp', 'bid_vol', 'ask_vol', 'bid_p', 'ask_p'}
        if not required.issubset(df.columns):
            st.error(f"CSV missing required columns. Found: {list(df.columns)}")
        else:
            st.markdown("---")

            col1, col2 = st.columns([1, 3])

            with col1:
                st.subheader("Analysis Parameters")
                window = st.slider(
                    "Rolling Window (seconds)",
                    min_value=30,
                    max_value=600,
                    value=60,
                    step=10,
                    help="Window size for Theta calculation. 30s=fast/noisy, 300s=stable"
                )

                threshold = st.slider(
                    "Instability Threshold",
                    min_value=-0.5,
                    max_value=0.5,
                    value=0.0,
                    step=0.05,
                    help="Theta above this value = UNSTABLE regime"
                )

                analyze_button = st.button(
                    "🔍 Analyze Liquidity Physics",
                    type="primary",
                    use_container_width=True
                )

            with col2:
                if analyze_button:
                    with st.spinner("Computing Orthogonal Distance Regression (ODR)..."):
                        try:
                            # Initialize Engine
                            engine = LiquidityThermodynamics(df, position_size_q=100)
                            engine.calculate_effective_liquidity()

                            # Compute Theta
                            results = engine.compute_elasticity_theta(
                                window_seconds=window,
                                verbose=False
                            )

                            # Detect Transitions
                            critical = engine.detect_phase_transitions(threshold=threshold)
                            stats = engine.get_regime_statistics()

                            # Display metrics
                            st.subheader("Regime Statistics")
                            s1, s2, s3, s4 = st.columns(4)
                            s1.metric("Unstable %", f"{stats['pct_unstable']:.1f}%")
                            s2.metric("Transitions", f"{stats['transition_count']}")
                            s3.metric("Max Theta", f"{stats['max_theta']:.4f}")
                            s4.metric("Mean Theta", f"{stats['mean_theta']:.4f}")

                            # Plot Theta series
                            st.subheader("Theta Time Series")
                            fig_theta = engine.plot_theta_series(figsize=(14, 6))
                            st.pyplot(fig_theta)

                            # Critical points table
                            if len(critical) > 0:
                                st.subheader(f"Detected Critical Points ({len(critical)} total)")
                                st.dataframe(
                                    critical.head(100),
                                    use_container_width=True
                                )
                            else:
                                st.success("No critical points detected - market remained stable.")

                        except Exception as e:
                            st.error(f"Analysis error: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                else:
                    st.info("👈 Configure parameters and click **Analyze** to begin.")

# =============================================================================
# TAB 3: THEORY
# =============================================================================
with tab_theory:
    st.header("Theoretical Framework")

    st.markdown("""
    ## The Core Insight

    In **stable markets**, liquidity **increases** when prices fall (mean reversion, "buying the dip").

    In **unstable markets**, liquidity **decreases** when prices fall (panic, reflexivity).

    **Theta** measures this elasticity:

    ```
    Δln(L) ≈ Θ · Δln(P)
    ```

    - **Θ < 0**: STABLE regime (negative feedback, mean-reverting)
    - **Θ > 0**: UNSTABLE regime (positive feedback, destabilizing)

    ---

    ## The Order Parameter (Phi)

    In the ABM simulation, we define systemic strain as:

    ```
    Φ = (Aggressive Selling Pressure) / (Available Depth)
    ```

    When Φ exceeds the critical threshold (0.5), the market is at risk of phase transition.

    ---

    ## Shear-Thickening Regulation

    Inspired by non-Newtonian fluids that become more viscous under stress,
    the intervention applies **adaptive latency** to aggressive orders:

    ```
    Latency = Base + Penalty × (Φ / Φ_critical)²
    ```

    This slows down market orders when systemic strain is high, allowing:
    1. Liquidity providers time to replenish depth
    2. Momentum traders to receive stale signals
    3. The cascade to "diffuse" into orderly decline

    ---

    ## Methodology

    1. **Total Least Squares (ODR)**: Unlike OLS, accounts for measurement error in both
       price and liquidity variables

    2. **Differencing**: We regress percentage changes (returns) rather than raw levels
       because Log-Price and Log-Liquidity are non-stationary

    3. **Liquidity Measure**: `L(t) = Depth / Spread` (Inverse Amihud approximation)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Thermodynamics of Liquidity | Phase Transition Detection & Prevention
</div>
""", unsafe_allow_html=True)
