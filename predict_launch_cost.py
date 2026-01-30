import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION: Domain Knowledge & Scenario Parameters
# ==============================================================================

# Physical/Economic Floor for Specific Launch Cost
# Even with full reusability, fundamental costs remain:
# - Propellant (~$200k per launch for heavy rockets)
# - Ground operations, insurance, labor
# - Rocket refurbishment (even reusable has maintenance)
COST_FLOOR = 150  # $/kg - long-term stable floor

# Convergence parameters
CONVERGENCE_START_YEAR = 2050  # Year when cost stabilization begins
CONVERGENCE_COMPLETE_YEAR = 2080  # Year when costs fully stabilize
STABLE_COST = 180  # $/kg - stable long-term cost (center of fluctuation band)

# Stable period fluctuation (±5% maximum)
STABLE_FLUCTUATION_PERCENT = 0.05  # 5% max variation

# Weight multipliers
HEAVY_WEIGHT_MULTIPLIER = 3.0
POST_2010_WEIGHT_MULTIPLIER = 5.0


def load_and_process_data(file_path='cost-space-launches-low-earth-orbit.csv'):
    """
    Loads the launch cost data and applies weighting for Heavy class and 2010+ data.
    """
    df = pd.read_csv(file_path)
    
    df['Is_Heavy'] = (df['Launch vehicle class'] == 'Heavy').astype(int)
    df['Weight'] = 1.0
    df.loc[df['Is_Heavy'] == 1, 'Weight'] *= HEAVY_WEIGHT_MULTIPLIER
    df.loc[df['Year'] >= 2010, 'Weight'] *= POST_2010_WEIGHT_MULTIPLIER
    
    heavy_df = df[df['Launch vehicle class'] == 'Heavy'].copy()
    
    return df, heavy_df


def fit_exponential_decay_model(heavy_df):
    """
    Fits an exponential decay model to the Heavy launch cost data.
    
    Model: Cost(t) = (C0 - C_floor) * exp(-λ * (t - t0)) + C_floor
    """
    years = heavy_df['Year'].values
    costs = heavy_df['Launch cost per kilogram of payload'].values
    weights = heavy_df['Weight'].values
    
    t0 = years.min()
    t = years - t0
    
    floor = COST_FLOOR
    min_cost = costs.min()
    if floor >= min_cost:
        floor = min_cost * 0.5
    
    log_costs = np.log(costs - floor)
    
    X = t.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, log_costs, sample_weight=weights)
    
    log_C0 = model.intercept_
    decay_rate = -model.coef_[0]
    
    if decay_rate < 0:
        decay_rate = 0.01
    
    C0 = np.exp(log_C0) + floor
    
    return {
        't0': t0,
        'C0': C0,
        'decay_rate': decay_rate,
        'floor': floor,
    }


def predict_launch_cost(target_year, model_params, heavy_df):
    """
    Predicts specific launch cost for a given year.
    
    Phases:
    1. Pre-2050: Exponential decay from historical data
    2. 2050-2080: Smooth transition to stable floor
    3. Post-2080: Stable plateau with minimal fluctuation (±5%)
    """
    t0 = model_params['t0']
    C0 = model_params['C0']
    decay_rate = model_params['decay_rate']
    floor = model_params['floor']
    
    t = target_year - t0
    
    # Phase 1: Exponential decay (before convergence starts)
    base_decay = (C0 - floor) * np.exp(-decay_rate * t) + floor
    
    if target_year < CONVERGENCE_START_YEAR:
        # Pure exponential decay phase
        return max(base_decay, floor)
    
    elif target_year < CONVERGENCE_COMPLETE_YEAR:
        # Phase 2: Smooth transition to stable cost
        # Use sigmoid-like transition for smoothness
        transition_duration = CONVERGENCE_COMPLETE_YEAR - CONVERGENCE_START_YEAR
        years_into_transition = target_year - CONVERGENCE_START_YEAR
        
        # Smooth S-curve transition (using cosine for smoothness)
        # Goes from 0 to 1 smoothly
        transition_factor = 0.5 * (1 - np.cos(np.pi * years_into_transition / transition_duration))
        
        # Value at start of transition
        t_start = CONVERGENCE_START_YEAR - t0
        cost_at_transition_start = (C0 - floor) * np.exp(-decay_rate * t_start) + floor
        
        # Blend between decay curve and stable cost
        predicted = cost_at_transition_start * (1 - transition_factor) + STABLE_COST * transition_factor
        
        return max(predicted, floor)
    
    else:
        # Phase 3: Stable plateau (post-2080)
        # Only tiny deterministic micro-fluctuations allowed (±5%)
        
        # Create a very smooth, slow oscillation using multiple low-frequency waves
        years_since_stable = target_year - CONVERGENCE_COMPLETE_YEAR
        
        # Primary wave: very slow (50-year period), small amplitude
        primary_wave = 0.02 * np.sin(2 * np.pi * years_since_stable / 50)
        
        # Secondary wave: medium (20-year period), smaller amplitude
        secondary_wave = 0.015 * np.sin(2 * np.pi * years_since_stable / 20 + 0.5)
        
        # Tertiary wave: faster but tiny (7-year period)
        tertiary_wave = 0.01 * np.sin(2 * np.pi * years_since_stable / 7 + 1.2)
        
        # Combined fluctuation (will be within ±4.5% typically)
        total_fluctuation = primary_wave + secondary_wave + tertiary_wave
        
        # Clamp to ±5%
        total_fluctuation = np.clip(total_fluctuation, -STABLE_FLUCTUATION_PERCENT, STABLE_FLUCTUATION_PERCENT)
        
        predicted = STABLE_COST * (1 + total_fluctuation)
        
        return max(predicted, floor)


def get_cost_trend_table(model_params, heavy_df, start_year=2020, end_year=2150, step=10):
    """
    Generates a table of predicted costs for visualization.
    """
    years = list(range(start_year, end_year + 1, step))
    predictions = [predict_launch_cost(y, model_params, heavy_df) for y in years]
    return list(zip(years, predictions))


def main():
    print("=" * 70)
    print("Heavy Launch Vehicle - Specific Launch Cost Predictor")
    print("  - Focus: Heavy class rockets")
    print("  - Weighting: 2010+ data emphasized (SpaceX era)")
    print("  - Model: Exponential decay → Smooth convergence → Stable plateau")
    print("=" * 70)
    
    try:
        if len(sys.argv) == 2:
            year = int(sys.argv[1])
        else:
            year_str = input("Enter Year to predict (e.g., 2050): ")
            year = int(year_str)
        
        all_df, heavy_df = load_and_process_data()
        
        if heavy_df.empty:
            print("Error: No Heavy class data found.")
            return
        
        model_params = fit_exponential_decay_model(heavy_df)
        predicted_cost = predict_launch_cost(year, model_params, heavy_df)
        
        print(f"\n{'─' * 50}")
        print(f"Predicted Specific Launch Cost for {year}: ${predicted_cost:.2f} / kg")
        print(f"{'─' * 50}")
        
        # Phase indicator
        if year < CONVERGENCE_START_YEAR:
            phase = "Exponential Decay Phase (Technology improvement)"
        elif year < CONVERGENCE_COMPLETE_YEAR:
            phase = f"Transition Phase ({CONVERGENCE_START_YEAR}-{CONVERGENCE_COMPLETE_YEAR})"
        else:
            phase = f"Stable Plateau Phase (±{STABLE_FLUCTUATION_PERCENT*100:.0f}% fluctuation)"
        print(f"📍 Current Phase: {phase}")
        
        print("\n📊 Heavy Launch Vehicle Historical Data:")
        print(heavy_df[['Entity', 'Year', 'Launch cost per kilogram of payload']].sort_values('Year').to_string(index=False))
        
        recent = heavy_df[heavy_df['Year'] >= 2010]
        if not recent.empty:
            print(f"\n📈 2010+ Average (Heavy): ${recent['Launch cost per kilogram of payload'].mean():.2f} / kg")
            print(f"📉 2010+ Minimum (Falcon Heavy, 2018): ${recent['Launch cost per kilogram of payload'].min():.2f} / kg")
        
        print(f"\n🔧 Model Parameters:")
        print(f"   - Initial Cost (C₀): ${model_params['C0']:.2f} / kg")
        print(f"   - Decay Rate (λ): {model_params['decay_rate']:.4f} / year")
        print(f"   - Cost Floor: ${model_params['floor']:.2f} / kg")
        print(f"   - Stable Plateau: ${STABLE_COST:.2f} / kg (after {CONVERGENCE_COMPLETE_YEAR})")
        
        print(f"\n📅 Cost Trend Preview (Smooth Monotonic Decline → Stable):")
        trend = get_cost_trend_table(model_params, heavy_df)
        prev_cost = None
        for y, c in trend:
            bar = "█" * max(1, int(c / 100))
            # Show trend direction
            if prev_cost is not None:
                if c < prev_cost * 0.99:
                    direction = "↓"
                elif c > prev_cost * 1.01:
                    direction = "↑"
                else:
                    direction = "→"
            else:
                direction = " "
            print(f"   {y}: ${c:>7.2f} / kg {direction} {bar}")
            prev_cost = c
        
    except ValueError as e:
        print(f"Invalid input: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
