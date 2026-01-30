import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION: Scenario Parameters for 2050+ Prediction
# ==============================================================================
SCENARIO_START_YEAR = 2050
SCENARIO_DURATION_YEARS = 100  # 2050 - 2150

# Price bands for the scenario period
BAND_LOW_MIN = 1.5
BAND_LOW_MAX = 2.0
BAND_HIGH_MIN = 2.0
BAND_HIGH_MAX = 3.0

# Oscillation period in months for band switching (e.g., 60 months = 5 years per cycle)
# One full cycle = low band phase + high band phase
# To ensure 50/50 split, each phase lasts half the period.
OSCILLATION_PERIOD_MONTHS = 120  # 10-year full cycle (5 years low, 5 years high)

# ==============================================================================
# Helper Functions
# ==============================================================================

def get_historical_statistics(file_path='jet_fuel.xlsx'):
    """
    Loads and processes the jet fuel data, focusing on 2005+ data.
    Returns monthly data, a trained linear model, and key statistics.
    """
    df = pd.read_excel(file_path)
    df = df.iloc[1:].copy()
    df.columns = ['Date', 'Price_Per_Gallon']
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Price_Per_Gallon'] = pd.to_numeric(df['Price_Per_Gallon'], errors='coerce')
    df.dropna(inplace=True)
    df.sort_values('Date', inplace=True)
    
    # Aggregate to Monthly
    df.set_index('Date', inplace=True)
    monthly_df = df.resample('ME').mean()
    monthly_df.reset_index(inplace=True)
    
    # Convert to $/Kg (1 US Gallon Jet Fuel ~= 3.06 kg)
    KG_PER_GALLON = 3.06
    monthly_df['Price_Per_Kg'] = monthly_df['Price_Per_Gallon'] / KG_PER_GALLON
    
    # Filter for 2005 onwards
    monthly_df = monthly_df[monthly_df['Date'].dt.year >= 2005].copy()
    
    if monthly_df.empty:
        return None, None, None
    
    # Calculate statistics
    base_date = monthly_df['Date'].min()
    monthly_df['Month_Index'] = (monthly_df['Date'].dt.year - base_date.year) * 12 + \
                                (monthly_df['Date'].dt.month - base_date.month)
    
    X = monthly_df[['Month_Index']]
    y = monthly_df['Price_Per_Kg']
    
    # Fit Linear Trend on 2005+ data
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate residual volatility for oscillation amplitude
    y_pred = model.predict(X)
    residuals = y.values - y_pred.flatten()
    volatility = residuals.std()
    
    stats = {
        'base_date': base_date,
        'volatility': volatility,
        'mean_price': y.mean(),
        'min_price': y.min(),
        'max_price': y.max(),
    }
    
    return monthly_df, model, stats


def predict_price(target_year, target_month, monthly_df, model, stats):
    """
    Predicts the jet fuel price for a given year and month.
    
    For years before 2050: Uses the historical linear trend with oscillation,
    transitioning to the 2050 scenario level.
    For years 2050+: Uses a trend + oscillation model.
      - Trend: Starts at ~2.5x historical mean, grows slowly (change < 1 over 50 years).
      - Oscillation: Small range sinusoidal fluctuation.
    """
    base_date = stats['base_date']
    volatility = stats['volatility']
    mean_price = stats['mean_price']
    
    # Calculate months since base date
    input_index = (target_year - base_date.year) * 12 + (target_month - base_date.month)
    
    # Define 2050 Scenario Parameters
    # 2050 Start Price = approx 2.5x historical mean (User request: 2-3x)
    start_price_2050 = mean_price * 2.5
    
    # Growth Trend: < 1.0 increase over 50 years (2050-2100)
    # Let's target ~0.8 increase by 2100
    growth_rate_per_year = 0.8 / 50.0  # 0.016 $/kg per year
    
    if target_year >= SCENARIO_START_YEAR:
        # Years past 2050
        years_into_scenario = (target_year - SCENARIO_START_YEAR) + (target_month - 1) / 12.0
        
        # 1. Trend Component
        trend = start_price_2050 + growth_rate_per_year * years_into_scenario
        
        # 2. Oscillation Component
        # 10-year period (120 months), amplitude ~0.2
        oscillation = 0.2 * np.sin(2 * np.pi * years_into_scenario / 10.0)
        
        # 3. Micro Noise
        micro_noise = 0.05 * np.sin(input_index * 0.7) + 0.02 * np.cos(input_index * 1.3)
        
        predicted_price = trend + oscillation + micro_noise
        
    else:
        # --- Pre-2050: Transition ---
        # Historical Linear Prediction
        linear_pred = model.predict([[input_index]])[0]
        
        # Historical Oscillation
        # 2-year cycle roughly matching historical volatility
        hist_oscillation = volatility * np.sin(2 * np.pi * (input_index / 24))
        
        if target_year >= 2026:
            # Transition period: 2026 to 2050
            # Blend from (Linear Pred + Hist Osc) -> (Start Price 2050)
            
            total_months_transition = (SCENARIO_START_YEAR - 2026) * 12
            months_passed = (target_year - 2026) * 12 + (target_month - 1)
            transition_progress = months_passed / total_months_transition
            transition_progress = np.clip(transition_progress, 0, 1)
            
            # Use a smoothstep function for smoother transition ?? Or just linear blend
            # Linear blend is robust enough
            
            # Base price moves from linear_projection to start_price_2050
            # We calculate what the scenario would be at exactly 2050 start (trend=start, osc=0 roughly)
            target_2050_val = start_price_2050
            
            # Blend
            base_price = linear_pred * (1 - transition_progress) + target_2050_val * transition_progress
            
            # Dampen historical oscillation as we approach 2050
            oscillation = hist_oscillation * (1 - transition_progress)
            
            # Add a "pre-shock" or transition turbulence? 
            # User asked for trend judgement. Let's keep it simple.
            
            predicted_price = base_price + oscillation
            
        else:
            # < 2026: Pure historical model
            predicted_price = linear_pred + hist_oscillation
            
        # Safety floor
        predicted_price = max(0.5, predicted_price)
    
    return predicted_price


def main():
    print("=" * 70)
    print("Jet Fuel Price Predictor")
    print("  - Training Data: 2005+ historical data")
    print("  - Scenario (2050-2150): Trend (Start ~2.5x HistMean, Growth <1/50y) + Oscillation")
    print("=" * 70)
    
    try:
        # Parse arguments
        if len(sys.argv) == 3:
            year = int(sys.argv[1])
            month = int(sys.argv[2])
        else:
            year_str = input("Enter Year (e.g., 2050): ")
            month_str = input("Enter Month (1-12): ")
            year = int(year_str)
            month = int(month_str)
        
        if month < 1 or month > 12:
            print("Error: Month must be between 1 and 12.")
            return
        
        # Load data and train model
        monthly_df, model, stats = get_historical_statistics()
        
        if monthly_df is None:
            print("Error: Could not load or process historical data.")
            return
        
        # Predict
        price = predict_price(year, month, monthly_df, model, stats)
        
        date_str = f"{year}-{month:02d}"
        print(f"\nPredicted Jet Fuel Price for {date_str}: ${price:.4f} / kg")
        
        # Context info
        if year >= SCENARIO_START_YEAR:
            print(f"(Scenario period: Trend + Oscillation model)")
        else:
            print(f"(Transition period: blending historical trend towards scenario)")
        
        # Show latest historical data
        last_date = monthly_df.iloc[-1]['Date']
        last_price = monthly_df.iloc[-1]['Price_Per_Kg']
        print(f"\nLatest Historical Data ({last_date.strftime('%Y-%m')}): ${last_price:.4f} / kg")
        print(f"Historical Mean (2005+): ${stats['mean_price']:.4f} / kg")
        print(f"Historical Range: ${stats['min_price']:.4f} - ${stats['max_price']:.4f} / kg")
        
    except ValueError as e:
        print(f"Invalid input: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
