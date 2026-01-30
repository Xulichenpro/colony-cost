"""
Total Launch Cost Calculator

Formula:
  Total Cost = Payload × (Construction/N + propellant_loading_ratio × (0.3 × FuelPrice + 0.7 × 0.15) + 0.01 × Construction)

Where:
  - Payload: Mass to orbit (default: 150,000 kg)
  - N: Number of reuses (default: 20)
  - Construction: Specific launch cost from predict_launch_cost ($/kg)
  - FuelPrice: Jet fuel price from predict_fuel_price ($/kg)
"""

import sys
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# Import prediction functions from existing modules
# ==============================================================================

# We need to import the core functions from our prediction scripts
# Since they're in the same directory, we can import them directly

from predict_launch_cost import (
    load_and_process_data as load_launch_data,
    fit_exponential_decay_model,
    predict_launch_cost
)

from predict_fuel_price import (
    get_historical_statistics,
    predict_price as predict_fuel_price_value
)


# ==============================================================================
# Default Parameters
# ==============================================================================
DEFAULT_PAYLOAD_KG = 150_000  # 150,000 kg
DEFAULT_N_REUSES = 20        # 20 reuses


def calculate_total_cost(year, month, payload=DEFAULT_PAYLOAD_KG, n=DEFAULT_N_REUSES, propellant_loading_ratio=0.91):
    """
    Calculates the total launch cost using the formula:
    
    Total = Payload × (Construction/N + propellant_loading_ratio × (0.3 × FuelPrice + 0.7 × 0.15) + 0.01 × Construction)
    
    Parameters:
    -----------
    year : int
        Target year
    month : int
        Target month (1-12)
    payload : float
        Payload mass in kg (default: 150,000)
    n : int
        Number of reuses (default: 20)
    propellant_loading_ratio : float
        Propellant loading ratio (default: 0.91)
    
    Returns:
    --------
    dict with all calculation details
    """
    
    # --- Get Construction Cost ($/kg) ---
    _, heavy_df = load_launch_data()
    launch_model_params = fit_exponential_decay_model(heavy_df)
    construction = predict_launch_cost(year, launch_model_params, heavy_df)
    
    # --- Get Fuel Price ($/kg) ---
    monthly_df, fuel_model, fuel_stats = get_historical_statistics()
    fuel_price = predict_fuel_price_value(year, month, monthly_df, fuel_model, fuel_stats)
    
    # --- Apply Formula ---
    # Total = Payload × (Construction/N + 0.89 × (0.3 × FuelPrice + 0.7 × 0.15) + 0.01 × Construction)
    
    term1 = construction / n                           # Construction amortized over N uses
    term2 = propellant_loading_ratio * (0.3 * fuel_price + 0.7 * 0.15)    # Fuel cost component
    term3 = 0.01 * construction                        # Maintenance/other (1% of construction)
    
    cost_per_kg = term1 + term2 + term3
    total_cost = payload * cost_per_kg
    
    return {
        'year': year,
        'month': month,
        'payload_kg': payload,
        'n_reuses': n,
        'propellant_loading_ratio': propellant_loading_ratio,
        'construction_per_kg': construction,
        'fuel_price_per_kg': fuel_price,
        'term1_construction_amortized': term1,
        'term2_fuel_component': term2,
        'term3_maintenance': term3,
        'cost_per_kg': cost_per_kg,
        'total_cost': total_cost,
    }


def format_currency(value):
    """Format large numbers with appropriate units."""
    if value >= 1e9:
        return f"${value/1e9:.2f} Billion"
    elif value >= 1e6:
        return f"${value/1e6:.2f} Million"
    elif value >= 1e3:
        return f"${value/1e3:.2f} Thousand"
    else:
        return f"${value:.2f}"


def main():
    print("=" * 70)
    print("Total Launch Cost Calculator")
    print("  Formula: Payload × (C/N + R×(0.3×F + 0.7×0.15) + 0.01×C)")
    print("  Where: C = Construction Cost, F = Fuel Price, N = Reuses, R = Propellant Loading Ratio")
    print("=" * 70)
    
    try:
        # Parse arguments
        if len(sys.argv) >= 3:
            year = int(sys.argv[1])
            month = int(sys.argv[2])
            payload = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_PAYLOAD_KG
            n = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_N_REUSES
        else:
            year_str = input(f"Enter Year (e.g., 2050): ")
            month_str = input(f"Enter Month (1-12): ")
            payload_str = input(f"Enter Payload in kg (default {DEFAULT_PAYLOAD_KG:,}): ").strip()
            n_str = input(f"Enter N (number of reuses, default {DEFAULT_N_REUSES}): ").strip()
            
            year = int(year_str)
            month = int(month_str)
            payload = int(payload_str) if payload_str else DEFAULT_PAYLOAD_KG
            n = int(n_str) if n_str else DEFAULT_N_REUSES
        
        if month < 1 or month > 12:
            print("Error: Month must be between 1 and 12.")
            return
        
        if n <= 0:
            print("Error: N (reuses) must be positive.")
            return
        
        # Calculate
        result = calculate_total_cost(year, month, payload, n)
        
        # Display results
        print(f"\n{'─' * 50}")
        print(f"📅 Prediction for: {year}-{month:02d}")
        print(f"{'─' * 50}")
        
        print(f"\n📦 Input Parameters:")
        print(f"   • Payload:     {result['payload_kg']:>15,} kg")
        print(f"   • N (reuses):  {result['n_reuses']:>15}")
        
        print(f"\n📊 Predicted Values:")
        print(f"   • Construction (C): ${result['construction_per_kg']:>10.2f} / kg")
        print(f"   • Fuel Price (F):   ${result['fuel_price_per_kg']:>10.4f} / kg")
        
        print(f"\n🔢 Formula Breakdown:")
        print(f"   • C/N (amortized construction):     ${result['term1_construction_amortized']:>10.4f} / kg")
        print(f"   • {result['propellant_loading_ratio']}×(0.3×F + 0.7×0.15) (fuel):   ${result['term2_fuel_component']:>10.4f} / kg")
        print(f"   • 0.01×C (maintenance):             ${result['term3_maintenance']:>10.4f} / kg")
        print(f"   ─────────────────────────────────────────────")
        print(f"   • Total Cost per kg:                ${result['cost_per_kg']:>10.4f} / kg")
        
        print(f"\n💰 TOTAL LAUNCH COST:")
        print(f"   {format_currency(result['total_cost'])}")
        print(f"   (${result['total_cost']:,.2f})")
        
    except ValueError as e:
        print(f"Invalid input: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
