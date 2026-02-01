
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import calendar
from matplotlib.ticker import FuncFormatter

# Import prediction modules
from predict_launch_cost import (
    load_and_process_data as load_launch_data,
    fit_exponential_decay_model,
    predict_launch_cost
)

from predict_fuel_price import (
    get_historical_statistics,
    predict_price as predict_fuel_price_value
)

# Configuration
START_YEAR = 2050
TARGET_CARGO_TONS = 100_000_000  # 100 million tons
LAUNCHES_PER_DAY = 10
DEFAULT_PAYLOAD_KG = 150_000
PAYLOAD_PER_LAUNCH_KG = DEFAULT_PAYLOAD_KG
PAYLOAD_PER_LAUNCH_TONS = PAYLOAD_PER_LAUNCH_KG / 1000.0
MAX_YEAR = 2100 # From original script for reuse calculation logic

def get_reuse_count(year):
    """
    Calculate the number of rocket reuses N as a logarithmic function of year.
    Formula: N = 10 + 90 * ln(1 + (year - 2050)) / ln(51)
    """
    if year >= MAX_YEAR:
        return 100
    if year <= START_YEAR:
        return 10
    
    # Logarithmic growth: 10 at 2050, 100 at 2100
    delta = year - START_YEAR
    n = 10 + 90 * np.log(1 + delta) / np.log(51)
    return int(round(n))

def format_large_currency(x, pos):
    """Format currency in Billions/Trillions"""
    if x >= 1e12:
        return f'${x/1e12:.1f}T'
    elif x >= 1e9:
        return f'${x/1e9:.1f}B'
    elif x >= 1e6:
        return f'${x/1e6:.1f}M'
    return f'${x:.0f}'

def format_large_mass(x, pos):
    """Format mass in Millions"""
    return f'{x/1e6:.0f}M'

def main():
    print("Initializing Simulation...")
    print(f"Target: Transport {TARGET_CARGO_TONS/1e6} million tons starting from {START_YEAR}")
    print(f"Rate: {LAUNCHES_PER_DAY} launches/day * {PAYLOAD_PER_LAUNCH_TONS} tons/launch = {LAUNCHES_PER_DAY * PAYLOAD_PER_LAUNCH_TONS} tons/day")

    # --- Pre-load Models (Optimization) ---
    print("Pre-loading models and data...")
    try:
        # Load Launch Cost Model
        _, heavy_df = load_launch_data()
        launch_model_params = fit_exponential_decay_model(heavy_df)
        
        # Load Fuel Price Model
        monthly_df, fuel_model, fuel_stats = get_historical_statistics()
        
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # Initialize simulation variables
    current_date = datetime.date(START_YEAR, 1, 1)
    accumulated_cargo_tons = 0
    accumulated_cost = 0
    propellant_loading_ratio = 0.91 # Default from optimize_rocket_cost.py
    
    # Data storage for plotting
    history = {
        'date': [],
        'fuel_price': [],
        'rocket_construction_cost': [],
        'cumulative_cargo': [],
        'cumulative_cost': []
    }
    
    print("Starting Monthly Simulation...")
    
    # Simulation Loop
    while accumulated_cargo_tons < TARGET_CARGO_TONS:
        year = current_date.year
        month = current_date.month
        
        # Get days in current month
        days_in_month = calendar.monthrange(year, month)[1]
        
        # --- Optimized Calculation Step ---
        
        # 1. Get Dynamic Parameters
        n_reuses = get_reuse_count(year)
        
        # 2. Predict Costs
        try:
            construction_price = predict_launch_cost(year, launch_model_params, heavy_df)
            fuel_price = predict_fuel_price_value(year, month, monthly_df, fuel_model, fuel_stats)
        except Exception as e:
            print(f"Error predicting for {year}-{month}: {e}")
            break
            
        # 3. Apply Formula
        # Total = Payload * (Construction/N + propellant_loading_ratio * (0.3 * FuelPrice + 0.7 * 0.15) + 0.01 * Construction)
        
        term1 = construction_price / n_reuses
        term2 = propellant_loading_ratio * (0.3 * fuel_price + 0.7 * 0.15)
        term3 = 0.01 * construction_price
        
        cost_per_kg = term1 + term2 + term3
        unit_launch_cost = PAYLOAD_PER_LAUNCH_KG * cost_per_kg
        
        # --------------------------------
        
        # Monthly totals
        monthly_launches = LAUNCHES_PER_DAY * days_in_month
        monthly_cargo_tons = monthly_launches * PAYLOAD_PER_LAUNCH_TONS
        monthly_cost = monthly_launches * unit_launch_cost
        
        # Check if we overshoot the target in this month
        if accumulated_cargo_tons + monthly_cargo_tons > TARGET_CARGO_TONS:
            remaining_tons = TARGET_CARGO_TONS - accumulated_cargo_tons
            fraction = remaining_tons / monthly_cargo_tons
            
            # Add fractional metrics
            accumulated_cargo_tons += remaining_tons
            accumulated_cost += monthly_cost * fraction
            
            # Approximate the day we finish
            days_needed = int(days_in_month * fraction)
            current_date += datetime.timedelta(days=days_needed)
            
            # Record final state
            history['date'].append(current_date)
            history['fuel_price'].append(fuel_price)
            history['rocket_construction_cost'].append(construction_price)
            history['cumulative_cargo'].append(accumulated_cargo_tons)
            history['cumulative_cost'].append(accumulated_cost)
            
            break
        else:
            accumulated_cargo_tons += monthly_cargo_tons
            accumulated_cost += monthly_cost
            
            # Record state (end of month)
            history['date'].append(current_date)
            history['fuel_price'].append(fuel_price)
            history['rocket_construction_cost'].append(construction_price)
            history['cumulative_cargo'].append(accumulated_cargo_tons)
            history['cumulative_cost'].append(accumulated_cost)
            
            # Advance to next month
            if month == 12:
                current_date = datetime.date(year + 1, 1, 1)
            else:
                current_date = datetime.date(year, month + 1, 1)

        # Progress indicator every 10 years
        if month == 1 and year % 10 == 0:
            print(f"Reached Year {year}: {accumulated_cargo_tons/1e6:.1f}M tons transported.")


    # Final Output Calculation
    end_date = history['date'][-1]
    start_date = datetime.date(START_YEAR, 1, 1)
    total_days = (end_date - start_date).days
    total_years = total_days / 365.25
    
    print("-" * 50)
    print("SIMULATION RESULTS")
    print("-" * 50)
    print(f"Completion Date: {end_date}")
    print(f"Total Time: {total_years:.2f} years ({total_days:,} days)")
    print(f"Total Cost: ${accumulated_cost:,.2f}")
    print(f"Final Cumulative Cargo: {accumulated_cargo_tons:,.2f} tons")
    
    # Plotting
    print("\nGenerating Plots...")
    
    dates = history['date']
    
    # Plotting separate files
    print("\nGenerating separate plots...")
    
    dates = history['date']
    
    # Common style settings
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    
    # 2. Rocket Construction Cost vs Time
    plt.figure()
    plt.plot(dates, history['rocket_construction_cost'], color='green', linewidth=2)
    plt.title('Rocket Construction Cost vs Time')
    plt.grid(True, alpha=0.3)
    plt.ylabel('Cost ($/kg)')
    plt.xlabel('Year')
    plt.tight_layout()
    plt.savefig('rocket_cost_vs_time.png', dpi=300)
    print("Saved rocket_cost_vs_time.png")
    
    # 3. Cumulative Cargo vs Time
    plt.figure()
    plt.plot(dates, history['cumulative_cargo'], color='blue', linewidth=2)
    plt.title('Cumulative Cargo Transported vs Time')
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_large_mass))
    plt.ylabel('Tons')
    plt.xlabel('Year')
    plt.fill_between(dates, history['cumulative_cargo'], color='blue', alpha=0.1)
    plt.tight_layout()
    plt.savefig('cargo_vs_time.png', dpi=300)
    print("Saved cargo_vs_time.png")
    
    # 4. Cumulative Cost vs Time
    plt.figure()
    plt.plot(dates, history['cumulative_cost'], color='red', linewidth=2)
    plt.title('Cumulative Transportation Cost vs Time')
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_large_currency))
    plt.ylabel('Cost ($)')
    plt.xlabel('Year')
    plt.tight_layout()
    plt.savefig('transport_cost_vs_time.png', dpi=300)
    print("Saved transport_cost_vs_time.png")

if __name__ == "__main__":
    main()
