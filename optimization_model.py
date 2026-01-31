

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

# Import prediction logic
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
# Constants & Constraints
# ==============================================================================
TARGET_TOTAL_CARGO_KG = 100_000_000 * 1000  # 100 million tons in kg
START_YEAR = 2050
ROCKET_MAX_LAUNCHES_PER_DAY = 10
ROCKET_PAYLOAD_KG = 150_000
SE_UNITS = 3
SE_CAPACITY_PER_UNIT_YEAR_TONS = 179_000
SE_TOTAL_CAPACITY_YEAR_KG = SE_UNITS * SE_CAPACITY_PER_UNIT_YEAR_TONS * 1000
SE_ANNUAL_COST = 100_000  # Fixed annual cost for Space Elevator system

class TransportOptimizationModel:
    def __init__(self):
        self.results = []
        self._init_models()

    def _init_models(self):
        """Pre-load models and data once for efficiency."""
        print("Initializing models...")
        # Launch Cost Model
        _, self.heavy_df = load_launch_data()
        self.launch_model_params = fit_exponential_decay_model(self.heavy_df)
        
        # Fuel Price Model
        self.fuel_monthly_df, self.fuel_model, self.fuel_stats = get_historical_statistics()
        print("Models initialized successfully.")

    def calculate_cost_fast(self, year, month, launches_count):
        """
        Fast cost calculation using pre-loaded models.
        Total = Launches * Payload * Cost_per_kg
        """
        if launches_count == 0:
            return 0
            
        # Get Unit Costs
        construction_cost = predict_launch_cost(year, self.launch_model_params, self.heavy_df)
        fuel_price = predict_fuel_price_value(year, month, self.fuel_monthly_df, self.fuel_model, self.fuel_stats)
        
        # Formula parameters (from calculate_total_cost.py)
        n = 20
        propellant_loading_ratio = 0.91
        payload = ROCKET_PAYLOAD_KG
        
        term1 = construction_cost / n
        term2 = propellant_loading_ratio * (0.3 * fuel_price + 0.7 * 0.15)
        term3 = 0.01 * construction_cost
        
        cost_per_kg = term1 + term2 + term3
        total_launch_cost = launches_count * payload * cost_per_kg
        
        return total_launch_cost

    def simulate_strategy(self, rocket_launches_per_day):
        """
        Simulate the transport process given a fixed daily rocket launch frequency.
        """
        current_year = START_YEAR
        current_month = 1
        remaining_cargo_kg = TARGET_TOTAL_CARGO_KG
        
        total_cost = 0
        total_rocket_cargo = 0
        total_se_cargo = 0
        
        # History log
        history = []
        
        # Monthly capacities (SE is constant per year, but monthly varies by days? No, assume flat monthly rate for SE)
        se_monthly_capacity = SE_TOTAL_CAPACITY_YEAR_KG / 12
        se_monthly_cost = SE_ANNUAL_COST / 12
        
        with tqdm(total=TARGET_TOTAL_CARGO_KG, desc=f"Simulating {rocket_launches_per_day}/day", leave=False) as pbar:
            while remaining_cargo_kg > 0:
                # 1. Determine Rocket Capacity
                days_in_month = pd.Period(f"{current_year}-{current_month}").days_in_month
                rocket_monthly_count = rocket_launches_per_day * days_in_month
                rocket_monthly_capacity = rocket_monthly_count * ROCKET_PAYLOAD_KG
                
                # 2. Allocation Strategy
                # Use SE first (cheaper)
                take_se = min(remaining_cargo_kg, se_monthly_capacity)
                remaining_cargo_after_se = remaining_cargo_kg - take_se
                
                take_rocket = 0
                rocket_cost = 0
                
                if remaining_cargo_after_se > 0 and rocket_monthly_capacity > 0:
                    take_rocket = min(remaining_cargo_after_se, rocket_monthly_capacity)
                    
                    # Calculate actual launches needed
                    needed_launches = math.ceil(take_rocket / ROCKET_PAYLOAD_KG)
                    
                    # Calculate cost using fast function
                    rocket_cost = self.calculate_cost_fast(current_year, current_month, needed_launches)
                
                # 3. Update Totals
                transported = take_se + take_rocket
                remaining_cargo_kg -= transported
                total_cost += (rocket_cost + se_monthly_cost)
                
                total_rocket_cargo += take_rocket
                total_se_cargo += take_se
                
                # 4. Log
                history.append({
                    'Year': current_year,
                    'Month': current_month,
                    'Rocket_Cost': rocket_cost,
                    'SE_Cost': se_monthly_cost,
                    'Transported_kg': transported
                })
                
                pbar.update(int(transported))
                
                # Time increment
                current_month += 1
                if current_month > 12:
                    current_month = 1
                    current_year += 1
                
                if current_year > 2500: # Safety break
                    break
                    
        df_history = pd.DataFrame(history)
        total_years = len(df_history) / 12
        
        return {
            'Strategy (Launches/Day)': rocket_launches_per_day,
            'Total Duration (Years)': total_years,
            'Total Cost (Billions)': total_cost / 1e9,
            'Finish Year': current_year,
            'Rocket Cargo (%)': (total_rocket_cargo / TARGET_TOTAL_CARGO_KG) * 100,
            'SE Cargo (%)': (total_se_cargo / TARGET_TOTAL_CARGO_KG) * 100
        }, df_history


    def run_optimization(self):
        print("Starting Optimization Simulation...")
        summary_results = []
        
        # Try strategies from 0 to 10 launches per day
        # We can also try float values if we interpret average launches/day, but integer is fine for discrete rockets.
        strategies = range(0, ROCKET_MAX_LAUNCHES_PER_DAY + 1)
        
        for n_launch in strategies:
            # If 0 launches, check if SE can handle it (Yes, but slow)
            summary, _ = self.simulate_strategy(n_launch)
            summary_results.append(summary)
            
        self.results_df = pd.DataFrame(summary_results)
        return self.results_df

def main():
    model = TransportOptimizationModel()
    results = model.run_optimization()
    
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS (Pareto Check)")
    print("="*80)
    print(results.to_string(index=False, float_format="%.2f"))
    
    # Identify Key Solutions
    min_cost_idx = results['Total Cost (Billions)'].idxmin()
    min_time_idx = results['Total Duration (Years)'].idxmin()
    
    print("\n" + "-"*40)
    print("RECOMMENDED STRATEGIES")
    print("-"*40)
    
    print(f"1. MINIMIZE COST Strategy:")
    print(results.iloc[min_cost_idx])
    
    print(f"\n2. MINIMIZE TIME Strategy:")
    print(results.iloc[min_time_idx])
    
    # Export results
    results.to_csv('optimization_results.csv', index=False)
    print("\nFull results saved to 'optimization_results.csv'")

if __name__ == "__main__":
    main()
