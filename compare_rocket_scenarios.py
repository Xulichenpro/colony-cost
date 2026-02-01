
import numpy as np
import pandas as pd
import datetime
import calendar
import math
import sys
import copy

# Import existing modules
from optimization_model import TransportOptimizationModel, TransportProblem, TARGET_TOTAL_CARGO_KG
from predict_rocket_accident import load_and_clean_data, fit_static_bayesian

# ==============================================================================
# Helper: Accident Rate Predictor
# ==============================================================================
class AccidentPredictor:
    def __init__(self):
        print("Initializing Accident Predictor...")
        self.df = load_and_clean_data('Space_Corrected.csv')
        max_year = self.df['Year'].max()
        start_year = max_year - 20
        df_recent = self.df[self.df['Year'] >= start_year].copy()
        self.base_theta = fit_static_bayesian(df_recent)
        self.base_year = max_year
        self.decay_rate = 0.01

    def get_rate(self, year):
        years_elapsed = max(0, year - self.base_year)
        return self.base_theta * ((1 - self.decay_rate) ** years_elapsed)

# Global predictor instance
predictor = None

def get_risk_multiplier(year):
    global predictor
    if predictor is None:
        predictor = AccidentPredictor()
    
    rate = predictor.get_rate(year)
    return 1.0 / (1.0 - rate)

# ==============================================================================
# 1. Rocket Only Simulation (Re-implementation for modification)
# ==============================================================================
def simulate_rocket_only_scenario(with_risk=False):
    """
    Simulate carrying 100M tons using ONLY rockets (10 launches/day).
    Re-implements logic from simulate_only_rocket.py but adds risk factor.
    """
    from simulate_only_rocket import (
        load_launch_data, fit_exponential_decay_model, predict_launch_cost,
        get_historical_statistics, predict_fuel_price_value, get_reuse_count,
        START_YEAR, TARGET_CARGO_TONS, LAUNCHES_PER_DAY, PAYLOAD_PER_LAUNCH_TONS, PAYLOAD_PER_LAUNCH_KG
    )

    print(f"\n--- Simulating Rocket Only (Risk Included: {with_risk}) ---")
    
    # Initialize predictor if needed
    global predictor
    if predictor is None:
        predictor = AccidentPredictor()
    
    # Models
    _, heavy_df = load_launch_data()
    launch_model_params = fit_exponential_decay_model(heavy_df)
    monthly_df, fuel_model, fuel_stats = get_historical_statistics()
    
    current_date = datetime.date(START_YEAR, 1, 1)
    accumulated_cargo_tons = 0
    accumulated_cost = 0
    
    while accumulated_cargo_tons < TARGET_CARGO_TONS:
        year = current_date.year
        month = current_date.month
        days_in_month = calendar.monthrange(year, month)[1]
        
        # Pricing
        n_reuses = get_reuse_count(year)
        construction_price = predict_launch_cost(year, launch_model_params, heavy_df)
        fuel_price = predict_fuel_price_value(year, month, monthly_df, fuel_model, fuel_stats)
        
        # Formula
        term1 = construction_price / n_reuses
        term2 = 0.91 * (0.3 * fuel_price + 0.7 * 0.15)
        term3 = 0.01 * construction_price
        cost_per_kg = term1 + term2 + term3
        
        # Risk Adjustment & Cargo Loss
        if with_risk:
            risk_rate = predictor.get_rate(year)
            # Cost per kg goes up? 
            # Actually, standard cost per launch is the same.
            # But "cost per effective kg" goes up.
            # Here we track raw cost and effective cargo separately.
            
            # Cargo Loss: We only deliver (1 - rate) of the payload
            effective_payload_ratio = 1.0 - risk_rate
        else:
            effective_payload_ratio = 1.0
            
        unit_launch_cost = PAYLOAD_PER_LAUNCH_KG * cost_per_kg
        
        # Monthly totals
        monthly_launches = LAUNCHES_PER_DAY * days_in_month
        # Nominal cargo (what we loaded)
        monthly_cargo_nominal_tons = monthly_launches * PAYLOAD_PER_LAUNCH_TONS
        # Effective cargo (what arrived)
        monthly_cargo_effective_tons = monthly_cargo_nominal_tons * effective_payload_ratio
        
        monthly_cost = monthly_launches * unit_launch_cost
        
        # Check finish
        # We need to reach TARGET based on ACCUMULATED EFFECTIVE CARGO
        if accumulated_cargo_tons + monthly_cargo_effective_tons > TARGET_CARGO_TONS:
            remaining_tons = TARGET_CARGO_TONS - accumulated_cargo_tons
            # Fraction of the month needed based on EFFECTIVE cargo rate
            fraction = remaining_tons / monthly_cargo_effective_tons
            
            accumulated_cargo_tons += remaining_tons
            accumulated_cost += monthly_cost * fraction
            
            days_needed = int(days_in_month * fraction)
            current_date += datetime.timedelta(days=days_needed)
            break
        else:
            accumulated_cargo_tons += monthly_cargo_effective_tons
            accumulated_cost += monthly_cost
            # Advance
            if month == 12:
                current_date = datetime.date(year + 1, 1, 1)
            else:
                current_date = datetime.date(year, month + 1, 1)
                
    total_days = (current_date - datetime.date(START_YEAR, 1, 1)).days
    total_years = total_days / 365.25
    
    return {
        'Total Cost (Billions)': accumulated_cost / 1e9,
        'Total Time (Years)': total_years,
        'Finish Year': current_date.year
    }

# ==============================================================================
# 2. Optimization Model Simulation (Inheritance)
# ==============================================================================
# Import constants for simulation
from optimization_model import START_YEAR, TARGET_TOTAL_CARGO_KG, ROCKET_PAYLOAD_KG, SE_TOTAL_CAPACITY_YEAR_KG 
from tqdm import tqdm

# ==============================================================================
# 2. Optimization Model Simulation (Inheritance)
# ==============================================================================
class RiskAwareOptimizationModel(TransportOptimizationModel):
    def simulate_strategy(self, rocket_launches_per_day, quiet=False):
        """
        Simulate with Risk: Cargo Loss Model.
        Rocket capacity is reduced by failure rate, forcing more time/launches to transport same amount.
        """
        current_year = START_YEAR
        current_month = 1
        remaining_cargo_kg = TARGET_TOTAL_CARGO_KG
        
        total_cost = 0
        total_rocket_cost_acc = 0
        total_se_cost_acc = 0
        total_rocket_cargo = 0
        total_se_cargo = 0
        
        history = []
        
        se_monthly_capacity = SE_TOTAL_CAPACITY_YEAR_KG / 12
        
        if not quiet:
             pbar = tqdm(total=TARGET_TOTAL_CARGO_KG, desc=f"Simulating {rocket_launches_per_day:.2f}/day", leave=False)
        
        while remaining_cargo_kg > 0:
            days_in_month = pd.Period(f"{current_year}-{current_month}").days_in_month
            
            # --- SE Strategy (First Priority) ---
            take_se = min(remaining_cargo_kg, se_monthly_capacity)
            remaining_after_se = remaining_cargo_kg - take_se
            
            # --- Rocket Strategy (Risk Modified) ---
            take_rocket_effective = 0
            rocket_cost = 0
            
            if remaining_after_se > 0:
                # 1. Calculate Theoretical Capacity
                rocket_monthly_count = rocket_launches_per_day * days_in_month
                rocket_monthly_capacity_nominal = rocket_monthly_count * ROCKET_PAYLOAD_KG
                
                # 2. Apply Cargo Loss
                risk_rate = predictor.get_rate(current_year)
                effective_ratio = 1.0 - risk_rate
                rocket_monthly_capacity_effective = rocket_monthly_capacity_nominal * effective_ratio
                
                # 3. Determine how much EFFECTIVE cargo we can take
                take_rocket_effective = min(remaining_after_se, rocket_monthly_capacity_effective)
                
                if take_rocket_effective > 0:
                    # 4. Back-calculate Nominal Load needed -> Launches
                    # If we need 'take_rocket_effective', we must have loaded 'take_rocket_nominal'
                    take_rocket_nominal = take_rocket_effective / effective_ratio
                    
                    needed_launches = math.ceil(take_rocket_nominal / ROCKET_PAYLOAD_KG)
                    
                    # 5. Cost is based on NUMBER OF LAUNCHES (Nominal)
                    # Use base class method (no risk multiplier on unit cost)
                    rocket_cost = self.calculate_cost_fast(current_year, current_month, needed_launches)
            
            # --- SE Cost ---
            se_monthly_cost = self._get_se_monthly_cost(current_year, take_se) if take_se > 0 else 0
            
            # --- Update State ---
            # We subtract EFFECTIVE cargo transported
            transported_effective = take_se + take_rocket_effective
            remaining_cargo_kg -= transported_effective
            
            total_rocket_cost_acc += rocket_cost
            total_se_cost_acc += se_monthly_cost
            total_cost += (rocket_cost + se_monthly_cost)
            
            total_rocket_cargo += take_rocket_effective
            total_se_cargo += take_se
            
            if not quiet:
                pbar.update(int(transported_effective))
            
            # Time increment
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1
            
            if current_year > 2500: # Safety break
                break

        if not quiet:
            pbar.close()
            
        total_months = (current_year - START_YEAR) * 12 + (current_month - 1)
        total_years = total_months / 12.0
        
        return {
            'Strategy (Launches/Day)': rocket_launches_per_day,
            'Total Duration (Years)': total_years,
            'Total Cost (Billions)': total_cost / 1e9,
            'Rocket Cost Weight': total_rocket_cost_acc / total_cost if total_cost > 0 else 0,
            'Elevator Cost Weight': total_se_cost_acc / total_cost if total_cost > 0 else 0,
            'Finish Year': current_year,
            'Rocket Cargo (%)': (total_rocket_cargo / TARGET_TOTAL_CARGO_KG) * 100,
            'SE Cargo (%)': (total_se_cargo / TARGET_TOTAL_CARGO_KG) * 100
        }, pd.DataFrame(history)

def run_optimization_comparison():
    print("\n" + "="*80)
    print("RUNNING OPTIMIZATION (IDEAL SCENARIO) TO FIND STRATEGIES...")
    print("="*80)
    
    # Initialize predictor
    global predictor
    if predictor is None:
        predictor = AccidentPredictor()
    
    # 1. Run Ideal Optimization
    model_ideal = TransportOptimizationModel()
    results_df = model_ideal.run_optimization()
    
    # 2. Extract Strategies
    # Min Cost
    min_cost_idx = results_df['Total Cost (Billions)'].idxmin()
    strategy_min_cost = results_df.iloc[min_cost_idx]
    rate_min_cost = strategy_min_cost['Strategy (Launches/Day)']
    
    # Min Time
    min_time_idx = results_df['Total Duration (Years)'].idxmin()
    strategy_min_time = results_df.iloc[min_time_idx]
    rate_min_time = strategy_min_time['Strategy (Launches/Day)']
    
    # Balanced (Median Cost)
    sorted_df = results_df.sort_values(by='Total Cost (Billions)').reset_index(drop=True)
    median_idx = len(sorted_df) // 2
    strategy_balanced = sorted_df.iloc[median_idx]
    rate_balanced = strategy_balanced['Strategy (Launches/Day)']
    
    strategies = [
        ("Min Cost", rate_min_cost, strategy_min_cost),
        ("Balanced", rate_balanced, strategy_balanced),
        ("Min Time", rate_min_time, strategy_min_time)
    ]
    
    print("\n" + "="*80)
    print("COMPARING STRATEGIES: IDEAL vs RISK")
    print("="*80)
    
    model_risk = RiskAwareOptimizationModel()
    
    comparison_data = []
    
    for name, rate, res_ideal in strategies:
        print(f"\nEvaluating Strategy: {name} ({rate:.4f} launches/day)...")
        
        # We already have Ideal result (res_ideal is a Series, let's normalize it to dict if needed or just use it)
        # However, res_ideal might lack 'Finish Year' or specific keys if the DF columns differ from simulate_strategy return dict.
        # To be safe and consistent, let's re-simulate Ideal too (it's fast).
        
        res_ideal_sim, _ = model_ideal.simulate_strategy(rate, quiet=True)
        res_risk_sim, _ = model_risk.simulate_strategy(rate, quiet=True)
        
        comparison_data.append({
            "Strategy": name,
            "Rate": rate,
            "Ideal": res_ideal_sim,
            "Risk": res_risk_sim
        })
        
    return comparison_data

def create_comparison_table(data, filename="strategy_comparison.png"):
    import matplotlib.pyplot as plt
    
    # Prepare table data
    # Columns: Strategy | Launch Rate | Metric | Ideal | With Risk | Change
    
    rows = []
    colors = []
    
    # Header color
    header_color = '#40466e'
    row_colors = ['#f1f1f2', 'white']
    
    for i, item in enumerate(data):
        name = item['Strategy']
        rate = item['Rate']
        ideal = item['Ideal']
        risk = item['Risk']
        
        # Metrics to show
        metrics = [
            ("Total Cost ($B)", 'Total Cost (Billions)', "${:.2f}"),
            ("Total Time (Years)", 'Total Duration (Years)', "{:.2f}"),
            ("Finish Year", 'Finish Year', "{}")
        ]
        
        # Logic for key names (Time vs Duration compatibility)
        if 'Total Duration (Years)' not in ideal:
             metrics[1] = ("Total Time (Years)", 'Total Time (Years)', "{:.2f}")

        for j, (label, key, fmt) in enumerate(metrics):
            val_ideal = ideal[key]
            val_risk = risk[key]
            
            # Calculate Change
            if isinstance(val_ideal, (int, float)):
                diff = val_risk - val_ideal
                if val_ideal != 0:
                    pct = (diff / val_ideal) * 100
                    change_str = f"{'+' if diff >=0 else ''}{pct:.2f}%"
                else:
                    change_str = "-"
            else:
                change_str = "-"
                
            # Format values
            val_ideal_str = fmt.format(val_ideal)
            val_risk_str = fmt.format(val_risk)
            
            # Row styling
            row_label = name if j == 0 else ""
            rate_label = f"{rate:.2f}/day" if j == 0 else ""
            
            rows.append([row_label, rate_label, label, val_ideal_str, val_risk_str, change_str])
            
            # Color logic (alternate by strategy group)
            colors.append(row_colors[i % 2])

    # Plot
    fig, ax = plt.subplots(figsize=(10, len(rows) * 0.6 + 2))
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=rows,
        colLabels=["Strategy", "Launch Rate", "Metric", "Ideal", "With Risk & Loss", "Change"],
        loc='center',
        cellLoc='center',
        colColours=[header_color] * 6
    )
    
    # Style
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Text formatting
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_color(colors[row-1])
            if col == 0: # Strategy Name bold
                cell.set_text_props(weight='bold')
            if col == 5: # Change column
                txt = cell.get_text().get_text()
                if '+' in txt and txt != "-":
                    cell.set_text_props(color='#d62728', weight='bold') # Red for increase
    
    plt.title("Optimization Strategies: Ideal vs Risk Scenario\n(Considering Rocket Accuracy & Cargo Loss)", 
              pad=20, fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nTable saved to {filename}")

# ==============================================================================
# Main Comparison Logic
# ==============================================================================
def main():
    import sys
    
    # B. Optimization Model Comparison
    comparison_data = run_optimization_comparison()
    
    # D. Visualize
    create_comparison_table(comparison_data, filename="compare_rockect_risk.png")

if __name__ == "__main__":
    main()
