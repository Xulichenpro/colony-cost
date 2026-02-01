
import pandas as pd
import numpy as np
import sys
import os

# Import prediction logic from existing scripts
from predict_rocket_accident import load_and_clean_data, fit_static_bayesian
from calculate_rocket_cost import calculate_total_cost, format_currency

def get_accident_rate(target_year, data_path='Space_Corrected.csv'):
    """
    Calculate the predicted accident rate for a target year.
    Uses Static Bayesian Model on last 20 years of data + 1% annual decay.
    """
    # Load and filter data (consistent with predict_rocket_accident.py)
    df = load_and_clean_data(data_path)
    max_year = df['Year'].max()
    start_year = max_year - 20
    df_recent = df[df['Year'] >= start_year].copy()
    
    # Fit Bayesian Model
    # Since fit_static_bayesian prints output, we might want to capture or ignore it if we want silent running,
    # but for this script, printed info is good.
    print(f"--- Accident Prediction Model (Base Year: {max_year}) ---")
    theta_bayes = fit_static_bayesian(df_recent)
    
    # Apply Decay
    decay_rate = 0.01
    years_elapsed = target_year - max_year
    if years_elapsed < 0:
        years_elapsed = 0
        
    prob_decayed = theta_bayes * ((1 - decay_rate) ** years_elapsed)
    
    return prob_decayed, theta_bayes

def calculate_annual_cost_with_risk(year, launches_per_day=10):
    """
    Calculate annual cost including risk factor.
    """
    print(f"\n{'='*60}")
    print(f"CALCULATING COST WITH RISK FOR YEAR {year}")
    print(f"{'='*60}")
    
    # 1. Calculate Base Cost (Single Launch)
    # We use month=1 for annual estimation (or could average over 12 months)
    print("\n[1] Calculating Base Launch Cost (No Failures)...")
    base_calc = calculate_total_cost(year=year, month=1) 
    cost_per_launch_ideal = base_calc['total_cost']
    cost_per_kg_ideal = base_calc['cost_per_kg']
    
    # 2. Calculate Accident Rate
    print("\n[2] Predicting Accident Rate...")
    accident_rate, base_rate = get_accident_rate(year)
    success_rate = 1 - accident_rate
    
    # 3. Calculate Risk-Adjusted Cost
    # Effective Cost = Cost / Success_Rate
    # This accounts for the fact that to get 1 successful payload, you need 1/Success_Rate launches on average.
    # Alternatively: Expected Cost = Cost_Success + (Prob_Fail * Cost_Fail_and_Retry) ... simplifies to Cost/P_Success if we assume infinite retries until success.
    
    cost_per_launch_real = cost_per_launch_ideal / success_rate
    cost_per_kg_real = cost_per_kg_ideal / success_rate
    
    # 4. Annual Totals
    days_in_year = 365 # Approx
    total_launches = launches_per_day * days_in_year
    # Expected successes
    expected_successes = total_launches * success_rate
    # Total Cargo Delivered
    total_cargo_kg = expected_successes * base_calc['payload_kg']
    
    # Total Expenditure (Fixed number of launches, some fail)
    # If we fix the number of launches to 10/day, then we just pay for 10/day.
    # But the *effective* cost per kg delivered goes up.
    # Total Annual Cost is simply: Launches * Cost_Per_Launch_Ideal (assuming we pay for the launch regardless of outcome)
    # But usually "Cost with Risk" implies "Cost to deliver X amount".
    # Let's stick to: We launch 10 times a day.
    
    total_annual_spending = total_launches * cost_per_launch_ideal
    
    # Cost per kg delivered (Risk Adjusted)
    # This is the key metric.
    
    print(f"\n{'='*60}")
    print(f"RESULTS FOR {year}")
    print(f"{'='*60}")
    
    print(f"1. Reliability:")
    print(f"   - Base Failure Rate (2024): {base_rate:.4%}")
    print(f"   - Predicted Failure Rate ({year}): {accident_rate:.4%}")
    print(f"   - Success Probability: {success_rate:.4%}")
    print(f"   - Risk Multiplier (1/Success): {1/success_rate:.4f}x")
    
    print(f"\n2. Unit Costs:")
    print(f"   - Ideal Cost per Launch: {format_currency(cost_per_launch_ideal)}")
    print(f"   - REAL Cost per Launch (Effective): {format_currency(cost_per_launch_real)}")
    print(f"     (Cost to ensure 1 successful mission)")
    
    print(f"\n3. Transport Efficiency:")
    print(f"   - Ideal Cost per kg: ${cost_per_kg_ideal:.2f}")
    print(f"   - REAL Cost per kg:  ${cost_per_kg_real:.2f}")
    
    print(f"\n4. Annual Projection ({launches_per_day} launches/day):")
    print(f"   - Total Launches: {total_launches:,}")
    print(f"   - Expected Failures: {int(total_launches * accident_rate):,}")
    print(f"   - Expected Successes: {int(expected_successes):,}")
    print(f"   - Total Cargo Delivered: {total_cargo_kg/1e6:,.2f} Million kg")
    print(f"   - Total Annual Spending: {format_currency(total_annual_spending)}")
    
    return {
        'year': year,
        'accident_rate': accident_rate,
        'ideal_cost_kg': cost_per_kg_ideal,
        'real_cost_kg': cost_per_kg_real,
        'annual_spending': total_annual_spending
    }

def main():
    if len(sys.argv) > 1:
        year = int(sys.argv[1])
    else:
        print("Usage: python calculate_rocket_cost_with_risk.py <year> [daily_launches]")
        print("Defaulting to year 2050...")
        year = 2050
        
    launches = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    calculate_annual_cost_with_risk(year, launches)

if __name__ == "__main__":
    main()
