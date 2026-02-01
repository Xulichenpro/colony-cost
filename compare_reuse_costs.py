import sys
import pandas as pd
# Import the calculation function
from calculate_rocket_cost import calculate_total_cost

def main():
    years = [2050, 2060, 2070, 2080, 2090, 2100]
    month = 1
    t_reuses_list = [20, 100]
    
    # Store results for table display
    results = []
    
    print(f"{'='*90}")
    print(f"Reuse Impact Analysis (N=20 vs N=100) - {min(years)}-{max(years)} - Payload: {150000:,} kg")
    print(f"{'='*90}")
    print(f"{'Year':<6} | {'Total Cost (Million $)':<25} | {'Savings (Million $)':<22} | {'Savings (%)':<10}")
    print(f"{'':<6} | {'N=20':<11} | {'N=100':<11} | {'':<22} | {'':<10}")
    print(f"{'-'*90}")
    
    for year in years:
        costs = {}
        
        for n in t_reuses_list:
            result = calculate_total_cost(year, month, n=n)
            costs[n] = result['total_cost']
            
        cost_20 = costs[20]
        cost_100 = costs[100]
        saving_amount = cost_20 - cost_100
        saving_percent = (saving_amount / cost_20) * 100
        
        # Convert to Millions for display
        c20_m = cost_20 / 1e6
        c100_m = cost_100 / 1e6
        sav_m = saving_amount / 1e6
        
        print(f"{year:<6} | ${c20_m:<10.2f} | ${c100_m:<10.2f} | ${sav_m:<21.2f} | {saving_percent:<9.2f}%")
        
        results.append({
            'Year': year,
            'Total Cost (N=20)': cost_20,
            'Total Cost (N=100)': cost_100,
            'Savings': saving_amount,
            'Savings %': saving_percent
        })
        
    print(f"{'-'*90}")
    
    # Optional: Calculate average savings
    avg_saving_percent = sum(r['Savings %'] for r in results) / len(results)
    print(f"\nAverage Savings by switching from 20 to 100 reuses: {avg_saving_percent:.2f}%")
    
    # Check if convergence/stability is visible
    # Taking the construction cost from the last run to see if it stabilized
    last_res = calculate_total_cost(2100, 1, n=20)
    print(f"\nNote: Underlying Construction Cost in 2100: ${last_res['construction_per_kg']:.2f}/kg")
    print(f"      Fuel Price in 2100: ${last_res['fuel_price_per_kg']:.2f}/kg")

if __name__ == "__main__":
    main()
