

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

# pymoo imports for NSGA-II
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination

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
# SE_ANNUAL_COST is now dynamic

# ==============================================================================
# Dynamic Reuse Count (Logarithmic Growth)
# ==============================================================================
MAX_YEAR_FOR_REUSE = 2100

def get_reuse_count(year):
    """
    Calculate the number of rocket reuses N as a logarithmic function of year.
    
    N grows from 10 (in 2050) to 100 (in 2100), then stays at 100.
    Formula: N = 10 + 90 * ln(1 + (year - 2050)) / ln(51)
    
    Parameters:
    -----------
    year : int
        Target year
        
    Returns:
    --------
    int: Number of reuses (10 to 100)
    """
    if year >= MAX_YEAR_FOR_REUSE:
        return 100
    if year <= START_YEAR:
        return 10
    
    # Logarithmic growth: 10 at 2050, 100 at 2100
    delta = year - START_YEAR
    n = 10 + 90 * np.log(1 + delta) / np.log(51)
    return int(round(n))

class TransportProblem(ElementwiseProblem):
    """
    Pymoo Problem Definition.
    Variables:
        x[0] = rocket_launches_per_day (Float, 0 to 10)
    Objectives:
        f1 = Total Cost (Billions)
        f2 = Total Time (Years)
    """
    def __init__(self, model_instance):
        self.model = model_instance
        super().__init__(n_var=1, n_obj=2, n_ieq_constr=0, xl=0.0, xu=ROCKET_MAX_LAUNCHES_PER_DAY)

    def _evaluate(self, x, out, *args, **kwargs):
        launch_rate = x[0]
        # Run simulation (quietly)
        summary, _ = self.model.simulate_strategy(launch_rate, quiet=True)
        
        # Objectives
        f1 = summary['Total Cost (Billions)']
        f2 = summary['Total Duration (Years)']
        
        out["F"] = [f1, f2]

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

    # ==============================================================================
    # Space Elevator Cost Constants (from calculate_elevator_cost.py)
    # ==============================================================================
    # Physical constants
    G0 = 9.81               # Earth surface gravity (m/s²)
    R_E = 6.371e6           # Earth radius (m)
    OMEGA = 2 * math.pi / 86400  # Earth's angular velocity (rad/s)
    H_ELEVATOR = 35786e3    # Space elevator height (m)
    
    # Economic constants
    NUM_ELEVATORS = 3
    C_ELEVATOR_LOW = 10e9 * 10       # Lower bound: $100B per elevator
    C_ELEVATOR_HIGH = 20e9 * 10      # Upper bound: $200B per elevator
    # Use average for optimization
    C_ELEVATOR_AVG = (C_ELEVATOR_LOW + C_ELEVATOR_HIGH) / 2  # $150B per elevator
    TOTAL_TRANSPORT_Q = 1e11        # Total transport quantity (kg) - 100 million tons
    ETA = 0.8                       # Energy conversion efficiency
    ELECTRIC_PRICE = 0.15           # Electricity price ($/kWh)
    OP_COST_INITIAL_RATIO = 0.01    # 1% of elevator construction cost
    OP_COST_DECAY_RATE = 0.05       # 5% annual decay
    
    # Lunar Tug constants
    M_PAYLOAD_TUG = 1e5             # 100 tons per tug
    C_TUG_BUILD_INITIAL = 5000e4    # $50M initial tug cost
    DELTA_V_CORRECTED = 1640        # 1.64 km/s
    I_SP_CORRECTED = 450            # LH2/LOX engine Isp (s)
    C_FUEL = 2.88                   # LH2 price ($/kg)
    TUG_COST_DECAY_RATE = 0.07      # 7% annual decay
    
    def _calculate_equivalent_gravity(self):
        """Calculate equivalent gravitational acceleration (g_avg)"""
        grav_term = self.G0 * self.R_E * self.H_ELEVATOR / (self.R_E + self.H_ELEVATOR)
        centri_term = self.OMEGA**2 * (self.R_E * self.H_ELEVATOR + (self.H_ELEVATOR**2)/2)
        g_avg = (grav_term - centri_term) / self.H_ELEVATOR
        return g_avg
    
    def _calculate_electric_cost_per_kg(self):
        """Calculate electricity cost per kg for space elevator transport"""
        g_avg = self._calculate_equivalent_gravity()
        work_per_kg = g_avg * self.H_ELEVATOR  # Work to lift 1kg (J)
        energy_per_kg_j = work_per_kg / self.ETA
        energy_per_kg_kwh = energy_per_kg_j / 3.6e6
        electric_cost_per_kg = energy_per_kg_kwh * self.ELECTRIC_PRICE
        return electric_cost_per_kg
    
    def _calculate_exponential_decay(self, initial_value, decay_rate, years_since_base):
        """Calculate exponential decay value over time"""
        if years_since_base <= 0:
            return initial_value
        return initial_value * math.exp(-decay_rate * years_since_base)
    
    def _calculate_tug_cost_per_kg(self, years_since_base, target_year):
        """
        Calculate lunar tug cost per kg with exponential decay and dynamic reuse count.
        """
        # Get dynamic reuse count based on year
        n_reuse = get_reuse_count(target_year)
        
        # Apply exponential decay to tug construction cost
        tug_build_current = self._calculate_exponential_decay(
            self.C_TUG_BUILD_INITIAL, self.TUG_COST_DECAY_RATE, years_since_base
        )
        # Amortized tug construction cost per launch
        tug_build_amortized = tug_build_current / n_reuse
        # Calculate fuel mass using Tsiolkovsky equation
        exp_term = math.exp(self.DELTA_V_CORRECTED / (self.G0 * self.I_SP_CORRECTED))
        fuel_mass_per_tug = self.M_PAYLOAD_TUG * (exp_term - 1)
        # Fuel cost per tug launch
        fuel_cost_per_tug = fuel_mass_per_tug * self.C_FUEL
        # Total cost per tug launch
        total_tug_cost = tug_build_amortized + fuel_cost_per_tug
        # Cost per kg payload
        tug_cost_per_kg = total_tug_cost / self.M_PAYLOAD_TUG
        
        return tug_cost_per_kg, n_reuse
    
    def _calculate_operational_cost_for_year(self, elevator_construction_cost, year):
        """Calculate annual operational cost with exponential decay for a specific year"""
        years_since_base = year - START_YEAR
        initial_op_cost = elevator_construction_cost * self.OP_COST_INITIAL_RATIO
        current_op_cost_annual = self._calculate_exponential_decay(
            initial_op_cost, self.OP_COST_DECAY_RATE, years_since_base
        )
        return current_op_cost_annual
    
    def _get_se_cost_per_kg(self, year, cargo_kg):
        """
        Calculate Space Elevator cost per kg based on calculate_elevator_cost.py logic.
        
        Components:
        1. Elevator construction cost (amortized over total transport quantity)
        2. Operational cost (annual, decaying exponentially)
        3. Electricity cost (per kg)
        4. Tug cost (construction + fuel, with decay)
        
        Returns cost per kg for the given year.
        """
        years_since_base = year - START_YEAR
        
        # 1. Elevator construction cost per kg (amortized over total mission)
        total_elevator_cost = self.NUM_ELEVATORS * self.C_ELEVATOR_AVG
        elevator_cost_per_kg = total_elevator_cost / self.TOTAL_TRANSPORT_Q
        
        # 2. Operational cost per kg for this year
        annual_op_cost = self._calculate_operational_cost_for_year(total_elevator_cost, year)
        # Approximate: spread annual op cost over SE annual capacity
        op_cost_per_kg = annual_op_cost / SE_TOTAL_CAPACITY_YEAR_KG if SE_TOTAL_CAPACITY_YEAR_KG > 0 else 0
        
        # 3. Electricity cost per kg
        elec_cost_per_kg = self._calculate_electric_cost_per_kg()
        
        # 4. Tug cost per kg
        tug_cost_per_kg, n_reuse = self._calculate_tug_cost_per_kg(years_since_base, year)
        
        # Total cost per kg
        total_cost_per_kg = elevator_cost_per_kg + op_cost_per_kg + elec_cost_per_kg + tug_cost_per_kg
        
        return total_cost_per_kg
    
    def _get_se_monthly_cost(self, year, cargo_kg):
        """
        Calculate Space Elevator monthly cost for a given cargo amount.
        Uses the cost per kg model from calculate_elevator_cost.py.
        """
        cost_per_kg = self._get_se_cost_per_kg(year, cargo_kg)
        return cargo_kg * cost_per_kg

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
        
        # Formula parameters (from calculate_rocket_cost.py)
        # N is now dynamic based on year
        n = get_reuse_count(year)
        propellant_loading_ratio = 0.91
        payload = ROCKET_PAYLOAD_KG
        
        term1 = construction_cost / n
        term2 = propellant_loading_ratio * (0.3 * fuel_price + 0.7 * 0.15)
        term3 = 0.01 * construction_cost
        
        cost_per_kg = term1 + term2 + term3
        total_launch_cost = launches_count * payload * cost_per_kg
        
        return total_launch_cost

    def simulate_strategy(self, rocket_launches_per_day, quiet=False):
        """
        Simulate the transport process given a fixed daily rocket launch frequency.
        Variable: rocket_launches_per_day (Optimization variable)
        """
        current_year = START_YEAR
        current_month = 1
        remaining_cargo_kg = TARGET_TOTAL_CARGO_KG
        
        total_cost = 0
        total_rocket_cost_acc = 0
        total_se_cost_acc = 0
        total_rocket_cargo = 0
        total_se_cargo = 0
        
        # History log
        history = []
        
        # Determine fixed monthly capacity for SE (Physical constraint)
        se_monthly_capacity = SE_TOTAL_CAPACITY_YEAR_KG / 12
        
        # Conditional Progress Bar
        iterator_wrapper = range(1000000) # Dummy large number
        if not quiet:
             pbar = tqdm(total=TARGET_TOTAL_CARGO_KG, desc=f"Simulating {rocket_launches_per_day:.2f}/day", leave=False)
        
        while remaining_cargo_kg > 0:
            # 1. Determine Rocket Capacity
            days_in_month = pd.Period(f"{current_year}-{current_month}").days_in_month
            rocket_monthly_count = rocket_launches_per_day * days_in_month
            rocket_monthly_capacity = rocket_monthly_count * ROCKET_PAYLOAD_KG
            
            # 2. Allocation Strategy
            # Use SE first (Optimization rule: SE is generally cheaper and cleaner)
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
            
            # 3. Calculate SE cost based on cargo transported (using new cost model)
            se_monthly_cost = self._get_se_monthly_cost(current_year, take_se) if take_se > 0 else 0
            
            # 4. Update Totals
            transported = take_se + take_rocket
            remaining_cargo_kg -= transported
            
            total_rocket_cost_acc += rocket_cost
            total_se_cost_acc += se_monthly_cost
            total_cost += (rocket_cost + se_monthly_cost)
            
            total_rocket_cargo += take_rocket
            total_se_cargo += take_se
            
            # 4. Log
            if not quiet:
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

        if not quiet:
            pbar.close()
            
        # If quiet (optimization loop), history might be skipped to save memory-time, 
        # but let's just return minimal info. 
        # Calculating duration:
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


    def run_optimization(self):
        print("Starting NSGA-II Optimization...")
        
        problem = TransportProblem(self)
        
        algorithm = NSGA2(
            pop_size=50,
            n_offsprings=20,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(prob=0.1, eta=20),
            eliminate_duplicates=True
        )
        
        termination = get_termination("n_gen", 40)
        
        res = minimize(
            problem,
            algorithm,
            termination,
            seed=1,
            save_history=True,
            verbose=True
        )
        
        print(f"Optimization finished. Time: {res.exec_time:.2f}s")
        
        # Process Results
        # res.X are the variable values (Launch Rate)
        # res.F are the objective values (Cost, Time)
        
        summary_results = []
        # Sort by Cost for better visualization
        sorted_indices = np.argsort(res.F[:, 0])
        
        for idx in sorted_indices:
            launch_rate = res.X[idx][0]
            # Re-simulate to get full details including weights
            details, _ = self.simulate_strategy(launch_rate, quiet=True)
            summary_results.append(details)
            
        self.results_df = pd.DataFrame(summary_results)
        return self.results_df

def main():
    model = TransportOptimizationModel()
    results = model.run_optimization()
    
    print("\n" + "="*80)
    print("NSGA-II OPTIMIZATION RESULTS (Pareto Front)")
    print("="*80)
    print(results.to_string(index=False, float_format="%.4f"))
    
    # Identify Key Solutions
    min_cost_idx = results['Total Cost (Billions)'].idxmin()
    min_time_idx = results['Total Duration (Years)'].idxmin()
    
    # Simple check for Trade-off (Middle ground) - closest to normalized origin if we normalized
    # Or just pick the median duration
    median_idx = len(results) // 2
    
    print("\n" + "-"*40)
    print("RECOMMENDED STRATEGIES")
    print("-"*40)
    
    print(f"1. MINIMIZE COST Strategy:")
    print(results.iloc[min_cost_idx])
    
    print(f"\n2. MINIMIZE TIME Strategy:")
    print(results.iloc[min_time_idx])

    print(f"\n3. BALANCED Strategy (Median):")
    print(results.iloc[median_idx])
    
    # Export results
    results.to_csv('optimization_results_nsga2.csv', index=False)
    print("\nFull results saved to 'optimization_results_nsga2.csv'")

if __name__ == "__main__":
    main()
