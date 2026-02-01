"""
Total Transportation Cost Calculator for Space Elevator + Lunar Tug (Scheme A)
Corrected: Lunar transfer cost (Apex Anchor to Moon) - aligned with low Δv physics
Currency: All in US Dollars ($)
Comments: English
Key Modifications:
1. Added exponential decay for operational costs (1% of initial elevator cost) and rocket construction costs
2. Added year-based total cost & cargo calculation function
3. Structured output with breakdown of all cost components
4. Fixed annual cargo capacity (537,000 tons/year = 537,000,000 kg/year)
"""

import sys
import math
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# Physical & Economic Constants (SI Units / US Dollars)
# ==============================================================================
# Earth physical constants
G0 = 9.81               # Earth surface gravity (m/s²)
R_E = 6.371e6           # Earth radius (m)
OMEGA = 2 * math.pi / 86400  # Earth's angular velocity (rad/s)
H_ELEVATOR = 35786e3    # Space elevator height (m)
ANNUAL_CARGO_CAPACITY = 537000 * 1000  # Fixed annual cargo: 537,000 tons = 537,000,000 kg/year

# Space elevator parameters
NUM_ELEVATORS = 3               # Number of space elevators
C_ELEVATOR_LOW = 10e9 * 10       # Lower bound of single elevator construction cost ($)
C_ELEVATOR_HIGH = 20e9 * 10      # Upper bound of single elevator construction cost ($)
TOTAL_TRANSPORT_Q = 1e11        # Total transport quantity (kg) - 100,000,000 tons
ETA = 0.8                       # Energy conversion efficiency (80%)
ELECTRIC_PRICE = 0.15           # Electricity price ($/kWh) - US industrial average
OP_COST_INITIAL_RATIO = 0.01    # Initial operational cost ratio (1% of elevator construction cost)
OP_COST_DECAY_RATE = 0.05       # Annual exponential decay rate for operational costs (5%)

# Lunar TUG parameters (Apex Anchor to Moon - corrected for low Δv)
M_PAYLOAD_TUG = 1e5             # Single tug payload (kg) - 100 tons 
# Increased initial tug construction cost (mature phase baseline + 20%) for decay calculation
C_TUG_BUILD_INITIAL = 5000e4  # Initial tug construction cost ($)
N_REUSE = 50                    # Tug reuse times
DELTA_V_CORRECTED = 1640        # Corrected Δv: 1.64 km/s = 1640 m/s
I_SP_CORRECTED = 450            # LH2/LOX engine vacuum Isp (s)
C_FUEL = 2.88                   # Fuel price: LH2 ($/kg)
FUEL_RATIO_PER_PAYLOAD = 0.495  # Fuel required per kg payload (kg)
TUG_COST_DECAY_RATE = 0.07      # Annual exponential decay rate for tug construction cost (7%)

# Base year
BASE_YEAR = 2050

# ==============================================================================
# Core Calculation Functions
# ==============================================================================
def calculate_equivalent_gravity():
    """Calculate equivalent gravitational acceleration (g_avg) via integral derivation"""
    # Integral of gravitational acceleration term
    grav_term = G0 * R_E * H_ELEVATOR / (R_E + H_ELEVATOR)
    # Integral of centrifugal acceleration term
    centri_term = OMEGA**2 * (R_E * H_ELEVATOR + (H_ELEVATOR**2)/2)
    # Average net acceleration
    g_avg = (grav_term - centri_term) / H_ELEVATOR
    return g_avg

def calculate_electric_cost_per_kg(g_avg):
    """Calculate electricity cost per kg for space elevator transport"""
    work_per_kg = 1 * g_avg * H_ELEVATOR  # Work to lift 1kg (J)
    energy_per_kg_j = work_per_kg / ETA   # Actual energy required (J)
    energy_per_kg_kwh = energy_per_kg_j / 3.6e6  # Convert to kWh
    electric_cost_per_kg = energy_per_kg_kwh * ELECTRIC_PRICE
    return energy_per_kg_kwh, electric_cost_per_kg

def calculate_elevator_cost_per_kg(c_per_elevator):
    """Calculate amortized construction cost per kg for space elevators"""
    total_elevator_cost = NUM_ELEVATORS * c_per_elevator
    elevator_cost_per_kg = total_elevator_cost / TOTAL_TRANSPORT_Q
    return elevator_cost_per_kg

def calculate_exponential_decay(initial_value, decay_rate, years_since_base):
    """Calculate exponential decay value over time"""
    if years_since_base <= 0:
        return initial_value
    return initial_value * math.exp(-decay_rate * years_since_base)

def calculate_tug_cost_per_kg(years_since_base):
    """
    Calculate corrected lunar tug cost per kg (Apex to Moon) with exponential decay
    Tsiolkovsky equation: m_fuel = m_payload × (exp(Δv/(G0×I_sp)) - 1)
    """
    # Apply exponential decay to tug construction cost
    tug_build_current = calculate_exponential_decay(
        C_TUG_BUILD_INITIAL, TUG_COST_DECAY_RATE, years_since_base
    )
    # Amortized tug construction cost per launch
    tug_build_amortized = tug_build_current / N_REUSE
    # Calculate fuel mass (corrected Δv and I_sp)
    exp_term = math.exp(DELTA_V_CORRECTED / (G0 * I_SP_CORRECTED))
    fuel_mass_per_tug = M_PAYLOAD_TUG * (exp_term - 1)
    # Fuel cost per tug launch (fixed price)
    fuel_cost_per_tug = fuel_mass_per_tug * C_FUEL
    # Total cost per tug launch (amortized build + fuel)
    total_tug_cost = tug_build_amortized + fuel_cost_per_tug
    # Cost per kg payload
    tug_cost_per_kg = total_tug_cost / M_PAYLOAD_TUG
    
    return tug_cost_per_kg, fuel_mass_per_tug, tug_build_current

def calculate_operational_cost(elevator_construction_cost, years_since_base):
    """Calculate annual operational cost with exponential decay"""
    initial_op_cost = elevator_construction_cost * OP_COST_INITIAL_RATIO
    current_op_cost_annual = calculate_exponential_decay(
        initial_op_cost, OP_COST_DECAY_RATE, years_since_base
    )
    # Total operational cost over the years
    total_op_cost = sum(
        calculate_exponential_decay(initial_op_cost, OP_COST_DECAY_RATE, y)
        for y in range(1, years_since_base + 1)
    ) if years_since_base > 0 else 0
    return current_op_cost_annual, total_op_cost

def calculate_total_cost_by_year(target_year):
    """
    Calculate total cost and cargo for space elevator scheme by target year
    Input: target_year (int) - year to calculate
    Output: dictionary with cost breakdown and cargo data (lower/upper bounds)
    """
    # Validate input year
    if target_year < BASE_YEAR:
        raise ValueError(f"Target year must be >= {BASE_YEAR}")
    
    years_since_base = target_year - BASE_YEAR
    total_cargo_kg = years_since_base * ANNUAL_CARGO_CAPACITY
    
    # Core constants
    g_avg = calculate_equivalent_gravity()
    elec_kwh_per_kg, elec_cost_per_kg = calculate_electric_cost_per_kg(g_avg)
    
    # Calculate for lower bound (C_ELEVATOR_LOW)
    ## Elevator construction cost (fixed)
    elev_construction_cost_low = NUM_ELEVATORS * C_ELEVATOR_LOW
    ## Operational cost (exponential decay)
    _, total_op_cost_low = calculate_operational_cost(elev_construction_cost_low, years_since_base)
    ## Electricity cost (fixed price, total over years)
    total_elec_cost_low = total_cargo_kg * elec_cost_per_kg
    ## Tug cost (with decay)
    tug_cost_per_kg_low, fuel_mass_low, tug_build_current_low = calculate_tug_cost_per_kg(years_since_base)
    total_tug_cost_low = total_cargo_kg * tug_cost_per_kg_low
    ## Fuel cost (fixed price, component of tug cost)
    total_fuel_cost_low = (fuel_mass_low / M_PAYLOAD_TUG) * total_cargo_kg * C_FUEL
    ## Tug construction cost (amortized total)
    total_tug_build_cost_low = total_tug_cost_low - total_fuel_cost_low
    ## Total cost breakdown
    total_cost_low = (
        elev_construction_cost_low + 
        total_op_cost_low + 
        total_elec_cost_low + 
        total_tug_cost_low
    )
    cost_per_kg_low = total_cost_low / total_cargo_kg if total_cargo_kg > 0 else 0
    
    # Calculate for upper bound (C_ELEVATOR_HIGH)
    ## Elevator construction cost (fixed)
    elev_construction_cost_high = NUM_ELEVATORS * C_ELEVATOR_HIGH
    ## Operational cost (exponential decay)
    _, total_op_cost_high = calculate_operational_cost(elev_construction_cost_high, years_since_base)
    ## Electricity cost (fixed price, total over years)
    total_elec_cost_high = total_cargo_kg * elec_cost_per_kg
    ## Tug cost (with decay)
    tug_cost_per_kg_high, fuel_mass_high, tug_build_current_high = calculate_tug_cost_per_kg(years_since_base)
    total_tug_cost_high = total_cargo_kg * tug_cost_per_kg_high
    ## Fuel cost (fixed price, component of tug cost)
    total_fuel_cost_high = (fuel_mass_high / M_PAYLOAD_TUG) * total_cargo_kg * C_FUEL
    ## Tug construction cost (amortized total)
    total_tug_build_cost_high = total_tug_cost_high - total_fuel_cost_high
    ## Total cost breakdown
    total_cost_high = (
        elev_construction_cost_high + 
        total_op_cost_high + 
        total_elec_cost_high + 
        total_tug_cost_high
    )
    cost_per_kg_high = total_cost_high / total_cargo_kg if total_cargo_kg > 0 else 0
    
    # Compile results
    results = {
        "target_year": target_year,
        "years_since_base": years_since_base,
        "total_cargo_kg": total_cargo_kg,
        "total_cargo_tons": total_cargo_kg / 1000,
        # Lower bound breakdown
        "lower_bound": {
            "elevator_construction": elev_construction_cost_low,
            "operational": total_op_cost_low,
            "electricity": total_elec_cost_low,
            "tug_construction": total_tug_build_cost_low,
            "tug_fuel": total_fuel_cost_low,
            "total_tug": total_tug_cost_low,
            "total_cost": total_cost_low,
            "cost_per_kg": cost_per_kg_low
        },
        # Upper bound breakdown
        "upper_bound": {
            "elevator_construction": elev_construction_cost_high,
            "operational": total_op_cost_high,
            "electricity": total_elec_cost_high,
            "tug_construction": total_tug_build_cost_high,
            "tug_fuel": total_fuel_cost_high,
            "total_tug": total_tug_cost_high,
            "total_cost": total_cost_high,
            "cost_per_kg": cost_per_kg_high
        }
    }
    
    return results

# ==============================================================================
# Helper Functions
# ==============================================================================
def format_currency(value):
    """Format currency for readability"""
    if value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    elif value >= 1e3:
        return f"${value/1e3:.2f}K"
    else:
        return f"${value:.2f}"

def print_cost_breakdown(results):
    """Print structured cost breakdown output"""
    target_year = results["target_year"]
    total_cargo_tons = results["total_cargo_tons"]
    
    print("=" * 80)
    print(f"Space Elevator + Lunar Tug Cost Summary - Year {target_year}")
    print("=" * 80)
    print(f"Total Cargo Transported: {total_cargo_tons:.2f} tons ({results['total_cargo_kg']:.2e} kg)")
    print(f"Years of Operation: {results['years_since_base']} years (Base Year: {BASE_YEAR})")
    print("-" * 80)
    
    # Lower bound breakdown
    print("\n[LOWER BOUND COSTS]")
    print(f"1. Elevator Construction Cost:    {format_currency(results['lower_bound']['elevator_construction'])}")
    print(f"2. Operational Cost (Total):      {format_currency(results['lower_bound']['operational'])}")
    print(f"3. Electricity Cost (Total):      {format_currency(results['lower_bound']['electricity'])}")
    print(f"4. Tug Construction Cost (Total): {format_currency(results['lower_bound']['tug_construction'])}")
    print(f"5. Tug Fuel Cost (Total):         {format_currency(results['lower_bound']['tug_fuel'])}")
    print(f"6. Total Tug Cost:                {format_currency(results['lower_bound']['total_tug'])}")
    print("-" * 60)
    print(f"Total Cost (Lower Bound):         {format_currency(results['lower_bound']['total_cost'])}")
    print(f"Cost per kg (Lower Bound):        {format_currency(results['lower_bound']['cost_per_kg'])}")
    
    # Upper bound breakdown
    print("\n[UPPER BOUND COSTS]")
    print(f"1. Elevator Construction Cost:    {format_currency(results['upper_bound']['elevator_construction'])}")
    print(f"2. Operational Cost (Total):      {format_currency(results['upper_bound']['operational'])}")
    print(f"3. Electricity Cost (Total):      {format_currency(results['upper_bound']['electricity'])}")
    print(f"4. Tug Construction Cost (Total): {format_currency(results['upper_bound']['tug_construction'])}")
    print(f"5. Tug Fuel Cost (Total):         {format_currency(results['upper_bound']['tug_fuel'])}")
    print(f"6. Total Tug Cost:                {format_currency(results['upper_bound']['total_tug'])}")
    print("-" * 60)
    print(f"Total Cost (Upper Bound):         {format_currency(results['upper_bound']['total_cost'])}")
    print(f"Cost per kg (Upper Bound):        {format_currency(results['upper_bound']['cost_per_kg'])}")
    print("=" * 80)

# ==============================================================================
# Main Execution
# ==============================================================================
def main():
    try:
        # Get target year input
        target_year = int(input(f"Enter target year (>= {BASE_YEAR}): "))
        # Calculate total costs by year
        cost_results = calculate_total_cost_by_year(target_year)
        # Print structured breakdown
        print_cost_breakdown(cost_results)
        
    except ValueError as ve:
        print(f"Input Error: {ve}")
    except Exception as e:
        print(f"Calculation Error: {e}")

if __name__ == "__main__":
    main()