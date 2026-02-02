import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 1. Core Parameters (Strictly follow user's assumptions) ----------------------
# Basic parameters
N = 100000  # Population of the lunar colony (persons)
DAYS_IN_YEAR = 365  # Number of days in a year
L_TO_M3 = 0.001  # Unit conversion: 1 liter = 0.001 cubic meters

# Domestic water consumption parameters
q_d = 45  # Daily domestic water consumption per person (L/person·day)
eta_d = 0.97  # Domestic water recovery rate (97%)

# Agricultural water consumption parameters
q_a = 1.5  # Daily net agricultural water consumption per person (L/person·day)

# Life support & oxygen production parameters
m_o = 0.84  # Daily oxygen demand per person (kg O₂/person·day)
O2_TO_WATER_RATIO = 9  # Theoretical water consumption to produce 1kg O₂ (kg water/kg O₂)
O2_WATER_LOSS_FACTOR = 0.15  # Equivalent net water loss coefficient for oxygen production

# Industrial & energy, construction & maintenance parameters (user's estimation)
Q_i = 2.0e7  # Annual net industrial water consumption (L/year)
Q_c = 1.5e7  # Annual net construction & maintenance water consumption (L/year)

# Ignore lunar local water: set extraction volume to 0
Q_LOCAL_YEAR = 0  

# ---------------------- 2. Core Calculation Function ----------------------
def calculate_steady_water_demand():
    """
    Calculate the annual water demand of the lunar colony in steady operation
    based on the five subsystems defined by the user.
    """
    # (1) Net domestic water consumption
    Q_d_total = N * q_d * DAYS_IN_YEAR  # Total annual domestic water circulation (L/year)
    Q_d = Q_d_total * (1 - eta_d)       # Net domestic water consumption (L/year)
    
    # (2) Net agricultural water consumption
    Q_a = N * q_a * DAYS_IN_YEAR        # Net agricultural water consumption (L/year)
    
    # (3) Net water consumption for life support & oxygen production
    M_o = N * m_o * DAYS_IN_YEAR        # Total annual oxygen demand (kg O₂/year)
    Q_l = M_o * O2_TO_WATER_RATIO * O2_WATER_LOSS_FACTOR  # Net life support water consumption (L/year)
    
    # (4) Summarize all subsystems' data (unit: L/year → m³/year for better understanding)
    Q_total_L = Q_d + Q_a + Q_l + Q_i + Q_c
    Q_total_m3 = Q_total_L * L_TO_M3    # Total annual net water consumption (m³/year)
    
    # (5) Water demand without recycling (for comparison)
    Q_gross_no_recycle_L = (Q_d_total) + (Q_a * 25) + (M_o * O2_TO_WATER_RATIO) + (Q_i * 10) + (Q_c * 10)
    Q_gross_no_recycle_m3 = Q_gross_no_recycle_L * L_TO_M3
    
    # (6) Data packaging for return
    return {
        # Net water consumption of each subsystem (m³/year)
        "Q_d_m3": Q_d * L_TO_M3,
        "Q_a_m3": Q_a * L_TO_M3,
        "Q_l_m3": Q_l * L_TO_M3,
        "Q_i_m3": Q_i * L_TO_M3,
        "Q_c_m3": Q_c * L_TO_M3,
        # Core output results
        "total_annual_consumption_m3": Q_total_m3,  # Total annual water consumption (m³/year)
        "transport_water_after_recycle_m3": Q_total_m3,  # Transport water after recycling (m³/year)
        "total_consumption_without_recycle_m3": Q_gross_no_recycle_m3
    }

# ---------------------- 3. Plot Annual Water Consumption Bar Chart ----------------------
def plot_annual_water_bar(water_data):
    """
    Plot a bar chart of annual net water consumption for each subsystem
    in a fully operational lunar colony.
    """

    import matplotlib.pyplot as plt

    # Configure font and rendering
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False

    # Subsystem labels (English)
    subsystems = [
        "Residential Domestic Water Use",
        "Agricultural System Water Use",
        "Life Support and Oxygen Production",
        "Industrial and Energy Systems",
        "Construction and Maintenance Systems"
    ]

    # Annual net water consumption data (m³/year)
    water_volumes = [
        water_data["Q_d_m3"],
        water_data["Q_a_m3"],
        water_data["Q_l_m3"],
        water_data["Q_i_m3"],
        water_data["Q_c_m3"]
    ]

    # Create figure
    plt.figure(figsize=(13, 6))

    bars = plt.bar(
        subsystems,
        water_volumes,
        edgecolor="black",
        linewidth=0.8
    )

    # Add numeric labels above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height * 1.01,
            f"{height:,.0f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    # Axis labels and title
    plt.xlabel("Water Consumption Subsystem", fontsize=12)
    plt.ylabel("Annual Net Water Consumption (m³/year)", fontsize=12)
    plt.title(
        "Annual Net Water Consumption by Subsystem\n"
        "Fully Operational 100,000-Person Lunar Colony",
        fontsize=14
    )

    # Grid and layout adjustments
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=15, fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.show()

# ---------------------- 4. Execute Calculation & Output Core Results ----------------------
if __name__ == "__main__":
    # Calculate steady-state water demand
    water_result = calculate_steady_water_demand()
    
    # Output core conclusions (total annual consumption & transport water after recycling)
    print("="*100)
    print("CORE CONCLUSIONS OF WATER DEMAND FOR LUNAR COLONY IN STEADY OPERATION")
    print("="*100)
    print(f"1. Total annual water consumption of the lunar colony: {water_result['total_annual_consumption_m3']:,.2f} cubic meters")
    print(f"2. Water volume to be transported from Earth after recycling: {water_result['transport_water_after_recycle_m3']:,.2f} cubic meters")
    print("="*100)
    
    # Plot annual water consumption bar chart
    plot_annual_water_bar(water_result)