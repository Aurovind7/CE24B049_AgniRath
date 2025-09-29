import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

# ==============================================================================
# 1. SETUP: PARAMETERS AND INPUT DATA
# ==============================================================================

def get_vehicle_parameters():
    """Returns a dictionary of all vehicle and environmental constants."""
    return {
        # Vehicle Parameters
        "vehicle_mass": 250.0,  # kg (car + driver)
        "drag_area_CdA": 0.12,  # m^2 (Cd * Frontal Area)
        "coeff_rolling_resistance_Crr": 0.006,
        "drivetrain_efficiency": 0.97,  # Motor, controller, and gearbox
        
        # Battery
        "battery_capacity_joules": 5 * 3.6e6,  # 5 kWh converted to Joules
        "initial_SoC": 1.0,  # Start with a full battery (100%)
        "min_SoC": 0.0,      # Minimum allowed SoC

        # Solar Array
        "solar_panel_area": 4.0,       # m^2
        "solar_panel_efficiency": 0.24,  # 24%

        # Environmental Constants
        "air_density": 1.225,  # kg/m^3
        "gravity": 9.81,       # m/s^2
    }

def get_solar_irradiance(time_of_day_hours, latitude_deg=-23.0):
    """
    A simple clear-sky solar irradiance model.
    Models a single race day (e.g., 8 AM to 5 PM) in Australia.
    Returns an array of irradiances in W/m^2.
    """
    sunrise = 8.0  # 8 AM
    sunset = 17.0  # 5 PM

    # Create a boolean "mask" to identify times that are during the day
    daylight_hours = (time_of_day_hours >= sunrise) & (time_of_day_hours <= sunset)

    # Calculate the angle for the sinusoidal model
    # The angle is calculated for all times, but we will only use it for daylight hours
    angle = np.pi * (time_of_day_hours - sunrise) / (sunset - sunrise)

    # Calculate the sinusoidal irradiance
    sinusoidal_irradiance = 1000 * np.sin(angle)

    # Use np.where to apply the condition:
    # where daylight_hours is True, use the calculated irradiance, otherwise use 0.0
    irradiance = np.where(daylight_hours, sinusoidal_irradiance, 0.0)

    return irradiance

def load_route_data(filename='Agnirath_app/route_data_resampled.csv'):
    """Loads and prepares the route data from the CSV file."""
    df = pd.read_csv(filename)
    
    # Calculate segment lengths and gradients from the route data
    # We use np.diff to find the difference between consecutive elements
    distances_m = haversine_distance(df['latitude'].iloc[:-1].values, 
                                     df['longitude'].iloc[:-1].values, 
                                     df['latitude'].iloc[1:].values, 
                                     df['longitude'].iloc[1:].values)
    
    # Prepend a 0 for the start of the first segment
    df['segment_distance_m'] = np.insert(distances_m, 0, 0)
    
    # Gradient is the change in altitude over the distance of the segment
    altitudes_m = df['altitude_m'].to_numpy()
    altitude_changes_m = np.diff(altitudes_m)
    # The sine of the slope angle is rise/run (delta_altitude / segment_distance)
    # We use np.clip to avoid division by zero for any zero-length segments
    sin_theta = np.divide(altitude_changes_m, distances_m, 
                          out=np.zeros_like(distances_m), 
                          where=distances_m!=0)
    df['gradient_sin_theta'] = np.insert(sin_theta, 0, 0)
    
    # The total distance is the cumulative sum of segment distances
    df['cumulative_distance_m'] = df['segment_distance_m'].cumsum()
    
    return df

def haversine_distance(lat1, lon1, lat2, lon2):
    """Helper function to calculate distance between GPS points."""
    R = 6371000  # Earth radius in meters
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# ==============================================================================
# 2. PHYSICS AND ENERGY SIMULATION MODEL
# ==============================================================================

def run_race_simulation(velocities_kph, route_df, params):
    """
    Simulates the entire race for a given velocity profile.
    
    Args:
        velocities_kph (np.array): An array of velocities (km/h) for each segment.
        route_df (pd.DataFrame): The prepared route data.
        params (dict): The vehicle parameters.

    Returns:
        tuple: (total_time_s, soc_profile, power_profile)
    """
    # Convert velocities from km/h to m/s for physics calculations
    velocities_mps = velocities_kph / 3.6
    
    # --- Calculate Forces and Power for ALL segments at once (Vectorization) ---
    
    # Resistive Forces
    F_drag = 0.5 * params['air_density'] * params['drag_area_CdA'] * velocities_mps**2
    F_rolling = params['coeff_rolling_resistance_Crr'] * params['vehicle_mass'] * params['gravity']
    F_gradient = params['vehicle_mass'] * params['gravity'] * route_df['gradient_sin_theta'].values
    
    F_resistive = F_drag + F_rolling + F_gradient
    # Ensure tractive force is never negative (we use brakes for that, not reverse thrust)
    F_tractive = np.maximum(0, F_resistive)

    # Power Calculations
    P_mech_watts = F_tractive * velocities_mps
    P_elec_watts = P_mech_watts / params['drivetrain_efficiency']

    # --- Time and Energy Simulation (Iterative) ---
    
    segment_distances = route_df['segment_distance_m'].values
    # Calculate time taken for each segment (delta_t = delta_d / v)
    # Add a small epsilon to velocity to avoid division by zero
    delta_t_s = segment_distances / (velocities_mps + 1e-9)
    
    total_time_s = np.sum(delta_t_s)
    
    # Calculate time of day for each segment to get solar irradiance
    time_of_day_hours = 8.0 + np.cumsum(delta_t_s) / 3600.0 # Start race at 8 AM
    
    P_solar_watts = get_solar_irradiance(time_of_day_hours) * \
                    params['solar_panel_area'] * \
                    params['solar_panel_efficiency']
    
    # Net power flow for each segment (solar generation - electrical consumption)
    P_net_watts = P_solar_watts - P_elec_watts
    
    # Calculate energy change in Joules for each segment (delta_E = P * delta_t)
    delta_energy_joules = P_net_watts * delta_t_s
    
    # Calculate the battery energy level over the whole race
    initial_energy = params['battery_capacity_joules'] * params['initial_SoC']
    cumulative_energy_joules = initial_energy + np.cumsum(delta_energy_joules)
    
    # Convert energy profile to State of Charge (SoC) profile
    soc_profile = cumulative_energy_joules / params['battery_capacity_joules']
    
    # Package power data for plotting
    power_profile = {
        'solar_gen_W': P_solar_watts,
        'elec_cons_W': P_elec_watts
    }
    
    return total_time_s, soc_profile, power_profile

# ==============================================================================
# 3. OPTIMIZATION SETUP
# ==============================================================================

def objective_function(velocities_kph, route_df, params):
    """The function the optimizer tries to minimize. Returns total race time."""
    total_time_s, _, _ = run_race_simulation(velocities_kph, route_df, params)
    return total_time_s

def constraint_function(velocities_kph, route_df, params):
    """
    The constraint the optimizer must obey.
    Returns the SoC profile. The optimizer will ensure all values are >= min_SoC.
    """
    _, soc_profile, _ = run_race_simulation(velocities_kph, route_df, params)
    # The constraint is soc_profile - min_SoC >= 0
    return soc_profile - params['min_SoC']

# ==============================================================================
# 4. MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == '__main__':
    # --- Load Inputs ---
    vehicle_params = get_vehicle_parameters()
    route_data = load_route_data()
    num_segments = len(route_data)
    
    print("--- AgniRath Race Strategy Optimization ---")
    print(f"Route distance: {route_data['cumulative_distance_m'].iloc[-1] / 1000:.1f} km")
    print(f"Number of segments: {num_segments}")
    
    # --- Define Optimization Problem ---
    # The decision variables are the velocities for each segment
    
    # 1. Initial Guess: A constant velocity of 80 km/h
    initial_velocity_guess = np.full(num_segments, 80.0)
    
    # 2. Bounds: Velocity must be between 10 km/h and 130 km/h
    velocity_bounds = [(10.0, 130.0) for _ in range(num_segments)]
    
    # 3. Constraints: The constraint function (SoC) must be non-negative
    soc_constraint = {'type': 'ineq', 'fun': constraint_function, 'args': (route_data, vehicle_params)}
    
    # --- Run the Optimizer ---
    print("\nStarting optimization... (This may take a few minutes)")
    start_time = time.time()
    
    # We use the SLSQP (Sequential Least Squares Programming) solver,
    # which is well-suited for constrained optimization problems.
    result = minimize(
        objective_function,
        initial_velocity_guess,
        args=(route_data, vehicle_params),
        method='SLSQP',
        bounds=velocity_bounds,
        constraints=[soc_constraint],
        options={'disp': True, 'maxiter': 100} # Display progress, limit iterations
    )
    
    end_time = time.time()
    print(f"Optimization finished in {end_time - start_time:.2f} seconds.")

    # --- Process and Display Results ---
    if result.success:
        optimal_velocities_kph = result.x
        
        # Rerun the simulation with the optimal profile to get final data
        final_time_s, final_soc, final_power = run_race_simulation(
            optimal_velocities_kph, route_data, vehicle_params
        )
        
        final_time_hr = final_time_s / 3600.0
        print(f"\n--- Optimal Strategy Found ---")
        print(f"Minimized Race Time: {final_time_hr:.2f} hours")
        
        # --- Visualization ---
        distance_km = route_data['cumulative_distance_m'] / 1000.0
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        fig.suptitle('Optimal Race Strategy Profile', fontsize=16)
        
        # 1. Velocity vs. Distance
        ax1.plot(distance_km, optimal_velocities_kph, label='Optimal Velocity', color='dodgerblue')
        ax1.set_ylabel('Velocity (km/h)')
        ax1.legend()
        
        # 2. SoC vs. Distance
        ax2.plot(distance_km, final_soc * 100, label='Battery SoC', color='limegreen')
        ax2.set_ylabel('State of Charge (%)')
        ax2.axhline(y=vehicle_params['min_SoC']*100, color='r', linestyle='--', label='Min SoC Limit')
        ax2.legend()
        
        # 3. Power vs. Distance
        ax3.plot(distance_km, final_power['solar_gen_W'], label='Solar Power Generated', color='gold')
        ax3.plot(distance_km, final_power['elec_cons_W'], label='Electrical Power Consumed', color='salmon', alpha=0.8)
        ax3.set_xlabel('Distance (km)')
        ax3.set_ylabel('Power (Watts)')
        ax3.legend()
        
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show()

    else:
        print("\nOptimization failed to find a solution.")
        print(result.message)