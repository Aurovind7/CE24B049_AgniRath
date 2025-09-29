import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
import requests

# ==============================================================================
# 1. SETUP: PARAMETERS AND INPUT DATA
# ==============================================================================

def get_vehicle_parameters():
    """Returns a dictionary of all vehicle and environmental constants."""
    return {
        # Vehicle Parameters (As per new requirements)
        "vehicle_mass": 330.0,
        "drag_area_CdA": 0.13,
        "coeff_rolling_resistance_Crr": 0.0045,
        "drivetrain_efficiency": 0.97,
        "regen_efficiency": 0.70, # ADDED: Efficiency of regenerative braking
        
        # Battery (As per new requirements)
        "battery_capacity_joules": 3 * 3.6e6, # 3 kWh
        "initial_SoC": 1.0,
        "min_SoC": 0.03,
        "max_SoC": 1.01, # ADDED: The battery cannot exceed 100% with a buffer
        "max_battery_current_a": 50.0,
        "battery_voltage_v": 120.0,

        # Solar Array (As per new requirements)
        "solar_panel_area": 5.85,
        "solar_panel_efficiency": 0.22,

        # Environmental
        "air_density": 1.225,
        "gravity": 9.81,

        # CHANGED: Replaced chunk-based limits with a physical force limit
        "max_acceleration_force_n": 500.0, # Max force in Newtons for acceleration
        
    }

def get_synthetic_solar_irradiance(time_of_day_hours):
    """
    A more realistic synthetic clear-sky solar model for Southern India (approx. 13° N)
    in May/June.
    """
    # Sunrise/sunset times for Chennai in May/June are approx. 5:45 AM and 6:30 PM
    sunrise = 5.75  # 5:45 AM
    sunset = 18.5   # 6:30 PM
    
    # Peak solar irradiance in India on a clear day is around 1050 W/m^2
    peak_irradiance = 1050.0

    # Create a boolean mask for daylight hours
    daylight_hours = (time_of_day_hours >= sunrise) & (time_of_day_hours <= sunset)
    
    # Calculate the angle for the sinusoidal model based on the new times
    angle = np.pi * (time_of_day_hours - sunrise) / (sunset - sunrise)
    
    # Calculate irradiance, ensuring it's never negative
    sinusoidal_irradiance = peak_irradiance * np.sin(angle)
    sinusoidal_irradiance = np.maximum(0, sinusoidal_irradiance)
    
    # Apply the daylight mask
    irradiance = np.where(daylight_hours, sinusoidal_irradiance, 0.0)
    return irradiance

# REPLACE your load_route_data function with this more robust version

def load_route_data(filename='Agnirath_app/route_data_resampled.csv'):
    """Loads, cleans, and prepares the route data."""
    df = pd.read_csv(filename)
    
    # --- Clean the GPS Data by removing duplicates ---
    is_duplicate = (df['latitude'].diff() == 0) & (df['longitude'].diff() == 0)
    df_cleaned = df[~is_duplicate].reset_index(drop=True)
    print(f"Cleaned route data: Removed {len(df) - len(df_cleaned)} duplicate GPS points.")
    
    # --- ADDED: Smooth the altitude data to remove noise ---
    # A window of 5 means we average the current point with 2 points before and 2 after.
    df_cleaned['altitude_m_smoothed'] = df_cleaned['altitude_m'].rolling(window=5, center=True, min_periods=1).mean()

    # All subsequent calculations use the cleaned and smoothed data
    distances_m = haversine_distance(df_cleaned['latitude'].iloc[:-1].values, 
                                     df_cleaned['longitude'].iloc[:-1].values, 
                                     df_cleaned['latitude'].iloc[1:].values, 
                                     df_cleaned['longitude'].iloc[1:].values)
    
    df_cleaned['segment_distance_m'] = np.insert(distances_m, 0, 0)
    
    # Use the new SMOOTHED altitude for gradient calculation
    altitudes_m = df_cleaned['altitude_m_smoothed'].to_numpy()
    altitude_changes_m = np.diff(altitudes_m)

    sin_theta = np.divide(altitude_changes_m, distances_m, 
                          out=np.zeros_like(distances_m), 
                          where=distances_m!=0)
    
    df_cleaned['gradient_sin_theta'] = np.insert(np.nan_to_num(sin_theta), 0, 0)
    df_cleaned['cumulative_distance_m'] = df_cleaned['segment_distance_m'].cumsum()
    
    return df_cleaned

def haversine_distance(lat1, lon1, lat2, lon2):
    """Helper function to calculate distance between GPS points."""
    R = 6371000  # Earth radius in meters
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    
    # --- THIS IS THE DEFINITIVE FIX ---
    # Clip 'a' to handle potential floating-point inaccuracies where a > 1.
    # This prevents taking the square root of a negative number.
    a = np.clip(a, 0, 1)
    
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# ==============================================================================
# 2. PHYSICS AND ENERGY SIMULATION MODEL
# ==============================================================================

def check_array(name, arr):
    """A helper function to print stats and check for invalid numbers in an array."""
    # Check for NaN or Inf values. This is the most important check.
    if np.isnan(arr).any() or np.isinf(arr).any():
        print(f"  !!!!!! FATAL WARNING: NaN or Inf detected in '{name}'! !!!!!!")
        # Print details about where the invalid values are
        nan_count = np.isnan(arr).sum()
        inf_count = np.isinf(arr).sum()
        print(f"  -> Contains {nan_count} NaN(s) and {inf_count} Inf(s).")
        return True # Return True to indicate an error was found
    
    # If the array is valid, print its basic stats
    print(f"  [OK] Checking {name}: min={np.min(arr):.2f}, max={np.max(arr):.2f}, mean={np.mean(arr):.2f}")
    return False

# REPLACE your run_race_simulation function with this
def run_race_simulation(velocities_kph, route_df, params, start_hour):
    """
    A true dynamic simulation that models acceleration and momentum between segments.
    """
    target_velocities_mps = velocities_kph / 3.6
    num_segments = len(route_df)
    
    # Initialize arrays to store the results of our step-by-step simulation
    actual_velocities_mps = np.zeros(num_segments)
    F_tractive = np.zeros(num_segments)
    P_elec_watts = np.zeros(num_segments)
    delta_t_s = np.zeros(num_segments)
    
    segment_distances = route_df['segment_distance_m'].values
    sin_theta = route_df['gradient_sin_theta'].values
    
    # --- Step-by-Step Dynamic Simulation Loop ---
    for i in range(1, num_segments):
        # The car's current speed is its speed from the end of the last segment
        current_v = actual_velocities_mps[i-1]
        
        # Calculate resistive forces at the CURRENT speed
        F_drag = 0.5 * params['air_density'] * params['drag_area_CdA'] * current_v**2
        F_rolling = params['coeff_rolling_resistance_Crr'] * params['vehicle_mass'] * params['gravity']
        F_gradient = params['vehicle_mass'] * params['gravity'] * sin_theta[i]
        F_resistive = F_drag + F_rolling + F_gradient
        
        # --- Kinematics: Calculate force needed to reach the TARGET speed ---
        target_v = target_velocities_mps[i]
        
        # Use kinematics (vf^2 = vi^2 + 2*a*d) to find the required acceleration 'a'
        # a = (vf^2 - vi^2) / (2*d)
        if segment_distances[i] > 0:
            required_accel = (target_v**2 - current_v**2) / (2 * segment_distances[i])
        else:
            required_accel = 0
            
        # Force needed for that acceleration (F=ma)
        F_inertial = params['vehicle_mass'] * required_accel
        
        # The total tractive force is the force to overcome resistance PLUS the force to accelerate
        total_force_needed = F_resistive + F_inertial
        
        # --- Apply Physical Limits ---
        # The motor can only provide a certain amount of acceleration force
        # and it cannot pull backwards (regen is handled by negative resistive force)
        F_tractive[i] = np.clip(total_force_needed, 0, F_resistive + params['max_acceleration_force_n'])
        
        # Recalculate the actual acceleration and final velocity based on the limited force
        actual_accel = (F_tractive[i] - F_resistive) / params['vehicle_mass']
        final_v_sq = current_v**2 + 2 * actual_accel * segment_distances[i]
        actual_velocities_mps[i] = np.sqrt(max(0, final_v_sq))
        
        # --- Final Power Calculation for this Segment ---
        P_mech_watts = F_tractive[i] * actual_velocities_mps[i]
        propulsion_power = P_mech_watts / params['drivetrain_efficiency']
        regen_power = P_mech_watts * params['regen_efficiency']
        P_elec_watts[i] = np.where(P_mech_watts >= 0, propulsion_power, regen_power)
        
        # Calculate time spent in this segment
        avg_v = (current_v + actual_velocities_mps[i]) / 2
        delta_t_s[i] = segment_distances[i] / (avg_v + 1e-9)

    # --- Energy simulation (this part remains vectorized) ---
    battery_current_a = P_elec_watts / params['battery_voltage_v']
    total_time_s = np.sum(delta_t_s)
    
    time_of_day_hours = start_hour + np.cumsum(delta_t_s) / 3600.0
    time_of_day_wrapped = time_of_day_hours % 24
    
    P_solar_watts = get_synthetic_solar_irradiance(time_of_day_wrapped) * \
                    params['solar_panel_area'] * \
                    params['solar_panel_efficiency']
                    
    P_net_watts = P_solar_watts - P_elec_watts
    delta_energy_joules = P_net_watts * delta_t_s
    
    initial_energy = params['battery_capacity_joules'] * params['initial_SoC']
    cumulative_energy_joules = initial_energy + np.cumsum(delta_energy_joules)
    soc_profile = cumulative_energy_joules / params['battery_capacity_joules']
    
    power_profile = {'solar_gen_W': P_solar_watts, 'elec_cons_W': P_elec_watts, 'actual_velocity_kph': actual_velocities_mps * 3.6}

    return total_time_s, soc_profile, power_profile, battery_current_a, delta_t_s

# ==============================================================================
# 3. OPTIMIZATION SETUP
# ==============================================================================

# ==============================================================================
# ENHANCED DEBUGGING SYSTEM
# ==============================================================================

class AdvancedOptimizationCallback:
    """
    A comprehensive callback that tracks constraints and identifies exactly why optimization fails.
    """
    def __init__(self, max_iterations, constraint_func, constraint_args, bounds, params):
        self.iteration_count = 0
        self.max_iterations = max_iterations
        self.constraint_func = constraint_func
        self.constraint_args = constraint_args
        self.bounds = bounds
        self.params = params
        self.last_valid_x = None
        self.best_constraint_violation = float('inf')
        self.constraint_history = []
        
    def __call__(self, xk):
        """This method is called by the optimizer at each iteration."""
        self.iteration_count += 1
        
        # Check constraints and get detailed violation info
        constraint_violations, violation_details = self.analyze_constraints(xk)
        total_violation = np.sum(np.abs(constraint_violations[constraint_violations < 0]))
        
        # Store the best solution found so far
        if total_violation < self.best_constraint_violation:
            self.best_constraint_violation = total_violation
            self.last_valid_x = xk.copy()
            self.best_violation_details = violation_details
        
        if self.iteration_count % 5 == 0:
            print(f"  ... Iteration {self.iteration_count}/{self.max_iterations} complete.")
            print(f"     Current constraint violation: {total_violation:.6f}")
            
            # Print worst violations every 10 iterations
            if self.iteration_count % 10 == 0 and violation_details:
                worst_violation = violation_details[0]
                print(f"     Worst violation: {worst_violation['type']} - {worst_violation['value']:.4f}")

    def analyze_constraints(self, xk):
        """Analyze exactly which constraints are violated and why."""
        constraints = self.constraint_func(xk, *self.constraint_args)
        violation_details = []
        
        # Analyze SOC constraints
        soc_constraints = constraints[:len(constraints)//3]  # First third are SOC constraints
        min_soc_violations = soc_constraints[soc_constraints < 0]
        if len(min_soc_violations) > 0:
            worst_violation = np.min(min_soc_violations)
            violation_details.append({
                'type': 'Minimum SOC Violation',
                'value': worst_violation,
                'message': f"Battery drops to {self.params['min_SoC'] + worst_violation:.3f} (limit: {self.params['min_SoC']})"
            })
        
        # Analyze Max SOC constraints  
        max_soc_constraints = constraints[len(constraints)//3:2*len(constraints)//3]
        max_soc_violations = max_soc_constraints[max_soc_constraints < 0]
        if len(max_soc_violations) > 0:
            worst_violation = np.min(max_soc_violations)
            violation_details.append({
                'type': 'Maximum SOC Violation', 
                'value': worst_violation,
                'message': f"Battery exceeds {self.params['max_SoC'] - worst_violation:.3f} (limit: {self.params['max_SoC']})"
            })
        
        # Analyze current constraints
        current_constraints = constraints[2*len(constraints)//3:]
        current_violations = current_constraints[current_constraints < 0]
        if len(current_violations) > 0:
            worst_violation = np.min(current_violations)
            violation_details.append({
                'type': 'Battery Current Violation',
                'value': worst_violation, 
                'message': f"Current exceeds {self.params['max_battery_current_a'] - worst_violation:.1f}A (limit: {self.params['max_battery_current_a']}A)"
            })
        
        # Sort by worst violation
        violation_details.sort(key=lambda x: x['value'])
        
        return constraints, violation_details

    def print_final_diagnostic(self, result):
        """Print comprehensive diagnostic information when optimization fails."""
        print("\n" + "="*80)
        print("COMPREHENSIVE OPTIMIZATION FAILURE ANALYSIS")
        print("="*80)
        
        print(f"\nOptimization status: {result.message}")
        print(f"Final function value: {result.fun:.2f}")
        print(f"Iterations: {result.nit}")
        print(f"Constraint violation: {self.best_constraint_violation:.6f}")
        
        if self.last_valid_x is not None:
            print(f"\nBest solution found (violation: {self.best_constraint_violation:.6f}):")
            print(f"  Start hour: {self.last_valid_x[0]:.2f}")
            print(f"  Velocities: {self.last_valid_x[1:].round(1)}")
            
            if hasattr(self, 'best_violation_details') and self.best_violation_details:
                print(f"\nCONSTRAINT VIOLATIONS IN BEST SOLUTION:")
                for i, violation in enumerate(self.best_violation_details[:3]):  # Top 3 violations
                    print(f"  {i+1}. {violation['type']}: {violation['message']}")
            
            # Run simulation on best solution to get detailed diagnostics
            self.detailed_simulation_analysis(self.last_valid_x)
        else:
            print("\n❌ NO VALID SOLUTION FOUND - Constraints too strict or initial guess infeasible")
            self.analyze_initial_feasibility()
    
    def detailed_simulation_analysis(self, xk):
        """Run simulation on the solution and analyze why constraints are violated."""
        start_hour = xk[0]
        chunk_velocities = xk[1:]
        route_df, params, chunk_size_m = self.constraint_args
        
        velocities_kph = map_chunk_velocities_to_segments(chunk_velocities, route_df, chunk_size_m)
        total_time_s, soc_profile, power_profile, battery_current, delta_t_s = run_race_simulation(
            velocities_kph, route_df, params, start_hour
        )
        
        print(f"\nDETAILED SIMULATION ANALYSIS:")
        print(f"  Race time: {total_time_s/3600:.2f} hours")
        print(f"  Min SOC: {np.min(soc_profile)*100:.1f}% (limit: {params['min_SoC']*100}%)")
        print(f"  Max SOC: {np.max(soc_profile)*100:.1f}% (limit: {params['max_SoC']*100}%)") 
        print(f"  Max battery current: {np.max(battery_current):.1f}A (limit: {params['max_battery_current_a']}A)")
        print(f"  Average velocity: {np.mean(velocities_kph):.1f} km/h")
        
        # Identify exactly where violations occur
        if np.min(soc_profile) < params['min_SoC']:
            min_idx = np.argmin(soc_profile)
            print(f"  ❌ SOC violation at segment {min_idx}, distance: {route_df['cumulative_distance_m'].iloc[min_idx]/1000:.1f}km")
        
        if np.max(soc_profile) > params['max_SoC']:
            max_idx = np.argmax(soc_profile) 
            print(f"  ❌ SOC overcharge at segment {max_idx}, distance: {route_df['cumulative_distance_m'].iloc[max_idx]/1000:.1f}km")
            
        if np.max(battery_current) > params['max_battery_current_a']:
            current_idx = np.argmax(battery_current)
            print(f"  ❌ Current violation at segment {current_idx}, distance: {route_df['cumulative_distance_m'].iloc[current_idx]/1000:.1f}km")
    
    def analyze_initial_feasibility(self):
        """Check if the initial guess and bounds are feasible."""
        print(f"\nINITIAL FEASIBILITY ANALYSIS:")
        
        # Check bounds
        print(f"  Start hour bounds: {self.bounds[0]}")
        print(f"  Velocity bounds: {self.bounds[1][0]} to {self.bounds[-1][1]} km/h")
        
        # Check if constraints are physically possible
        route_df, params, chunk_size_m = self.constraint_args
        total_distance = route_df['cumulative_distance_m'].iloc[-1]
        min_energy_required = total_distance * params['vehicle_mass'] * params['gravity'] * params['coeff_rolling_resistance_Crr'] / params['drivetrain_efficiency']
        battery_capacity_joules = params['battery_capacity_joules'] * (params['initial_SoC'] - params['min_SoC'])
        
        print(f"  Total distance: {total_distance/1000:.1f} km")
        print(f"  Minimum energy required: {min_energy_required/3.6e6:.2f} kWh")
        print(f"  Usable battery capacity: {battery_capacity_joules/3.6e6:.2f} kWh")
        
        if min_energy_required > battery_capacity_joules:
            print("  ❌ PHYSICAL IMPOSSIBILITY: Required energy exceeds battery capacity!")
            print("     Solution: Increase battery capacity or reduce distance/rolling resistance")


def map_chunk_velocities_to_segments(chunk_velocities, route_df, chunk_size_m):
    """
    Takes a small array of velocities and maps them to the full route.
    """
    cumulative_dist = route_df['cumulative_distance_m'].values
    segment_chunk_indices = np.floor(cumulative_dist / chunk_size_m).astype(int)
    segment_chunk_indices = np.clip(segment_chunk_indices, 0, len(chunk_velocities) - 1)
    full_velocity_profile = chunk_velocities[segment_chunk_indices]
    
    # Force start at 0, which is physically correct. The simulation will handle the ramp up.
    full_velocity_profile[0] = 0.0 
    
    return full_velocity_profile

# REPLACE the old objective_function with this one
def objective_function(decision_vars, route_df, params, chunk_size_m): # No interpolator
    start_hour = decision_vars[0]
    chunk_velocities = decision_vars[1:]
    velocities_kph = map_chunk_velocities_to_segments(chunk_velocities, route_df, chunk_size_m)
    total_time_s, _, _, _, _ = run_race_simulation(
        velocities_kph, route_df, params, start_hour
    )
    return total_time_s

# ==============================================================================
# IMPROVED CONSTRAINT FUNCTION WITH BETTER DEBUGGING
# ==============================================================================

def constraint_function(decision_vars, route_df, params, chunk_size_m):
    """
    The constraint function with better error handling and validation.
    """
    try:
        start_hour = decision_vars[0]
        chunk_velocities = decision_vars[1:]
        
        # Validate inputs
        if start_hour < 0 or start_hour > 24:
            raise ValueError(f"Invalid start hour: {start_hour}")
        
        if np.any(chunk_velocities < 0):
            raise ValueError("Negative velocities detected")
        
        velocities_kph = map_chunk_velocities_to_segments(chunk_velocities, route_df, chunk_size_m)
        
        _, soc_profile, _, battery_current_profile, _ = run_race_simulation(
            velocities_kph, route_df, params, start_hour
        )
        
        # Constraint 1: SoC must be ABOVE the minimum
        c1_min_soc = soc_profile - params['min_SoC']
        
        # Constraint 2: SoC must be BELOW the maximum  
        c2_max_soc = params['max_SoC'] - soc_profile
        
        # Constraint 3: Battery current must be below the maximum
        c3_battery_current = params['max_battery_current_a'] - battery_current_profile
        
        return np.hstack([c1_min_soc, c2_max_soc, c3_battery_current])
        
    except Exception as e:
        print(f"❌ Constraint function error: {e}")
        # Return a large violation to guide optimizer away from problematic regions
        return np.array([-1000.0])
    

# ==============================================================================
# 4. MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == '__main__':
    # --- Load Inputs ---
    vehicle_params = get_vehicle_parameters()
    route_data = load_route_data()
    
    # --- Define Optimization Problem ---
    CHUNK_SIZE_M = 500
    num_chunks = int(np.ceil(route_data['cumulative_distance_m'].iloc[-1] / CHUNK_SIZE_M))
    
    print("--- AgniRath Race Strategy Optimization (Dynamic Model) ---")
    print(f"Route distance: {route_data['cumulative_distance_m'].iloc[-1] / 1000:.1f} km")
    print(f"Optimizing {num_chunks} velocity chunks + 1 start time variable.")
    
    initial_start_hour = 10.0
    initial_velocities = np.full(num_chunks, 60.0)
    initial_guess = np.insert(initial_velocities, 0, initial_start_hour)
    
    start_hour_bounds = (8.0, 13.0)
    velocity_bounds = [(10.0, 130.0) for _ in range(num_chunks)]
    bounds = [start_hour_bounds] + velocity_bounds
    
    # --- Setup Arguments and Constraints ---
    args_tuple = (route_data, vehicle_params, CHUNK_SIZE_M)
    
    # The constraints list is now simpler, containing only the main simulation constraint
    constraints = [{'type': 'ineq', 'fun': constraint_function, 'args': args_tuple}]
    # REMOVED: The accel_constraint_def line has been deleted.

    # --- Setup the Progress Callback ---
    MAX_ITER = 20
    progress_callback = AdvancedOptimizationCallback(
        max_iterations=MAX_ITER,
        constraint_func=constraint_function,
        constraint_args=args_tuple,
        bounds=bounds,
        params=vehicle_params
    )
    
    # --- Run the Optimizer ---
    print("\nStarting optimization...")
    start_time = time.time()
    
    result = minimize(
        objective_function,
        initial_guess,
        args=args_tuple,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints, # Use the simplified constraints list
        callback=progress_callback,
        options={'disp': True, 'maxiter': MAX_ITER, 'ftol': 1e-6}
    )
    
    end_time = time.time()
    print(f"Optimization finished in {end_time - start_time:.2f} seconds.")

    # --- Run Final Analysis and Plotting ---
    if result is not None:
        progress_callback.print_final_diagnostic(result)
        
        solution_x = None
        if result.success:
            solution_x = result.x
        elif progress_callback.last_valid_x is not None:
            solution_x = progress_callback.last_valid_x

        if solution_x is not None:
            # Rerun the simulation with the final solution to get all data for plotting
            optimal_start_hour = solution_x[0]
            optimal_chunk_velocities = solution_x[1:]
            
            # NOTE: This is the TARGET profile. The SIMULATION will give the ACTUAL profile.
            target_velocities_kph = map_chunk_velocities_to_segments(
                optimal_chunk_velocities, route_data, CHUNK_SIZE_M
            )
            
            final_time_s, final_soc, final_power, _, _ = run_race_simulation(
                target_velocities_kph, route_data, vehicle_params, optimal_start_hour
            )
            
            # Get the actual velocity profile, which includes the effects of acceleration limits
            actual_velocities_kph = final_power['actual_velocity_kph']
            
            # --- Visualization ---
            distance_km = route_data['cumulative_distance_m'] / 1000.0
            
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
            fig.suptitle('Optimal Race Strategy Analysis (Dynamic Model)', fontsize=16)
            
            # Plot the ACTUAL velocity
            ax[0, 0].plot(distance_km, actual_velocities_kph, label='Actual Velocity', color='dodgerblue')
            ax[0, 0].plot(distance_km, target_velocities_kph, label='Target Velocity', color='lightskyblue', linestyle='--')
            ax[0, 0].set_ylabel('Velocity (km/h)')
            ax[0, 0].set_title('Velocity Profile')
            ax[0, 0].legend()
                
            # 2. State of Charge (SoC)
            ax[1, 0].plot(distance_km, final_soc * 100, label='Battery SoC', color='limegreen')
            ax[1, 0].set_ylabel('State of Charge (%)')
            ax[1, 0].set_title('Battery State of Charge')
            ax[1, 0].axhline(y=vehicle_params['min_SoC']*100, color='r', linestyle='--', label=f"{vehicle_params['min_SoC']*100}% Limit")
            ax[1, 0].set_xlabel('Distance (km)')
            ax[1, 0].legend()
            
            # 3. Power Balance
            ax[0, 1].plot(distance_km, final_power['solar_gen_W'] / 1000, label='Solar Power Generated', color='gold')
            ax[0, 1].plot(distance_km, final_power['elec_cons_W'] / 1000, label='Electrical Power Consumed', color='salmon', alpha=0.8)
            ax[0, 1].set_ylabel('Power (kW)')
            ax[0, 1].set_title('Power Balance')
            ax[0, 1].legend()
            # 4. ADDED: Environmental Conditions
            gradient_degrees = np.rad2deg(np.arcsin(route_data['gradient_sin_theta']))
            irradiance_w_m2 = final_power['solar_gen_W'] / (vehicle_params['solar_panel_area'] * vehicle_params['solar_panel_efficiency'])
            ax_env = ax[1, 1]
            ax_solar = ax_env.twinx() # Create a second y-axis
            ax_env.plot(distance_km, gradient_degrees, label='Road Gradient', color='grey', alpha=0.9)
            ax_env.set_ylabel('Gradient (Degrees)', color='grey')
            ax_env.tick_params(axis='y', labelcolor='grey')
            ax_env.set_title('Environmental Conditions')
            ax_env.set_xlabel('Distance (km)')
            ax_solar.plot(distance_km, irradiance_w_m2, label='Solar Irradiance', color='orange', linestyle='--')
            ax_solar.set_ylabel('Irradiance (W/m²)', color='orange')
            ax_solar.tick_params(axis='y', labelcolor='orange')
            # To get legends from both axes to show up
            lines, labels = ax_env.get_legend_handles_labels()
            lines2, labels2 = ax_solar.get_legend_handles_labels()
            ax_solar.legend(lines + lines2, labels + labels2, loc=0)
            
            plt.tight_layout(rect=(0, 0, 1, 0.96))
            plt.show()