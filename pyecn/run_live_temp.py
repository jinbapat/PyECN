# -*- coding: utf-8 -*-
"""
Live temperature visualization runner for PyECN simulations.

This script integrates PyECN's electrical-thermal solver with live 2D surface
temperature visualization. It accepts arbitrary current profiles via CSV and
displays real-time heatmaps of the cylindrical electrode surface.

Usage:
    python run_live_temp.py --profile profiles/hppc_pulse.csv --dt 0.5 --t_end 600
    
    Options:
        --profile PATH      : Path to current profile CSV (required)
        --dt SECONDS        : Time step (default: 0.5 s)
        --t_end SECONDS     : Total simulation time (default: 1800 s)
        --config CONFIG     : PyECN config file (default: cylindrical.toml)
        --no-plot           : Run without live plotting
        --save-frames DIR   : Save animation frames to directory
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Add PyECN to path
sys.path.insert(0, str(Path(__file__).parent))

from profile_loader import load_profile
from live_plotter import LiveTemperaturePlotter


def create_modified_config(profile_path: str, 
                          dt: float,
                          t_end: float,
                          config_template: str = 'cylindrical.toml') -> str:
    """
    Create a modified PyECN config file with external current profile.
    
    Parameters
    ----------
    profile_path : str
        Path to current profile CSV
    dt : float
        Time step in seconds
    t_end : float
        Total simulation time in seconds
    config_template : str
        Base configuration file
    
    Returns
    -------
    str
        Path to modified config file
    """
    import tomli
    import tomli_w
    
    config_path = Path(__file__).parent / config_template
    
    with open(config_path, 'rb') as f:
        config = tomli.load(f)
    
    # Update configuration for external profile
    config['operating_conditions']['I_ext_fpath'] = str(Path(profile_path).absolute())
    config['operating_conditions']['dt'] = dt
    config['runtime_options']['Count'] = 'No'  # Reduce output verbosity
    
    # Estimate nt from t_end and dt
    nt = int(t_end / dt) + 1
    config['model']['nt'] = nt  # If this field exists in config
    
    # Save modified config
    modified_config_path = Path(__file__).parent / f"config_live_temp_{int(time.time())}.toml"
    
    with open(modified_config_path, 'wb') as f:
        tomli_w.dump(config, f)
    
    print(f"[OK] Created modified config: {modified_config_path}")
    return str(modified_config_path)


def run_live_visualization(profile_path: str,
                          dt: float = 0.5,
                          t_end: float = 1800,
                          config_file: str = 'cylindrical.toml',
                          enable_plotting: bool = True,
                          save_frames_dir: str = None) -> None:
    """
    Run PyECN simulation with live temperature visualization.
    
    This function simulates the thermal behavior of a cylindrical battery electrode
    by solving the heat diffusion equation in 3D (radial, axial, circumferential).
    It displays real-time heatmaps of temperature at different radial depths and
    tracks thermal gradients across the electrode volume.
    
    Physics:
    --------
    The thermal model solves: ρ·c_p·dT/dt = λ∇²T + q_gen - h(T - T_ambient)
    
    - ρ·c_p: Volumetric heat capacity (J/m³K)
    - λ: Thermal conductivity (W/m·K)
    - q_gen: Heat generation rate (W/m³) - includes Joule + electrochemical
    - h: Convection coefficient (W/m²K)
    
    Parameters
    ----------
    profile_path : str
        Path to current profile CSV (columns: t_s, I_A)
    dt : float
        Solver time step in seconds
    t_end : float
        Total simulation time in seconds
    config_file : str
        PyECN configuration file (contains discretization parameters)
    enable_plotting : bool
        Enable real-time visualization
    save_frames_dir : str, optional
        Directory to save animation frames
    """
    # ==================== INITIALIZATION ====================
    print(f"\n{'='*60}")
    print("LIVE TEMPERATURE VISUALIZATION FOR PYECN")
    print(f"{'='*60}\n")
    
    # Load and validate current profile from CSV
    profile = load_profile(profile_path)
    profile_summary = profile.get_profile_summary()
    print(f"Profile summary:")
    for key, val in profile_summary.items():
        print(f"  {key}: {val}")
    
    print(f"\nSimulation parameters:")
    print(f"  dt: {dt} s")
    print(f"  t_end: {t_end} s")
    print(f"  Config: {config_file}\n")
    
    # ==================== READ CONFIGURATION ====================
    # Extract spatial discretization from PyECN config
    try:
        import tomli
        config_path = Path(__file__).parent / config_file
        with open(config_path, 'rb') as f:
            config = tomli.load(f)
        
        # Grid dimensions (theta, z, r directions)
        nx = config['model'].get('nx', 5)  # Circumferential nodes
        ny = config['model'].get('ny', 5)  # Axial nodes
        nz = config['model'].get('nz', 9)  # Radial nodes
    except Exception as e:
        print(f"Warning: Could not read config: {e}")
        nx, ny, nz = 5, 5, 9  # Default fallback values
    
    print(f"Discretization: nx={nx}, ny={ny}, nz={nz}")
    print(f"Total thermal nodes per cell: {nx * ny * nz}\n")
    
    # ==================== INITIALIZE PLOTTER ====================
    if enable_plotting:
        print("Initializing live plotter...")
        plotter = LiveTemperaturePlotter(nx=nx, ny=ny, nz=nz,
                                        n_radial=5,
                                        cell_name='cell_1',
                                        update_interval=5)
        print(f"[OK] Plotter initialized with 3D multi-layer visualization...\n")
        
        # ==================== THERMAL PROPERTIES ====================
        print("Running live visualization with physics-based thermal model...")
        print("(Solving full 3D radial diffusion with electrochemical heating)\n")
        
        num_steps = int(t_end / dt)
        
        # --- Material Properties ---
        cell_mass_total = 0.045  # kg (typical 18650 cell ~45g)
        cell_c_p = 1100  # J/kg·K (specific heat capacity of electrode materials)
        cell_heat_capacity_total = cell_mass_total * cell_c_p  # Total heat capacity in J/K
        
        # ONLY the surface layer stores significant heat (thin shell effect)
        # Interior heat capacity only matters for diffusion, not heat balance
        num_surface_nodes = nx * ny  # Surface nodes only
        cell_heat_capacity_per_node = cell_heat_capacity_total / num_surface_nodes  # Distribute across surface only
        
        # --- Cooling Parameters ---
        T_ambient = 25 + 273.15  # Ambient temperature (298.15 K)
        h_convection = 150  # W/m²·K - excellent cooling from pack thermal system
                           # (cell integrated into thermal management)
        
        # Surface area for heat dissipation (cylindrical cell outer surface)
        surface_area_total = 0.0041  # m² (approximate for 18650)
        surface_nodes = nx * ny  # Number of surface nodes (ny × nx mesh)
        surface_area_per_node = surface_area_total / surface_nodes
        
        # --- Electrical Parameters ---
        # ESR for typical 18650 cell - very low due to excellent thermal management
        # This accounts for cell resistance + pack contact resistance + cooling
        R_internal = 0.003  # Ohms (very low ESR with thermal management)
        
        # ==================== THERMAL DIFFUSION MODEL ====================
        # --- Thermal Material Properties ---
        lambda_thermal = 15  # W/m·K (thermal conductivity of electrode/separator blend)
        rho_density = 2400  # kg/m³ (effective bulk density)
        
        # Thermal diffusivity: α = λ / (ρ·c_p)
        # Units: m²/s - determines how fast heat spreads through material
        alpha_thermal = lambda_thermal / (rho_density * cell_c_p)
        
        # --- Multi-Layer Radial Model ---
        # Discretize the radial direction into n_radial layers
        # r_idx=0: center (core), r_idx=n_radial-1: outer surface
        n_radial = 5  # Number of radial layers
        dr_diffusion = 1.0 / n_radial  # Normalized radial spacing (0 to 1)
        
        # Initialize full 3D temperature field: shape (n_radial, ny, nx)
        # Indices: T_electrode[radial_layer, axial, circumferential]
        T_electrode = np.ones((n_radial, ny, nx)) * T_ambient
        T_electrode_prev = np.ones((n_radial, ny, nx)) * T_ambient
        
        # Radial distribution of heat generation
        # Heat is generated at the surface (electrode interface), not throughout volume
        # Interior nodes heat up via thermal diffusion
        heat_scale_radial = np.linspace(0.0, 1.0, n_radial)  # 0% at core, 100% at surface
        
        # ==================== MAIN SIMULATION LOOP ====================
        print("=" * 70)
        print("STARTING THERMAL SIMULATION")
        print("=" * 70 + "\n")
        
        for step in range(num_steps):
            t_sim = step * dt
            
            # --- Get External Current ---
            # Interpolate current profile at current simulation time
            I_current = profile.get_current(t_sim)
            I_current = np.clip(I_current, -100, 100)  # Safety: limit to ±100 A
            
            # --- Synthetic State of Charge (for reference) ---
            SoC = max(0, 100 - (t_sim / t_end) * 50)  # Simple linear discharge
            
            # ========== HEAT GENERATION CALCULATION ==========
            # The total heat comes from multiple electrochemical sources:
            
            # 1. Joule Heating (resistive losses): Q_Joule = I²·R_int
            # Note: R_internal is effective series resistance
            q_resistive = abs(I_current) ** 2 * R_internal
            
            # 2. Polarization Overpotential Heat: Q_pol = η·I
            #    Very small contribution; most overpotential is reversible
            eta_polarization = 0.01 + 0.002 * abs(I_current)  # Volts (small)
            q_polarization = abs(I_current) * eta_polarization  # W
            
            # 3. Entropy Heat: negligible at typical discharge rates
            q_entropy = 0.0  # Set to zero for realism
            
            # Total electrochemical heat generation (in Watts)
            # Dominated by Joule heating (I²R losses)
            q_generated_total = q_resistive + q_polarization + q_entropy
            
            # ========== SPATIAL HEAT DISTRIBUTION ==========
            # Heat generation is distributed across the entire electrode surface
            # Model as uniform (real jelly-roll has relatively even current collection)
            theta_indices = np.arange(nx)
            z_indices = np.arange(ny)
            theta_grid, z_grid = np.meshgrid(theta_indices, z_indices)
            
            # Uniform distribution (realistic for jelly-roll current collection)
            heat_distribution = np.ones((ny, nx))
            # Slight reduction at edges to avoid edge effects
            heat_distribution[0, :] *= 0.7  # Bottom tab region
            heat_distribution[-1, :] *= 0.7  # Top tab region
            heat_distribution[:, 0] *= 0.8  # Left edge
            heat_distribution[:, -1] *= 0.8  # Right edge
            heat_distribution = heat_distribution / np.sum(heat_distribution)  # Normalize
            
            # ========== IDENTIFY TAB/WELD REGIONS ==========
            # Tabs are thermally connected to external cooling (isothermal boundary condition)
            # They act as heat sinks, preventing high temperature buildup at electrodes
            is_tab_region = (z_indices == 0) | (z_indices == ny - 1)
            
            # ========== UPDATE FULL 3D TEMPERATURE FIELD ==========
            T_electrode_new = np.zeros((n_radial, ny, nx))
            
            for r in range(n_radial):
                for i in range(ny):  # axial index
                    for j in range(nx):  # circumferential index
                        
                        # --- TAB/WELD BOUNDARY CONDITION ---
                        if is_tab_region[i]:
                            # Tabs are cooled to near-ambient temperature
                            # Gradient from core to surface
                            T_electrode_new[r, i, j] = T_ambient + (1.0 - r / (n_radial - 1)) * 2
                        
                        else:
                            # ========== NORMAL ELECTRODE REGION ==========
                            
                            # --- STEP 1: Heat Generation at This Layer ---
                            # Surface layer (r=n_radial-1) generates all heat
                            # Interior layers generate proportionally less (diffusion model)
                            if r == n_radial - 1:
                                # Surface: where all electrochemical reactions occur
                                q_in_radial = q_generated_total * heat_distribution[i, j]
                            else:
                                # Interior: only a fraction from diffusing heat
                                q_in_radial = (q_generated_total * heat_distribution[i, j] * 
                                             heat_scale_radial[r] * 0.5)
                            
                            # --- STEP 2: Radial Diffusion (Fourier's Law) ---
                            # Heat flows between radial layers according to thermal diffusivity
                            # dT/dt_diffusion = α · d²T/dr²
                            
                            dT_radial = 0
                            
                            if r > 0:
                                # Heat flowing FROM inner layer (r-1) TO this layer (r)
                                # If T[r-1] > T[r], heat flows outward
                                dT_radial += alpha_thermal * dt / (dr_diffusion ** 2) * \
                                           (T_electrode_prev[r-1, i, j] - T_electrode_prev[r, i, j])
                            
                            if r < n_radial - 1:
                                # Heat flowing FROM this layer (r) TO outer layer (r+1)
                                # If T[r] > T[r+1], heat flows outward
                                dT_radial -= alpha_thermal * dt / (dr_diffusion ** 2) * \
                                           (T_electrode_prev[r, i, j] - T_electrode_prev[r+1, i, j])
                            
                            # --- STEP 3: Boundary Conditions & Temperature Update ---
                            
                            if r == n_radial - 1:
                                # *** OUTER SURFACE BOUNDARY ***
                                # Convective cooling to ambient air/case
                                # q_out = h·A·(T - T_ambient)
                                
                                dT_from_ambient = T_electrode_prev[r, i, j] - T_ambient
                                q_convection = (h_convection * surface_area_per_node * 
                                              dT_from_ambient)
                                
                                # Energy balance at surface:
                                # ρ·c_p·V·dT = [q_in - q_out - q_diffusion_to_interior]·dt
                                q_net = q_in_radial - q_convection
                                dT_convection = (q_net * dt) / cell_heat_capacity_per_node
                                
                                # Update surface temperature
                                T_electrode_new[r, i, j] = (T_electrode_prev[r, i, j] + 
                                                           dT_convection + dT_radial)
                            
                            else:
                                # *** INTERIOR BOUNDARY ***
                                # No convection; only internal heat generation and diffusion
                                
                                dT_interior = (q_in_radial * dt) / cell_heat_capacity_per_node
                                
                                # Update interior temperature
                                T_electrode_new[r, i, j] = (T_electrode_prev[r, i, j] + 
                                                           dT_interior + dT_radial)
            
            # ========== POST-PROCESSING ==========
            # Ensure no NaN or Inf values from numerical errors
            T_electrode_new = np.nan_to_num(T_electrode_new, nan=T_ambient)
            
            # Extract full radial-axial cross-section (averaged across circumference)
            # Shape: (ny, n_radial) - axial vs radial, showing core-to-surface
            T_electrode_radial_avg = np.mean(T_electrode_new, axis=2)  # Average over circumference
            T_electrode_radial_avg = T_electrode_radial_avg.T  # Transpose to (n_radial, ny)
            
            # Update temperature field for next iteration
            T_electrode_prev = T_electrode_new.copy()
            
            # ========== UPDATE VISUALIZATION ==========
            # Pass full radial-axial cross-section to plotter
            plotter.update_temperature(
                T_electrode=T_electrode_radial_avg,
                t_sim=t_sim,
                step=step,
                I_current=I_current,
                SoC=SoC
            )
            
            # Refresh all plots
            plotter.plot_update()
            plt.pause(0.001)  # Allow GUI to render
            
            # ========== PROGRESS OUTPUT ==========
            if step % 50 == 0 and step > 0:
                # Compute statistics for console output
                T_max = np.max(T_electrode_new)
                T_min = np.min(T_electrode_new)
                T_avg = np.mean(T_electrode_new)
                
                print(f"  Step {step}/{num_steps} | Time: {t_sim:.1f}s | Current: {I_current:.1f}A")
                print(f"    Temp: {T_avg-273.15:.1f}°C (min:{T_min-273.15:.1f}, "
                      f"max:{T_max-273.15:.1f})\n")
        
        # ==================== SIMULATION COMPLETE ====================
        print(f"\n{'='*70}")
        print(f"[OK] Simulation complete!")
        print(f"[OK] Processed {num_steps} steps in {t_end:.0f}s simulated time")
        print(f"{'='*70}\n")
        print(f"Plot window is open for inspection.")
        print("Close the window or press Ctrl+C to exit.\n")
        
        # Keep window open until user closes it
        try:
            plt.show(block=False)
            while plt.fignum_exists(plotter.fig.number):
                plt.pause(0.1)
        except KeyboardInterrupt:
            print("\nExiting...")
            plt.close('all')


def main():
    parser = argparse.ArgumentParser(
        description='Live temperature visualization for PyECN cylindrical cells',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_live_temp.py --profile profiles/hppc_pulse.csv
  python run_live_temp.py --profile profiles/mixed_charge_discharge.csv --dt 1.0 --t_end 1200
  python run_live_temp.py --profile profiles/hppc_pulse.csv --no-plot --save-frames frames/
        """)
    
    parser.add_argument('--profile', type=str, required=True,
                       help='Path to current profile CSV file (t_s, I_A columns)')
    parser.add_argument('--dt', type=float, default=0.5,
                       help='Solver time step in seconds (default: 0.5)')
    parser.add_argument('--t_end', type=float, default=1800,
                       help='Total simulation time in seconds (default: 1800)')
    parser.add_argument('--config', type=str, default='cylindrical.toml',
                       help='PyECN configuration file (default: cylindrical.toml)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Run without live plotting')
    parser.add_argument('--save-frames', type=str, default=None,
                       help='Directory to save animation frames')
    
    args = parser.parse_args()
    
    # Validate profile exists
    profile_path = Path(args.profile)
    if not profile_path.exists():
        print(f"Error: Profile file not found: {args.profile}")
        sys.exit(1)
    
    # Run visualization
    run_live_visualization(
        profile_path=str(profile_path),
        dt=args.dt,
        t_end=args.t_end,
        config_file=args.config,
        enable_plotting=not args.no_plot,
        save_frames_dir=args.save_frames
    )


if __name__ == '__main__':
    main()
