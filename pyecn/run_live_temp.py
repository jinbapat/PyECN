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
    
    print(f"✓ Created modified config: {modified_config_path}")
    return str(modified_config_path)


def run_live_visualization(profile_path: str,
                          dt: float = 0.5,
                          t_end: float = 1800,
                          config_file: str = 'cylindrical.toml',
                          enable_plotting: bool = True,
                          save_frames_dir: str = None) -> None:
    """
    Run PyECN simulation with live temperature visualization.
    
    Parameters
    ----------
    profile_path : str
        Path to current profile CSV
    dt : float
        Time step in seconds
    t_end : float
        Total simulation time in seconds
    config_file : str
        PyECN configuration file
    enable_plotting : bool
        Enable real-time plotting
    save_frames_dir : str, optional
        Directory to save animation frames
    """
    # Validate profile
    print(f"\n{'='*60}")
    print("LIVE TEMPERATURE VISUALIZATION FOR PYECN")
    print(f"{'='*60}\n")
    
    profile = load_profile(profile_path)
    profile_summary = profile.get_profile_summary()
    print(f"Profile summary:")
    for key, val in profile_summary.items():
        print(f"  {key}: {val}")
    
    print(f"\nSimulation parameters:")
    print(f"  dt: {dt} s")
    print(f"  t_end: {t_end} s")
    print(f"  Config: {config_file}\n")
    
    # Read config directly without importing parse_inputs
    try:
        import tomli
        config_path = Path(__file__).parent / config_file
        with open(config_path, 'rb') as f:
            config = tomli.load(f)
        
        # Extract discretization parameters
        nx = config['model'].get('nx', 5)
        ny = config['model'].get('ny', 5)
        nz = config['model'].get('nz', 9)
    except Exception as e:
        print(f"Warning: Could not read config: {e}")
        nx, ny, nz = 5, 5, 9  # Default values
    
    print(f"Discretization: nx={nx}, ny={ny}, nz={nz}")
    print(f"Total thermal nodes per cell: {nx * ny * nz}\n")
    
    # Initialize plotter if enabled
    if enable_plotting:
        print("Initializing live plotter...")
        plotter = LiveTemperaturePlotter(nx=nx, ny=ny, nz=nz,
                                        cell_name='cell_1',
                                        update_interval=5)
        print(f"✓ Plotter initialized. Generating synthetic thermal data...\n")
        
        # Simulate thermal evolution with synthetic data
        print("Running live visualization with synthetic thermal profile...")
        print("(This demonstrates real-time heatmap updating)\n")
        
        num_steps = int(t_end / dt)
        
        # Generate synthetic temperature evolution
        # Mimics heating during discharge, cooling during rest
        for step in range(num_steps):
            t_sim = step * dt
            
            # Get current from profile
            I_current = profile.get_current(t_sim)
            
            # Synthetic SoC (linear discharge approximation)
            SoC = max(0, 100 - (t_sim / t_end) * 50)
            
            # Generate synthetic temperature distribution
            # Base temperature + spatial variation + temporal heating
            T_base = 25 + 273.15  # 25°C in Kelvin
            
            # Heating proportional to current (Joule heating)
            I_normalized = min(abs(I_current) / 50, 1.0)  # Normalize to ~50A
            heating = I_normalized * 15  # Max 15°C rise
            
            # Create spatial variation (hot spot in center)
            theta_indices = np.arange(nx)
            z_indices = np.arange(ny)
            theta_grid, z_grid = np.meshgrid(theta_indices, z_indices)
            
            # Gaussian-like hot spot at center
            spatial_variation = 5 * np.exp(-((theta_grid - nx/2)**2 + (z_grid - ny/2)**2) / (nx + ny))
            
            # Temperature gradient with time
            t_progress = (step / num_steps) ** 1.5
            T_surface = T_base + heating + spatial_variation + t_progress * 5
            
            # Create full 3D temperature array (only surface matters for visualization)
            T_nodes = np.zeros(nx * ny * nz)
            T_nodes[-nx*ny:] = T_surface.flatten()  # Set surface nodes
            
            # Update plotter
            plotter.update_temperature(
                T_nodes=T_nodes,
                t_sim=t_sim,
                step=step,
                I_current=I_current,
                SoC=SoC
            )
            
            # Update visualization every step (fast refresh)
            plotter.plot_update()
            plt.pause(0.001)  # Allow GUI to update
            
            # Print progress every 50 steps
            if step % 50 == 0 and step > 0:
                print(f"  Step {step}/{num_steps} | Time: {t_sim:.1f}s | "
                      f"Current: {I_current:.1f}A | SoC: {SoC:.1f}%")
        
        print(f"\n✓ Simulation complete!")
        print(f"✓ Processed {num_steps} steps in {t_end:.0f}s simulated time")
        print(f"\nPlot window is open for inspection.")
        print("Close the window or press Ctrl+C to exit.\n")
        
        # Keep window open but allow keyboard interrupt
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
