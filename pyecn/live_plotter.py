# -*- coding: utf-8 -*-
"""
Live temperature visualization for cylindrical electrode surface.

Creates a 2D unwrapped surface map (theta vs z) with real-time animation
and supporting plots for temperature statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from typing import Optional, Callable, Tuple
import collections


class LiveTemperaturePlotter:
    """Real-time 2D surface temperature visualization."""
    
    def __init__(self, 
                 nx: int, 
                 ny: int,
                 nz: int,
                 cell_name: str = 'cell_1',
                 update_interval: int = 5,
                 figsize: Tuple[int, int] = (16, 10)):
        """
        Initialize the live temperature plotter.
        
        Parameters
        ----------
        nx : int
            Number of circumferential nodes (theta direction)
        ny : int
            Number of axial nodes (z direction)
        nz : int
            Number of radial nodes (r direction)
        cell_name : str
            Name of the cell to monitor
        update_interval : int
            Update plot every N simulation steps
        figsize : tuple
            Figure size in inches
        """
        self.nx = nx  # circumferential (theta)
        self.ny = ny  # axial (z)
        self.nz = nz  # radial (r)
        self.cell_name = cell_name
        self.update_interval = update_interval
        self.figsize = figsize
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=figsize)
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Main heatmap: surface temperature (theta vs z)
        self.ax_heatmap = self.fig.add_subplot(gs[0, :])
        
        # Time series: average and max temperature
        self.ax_time_series = self.fig.add_subplot(gs[1, 0])
        
        # Statistics display
        self.ax_stats = self.fig.add_subplot(gs[1, 1])
        self.ax_stats.axis('off')
        
        # Data buffers
        self.time_buffer = collections.deque(maxlen=1000)
        self.T_avg_buffer = collections.deque(maxlen=1000)
        self.T_max_buffer = collections.deque(maxlen=1000)
        self.T_min_buffer = collections.deque(maxlen=1000)
        
        # State
        self.step_count = 0
        self.time_elapsed = 0.0
        self.current_T_surface = None
        self.im = None
        self.cbar = None
        
        self._setup_heatmap()
        self._setup_time_series()
        
    def _setup_heatmap(self) -> None:
        """Initialize the main heatmap axes."""
        self.ax_heatmap.set_xlabel('Circumferential Index (θ)', fontsize=12, fontweight='bold')
        self.ax_heatmap.set_ylabel('Axial Index (z)', fontsize=12, fontweight='bold')
        self.ax_heatmap.set_title('Electrode Surface Temperature [°C]', 
                                  fontsize=14, fontweight='bold')
        
        # Initialize empty heatmap
        dummy_data = np.zeros((self.ny, self.nx))
        self.im = self.ax_heatmap.imshow(dummy_data, cmap='RdYlBu_r', 
                                          aspect='auto', origin='lower')
        self.cbar = plt.colorbar(self.im, ax=self.ax_heatmap, label='Temperature (°C)')
    
    def _setup_time_series(self) -> None:
        """Initialize the time series plot."""
        self.ax_time_series.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        self.ax_time_series.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
        self.ax_time_series.set_title('Surface Temperature History', 
                                      fontsize=12, fontweight='bold')
        self.ax_time_series.grid(True, alpha=0.3)
        self.line_avg, = self.ax_time_series.plot([], [], 'b-', label='Average', linewidth=2)
        self.line_max, = self.ax_time_series.plot([], [], 'r-', label='Maximum', linewidth=2)
        self.line_min, = self.ax_time_series.plot([], [], 'c-', label='Minimum', linewidth=2)
        self.ax_time_series.legend(loc='best', fontsize=10)
    
    def update_temperature(self, 
                          T_nodes: np.ndarray,
                          t_sim: float,
                          step: int,
                          I_current: Optional[float] = None,
                          SoC: Optional[float] = None) -> None:
        """
        Update temperature data from simulation.
        
        Parameters
        ----------
        T_nodes : np.ndarray
            Temperature array of shape (ntotal,) or similar
        t_sim : float
            Current simulation time in seconds
        step : int
            Current simulation step number
        I_current : float, optional
            Current magnitude in Amps
        SoC : float, optional
            State of charge as fraction
        """
        self.step_count = step
        self.time_elapsed = t_sim
        self.I_current = I_current
        self.SoC = SoC
        
        # Extract surface temperature (outermost radial layer)
        # Assumption: T_nodes is shaped such that surface is the last ny*nx elements
        # or we extract every nz-th node for surface
        try:
            # For cylindrical cell: assume nodes are ordered (theta, z, r)
            # Surface = r_max layer, i.e., last nx*ny nodes
            if len(T_nodes) >= self.nx * self.ny:
                T_surface = T_nodes[-self.nx*self.ny:].reshape(self.ny, self.nx)
            else:
                # Fallback: reshape available data
                n_available = len(T_nodes)
                T_surface = T_nodes[:self.nx*self.ny].reshape(self.ny, self.nx)
        except Exception as e:
            print(f"Warning: Failed to extract surface temperature: {e}")
            T_surface = np.full((self.ny, self.nx), np.mean(T_nodes) - 273.15)
        
        # Convert to Celsius
        self.current_T_surface = T_surface - 273.15  # Kelvin to Celsius
        
        # Compute statistics
        T_avg = np.mean(self.current_T_surface)
        T_max = np.max(self.current_T_surface)
        T_min = np.min(self.current_T_surface)
        
        # Store in buffers
        self.time_buffer.append(t_sim)
        self.T_avg_buffer.append(T_avg)
        self.T_max_buffer.append(T_max)
        self.T_min_buffer.append(T_min)
    
    def plot_update(self) -> None:
        """Update all plots with current data."""
        if self.current_T_surface is None:
            return
        
        # Update heatmap
        self.im.set_data(self.current_T_surface)
        vmin, vmax = self.current_T_surface.min(), self.current_T_surface.max()
        self.im.set_clim(vmin, vmax)
        
        # Update time series
        if len(self.time_buffer) > 0:
            times = list(self.time_buffer)
            self.line_avg.set_data(times, list(self.T_avg_buffer))
            self.line_max.set_data(times, list(self.T_max_buffer))
            self.line_min.set_data(times, list(self.T_min_buffer))
            
            self.ax_time_series.set_xlim(times[0], times[-1])
            T_all = list(self.T_max_buffer) + list(self.T_min_buffer)
            self.ax_time_series.set_ylim(min(T_all) - 1, max(T_all) + 1)
        
        # Update statistics
        self._update_stats_text()
        
        self.fig.canvas.draw_idle()
    
    def _update_stats_text(self) -> None:
        """Update the statistics text box."""
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        if self.current_T_surface is None:
            return
        
        T_avg = np.mean(self.current_T_surface)
        T_max = np.max(self.current_T_surface)
        T_min = np.min(self.current_T_surface)
        T_delta = T_max - T_min
        
        stats_text = (
            f"{'─'*30}\n"
            f"{'SIMULATION STATUS':^30}\n"
            f"{'─'*30}\n"
            f"Step: {self.step_count}\n"
            f"Time: {self.time_elapsed:.2f} s\n"
            f"\n{'SURFACE TEMPERATURE':^30}\n"
            f"{'─'*30}\n"
            f"Average: {T_avg:.2f} °C\n"
            f"Maximum: {T_max:.2f} °C\n"
            f"Minimum: {T_min:.2f} °C\n"
            f"ΔT: {T_delta:.2f} °C\n"
        )
        
        if self.I_current is not None:
            stats_text += f"\nCurrent: {self.I_current:.3f} A\n"
        
        if self.SoC is not None:
            stats_text += f"SoC: {self.SoC*100:.1f}%\n"
        
        self.ax_stats.text(0.1, 0.95, stats_text, transform=self.ax_stats.transAxes,
                          fontsize=10, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def show(self) -> None:
        """Display the plot."""
        plt.show()
    
    def save_frame(self, filename: str) -> None:
        """Save current figure to file."""
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {filename}")


class LivePlotterAnimated:
    """Animated version using matplotlib animation."""
    
    def __init__(self, 
                 nx: int,
                 ny: int, 
                 nz: int,
                 cell_name: str = 'cell_1',
                 update_interval: int = 5,
                 figsize: Tuple[int, int] = (16, 10)):
        """Initialize animated plotter."""
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.plotter = LiveTemperaturePlotter(nx, ny, nz, cell_name, 
                                             update_interval, figsize)
        self.animation = None
        self.should_update = False
    
    def add_data_callback(self, 
                         T_nodes: np.ndarray,
                         t_sim: float,
                         step: int,
                         I_current: Optional[float] = None,
                         SoC: Optional[float] = None) -> None:
        """Add data and trigger update if interval reached."""
        self.plotter.update_temperature(T_nodes, t_sim, step, I_current, SoC)
        
        if step % self.plotter.update_interval == 0:
            self.plotter.plot_update()
    
    def show(self) -> None:
        """Display the animated plot."""
        self.plotter.show()
    
    def save_frame(self, filename: str) -> None:
        """Save current frame."""
        self.plotter.save_frame(filename)
