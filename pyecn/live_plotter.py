# -*- coding: utf-8 -*-
"""
Live temperature visualization for cylindrical electrode (jelly roll unwrapped).

Creates 2D unwrapped heatmap showing:
- Full radial-axial cross-section (core to surface vs electrode height)
- Like unwrapping a jelly roll to see the spiral from center to outside
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from typing import Optional, Callable, Tuple
import collections


class LiveTemperaturePlotter:
    """Real-time electrode temperature visualization as unwrapped jelly roll."""
    
    def __init__(self, 
                 nx: int, 
                 ny: int,
                 nz: int,
                 n_radial: int = 5,
                 cell_name: str = 'cell_1',
                 update_interval: int = 5,
                 figsize: Tuple[int, int] = (14, 8)):
        """
        Initialize the live temperature plotter for unwrapped electrode.
        
        Parameters
        ----------
        nx : int
            Number of circumferential nodes (theta direction)
        ny : int
            Number of axial nodes (z direction)
        nz : int
            Number of radial nodes (r direction)
        n_radial : int
            Number of radial layers being tracked
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
        self.n_radial = n_radial
        self.cell_name = cell_name
        self.update_interval = update_interval
        self.figsize = figsize
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=figsize)
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Main heatmap: full radial-axial cross-section (unwrapped jelly roll)
        self.ax_main = self.fig.add_subplot(gs[0, :])
        
        # Time series: temperature history
        self.ax_time_series = self.fig.add_subplot(gs[1, 0])
        
        # Statistics display
        self.ax_stats = self.fig.add_subplot(gs[1, 1])
        self.ax_stats.axis('off')
        
        # Data buffers
        self.time_buffer = collections.deque(maxlen=1000)
        self.T_avg_buffer = collections.deque(maxlen=1000)
        self.T_max_buffer = collections.deque(maxlen=1000)
        
        # Current data
        self.step_count = 0
        self.time_elapsed = 0.0
        self.T_radial_axial = None
        self.I_current = None
        self.SoC = None
        
        # Image object
        self.im_main = None
        self.cbar = None
        
        self._setup_main_heatmap()
        self._setup_time_series()
        
    def _setup_main_heatmap(self) -> None:
        """Initialize the main unwrapped jelly roll heatmap."""
        self.ax_main.set_xlabel('Radial Layer (Core → Surface)', fontsize=12, fontweight='bold')
        self.ax_main.set_ylabel('Axial Position (z)', fontsize=12, fontweight='bold')
        self.ax_main.set_title('Electrode Temperature - Unwrapped Jelly Roll (Core to Surface)', 
                              fontsize=13, fontweight='bold')
        
        # Initialize empty heatmap
        dummy_data = np.zeros((self.ny, self.n_radial))
        self.im_main = self.ax_main.imshow(dummy_data, cmap='RdYlBu_r', 
                                          aspect='auto', origin='lower',
                                          interpolation='bilinear')
        self.cbar = plt.colorbar(self.im_main, ax=self.ax_main, label='Temperature (°C)')
    
    def _setup_time_series(self) -> None:
        """Initialize the time series plot."""
        self.ax_time_series.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        self.ax_time_series.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
        self.ax_time_series.set_title('Temperature History', 
                                      fontsize=12, fontweight='bold')
        self.ax_time_series.grid(True, alpha=0.3)
        self.line_avg, = self.ax_time_series.plot([], [], 'b-', label='Average', linewidth=2)
        self.line_max, = self.ax_time_series.plot([], [], 'r-', label='Maximum', linewidth=2)
        self.ax_time_series.legend(loc='best', fontsize=10)
    
    def update_temperature(self, 
                          T_electrode: np.ndarray,
                          t_sim: float,
                          step: int,
                          I_current: Optional[float] = None,
                          SoC: Optional[float] = None) -> None:
        """
        Update temperature data from simulation.
        
        Parameters
        ----------
        T_electrode : np.ndarray
            Radial-axial temperature array of shape (n_radial, ny) in Kelvin
            Columns: radial layers (0=core, n-1=surface)
            Rows: axial positions (z direction)
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
        
        try:
            # T_electrode shape should be (n_radial, ny) - radial vs axial
            if T_electrode.ndim == 2:
                # Convert to Celsius for display
                self.T_radial_axial = T_electrode.T - 273.15  # Transpose to (ny, n_radial)
                
                # Compute statistics
                T_avg = np.mean(T_electrode)
                T_max = np.max(T_electrode)
                
                # Store in buffers
                self.time_buffer.append(t_sim)
                self.T_avg_buffer.append(T_avg - 273.15)
                self.T_max_buffer.append(T_max - 273.15)
            else:
                print(f"Warning: T_electrode has unexpected shape {T_electrode.shape}")
        except Exception as e:
            print(f"Warning: Failed to process T_electrode: {e}")
    
    def plot_update(self) -> None:
        """Update all plots with current data."""
        if self.T_radial_axial is None:
            return
        
        # Update main heatmap (unwrapped jelly roll)
        self.im_main.set_data(self.T_radial_axial)
        vmin, vmax = self.T_radial_axial.min(), self.T_radial_axial.max()
        self.im_main.set_clim(vmin, vmax)
        
        # Update time series
        if len(self.time_buffer) > 0:
            times = list(self.time_buffer)
            self.line_avg.set_data(times, list(self.T_avg_buffer))
            self.line_max.set_data(times, list(self.T_max_buffer))
            
            self.ax_time_series.set_xlim(times[0], times[-1])
            T_all = list(self.T_max_buffer) + list(self.T_avg_buffer)
            self.ax_time_series.set_ylim(min(T_all) - 1, max(T_all) + 1)
        
        # Update statistics
        self._update_stats_text()
        
        self.fig.canvas.draw_idle()
    
    def _update_stats_text(self) -> None:
        """Update the statistics text box."""
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        if self.T_radial_axial is None:
            return
        
        T_avg = np.mean(self.T_radial_axial)
        T_max = np.max(self.T_radial_axial)
        T_min = np.min(self.T_radial_axial)
        T_delta = T_max - T_min
        
        stats_text = (
            f"{'─'*35}\n"
            f"{'SIMULATION STATUS':^35}\n"
            f"{'─'*35}\n"
            f"Step: {self.step_count}\n"
            f"Time: {self.time_elapsed:.2f} s\n"
            f"\n{'ELECTRODE TEMPERATURE':^35}\n"
            f"{'─'*35}\n"
            f"Average: {T_avg:.2f} °C\n"
            f"Maximum: {T_max:.2f} °C\n"
            f"Minimum: {T_min:.2f} °C\n"
            f"ΔT: {T_delta:.2f} °C\n"
        )
        
        if self.I_current is not None:
            stats_text += f"\nCurrent: {self.I_current:.2f} A\n"
        
        if self.SoC is not None:
            stats_text += f"SoC: {self.SoC*100:.1f}%\n"
        
        self.ax_stats.text(0.1, 0.95, stats_text, transform=self.ax_stats.transAxes,
                          fontsize=10, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    def show(self) -> None:
        """Display the plot."""
        plt.show()
    
    def save_frame(self, filename: str) -> None:
        """Save current frame to file."""
        self.fig.savefig(filename, dpi=100, bbox_inches='tight')


