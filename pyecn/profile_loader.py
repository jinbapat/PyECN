# -*- coding: utf-8 -*-
"""
Current profile loader for PyECN simulations.

Supports CSV input with columns:
    - t_s: time in seconds (monotonic increasing)
    - I_A: current in Amps (positive=discharge, negative=charge)

Features:
    - Linear interpolation at solver time steps
    - Piecewise-constant profile support
    - Input validation (monotonic time, numeric values, non-empty)
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Tuple, Callable


class CurrentProfileLoader:
    """Load and interpolate current profiles from CSV files."""
    
    def __init__(self, profile_path: str, interpolation_method: str = 'linear'):
        """
        Initialize the current profile loader.
        
        Parameters
        ----------
        profile_path : str
            Path to CSV file with columns 't_s' and 'I_A'
        interpolation_method : str, optional
            Interpolation method: 'linear' or 'zero' (piecewise-constant)
        """
        self.profile_path = profile_path
        self.interpolation_method = interpolation_method
        self.t_profile = None
        self.I_profile = None
        self.interp_func = None
        self.t_max = None
        self.I_min = None
        self.I_max = None
        
        self._load_and_validate()
        self._create_interpolator()
    
    def _load_and_validate(self) -> None:
        """Load CSV and validate input data."""
        try:
            df = pd.read_csv(self.profile_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Profile file not found: {self.profile_path}")
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")
        
        # Check required columns
        required_cols = ['t_s', 'I_A']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(
                f"CSV must contain columns {required_cols}. "
                f"Found columns: {list(df.columns)}"
            )
        
        # Extract and validate time
        self.t_profile = df['t_s'].values
        if len(self.t_profile) < 2:
            raise ValueError("Profile must have at least 2 time points")
        
        if not np.all(np.diff(self.t_profile) > 0):
            raise ValueError("Time values (t_s) must be strictly monotonic increasing")
        
        if not np.all(np.isfinite(self.t_profile)):
            raise ValueError("Time values contain non-finite numbers (NaN or Inf)")
        
        # Extract and validate current
        self.I_profile = df['I_A'].values
        if len(self.I_profile) != len(self.t_profile):
            raise ValueError("Time and current arrays must have same length")
        
        if not np.all(np.isfinite(self.I_profile)):
            raise ValueError("Current values contain non-finite numbers (NaN or Inf)")
        
        self.t_max = self.t_profile[-1]
        self.I_min = self.I_profile.min()
        self.I_max = self.I_profile.max()
        
        print(f"[OK] Loaded profile: t_s in [0, {self.t_max:.2f}] s, "
              f"I_A in [{self.I_min:.2f}, {self.I_max:.2f}] A")
    
    def _create_interpolator(self) -> None:
        """Create interpolation function for current profile."""
        if self.interpolation_method == 'linear':
            self.interp_func = interp1d(
                self.t_profile, self.I_profile,
                kind='linear', bounds_error=False,
                fill_value=(self.I_profile[0], self.I_profile[-1])
            )
        elif self.interpolation_method == 'zero':
            # Piecewise constant: use 'nearest' with step='pre'
            self.interp_func = interp1d(
                self.t_profile, self.I_profile,
                kind='nearest', bounds_error=False,
                fill_value=(self.I_profile[0], self.I_profile[-1])
            )
        else:
            raise ValueError(
                f"Unknown interpolation method: {self.interpolation_method}"
            )
    
    def get_current(self, t_sim: float) -> float:
        """
        Get current at a given simulation time.
        
        Parameters
        ----------
        t_sim : float
            Simulation time in seconds
        
        Returns
        -------
        float
            Current in Amps at time t_sim
        """
        if t_sim < 0:
            raise ValueError(f"Negative time not allowed: {t_sim}")
        
        if t_sim > self.t_max:
            # Extrapolate: hold last value
            return float(self.I_profile[-1])
        
        return float(self.interp_func(t_sim))
    
    def get_current_array(self, t_array: np.ndarray) -> np.ndarray:
        """
        Get current at multiple times.
        
        Parameters
        ----------
        t_array : np.ndarray
            Array of simulation times in seconds
        
        Returns
        -------
        np.ndarray
            Array of currents in Amps
        """
        t_clipped = np.clip(t_array, 0, self.t_max)
        return self.interp_func(t_clipped)
    
    def get_profile_summary(self) -> dict:
        """
        Return summary statistics of the profile.
        
        Returns
        -------
        dict
            Summary containing duration, current range, and statistics
        """
        return {
            'duration_s': self.t_max,
            'I_min_A': float(self.I_min),
            'I_max_A': float(self.I_max),
            'I_mean_A': float(np.mean(self.I_profile)),
            'I_std_A': float(np.std(self.I_profile)),
            'n_points': len(self.t_profile),
        }


def load_profile(profile_path: str, 
                interpolation_method: str = 'linear') -> CurrentProfileLoader:
    """
    Convenience function to load a current profile.
    
    Parameters
    ----------
    profile_path : str
        Path to CSV file with columns 't_s' and 'I_A'
    interpolation_method : str, optional
        Interpolation method: 'linear' or 'zero'
    
    Returns
    -------
    CurrentProfileLoader
        Loader instance with callable get_current() method
    """
    return CurrentProfileLoader(profile_path, interpolation_method)
