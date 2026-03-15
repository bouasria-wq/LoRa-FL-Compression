"""
Data Loading Module for Federated Learning HVAC Project
========================================================

Equation 1: Input Feature Vector
x_t = [T_in, T_out, H, O, S_HVAC, E, sin(θ), cos(θ)]^T ∈ ℝ^8

This module loads the first 7 days of data from each home's CSV file.
Data is downsampled from 5-minute to 15-minute intervals (every 3rd row).

File: local_home/load_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


class DataLoader:

    def __init__(self, data_dir='data', n_homes=10, n_days=7):
        self.data_dir = Path(data_dir)
        self.n_homes = n_homes
        self.n_days = n_days

        # Data sampled every 15 minutes: 96 samples per day
        self.samples_per_day = 96
        self.total_samples = n_days * self.samples_per_day  # 7 days = 672 samples

        self.feature_columns = [
            'T_indoor', 'T_outdoor', 'humidity', 'occupancy',
            'HVAC_state', 'energy', 'sin_hour', 'cos_hour'
        ]
        self.target_column = 'T_indoor'

    def load_home_data(self, home_id):
        filename = self.data_dir / f'home_{home_id:02d}.csv'

        if not filename.exists():
            raise FileNotFoundError(f"Data file not found: {filename}")

        df = pd.read_csv(filename)

        # Downsample from 5-min to 15-min (keep every 3rd row)
        df = df.iloc[::3].reset_index(drop=True)

        # Take first 7 days at 15-minute intervals
        # 7 days x 96 samples/day = 672 rows
        df_7days = df.iloc[:self.total_samples].copy()

        print(f"Loaded Home {home_id:02d}: {len(df_7days)} samples ({self.n_days} days @ 15-min intervals)")

        return df_7days

    def load_all_homes(self):
        all_data = {}
        for home_id in range(1, self.n_homes + 1):
            all_data[home_id] = self.load_home_data(home_id)
        print(f"\nLoaded {self.n_homes} homes, {self.total_samples} samples each ({self.n_days} days)")
        return all_data

    def get_features_target(self, df):
        X = df[self.feature_columns].values
        y = df[self.target_column].values.reshape(-1, 1)
        return X, y

    def get_temperature_range(self, df):
        T_indoor = df['T_indoor'].values
        return T_indoor.max() - T_indoor.min()

    def print_data_summary(self, home_id, df):
        print(f"\n{'='*60}")
        print(f"Home {home_id:02d} Data Summary (First {self.n_days} Days = {self.total_samples} Samples)")
        print(f"{'='*60}")
        print(f"Shape: {df.shape} (rows x columns)")
        print(f"Sampling: Every 15 minutes ({self.samples_per_day} samples/day)")
        print(f"\nFeature Statistics:")
        print(df.describe().T[['mean', 'std', 'min', 'max']])
        temp_range = self.get_temperature_range(df)
        print(f"\nIndoor Temp Range: {temp_range:.2f} degrees C")
        print(f"{'='*60}")


if __name__ == "__main__":
    loader = DataLoader(data_dir='data', n_homes=10, n_days=7)
    df = loader.load_home_data(home_id=1)
    loader.print_data_summary(home_id=1, df=df)
    X, y = loader.get_features_target(df)
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"First sample X: {X[0]}")
    print(f"First sample y: {y[0][0]:.1f} degrees C")
