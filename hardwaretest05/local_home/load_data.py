"""
Data Loading Module for Federated Learning HVAC Project
========================================================
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
        self.samples_per_day = 96
        self.total_samples = n_days * self.samples_per_day

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
        df = df.iloc[::3].reset_index(drop=True)
        df_7days = df.iloc[:self.total_samples].copy()
        print(f"Loaded Home {home_id:02d}: {len(df_7days)} samples ({self.n_days} days @ 15-min intervals)")
        return df_7days

    def load_all_homes(self):
        all_data = {}
        for home_id in range(1, self.n_homes + 1):
            all_data[home_id] = self.load_home_data(home_id)
        return all_data

    def get_features_target(self, df):
        X = df[self.feature_columns].values
        y = df[self.target_column].values.reshape(-1, 1)
        return X, y

    def get_temperature_range(self, df):
        T_indoor = df['T_indoor'].values
        return T_indoor.max() - T_indoor.min()
