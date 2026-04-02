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
        self.feature_columns = ['T_indoor','T_outdoor','humidity','occupancy','HVAC_state','energy','sin_hour','cos_hour']
        self.target_column = 'T_indoor'

    def load_home_data(self, home_id):
        filename = self.data_dir / f'home_{home_id:02d}.csv'
        if not filename.exists():
            raise FileNotFoundError(f"Data file not found: {filename}")
        df = pd.read_csv(filename).iloc[::3].reset_index(drop=True)
        df = df.iloc[:self.total_samples].copy()
        print(f"Loaded Home {home_id:02d}: {len(df)} samples")
        return df

    def get_features_target(self, df):
        return df[self.feature_columns].values, df[self.target_column].values.reshape(-1, 1)
