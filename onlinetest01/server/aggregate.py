"""
Server Aggregation Module
=========================

Implements Equation 35: Federated Averaging
"""

import numpy as np


class FederatedServer:

    def __init__(self, n_params=553, alpha=0.25, n_clients=10):
        self.n_params = n_params
        self.alpha = alpha
        self.n_clients = n_clients
        self.theta_global = np.zeros(n_params)
        self.round = 0

        print("\n" + "="*60)
        print("Federated Server Initialized")
        print("="*60)
        print(f"Parameters: {n_params}")
        print(f"Aggregation rate alpha: {alpha}")
        print(f"Number of clients: {n_clients}")
        print("="*60)

    def aggregate_synchronous(self, client_params_list):
        self.theta_global = np.mean(client_params_list, axis=0)
        self.round += 1
        return self.theta_global

    def aggregate_round(self, client_params_list, method='synchronous'):
        print(f"\nAggregating {len(client_params_list)} clients...")
        result = self.aggregate_synchronous(client_params_list)
        return result

    def get_global_model(self):
        return self.theta_global.copy()

    def set_global_model(self, params):
        self.theta_global = params.copy()

