"""
Federated Server - ME-CFL Version
===================================
File: server/aggregate.py
"""

import numpy as np


class FederatedServer:

    def __init__(self, n_clients=10, alpha=0.25, beta=0.9, eta=0.01):
        self.n_clients = n_clients
        self.alpha = alpha
        self.beta = beta
        self.eta = eta

        self.global_model = None
        self.global_shift = None
        self.momentum = None
        self.client_shifts = {}
        self.round_history = []
        self.aggregation_count = 0

    def initialize(self, param_size):
        if self.global_model is None:
            self.global_model = np.zeros(param_size, dtype=np.float32)
            self.global_shift = np.zeros(param_size, dtype=np.float32)
            self.momentum = np.zeros(param_size, dtype=np.float32)
            print(f"Server initialized: {param_size} parameters")

    def set_global_model(self, params):
        flat = np.concatenate([p.flatten() for p in params])
        if self.global_model is None:
            self.initialize(len(flat))
        self.global_model = flat.copy()

    def get_global_model(self):
        return self.global_model.copy() if self.global_model is not None else None

    def update_client_shift(self, client_id, client_params, alpha_shift=None):
        if alpha_shift is None:
            alpha_shift = self.alpha
        if client_id not in self.client_shifts:
            self.client_shifts[client_id] = np.zeros_like(client_params)
        self.client_shifts[client_id] = (
            self.client_shifts[client_id] +
            alpha_shift * (client_params - self.client_shifts[client_id])
        )

    def compute_variance_reduced_gradient(self, client_updates, participating_ids):
        S = len(participating_ids)
        N = self.n_clients
        sum_updates = np.zeros_like(self.global_model)
        for client_id, params in zip(participating_ids, client_updates):
            h_i = self.client_shifts.get(client_id, np.zeros_like(params))
            sum_updates += (params + h_i)
        g_t = (1.0 / S) * sum_updates + (1.0 - S / N) * self.global_shift
        return g_t

    def compute_momentum(self, g_t):
        self.momentum = self.beta * self.momentum + (1 - self.beta) * g_t
        return self.momentum

    def aggregate_round(self, client_params_dict, day):
        if not client_params_dict:
            return self.global_model
        participating_ids = list(client_params_dict.keys())
        client_updates = [client_params_dict[cid] for cid in participating_ids]
        param_size = len(client_updates[0])
        if self.global_model is None:
            self.initialize(param_size)
        for client_id, params in zip(participating_ids, client_updates):
            self.update_client_shift(client_id, params)
        g_t = self.compute_variance_reduced_gradient(client_updates, participating_ids)
        self.global_shift = (
            self.global_shift +
            (self.alpha / self.n_clients) *
            sum(client_updates[i] - self.client_shifts.get(participating_ids[i],
                np.zeros(param_size)) for i in range(len(participating_ids)))
        )
        M_t = self.compute_momentum(g_t)
        x_new = self.global_model - self.eta * M_t
        mean_params = np.mean(client_updates, axis=0)
        self.global_model = (1 - self.alpha) * x_new + self.alpha * mean_params
        self.aggregation_count += 1
        self.round_history.append({
            'day': day,
            'n_participants': len(participating_ids),
            'participation_rate': len(participating_ids) / self.n_clients
        })
        print(f"Day {day}: Aggregated {len(participating_ids)}/{self.n_clients} homes")
        return self.global_model

    def get_summary(self):
        print("\n" + "="*40)
        print("SERVER AGGREGATION SUMMARY")
        print("="*40)
        for r in self.round_history:
            print(f"Day {r['day']}: {r['n_participants']} homes | "
                  f"Participation: {r['participation_rate']*100:.0f}%")
