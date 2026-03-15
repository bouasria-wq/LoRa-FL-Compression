"""
Federated Server - ME-CFL Version
===================================
Variance-reduced aggregation with global shift h^t.
Momentum-based model update in Hilbert space.

Equations implemented:
- ME-CFL Eq 9:  g^t = (1/|S^t|) sum[v_i^t + h_i^t] + (1 - |S^t|/N) * h^{t-1}
- ME-CFL Eq 10: x^{t+1} = x^t - eta_t * M(g^t, beta)
- ME-CFL Eq 3:  Bounded heterogeneous variance zeta_i

File: server/aggregate.py
"""

import numpy as np


class FederatedServer:

    def __init__(self, n_clients=10, alpha=0.25, beta=0.9, eta=0.01):
        self.n_clients = n_clients
        self.alpha = alpha        # Aggregation rate
        self.beta = beta          # Momentum coefficient
        self.eta = eta            # Learning rate for momentum update

        self.global_model = None
        self.global_shift = None       # h^t — global shift for variance reduction
        self.momentum = None           # M(g^t, beta) — recursive momentum
        self.client_shifts = {}        # h_i^t — per client local shifts
        self.zeta_values = {}          # zeta_i — heterogeneous variance per client

        self.round_history = []
        self.aggregation_count = 0

    def initialize(self, param_size):
        """Initialize global model, shift and momentum vectors."""
        if self.global_model is None:
            self.global_model = np.zeros(param_size, dtype=np.float32)
            self.global_shift = np.zeros(param_size, dtype=np.float32)
            self.momentum = np.zeros(param_size, dtype=np.float32)
            print(f"Server initialized: {param_size} parameters")
            print(f"Aggregation rate alpha: {self.alpha}")
            print(f"Momentum beta: {self.beta}")
            print(f"Learning rate eta: {self.eta}")

    def set_global_model(self, params):
        """Set global model from list of arrays."""
        flat = np.concatenate([p.flatten() for p in params])
        if self.global_model is None:
            self.initialize(len(flat))
        self.global_model = flat.copy()

    def get_global_model(self):
        """Return global model as flat array."""
        return self.global_model.copy() if self.global_model is not None else None

    def update_client_shift(self, client_id, client_params, alpha_shift=None):
        """
        Update local shift h_i^t for client.
        h_i^{t+1} = h_i^t + alpha * (v_i^t - h_i^t)
        """
        if alpha_shift is None:
            alpha_shift = self.alpha

        if client_id not in self.client_shifts:
            self.client_shifts[client_id] = np.zeros_like(client_params)

        self.client_shifts[client_id] = (
            self.client_shifts[client_id] +
            alpha_shift * (client_params - self.client_shifts[client_id])
        )

    def compute_variance_reduced_gradient(self, client_updates, participating_ids):
        """
        ME-CFL Eq 9: Variance-reduced gradient estimator.
        g^t = (1/|S^t|) * sum[v_i^t + h_i^t] + (1 - |S^t|/N) * h^{t-1}
        """
        S = len(participating_ids)
        N = self.n_clients

        # Sum of (v_i^t + h_i^t) for participating clients
        sum_updates = np.zeros_like(self.global_model)
        for client_id, params in zip(participating_ids, client_updates):
            h_i = self.client_shifts.get(client_id, np.zeros_like(params))
            sum_updates += (params + h_i)

        # ME-CFL Eq 9
        g_t = (1.0 / S) * sum_updates + (1.0 - S / N) * self.global_shift

        return g_t

    def compute_momentum(self, g_t):
        """
        ME-CFL Eq 10: Recursive momentum in Hilbert space.
        M = (1-beta) * sum_{j=0}^{t} beta^{t-j} * g^j
        Implemented recursively as:
        M^{t+1} = beta * M^t + (1-beta) * g^t
        """
        self.momentum = self.beta * self.momentum + (1 - self.beta) * g_t
        return self.momentum

    def aggregate_round(self, client_params_dict, day):
        """
        Full ME-CFL aggregation for one round.
        client_params_dict: {client_id: flat_params_array}
        """
        if not client_params_dict:
            print(f"Day {day}: No client updates received!")
            return self.global_model

        participating_ids = list(client_params_dict.keys())
        client_updates = [client_params_dict[cid] for cid in participating_ids]

        # Initialize if first round
        param_size = len(client_updates[0])
        if self.global_model is None:
            self.initialize(param_size)

        # Update client shifts h_i^t
        for client_id, params in zip(participating_ids, client_updates):
            self.update_client_shift(client_id, params)

        # ME-CFL Eq 9: Variance-reduced gradient
        g_t = self.compute_variance_reduced_gradient(client_updates, participating_ids)

        # Update global shift h^t
        self.global_shift = (
            self.global_shift +
            (self.alpha / self.n_clients) *
            sum(client_updates[i] - self.client_shifts.get(participating_ids[i],
                np.zeros(param_size)) for i in range(len(participating_ids)))
        )

        # ME-CFL Eq 10: Momentum update
        M_t = self.compute_momentum(g_t)
        x_new = self.global_model - self.eta * M_t

        # Standard FedAvg blend for stability
        mean_params = np.mean(client_updates, axis=0)
        self.global_model = (1 - self.alpha) * x_new + self.alpha * mean_params

        # Track zeta_i values
        for client_id in participating_ids:
            pass  # zeta tracked in hegazy.py

        self.aggregation_count += 1
        self.round_history.append({
            'day': day,
            'n_participants': len(participating_ids),
            'participating_ids': participating_ids,
            'participation_rate': len(participating_ids) / self.n_clients
        })

        print(f"Day {day}: Aggregated {len(participating_ids)}/{self.n_clients} homes | "
              f"Participation: {len(participating_ids)/self.n_clients*100:.0f}%")

        return self.global_model

    def aggregate_synchronous(self, all_params, day):
        """
        Synchronous aggregation — all homes participated.
        Wraps aggregate_round with sequential IDs.
        """
        client_params_dict = {i+1: params for i, params in enumerate(all_params)}
        return self.aggregate_round(client_params_dict, day)

    def get_summary(self):
        """Print aggregation summary."""
        print("\n" + "="*40)
        print("SERVER AGGREGATION SUMMARY")
        print("="*40)
        for r in self.round_history:
            print(f"Day {r['day']}: {r['n_participants']} homes | "
                  f"Participation: {r['participation_rate']*100:.0f}%")
        print("="*40)
