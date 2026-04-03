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

    def initialize(self, param_size):
        if self.global_model is None:
            self.global_model = np.zeros(param_size, dtype=np.float32)
            self.global_shift = np.zeros(param_size, dtype=np.float32)
            self.momentum     = np.zeros(param_size, dtype=np.float32)
            print(f"Server initialized: {param_size} parameters")

    def aggregate_round(self, client_params_dict, day):
        if not client_params_dict:
            return self.global_model
        ids    = list(client_params_dict.keys())
        params = [client_params_dict[i] for i in ids]
        if self.global_model is None:
            self.initialize(len(params[0]))
        for cid, p in zip(ids, params):
            if cid not in self.client_shifts:
                self.client_shifts[cid] = np.zeros_like(p)
            self.client_shifts[cid] += self.alpha * (p - self.client_shifts[cid])
        S = len(ids)
        N = self.n_clients
        g_t = (1/S) * sum(params[i] + self.client_shifts.get(ids[i], np.zeros_like(params[i])) for i in range(S)) + (1 - S/N) * self.global_shift
        self.global_shift += (self.alpha/N) * sum(params[i] - self.client_shifts.get(ids[i], np.zeros_like(params[i])) for i in range(S))
        self.momentum = self.beta * self.momentum + (1-self.beta) * g_t
        x_new = self.global_model - self.eta * self.momentum
        self.global_model = (1-self.alpha)*x_new + self.alpha*np.mean(params, axis=0)
        self.round_history.append({'day': day, 'n': S})
        print(f"Day {day}: Aggregated {S}/{self.n_clients} homes")
        return self.global_model

    def get_summary(self):
        print("\n" + "="*40 + "\nSERVER AGGREGATION SUMMARY\n" + "="*40)
        for r in self.round_history:
            print(f"Day {r['day']}: {r['n']} homes | Participation: {r['n']/self.n_clients*100:.0f}%")
