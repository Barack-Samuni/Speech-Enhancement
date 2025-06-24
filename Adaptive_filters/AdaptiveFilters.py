import numpy as np


class RLSFilter:
    def __init__(self, n_taps, lam=0.99, delta=1.0):
        self.n_taps = n_taps
        self.lam = lam
        self.delta = delta
        self.w = np.zeros(n_taps)
        self.P = (1.0 / delta) * np.eye(n_taps)

    def adapt(self, x, d):
        x = np.array(x)
        if x.shape[0] != self.n_taps:
            raise ValueError("Input vector length must match n_taps")

        # Gain vector
        Pi_x = self.P @ x
        k = Pi_x / (self.lam + x.T @ Pi_x)

        # Error
        y = self.w @ x
        e = d - y

        # Update weights
        self.w += k * e

        # Update inverse correlation matrix
        self.P = (self.P - np.outer(k, x) @ self.P) / self.lam

        return y, e

    def predict(self, x):
        return self.w @ np.array(x)
