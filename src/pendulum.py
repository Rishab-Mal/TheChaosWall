import numpy as np
import pandas as pd

class DoublePendulum:
    def __init__(self, theta1, theta2, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81, dt=0.01, T=1.0, substeps=5):
        self.theta1 = theta1
        self.theta2 = theta2
        self.m1, self.m2 = m1, m2
        self.l1, self.l2 = l1, l2
        self.g = g
        self.dt = dt
        self.T = T
        self.substeps = substeps  # internal steps per output step (reduces O(dt²) energy error)

    def _hamiltonian_rhs(self, y):
        t1, t2, p1, p2 = y
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        delta = t1 - t2
        
        den = l1 * l2 * (m1 + m2 * np.sin(delta)**2)
        
        # dH/dp = dot_theta
        t1_dot = (l2 * p1 - l1 * np.cos(delta) * p2) / (l1 * den)
        t2_dot = (l1 * (m1 + m2
        ) * p2 - m2 * l2 * np.cos(delta) * p1) / (m2 * l2 * den)
        
        # -dH/dtheta = dot_p
        # Potential derivative
        p1_dot = -(m1 + m2) * g * l1 * np.sin(t1)
        p2_dot = -m2 * g * l2 * np.sin(t2)
        
        # Kinetic coupling: -∂T_H/∂θ1 = -C1, -∂T_H/∂θ2 = +C1
        C1 = t1_dot * t2_dot * m2 * l1 * l2 * np.sin(delta)
        p1_dot -= C1
        p2_dot += C1
        
        return np.array([t1_dot, t2_dot, p1_dot, p2_dot])

    def generateTimeData(self):
        t_eval = np.arange(0, self.T, self.dt)
        n = len(t_eval)
        # Initial state: [theta1, theta2, p1, p2]
        # p=0 because we assume starting from rest (omega=0)
        y = np.array([self.theta1, self.theta2, 0.0, 0.0])
        h = self.dt / self.substeps  # internal step size (substeps reduces O(dt²) energy error by substeps²)

        results = np.empty((n, 6)) # [theta1, theta2, t1_dot, t2_dot, p1, p2]

        for i in range(n):
            # Record current state
            rhs_vals = self._hamiltonian_rhs(y)
            results[i] = [y[0], y[1], rhs_vals[0], rhs_vals[1], y[2], y[3]]

            # Advance by self.dt using self.substeps implicit midpoint steps of size h
            for _ in range(self.substeps):
                rhs_i = self._hamiltonian_rhs(y)
                y_next = y + h * rhs_i  # initial guess: explicit Euler
                y_new = y_next
                for _ in range(100):
                    y_new = y + h * self._hamiltonian_rhs((y + y_next) * 0.5)
                    if np.max(np.abs(y_new - y_next)) < 1e-12:
                        break
                    y_next = y_new
                y = y_new

        df = pd.DataFrame({
            "t": t_eval,
            "theta1": results[:, 0],
            "theta2": results[:, 1],
            "theta1_dot": results[:, 2],
            "theta2_dot": results[:, 3],
            # Metadata for visualization script
            "m1": self.m1, "m2": self.m2,
            "l1": self.l1, "l2": self.l2,
            "g": self.g, "dt": self.dt
        })
        return df