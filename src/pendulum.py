# //src/pendulum.py

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


class DoublePendulum:

    def __init__(
        self,
        theta1,
        theta2,
        m1=1.0,
        m2=1.0,
        l1=1.0,
        l2=1.0,
        g=9.81,
        dt=0.01,
        T=1.0
    ):
        """
        All angles must be in radians.
        """
        
        self.theta1 = theta1
        self.theta2 = theta2
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g
        self.dt = dt
        self.T = T

    # -------------------------------------------------
    # System of ODEs
    # -------------------------------------------------
    def equations(self, t, y):
        theta1, theta2, theta1_dot, theta2_dot = y

        m1, m2 = self.m1, self.m2
        l1, l2 = self.l1, self.l2
        g = self.g

        delta = theta1 - theta2
        sin_delta = np.sin(delta)
        cos_delta = np.cos(delta)

        D = (m1 + m2) * l1 - m2 * l1 * cos_delta**2

        theta1_ddot = (
            m2 * l1 * theta1_dot**2 * sin_delta * cos_delta
            + m2 * g * np.sin(theta2) * cos_delta
            + m2 * l2 * theta2_dot**2 * sin_delta
            - (m1 + m2) * g * np.sin(theta1)
        ) / D

        theta2_ddot = (
            -m2 * l2 * theta2_dot**2 * sin_delta * cos_delta
            + (m1 + m2) * (
                g * np.sin(theta1) * cos_delta
                - l1 * theta1_dot**2 * sin_delta
                - g * np.sin(theta2)
            )
        ) / (l2 / l1 * D)

        return [theta1_dot, theta2_dot, theta1_ddot, theta2_ddot]

    # -------------------------------------------------
    # Generate Time Data (RK45)
    # -------------------------------------------------
    def generateTimeData(self):

        y0 = [self.theta1, self.theta2, 0.0, 0.0]

        t_eval = np.arange(0, self.T, self.dt)

        sol = solve_ivp(
            self.equations,
            [0, self.T],
            y0,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-9
        )

        theta1 = sol.y[0]
        theta2 = sol.y[1]
        theta1_dot = sol.y[2]
        theta2_dot = sol.y[3]

        df = pd.DataFrame({
            "t": sol.t,
            "theta1": theta1,
            "theta2": theta2,
            "theta1_dot": theta1_dot,
            "theta2_dot": theta2_dot
        })

        return df
