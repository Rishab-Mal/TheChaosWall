import math
import pandas as pd

class DoublePendulum:
    def __init__(self, theta1_deg, theta2_deg, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
        self.init_theta1_deg = theta1_deg
        self.init_theta2_deg = theta2_deg
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g
    
    def generateTimeData(self):
        data = []
        dt = 0.01
        T = 1
        cur_t = 0
        
        # Convert to radians for calculation
        theta1 = math.radians(self.init_theta1_deg)
        theta2 = math.radians(self.init_theta2_deg)
        theta1_dot = 0.0
        theta2_dot = 0.0
        
        while cur_t < T:
            # EULER-LAGRANGE
            delta = theta1 - theta2
            cos_delta = math.cos(delta)
            sin_delta = math.sin(delta)
            
            D = (self.m1 + self.m2) * self.l1 - self.m2 * self.l1 * cos_delta**2
            
            # ANGULAR ACCELERATIONS
            theta1_ddot = (
                self.m2 * self.l1 * theta1_dot**2 * sin_delta * cos_delta
                + self.m2 * self.g * math.sin(theta2) * cos_delta
                + self.m2 * self.l2 * theta2_dot**2 * sin_delta
                - (self.m1 + self.m2) * self.g * math.sin(theta1)
            ) / D
            
            theta2_ddot = (
                -self.m2 * self.l2 * theta2_dot**2 * sin_delta * cos_delta
                + (self.m1 + self.m2) * (
                    self.g * math.sin(theta1) * cos_delta
                    - self.l1 * theta1_dot**2 * sin_delta
                    - self.g * math.sin(theta2)
                )
            ) / (self.l2 / self.l1 * D)
            
            # Store current state (all in degrees)
            data.append({
                'init_theta1': self.init_theta1_deg,
                'init_theta2': self.init_theta2_deg,
                't': cur_t,
                'theta1': math.degrees(theta1),
                'theta2': math.degrees(theta2),
                'theta1_dot': math.degrees(theta1_dot),
                'theta2_dot': math.degrees(theta2_dot)
            })
            
            # TIME INTEGRATION
            theta1_dot += theta1_ddot * dt
            theta2_dot += theta2_ddot * dt
            theta1 += theta1_dot * dt
            theta2 += theta2_dot * dt
            cur_t += dt
        
        return pd.DataFrame(data)