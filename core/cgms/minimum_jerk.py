import numpy as np

class MinimumJerk:
    """
    Generate minimum-jerk trajectory from start to goal in time tau with dt step.
    """
    def __init__(self, start, goal, tau, dt):
        """
        @param start (np.ndarray)
            Initial position vector.
        @param goal (np.ndarray)
            Final position vector.
        @param tau (float)
            Duration of the trajectory.
        @param dt (float)
            Time step for discretization.
        """
        self.start  =   np.asarray(start, float).reshape(3)
        self.goal   =   np.asarray(goal, float).reshape(3)
        self.tau    =   float(tau)
        self.dt     =   float(dt)
        self.ts     =   np.arange(0.0, self.tau+1e-12, self.dt)

    def generate(self):
        """
        @return:
            y   (np.ndarray): Positions over time, shape (T, 3)
            yd  (np.ndarray): Velocities over time, shape (T, 3)
            ydd (np.ndarray): Accelerations over time, shape (T, 3)
            ts  (np.ndarray): Time stamps, shape (T,)
        """
        ts = self.ts
        tau_safe = max(self.tau, np.finfo(float).eps)
        s = ts / tau_safe                       
        A = (self.goal - self.start)[None, :]       

        phi   = 10*s**3 - 15*s**4 + 6*s**5
        dphi  = (30*s**2 - 60*s**3 + 30*s**4) / tau_safe
        ddphi = (60*s - 180*s**2 + 120*s**3) / (tau_safe**2)

        y   = self.start[None, :] + phi[:, None]  * A
        yd  = dphi[:, None] * A
        ydd = ddphi[:, None]* A
        return y, yd, ydd, ts