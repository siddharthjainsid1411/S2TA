import numpy as np

class DynamicalSystems:
    """
    Generates the canonical phase and gating signal for DMPs.
    """
    def __init__(self, tau, decay = 0.1, D0 = 1e-7):
        """
        @param tau (float)
            Time constant for the phase variable.
        @param decay (float)
            Decay rate for the sigmoid gating function.
        @param D0 (float)
            Offset for the sigmoid gating function.
        """
        self.tau    = float(tau)
        self.decay  = float(decay) 
        self.K      = 1.0 + D0 
        self.D0     = D0
        num         = ((self.K / self.decay) - 1.0) / self.D0
        self.r      = -np.log(num) / self.tau

    def time_system(self, ts): 
        """
        @brief
            A dynamical system with linear decay.
            Dynamics:               x'   = -1 / tau
            Analytical solution:    x(t) = 1 - (t / tau)

        @param ts (np.ndarray)
            Time stamps.

        @return (np.ndarray)
            Phase variable values at the given time stamps.
        """
        ts = np.asarray(ts, float)
        return np.clip(1.0 - ts/self.tau, 0.0, 1.0) 
    
    def sigmoid_system(self, ts):
        """
        @brief
            A dynamical system with sigmoid decay.
            Dynamics:               x'   = r * x * (1 - x / K)
            Analytical solution:    x(t) = K / (1 + D0 * exp(-r * t))

        @param ts (np.ndarray)
            Time stamps.

        @return (np.ndarray)
            Phase variable values at the given time stamps.
        """
        ts = np.asarray(ts, float)
        return self.K / (1.0 + self.D0*np.exp(-self.r*ts))
    
    def exponential_system(self, ts, start, goal, alpha=15.0):
        """
        @brief
            A dynamical system with exponential decay.
            Dynamics:               x'   = -(alpha / tau) * (x - goal)
            Analytical solution:    x(t) = goal + (start - goal) * exp(-(alpha / tau) * t)

        @param ts (np.ndarray)
            Time stamps.
        @param start (np.ndarray)
            Initial position vector.
        @param goal (np.ndarray)
            Final position vector.
        @param alpha (float)
            Decay rate.

        @return (np.ndarray)
            Phase variable values at the given time stamps.
        """
        ts      = np.asarray(ts, float)
        start   = np.asarray(start, float).reshape(3) 
        goal    = np.asarray(goal, float).reshape(3)
        return goal[None,:] + (start - goal)[None,:] * np.exp(-(alpha / self.tau) * ts)[:,None]
    
    def polynomial_system(self, ts, start, goal, alpha=15.0):
        """
        @brief
            A dynamical system with polynomial decay.
            Dynamics:               x'   = -(alpha / tau) * (x - goal)^(1 - 1/alpha)
            Analytical solution:    x(t) = goal + (start - goal) * (1 - t / tau)^alpha

        @param ts (np.ndarray)
            Time stamps.
        @param start (np.ndarray)
            Initial position vector.
        @param goal (np.ndarray)
            Final position vector.
        @param alpha (float)
            Decay rate.

        @return (np.ndarray)
            Phase variable values at the given time stamps.
        """
        ts    = np.asarray(ts, float)
        s     = np.clip(ts / self.tau, 0.0, 1.0)          
        start = np.asarray(start, float).reshape(3)
        goal  = np.asarray(goal,  float).reshape(3)
        return goal[None,:] + (start - goal)[None,:] * (1.0 - s)[:,None]**alpha
