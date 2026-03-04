import numpy as np

class PI2:
    """
    PI^2 style updater over a parameter vector theta.
    """
    def __init__(self, theta, sigma, lam=1.0, decay=0.98, seed=0):
        """
        @param theta    (np.ndarray)
            Initial guess of parameters
        @param sigma    (np.ndarray)
            Exploration noise per parameter 
        @param lam      (float)
            Temperature parameter 
        @param decay    (float)
            Moving average for smoothing of covariance
        """
        self.mean  = theta.copy()
        self.sigma = sigma.copy()
        self.lam   = float(lam)
        self.decay = float(decay)
        self.rng   = np.random.default_rng(int(seed))

    def sample(self, n):
        """
        @param n    (float)
            Number of sample in each iteration

        @return     (np.ndarray (n, p))
            Samples of p dimensional parameter vector drawn from ~ N(mean, diag(covariance))
        """
        # unscaled noise matrix of shape (n, p) where each element is ~ N(0, 1)
        z = self.rng.normal(0.0, 1.0, size=(n, self.mean.size))
        return self.mean[None, :] + z * self.sigma[None, :]

    def _weights_from_costs(self, costs):
        """
        @param      (np.ndarray (n, 1))
            Vector of cost for each sample of an iteration
        """
        # PI^2 uses a temperature λ without max-min range scaling
        cmin = float(np.min(costs))
        lam  = max(1e-12, self.lam)
        w = np.exp(-(costs - cmin) / lam)
        return w / (np.sum(w) + 1e-12)

    def update(self, samples, costs):
        """
        @brief 
            Update the mean and standard deviation of the distribution based on sample rollouts.

        @param samples      (np.ndarray (n, p))
            Sample drawn from ~ N(mean, diag(covariance))
        @param costs        (np.ndarray (n,))
            Cost associated with each sampled parameter vector
        
        @return new_mean    (np.ndarray, shape = (p,))
            Updated parameter mean.
        @return new_sigma   (np.ndarray, shape = (p,))
            Updated standard deviation per parameter dimension.
        @return w           (np.ndarray, shape = (n,))
            Normalized importance weights for each sample.
        """
        w = self._weights_from_costs(costs)                             # (N,)
        new_mean = np.sum(samples * w[:, None], axis=0)                 # (P,)
        # weighted diagonal covariance estimate with exponential weights
        diff2 = np.sum(w[:, None] * (samples - new_mean[None, :])**2, axis=0)
        new_sigma = np.sqrt(self.decay * self.sigma**2 + (1.0 - self.decay) * diff2 + 1e-12)
        self.mean, self.sigma = new_mean, new_sigma
        return new_mean, new_sigma, w


class PIBB:
    """
    PI-BB style updater over a parameter vector theta.
    """
    def __init__(self, theta, sigma, beta=8.0, decay=0.98, seed=0):
        """
        @param theta    (np.ndarray)
            Initial guess of parameters
        @param sigma    (np.ndarray)
            Exploration noise per parameter 
        @param beta     (float)
            Softmax sharpness in cost-to-weight 
        @param decay    (float)
            Moving average for smoothing of covariance
        """
        self.mean  = theta.copy()
        self.sigma = sigma.copy()
        self.beta  = float(beta)
        self.decay = float(decay)
        self.rng   = np.random.default_rng(int(seed))

    def sample(self, n):
        """
        @param n    (float)
            Number of sample in each iteration

        @return     (np.ndarray (n, p))
            Samples of p dimensional parameter vector drawn from ~ N(mean, diag(covariance))
        """
        # unscaled noise matrix of shape (n, p) where each element is ~ N(0, 1)
        z = self.rng.normal(0.0, 1.0, size=(n, self.mean.size))
        return self.mean[None, :] + z * self.sigma[None, :]

    def _weights_from_costs(self, costs):
        """
        @param      (np.ndarray (n, 1))
            Vector of cost for each sample of an iteration
        """
        # minimum and maximum cost values among all sampled rollouts within a single iteration 
        cmin, cmax = float(np.min(costs)), float(np.max(costs))
        scale = max(1e-12, cmax - cmin)
        w = np.exp(-self.beta * (costs - cmin) / scale)
        return w / (np.sum(w) + 1e-12)

    def update(self, samples, costs):
        """
        @brief 
            Update the mean and standard deviation of the distribution based on sample rollouts.

        @param samples      (np.ndarray (n, p))
            Sample drawn from ~ N(mean, diag(covariance))
        @param costs        (np.ndarray (n,))
            Cost associated with each sampled parameter vector
        
        @return new_mean    (np.ndarray, shape = (p,))
            Updated parameter mean.
        @return new_sigma   (np.ndarray, shape = (p,))
            Updated standard deviation per parameter dimension.
        @return w           (np.ndarray, shape = (n,))
            Normalized importance weights for each sample.
        """
        w = self._weights_from_costs(costs)                             # (N,)
        new_mean = np.sum(samples * w[:, None], axis=0)                 # (P,)
        diff2 = np.sum(w[:, None] * (samples - new_mean[None, :])**2, axis=0)
        new_sigma = np.sqrt(self.decay * self.sigma**2 + (1.0 - self.decay) * diff2 + 1e-12)
        self.mean, self.sigma = new_mean, new_sigma
        return new_mean, new_sigma, w

