import numpy as np

class FunctionApproximatorRBFN:
    """
    Radial Basis Function (RBF) approximator with Gaussian kernels.
    """
    def __init__(self, n_bfs: int, intersection_height: float = 0.7, regularization: float = 1e-6, normalize: bool = True):
        """
        @param n_bfs (int)
            Number of RBFs.
        @param intersection_height (float)
            Height at which two neighbouring basis functions intersect.
        @param regularization (float)
            Regularization term for the least squares solution.
        @param normalize (bool)
            Whether to normalize the RBF outputs to sum to 1.
        """
        self.M          = int(n_bfs)
        self.h          = float(intersection_height)
        self.reg        = float(regularization)
        self.normalize  = bool(normalize)

        self.centers = None    # shape (M, 1)
        self.widths  = None    # shape (M, 1)
        self.W       = None    # shape (M, D)

    def _compute_centers_widths(self):
        """
        @brief
            Evenly space centers in [0, 1], calculate widths from intersection height h.
        """
        if self.M > 1:
            self.centers    = np.linspace(0, 1, self.M).reshape(-1,1)
            delta           = self.centers[1] - self.centers[0]
            # Standard deviation (sigma) for Gaussian basis functions such that two neighbouring functions intersect at height h
            sigma           = float(delta) / np.sqrt(-8.0 * np.log(self.h))
            self.widths     = np.full((self.M, 1), sigma)
        else:
            self.centers = np.array([[0.5]])
            self.widths  = np.array([[1.0]])

    def _activations(self, x):
        """
        @brief
            Compute the RBF activations for input x.

        @param x (np.ndarray)
            Input values, shape (T,).
        """
        X   = np.asarray(x, float).reshape(-1,1)        # shape (T, 1)
        C   = self.centers.T                            # shape (1, M)
        W   = self.widths.T                             # shape (1, M)
        phi = np.exp(-0.5 * ((X - C) / W)**2)           # shape (T, M)
        if self.normalize:
            s   = np.sum(phi, axis=1, keepdims=True) + 1e-12
            phi = phi/s
        return phi
    
    def train(self, x, fx):
        """
        @brief
            Train the RBF weights W such that psi(x) @ W = fx, using regularized least squares.

        @param x (np.ndarray)
            Input values, shape (T,).
        @param fx (np.ndarray)
            Target output values, shape (T, D).
        """
        x   = np.asarray(x, float).reshape(-1)
        FX  = np.asarray(fx, float)

        if FX.ndim == 1:
            FX = FX[:,None]

        # 1) set up Gaussians 
        self._compute_centers_widths()

        # 2) build activation matrix
        PSI = self._activations(x)

        # 3) solve for W using regularized least squares
        A   = PSI.T @ PSI + self.reg * np.eye(self.M)
        B   = PSI.T @ FX
        try: 
            self.W = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            self.W, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

    def predict(self,x):
        """
        @brief
            Predict output for input x using the trained RBF weights.

        @param x (np.ndarray)
            Input values, shape (T,).
            
        @return (np.ndarray)
            Predicted output values, shape (T, D).
        """
        if self.W is None: 
            raise RuntimeError("RBF not trained")
        PSI = self._activations(np.asarray(x, float).reshape(-1))
        return PSI @ self.W
    
    def activations_and_time_derivative(self, x, xdot):
        X  = np.asarray(x, float).reshape(-1, 1)            # (T,1)
        xd = np.asarray(xdot, float).reshape(-1, 1)         # (T,1)
        C  = self.centers.T                                 # (1,M)
        W  = self.widths.T                                  # (1,M)

        G   = np.exp(-0.5 * ((X - C) / W)**2)               # (T,M)
        dGdx = G * (-(X - C) / (W**2))                      # (T,M)
        dGdt = dGdx * xd                                    # (T,M)  chain rule

        if self.normalize:
            s   = np.sum(G, axis=1, keepdims=True) + 1e-12  # (T,1)
            sd  = np.sum(dGdt, axis=1, keepdims=True)       # (T,1)
            Phi = G / s
            dPhi_dt = (dGdt * s - G * sd) / (s**2)          # quotient rule
        else:
            Phi = G
            dPhi_dt = dGdt

        return Phi, dPhi_dt

    def predict_with_time_derivative(self, x, xdot):
        if self.W is None:
            raise RuntimeError("RBF not trained")
        Phi, dPhi_dt = self.activations_and_time_derivative(x, xdot)
        return Phi @ self.W, dPhi_dt @ self.W
