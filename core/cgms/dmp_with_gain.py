import numpy as np
from .utils import lt_pack, lt_unpack, sym
from .minimum_jerk import MinimumJerk
from .dynamical_systems import DynamicalSystems
from .function_approximator import FunctionApproximatorRBFN

class DMPWithGainScheduling:
    """
    DMP with gain scheduling via RBFs for trajectory and gain modulation.
    """
    def __init__(self, start, end, tau, dt, n_bfs_traj, n_bfs_slack, K0, D0, alpha, H, 
                normalize_rbfs_traj=True, normalize_rbfs_slack=True, 
                slack_mag=20.0, slack_rate_limit=200.0
        ):
        """
        @param start                    (np.ndarray)
            Start position vector. 
        @param end                      (np.ndarray)
            End position vector.
        @param tau                      (float)
            Duration of the trajectory.
        @param dt                       (float)   
            Time step for discretization.
        @param n_bfs_traj               (int)
            Number of RBFs for trajectory generation.
        @param n_bfs_slack              (int)    
            Number of RBFs for gain scheduling.
        @param K0                       (float)   
            Nominal stiffness gain.
        @param D0                       (float)
            Nominal damping gain.
        @param alpha                    (float)
            Gain scheduling scaling factor.
        @param H                        (np.ndarray)
            Desired stiffness matrix for the task.
        @param normalize_rbfs_traj      (bool)
            Whether to normalize trajectory RBF outputs.
        @param normalize_rbfs_slack     (bool)
            Whether to normalize gain scheduling RBF outputs.
        @param slack_mag                (float)    
            Maximum magnitude for slack variables.
        @param slack_rate_limit         (float) 
            Maximum rate of change for slack variables.
        """
        self.start      = np.asarray(start, float).reshape(3)
        self.end        = np.asarray(end, float).reshape(3)
        self.tau        = float(tau)
        self.dt         = float(dt)
        self.ts         = np.arange(0.0, self.tau+1e-12, self.dt)
        self.T          = self.ts.size
        self.alpha      = float(alpha)
        self.H          = np.asarray(H, float).reshape(3, 3)
        self.K0         = float(K0)
        self.D0         = float(D0)
        self.slack_mag  = float(slack_mag)
        self.slack_rate = float(slack_rate_limit)
        self.Q_init     = None   # set by multi_phase_policy for phase continuity
        # Repulsive obstacles: list of {"center": np.array(3), "radius": float,
        #                                "strength": float}
        # Injected by MultiPhaseCertifiedPolicy.set_obstacles() before rollout.
        # Applied INSIDE the DMP ODE so the spring-damper attractor routes around
        # the obstacle organically — goal-seeking is preserved.
        # The radial projector in obstacle_projection.py remains as the hard
        # backstop guarantee; this repulsion only shapes the DMP trajectory.
        self.repulsive_obstacles = []
        self.ds         = DynamicalSystems(self.tau)
        
        y, yd, ydd, ts  = MinimumJerk(self.start, self.end, self.tau, self.dt).generate()
        phase           = self.ds.time_system(ts)
        goal            = self.ds.polynomial_system(ts, self.start, self.end, 3)

        self.d = 20.0
        self.m = 1.0
        d, m = self.d, self.m
        k    = (d**2) / 4.0
        
        """
        Initialize trajectory RBFs by computing target forcing term
        """
        self.rbf_traj   = [FunctionApproximatorRBFN(n_bfs_traj, normalize=normalize_rbfs_traj, intersection_height = 0.95) for _ in range(3)]
        spring      = k * (y - goal)
        damper      = d * self.tau * yd
        f_target    = (self.tau**2) * ydd + (spring + damper) / m
        f_target    = f_target / (phase[:,None] + 1e-12)
        for i in range(3): 
            self.rbf_traj[i].train(phase, f_target[:,i])

        """
        Initialize slacks for constant gains pre-sampling 
        """
        self.rbf_SD = FunctionApproximatorRBFN(n_bfs_slack, normalize=normalize_rbfs_slack, intersection_height = 0.7)
        self.rbf_SK = FunctionApproximatorRBFN(n_bfs_slack, normalize=normalize_rbfs_slack, intersection_height = 0.7)
        I = np.eye(3)
        H = self.H
        # We want to find SK0 such that SK0^2 = 2*alpha*K0*I or SK0 = sqrt(2*alpha*K0)*I
        SK0 = np.sqrt(max(0.0, 2*alpha*K0)) * I
        # We want to find SD0 such that SD0^2 = D0*I - alpha*H or SD0 = sqrt(D0*I - alpha*H)
        # Perform eigen-decomposition to calculate square root of SD0
        w, V = np.linalg.eigh(sym(D0 * I - alpha*H))
        w = np.clip(w, 0, None)
        SD0 = (V * np.sqrt(w)) @ V.T
        SK = np.tile(lt_pack(SK0)[None,:], (ts.size,1))
        SD = np.tile(lt_pack(SD0)[None,:], (ts.size,1))
        self.rbf_SK.train(phase, SK)
        self.rbf_SD.train(phase, SD)
        
    def initial_weights(self):
        """
        @brief
            Concatenate the weight matrices into a single vector for optimization.
        """
        theta = np.concatenate([r.W.ravel() for r in self.rbf_traj] + [self.rbf_SD.W.ravel(), self.rbf_SK.W.ravel()])
        n_forcing_weights   = sum(r.W.size for r in self.rbf_traj)
        n_damping_weights   = self.rbf_SD.W.size
        n_stiffness_weights = self.rbf_SK.W.size
        return theta, n_forcing_weights, n_damping_weights, n_stiffness_weights
    
    def set_theta(self, theta, sizes):
        """
        @brief
            Slice the flat theta back into the weight matrices in the same order as initial_weights().

        @param theta (np.ndarray)
            Flat weight vector.
        @param sizes (Tuple[int, int, int])
            Sizes of the weight matrices: (n_forcing_weights, n_damping_weights, n_stiffness_weights).

        @note
            SD and SK weights are hard-clipped to ±SK_CLIP before being written
            into the RBF approximators.  This keeps the Cholesky ODE inputs bounded
            (B = -α Ḋ - SK SK^T stays finite) so that K = Q^T Q never diverges
            during exploration.  The stiffness penalty in compiler.py provides the
            honest PI2 cost signal that steers the mean away from this boundary;
            the clip only acts as a safety net for extreme samples.
        """
        SK_CLIP = 15.0   # ±15 → max ||SK||_F^2 ≈ 1350 → tr(K) stays in low thousands (N/m)
        _, n_damping_weights, n_stiffness_weights = sizes
        off = 0
        for r in self.rbf_traj:
            n   = r.W.size
            r.W = theta[off:off + n].reshape(r.W.shape)
            off += n
        self.rbf_SD.W   = np.clip(theta[off:off + n_damping_weights], -SK_CLIP, SK_CLIP).reshape(self.rbf_SD.W.shape)
        off             +=n_damping_weights
        self.rbf_SK.W   = np.clip(theta[off:off + n_stiffness_weights], -SK_CLIP, SK_CLIP).reshape(self.rbf_SK.W.shape)

    def rollout_traj(self, sample_unsafe: bool = False):
        ts   = self.ts
        T    = self.T
        y    = np.zeros((T,3))
        yd   = np.zeros((T,3))
        ydd  = np.zeros((T,3))
        y[0] = self.start

        def dmp(t, y, yd):
            phase = self.ds.time_system(np.array([t]))[0]
            gate  = phase
            goal  = self.ds.polynomial_system(np.array([t]), self.start, self.end, 3)[0]
            fhat  = np.array([self.rbf_traj[i].predict(phase)[0,0] for i in range(3)])
            d, m  = self.d, self.m
            k     = (d**2)/4.0
            spring = k * (y - goal)
            damper = d * self.tau * yd
            net_accel = ((fhat * gate) - (spring + damper) / m) / (self.tau**2)

            # ── Repulsive obstacle forcing (inside ODE — shapes the path) ──
            # For each registered obstacle, add a repulsive acceleration when
            # y is within the influence zone (d < r * INFL_FACTOR).
            # Repulsion tapers smoothly: zero at influence boundary, max at surface.
            # This steers the spring-damper ODE organically — the goal attractor
            # remains active and the path routes AROUND the obstacle.
            # The K=Q^T Q Cholesky ODE runs AFTER this position loop — untouched.
            for obs in self.repulsive_obstacles:
                diff = y - obs["center"]
                dist = float(np.linalg.norm(diff))
                r    = obs["radius"]
                r_infl = obs["r_infl"]
                if dist < 1e-9:
                    # Degenerate: exactly at centre — push in +X+Y direction
                    net_accel += obs["strength"] * np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
                elif dist < r_infl:
                    n      = diff / dist                          # outward unit normal
                    # Smooth cubic taper: alpha=1 at sphere surface (dist=r), 0 at r_infl
                    # alpha = ((r_infl - dist) / (r_infl - r))^3
                    # Bounded — no 1/dist^2 singularity.
                    # Strength is scaled by DMP spring constant so the repulsion is
                    # commensurable with the attractor force (k = d^2/4 = 100 here).
                    alpha  = ((r_infl - dist) / (r_infl - r)) ** 3
                    k_dmp  = (self.d ** 2) / 4.0   # DMP spring constant
                    mag    = obs["strength"] * k_dmp * alpha
                    net_accel += mag * n

            return net_accel

        for k in range(T-1):
            t0 = ts[k]; h = ts[k+1] - ts[k]
            k1y = yd[k];               k1v = dmp(t0,            y[k],                    yd[k])
            k2y = yd[k] + 0.5*h*k1v;   k2v = dmp(t0 + 0.5*h,    y[k] + 0.5*h*k1y,        yd[k] + 0.5*h*k1v)
            k3y = yd[k] + 0.5*h*k2v;   k3v = dmp(t0 + 0.5*h,    y[k] + 0.5*h*k2y,        yd[k] + 0.5*h*k2v)
            k4y = yd[k] + 1.0*h*k3v;   k4v = dmp(t0 + 1.0*h,    y[k] + 1.0*h*k3y,        yd[k] + 1.0*h*k3v)
            y[k+1]  = y[k]  + (h/6.0)*(k1y + 2*k2y + 2*k3y + k4y)
            yd[k+1] = yd[k] + (h/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
            ydd[k]  = k1v
        ydd[-1] = dmp(ts[-1], y[-1], yd[-1])

        x    = self.ds.time_system(ts)
        xdot = -np.ones_like(x) / self.tau

        SD_vecs, SDdot_vecs = self.rbf_SD.predict_with_time_derivative(x, xdot)     # (T,6), (T,6)
        SK_vecs              = self.rbf_SK.predict(x)                               # (T,6)

        SD   = np.array([lt_unpack(v) for v in SD_vecs])        # (T,3,3)
        SDot = np.array([lt_unpack(v) for v in SDdot_vecs])     # (T,3,3)
        SK   = np.array([lt_unpack(v) for v in SK_vecs])        # (T,3,3)

        H = self.H
        D    = np.array([sym(self.alpha*H + SD[k]@SD[k].T) for k in range(T)])                  # (T,3,3)
        Ddot = np.array([SDot[k]@SD[k].T + SD[k]@SDot[k].T for k in range(T)])                  # (T,3,3)

        def _B_at(t):
            x    = max(0.0, 1.0 - t/self.tau); xdot = -1.0/self.tau
            SDv, SDdv = self.rbf_SD.predict_with_time_derivative(np.array([x]), np.array([xdot]))
            SKv       = self.rbf_SK.predict(np.array([x]))
            SDt   = lt_unpack(SDv[0]); SDdt = lt_unpack(SDdv[0]); SKt = lt_unpack(SKv[0])
            Ddt   = SDdt@SDt.T + SDt@SDdt.T
            return sym(-self.alpha*Ddt - SKt@SKt.T)

        Q    = np.zeros((T,3,3))
        if self.Q_init is not None:
            Q[0] = self.Q_init
        else:
            Q[0] = np.linalg.cholesky(sym(self.K0*np.eye(3)) + 1e-9*np.eye(3))

        def fQ(Qk, t):
            Bk = _B_at(t)
            X  = np.linalg.solve(Qk.T, Bk)         # Q^{-T} B
            return self.alpha*Qk + 0.5*X

        for k in range(T-1):
            t = ts[k]; h = ts[k+1]-ts[k]
            k1 = fQ(Q[k],            t)
            k2 = fQ(Q[k] + 0.5*h*k1, t + 0.5*h)
            k3 = fQ(Q[k] + 0.5*h*k2, t + 0.5*h)
            k4 = fQ(Q[k] + h*k3,     t + h)
            Q[k+1] = Q[k] + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        K = np.array([sym(Q[k].T @ Q[k]) for k in range(T)])

        return {
            "ts": ts, "y_des": y, "yd_des": yd, "ydd_des": ydd,
            "SD": SD, "SK": SK,
            "D": D, "Ddot": Ddot, "K": K,
            "Q_final": Q[-1].copy()   # pass to next phase for K continuity
        }