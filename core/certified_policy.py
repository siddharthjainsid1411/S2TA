import numpy as np

from core.cgms.dmp_with_gain import DMPWithGainScheduling
from core.cgms.dynamical_systems import DynamicalSystems


class Trace:
    """
    Lightweight container for trajectory trace.
    """
    def __init__(self, time, position, velocity, gains,
                 raw_sk_weights=None, raw_sd_weights=None,
                 orientation=None, angular_velocity=None):
        self.time = time
        self.position = position
        self.velocity = velocity
        self.gains = gains
        # Pre-clip raw weights — used by compiler for honest stiffness penalty
        self.raw_sk_weights = raw_sk_weights
        self.raw_sd_weights = raw_sd_weights
        # Orientation (quaternion) and angular velocity — None when not used
        self.orientation = orientation          # (T, 4) quaternions [w,x,y,z]
        self.angular_velocity = angular_velocity  # (T, 3) rad/s


class CertifiedPolicy:

    def __init__(self, tau):

        # ---- Hardcoded for now ----
        start_pos = np.array([0.55, 0.00, 0.11])
        goal_pos  = np.array([0.05, 0.72, 0.11])

        # ---- Same hyperparameters as CGMS ----
        self.TAU = tau
        self.DT = 0.01
        self.ALPHA = 0.05
        self.K0 = 200.0
        self.D0 = 30.0

        # ---- Instantiate CGMS DMP ----
        self.dmp = DMPWithGainScheduling(
            start=start_pos,
            end=goal_pos,
            tau=self.TAU,
            dt=self.DT,
            n_bfs_traj=51,
            n_bfs_slack=7,
            K0=self.K0,
            D0=self.D0,
            alpha=self.ALPHA,
            H=np.eye(3)
        )

        # ---- Extract parameter vector info ----
        theta_init, n_traj, n_damp, n_stiff = self.dmp.initial_weights()
        self.theta_dim = len(theta_init)
        self.sizes = (n_traj, n_damp, n_stiff)

    # def parameter_dimension(self):
    #     # +1 for time scaling parameter
    #     return self.dmp.param_dim
    
    def parameter_dimension(self):
    # +1 for time scaling parameter
        return self.theta_dim 

    def structured_sigma(self, sigma_traj_xy=5.0, sigma_traj_z=5.0,
                         sigma_sd=5.0, sigma_sk=5.0):
        """
        Build a per-parameter exploration noise vector.

        Default: uniform σ=5.0 across all groups (same as original behaviour).
        Individual groups can be overridden if needed, but the clip in
        set_theta() (Fix B, ±15 on SD/SK weights) is the hard manifold
        boundary that keeps K bounded — not sigma reduction.

        Parameter layout (total = theta_dim):
          [traj_X (51)] [traj_Y (51)] [traj_Z (51)] [SD (42)] [SK (42)]
        """
        n_traj, n_sd, n_sk = self.sizes
        n_per_axis = n_traj // 3                    # 51 weights per axis

        sigma = np.empty(self.theta_dim)
        off = 0
        # X-axis trajectory weights
        sigma[off:off + n_per_axis] = sigma_traj_xy;  off += n_per_axis
        # Y-axis trajectory weights
        sigma[off:off + n_per_axis] = sigma_traj_xy;  off += n_per_axis
        # Z-axis trajectory weights
        sigma[off:off + n_per_axis] = sigma_traj_z;   off += n_per_axis
        # SD (damping slack) weights
        sigma[off:off + n_sd] = sigma_sd;              off += n_sd
        # SK (stiffness slack) weights
        sigma[off:off + n_sk] = sigma_sk;              off += n_sk

        return sigma

    # def rollout(self, theta):

    #     # ----------------------------
    #     # Split parameters
    #     # ----------------------------
    #     theta_dmp = theta[:-1]   # all except last
    #     theta_time = theta[-1]   # last element controls time

    #     # ----------------------------
    #     # Map time parameter to bounded duration
    #     # ----------------------------
    #     tau_min = 1.0
    #     tau_max = 6.0

    #     # sigmoid mapping to keep tau positive and bounded
    #     tau = tau_min + (tau_max - tau_min) * (1 / (1 + np.exp(-theta_time)))

    #     # Set DMP duration
    #     self.dmp.tau = tau
    #     self.dmp.tau = tau
    #     self.dmp.ts = np.arange(0.0, tau + 1e-12, self.dmp.dt)
    #     self.dmp.T  = self.dmp.ts.size
    #     self.dmp.ds = DynamicalSystems(tau)

    #     # ----------------------------
    #     # Set DMP parameters
    #     # ----------------------------
    #     self.dmp.set_theta(theta_dmp, self.sizes)

    #     # ----------------------------
    #     # Rollout
    #     # ----------------------------
    #     plan = self.dmp.rollout_traj()

    #     trace = Trace(
    #         time=plan["ts"],
    #         position=plan["y_des"],
    #         velocity=plan["yd_des"],
    #         gains={
    #             "K": plan["K"],
    #             "D": plan["D"]
    #         }
    #     )

    #     return trace


    def rollout(self, theta):

        # ----------------------------
        # Use full theta as DMP weights
        # ----------------------------
        theta_dmp = theta

        # ----------------------------
        # Deterministic time scaling
        # ----------------------------
        tau = self.TAU   # fixed duration (e.g., 2.0, 5.0, 10.0)

        # Update DMP internal timing
        self.dmp.tau = tau
        self.dmp.ts = np.arange(0.0, tau + 1e-12, self.dmp.dt)
        self.dmp.T  = self.dmp.ts.size
        self.dmp.ds = DynamicalSystems(tau)

        # ----------------------------
        # Set DMP parameters
        # ----------------------------
        # Extract raw SK/SD weights BEFORE set_theta clips them — needed for
        # the honest stiffness penalty in compiler.py.
        n_traj, n_sd, n_sk = self.sizes
        raw_sd = theta_dmp[n_traj:n_traj + n_sd].copy()
        raw_sk = theta_dmp[n_traj + n_sd:n_traj + n_sd + n_sk].copy()

        self.dmp.set_theta(theta_dmp, self.sizes)

        # ----------------------------
        # Rollout
        # ----------------------------
        plan = self.dmp.rollout_traj()

        trace = Trace(
            time=plan["ts"],
            position=plan["y_des"],
            velocity=plan["yd_des"],
            gains={
                "K": plan["K"],
                "D": plan["D"]
            },
            raw_sk_weights=raw_sk,
            raw_sd_weights=raw_sd,
        )

        return trace