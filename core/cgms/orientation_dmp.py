"""
Orientation DMP — spring-damper system in quaternion tangent space.

Operates on the log-map error between current and goal quaternion.
Uses same RBF basis as position DMP for the forcing term.

Equations (in tangent space ℝ³):
    e(t) = log(q_goal(t)⁻¹ ⊗ q(t))         — orientation error
    τ² ω̇ = k_ori · e(t) - τ · d_ori · ω + γ(t) · f_ori(s)
    q̇   integrated via exponential map

Convention:  q = [w, x, y, z]  (scalar-first)
"""

import numpy as np
from .quat_utils import (
    quat_normalize, quat_error, quat_integrate, quat_slerp,
)
from .function_approximator import FunctionApproximatorRBFN
from .dynamical_systems import DynamicalSystems


class OrientationDMP:
    """
    Orientation DMP with RBF forcing in quaternion log-space.
    """

    def __init__(self, q_start, q_end, tau, dt, n_bfs_ori=15):
        """
        Parameters
        ----------
        q_start   : array (4,)  — start quaternion [w,x,y,z]
        q_end     : array (4,)  — goal quaternion  [w,x,y,z]
        tau       : float       — phase duration (seconds)
        dt        : float       — timestep
        n_bfs_ori : int         — number of RBF basis functions for orientation forcing
        """
        self.q_start = quat_normalize(np.asarray(q_start, float))
        self.q_end   = quat_normalize(np.asarray(q_end, float))
        self.tau     = float(tau)
        self.dt      = float(dt)
        self.ts      = np.arange(0.0, self.tau + 1e-12, self.dt)
        self.T       = self.ts.size
        self.n_bfs   = n_bfs_ori

        # Spring-damper gains (critically damped in log-space)
        self.d_ori = 20.0
        self.k_ori = (self.d_ori ** 2) / 4.0

        self.ds = DynamicalSystems(self.tau)

        # 3 independent RBF networks — one per tangent-space axis
        self.rbf_ori = [
            FunctionApproximatorRBFN(n_bfs_ori, normalize=True, intersection_height=0.95)
            for _ in range(3)
        ]

        # Train RBFs to reproduce zero forcing (identity rollout = SLERP-like)
        phase = self.ds.time_system(self.ts)
        zero_forcing = np.zeros((self.T, 3))
        for i in range(3):
            self.rbf_ori[i].train(phase, zero_forcing[:, i])

    # ------------------------------------------------------------------ #
    def n_weights(self):
        """Total learnable weights: n_bfs × 3 axes."""
        return self.n_bfs * 3

    def initial_weights(self):
        """Return concatenated weight vector [axis_x | axis_y | axis_z]."""
        return np.concatenate([r.W.ravel() for r in self.rbf_ori])

    def set_weights(self, w_ori):
        """Set orientation RBF weights from flat vector."""
        off = 0
        for r in self.rbf_ori:
            n = r.W.size
            r.W = w_ori[off:off + n].reshape(r.W.shape)
            off += n

    # ------------------------------------------------------------------ #
    def _goal_at_time(self, t):
        """
        Time-varying goal quaternion.

        Default: constant q_end for the whole phase.
        Override this for time-varying goals (e.g., SLERP transition
        in the last N seconds).
        """
        return self.q_end.copy()

    # ------------------------------------------------------------------ #
    def rollout(self):
        """
        Integrate orientation DMP using RK4 in log-space.

        Returns
        -------
        dict with keys:
            q_des   : ndarray (T, 4)  — desired quaternion trajectory
            omega   : ndarray (T, 3)  — angular velocity trajectory
        """
        ts = self.ts
        T  = self.T
        tau = self.tau

        q_traj = np.zeros((T, 4))
        omega  = np.zeros((T, 3))
        q_traj[0] = self.q_start.copy()

        def _dmp_accel(t, q, w):
            """Compute angular acceleration  ω̇  at state (q, ω)."""
            phase = self.ds.time_system(np.array([t]))[0]
            gate  = phase  # gating decays from 1→0

            # Orientation error in tangent space
            # e = log(q_goal⁻¹ ⊗ q) points FROM goal TO current
            # Spring should pull TOWARD goal → use -e
            q_goal = self._goal_at_time(t)
            e = quat_error(q, q_goal)  # ℝ³ log-error

            # RBF forcing (per axis)
            f_ori = np.array([
                self.rbf_ori[i].predict(phase)[0, 0] for i in range(3)
            ])

            # Spring-damper + forcing
            # τ² ω̇ = -k·e - τ·d·ω + gate·f_ori
            acc = (-self.k_ori * e - tau * self.d_ori * w + gate * f_ori) / (tau ** 2)
            return acc

        for k in range(T - 1):
            t0 = ts[k]
            h  = ts[k + 1] - ts[k]
            q0 = q_traj[k]
            w0 = omega[k]

            # RK4 for angular velocity (ω)
            k1w = _dmp_accel(t0,           q0,                              w0)
            k1q_mid = quat_integrate(q0, w0 + 0.5 * h * k1w, 0.0)  # just use q0 for mid
            k2w = _dmp_accel(t0 + 0.5 * h, q0,                              w0 + 0.5 * h * k1w)
            k3w = _dmp_accel(t0 + 0.5 * h, q0,                              w0 + 0.5 * h * k2w)
            k4w = _dmp_accel(t0 + h,        q0,                              w0 + h * k3w)

            omega[k + 1] = w0 + (h / 6.0) * (k1w + 2 * k2w + 2 * k3w + k4w)

            # Integrate quaternion using midpoint angular velocity
            w_mid = 0.5 * (omega[k] + omega[k + 1])
            q_traj[k + 1] = quat_integrate(q0, w_mid, h)

        return {"q_des": q_traj, "omega": omega}
