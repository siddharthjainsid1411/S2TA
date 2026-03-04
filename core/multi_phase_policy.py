"""
Multi-phase certified policy.

Chains N DMP segments end-to-end.  Each segment has its own start/end,
duration, and learnable weights.  The traces are concatenated into a
single Trace so the existing temporal-logic compiler works unchanged.

Phase layout (example — 3 phases):
    Phase 1: start → waypoint       (duration_1)
    Phase 2: waypoint → waypoint    (hold — duration_2, same start/end)
    Phase 3: waypoint → goal        (duration_3)

Total horizon = sum(duration_i).

Theta layout:
    [phase_1 weights | phase_2 weights | phase_3 weights]
Each phase has its own theta_dim (may differ if n_bfs differ, but
we keep them identical for simplicity).
"""

import numpy as np
from core.cgms.dmp_with_gain import DMPWithGainScheduling
from core.cgms.dynamical_systems import DynamicalSystems
from core.cgms.orientation_dmp import OrientationDMP
from core.cgms.quat_utils import quat_normalize
from core.certified_policy import Trace
from core.obstacle_projection import ObstacleProjector


class MultiPhaseCertifiedPolicy:
    """
    Chains multiple DMP segments.  Each phase is an independent
    DMPWithGainScheduling instance with its own weights.
    """

    def __init__(self, phases, K0=200.0, D0=30.0):
        """
        Parameters
        ----------
        phases : list of dict
            Each dict contains:
                start       : np.ndarray (3,)
                end         : np.ndarray (3,)
                duration    : float        (seconds)
                n_bfs_traj  : int          (default 51)
                n_bfs_slack : int          (default 7)
            Optional (orientation):
                start_quat  : list/array (4,)  [w,x,y,z]
                end_quat    : list/array (4,)  [w,x,y,z]
                n_bfs_ori   : int          (default 15)
        K0 : float
            Nominal stiffness per axis (N/m).  Default 200.
        D0 : float
            Nominal damping per axis (Ns/m).  Default 30.
        """
        self.phases = phases
        self.dmps = []
        self.ori_dmps = []           # OrientationDMP or None per phase
        self.sizes_list = []         # (n_traj, n_sd, n_sk) per phase
        self.ori_dims = []           # orientation theta dim per phase (0 if no ori)
        self.theta_dims = []         # total theta_dim per phase (pos + ori + SK/SD)
        self.offsets = []            # cumulative offsets into flat theta
        self.has_orientation = False # True if ANY phase has quaternions

        DT    = 0.01
        ALPHA = 0.05
        H     = np.eye(3)

        offset = 0
        for p in phases:
            dmp = DMPWithGainScheduling(
                start=np.asarray(p["start"], float),
                end=np.asarray(p["end"], float),
                tau=p["duration"],
                dt=DT,
                n_bfs_traj=p.get("n_bfs_traj", 51),
                n_bfs_slack=p.get("n_bfs_slack", 7),
                K0=K0, D0=D0, alpha=ALPHA, H=H,
            )
            theta_init, n_traj, n_sd, n_sk = dmp.initial_weights()

            # --- Orientation DMP (optional) ---
            ori_dmp = None
            ori_dim = 0
            if "start_quat" in p and "end_quat" in p:
                self.has_orientation = True
                q_start = quat_normalize(np.asarray(p["start_quat"], float))
                q_end   = quat_normalize(np.asarray(p["end_quat"], float))
                n_bfs_ori = p.get("n_bfs_ori", 15)
                ori_dmp = OrientationDMP(
                    q_start=q_start, q_end=q_end,
                    tau=p["duration"], dt=DT,
                    n_bfs_ori=n_bfs_ori,
                )
                ori_dim = ori_dmp.n_weights()

            # Theta layout per phase: [traj_xyz | ori_xyz | SD | SK]
            dim = len(theta_init) + ori_dim

            self.dmps.append(dmp)
            self.ori_dmps.append(ori_dmp)
            self.sizes_list.append((n_traj, n_sd, n_sk))
            self.ori_dims.append(ori_dim)
            self.theta_dims.append(dim)
            self.offsets.append(offset)
            offset += dim

        self.total_theta_dim = offset
        self.DT = DT
        # Hard obstacle avoidance projector (off by default — backward compatible)
        self._projector = ObstacleProjector()

    # ------------------------------------------------------------------ #
    def setup_hard_obstacles_from_taskspec(self, taskspec):
        """
        Automatically wire Layers 1+2 (DMP repulsion + radial projector) for
        all clauses declared with modality="HARD" in the JSON task spec.

        Call this immediately after constructing the policy, before rollout.
        The obstacle specs (center, radius, strength, infl_factor) are
        extracted by json_parser and stored on taskspec.hard_obstacle_specs.

        Example
        -------
            policy = MultiPhaseCertifiedPolicy(taskspec.phases)
            policy.setup_hard_obstacles_from_taskspec(taskspec)
            # No manual set_obstacles() needed.
        """
        specs = getattr(taskspec, "hard_obstacle_specs", [])
        if specs:
            self.set_obstacles(specs)

    # ------------------------------------------------------------------ #
    def set_obstacles(self, obstacles):
        """
        Register spherical obstacles with three-tier avoidance control.

        Parameters
        ----------
        obstacles : list of dict, each with keys:
            "center"      : array-like (3,)
            "radius"      : float              — safe clearance radius
            "avoidance"   : str    (optional)  — one of "HARD", "SOFT", "NONE".
                                                 Default: "HARD".
            "strength"    : float  (optional)  — DMP repulsion strength (default 0.05).
            "infl_factor" : float  (optional)  — influence zone = radius * infl_factor
                                                 (default 2.5)

        Backward-compatible alias (deprecated — prefer "avoidance"):
            "hard"        : bool   (optional)  — True → "HARD", False → "SOFT".
                                                 Ignored if "avoidance" is also set.

        Three tiers
        -----------
        HARD  (avoidance="HARD", default)
          Layer 1 — DMP repulsive forcing inside ODE: organically routes the
                    spring-damper attractor around obstacle. Smooth arc, no C-turn.
          Layer 2 — Hard radial projector post-rollout: backstop guarantee.
                    ∀t: ||p(t) − c|| ≥ radius  — by construction, unbreakable.
          Use for: objects that must not be hit (fragile mug, human, wall).

        SOFT  (avoidance="SOFT")
          Layer 1 — DMP repulsive forcing inside ODE only: path PREFERS to arc
                    around the obstacle but is NOT guaranteed to stay outside.
                    No projector backstop. Optimizer (PIBB) sees soft cost.
          Use for: preferred avoidance but occasional penetration is acceptable.

        NONE  (avoidance="NONE")
          No DMP repulsion, no projector — pure original spring-damper behavior.
          The optimizer sees only the soft ObstacleAvoidance clause cost (if
          registered in the task spec) but there is no geometric forcing at all.
          The trajectory may freely penetrate the obstacle zone.
          Use for: harmless objects where the ball can pass through freely
                   (e.g. ball delivery through a gate or past a marker).

        Examples
        --------
        # Hard avoidance (guaranteed — mug-carry scene):
        policy.set_obstacles([
            {"center": [0.40, 0.30, 0.30], "radius": 0.12,
             "avoidance": "HARD", "strength": 0.05, "infl_factor": 2.0},
        ])

        # None avoidance (raw, may penetrate freely — ball delivery scene):
        policy.set_obstacles([
            {"center": [0.40, 0.30, 0.30], "radius": 0.12,
             "avoidance": "NONE"},
        ])
        """
        def _mode(obs):
            """Resolve avoidance mode string, with backward-compat for 'hard' key."""
            if "avoidance" in obs:
                return obs["avoidance"].upper()
            # Legacy: hard=True → HARD, hard=False → SOFT (NOT NONE)
            if "hard" in obs:
                return "HARD" if obs["hard"] else "SOFT"
            return "HARD"   # default

        # ── Hard projector: only for HARD obstacles ─────────────────────
        hard_obs = [obs for obs in obstacles if _mode(obs) == "HARD"]
        self._projector = ObstacleProjector(hard_obs)

        # ── DMP repulsive forcing: HARD and SOFT obstacles only.
        # NONE obstacles get neither projector nor repulsion — raw behavior.
        rep_obs = []
        for obs in obstacles:
            if _mode(obs) == "NONE":
                continue          # skip — no forcing at all for this obstacle
            c      = np.asarray(obs["center"], float)
            r      = float(obs["radius"])
            s      = float(obs.get("strength",    0.05))
            ifact  = float(obs.get("infl_factor", 2.5))
            rep_obs.append({
                "center":  c,
                "radius":  r,
                "r_infl":  r * ifact,
                "strength": s,
            })

        # Inject into every phase DMP
        for dmp in self.dmps:
            dmp.repulsive_obstacles = rep_obs

    # ------------------------------------------------------------------ #
    def parameter_dimension(self):
        return self.total_theta_dim

    # ------------------------------------------------------------------ #
    def structured_sigma(self, sigma_traj_xy=5.0, sigma_traj_z=5.0,
                         sigma_sd=5.0, sigma_sk=5.0, sigma_ori=2.0):
        """Per-parameter exploration noise — concatenated over all phases."""
        parts = []
        for idx, (dmp, sizes) in enumerate(zip(self.dmps, self.sizes_list)):
            n_traj, n_sd, n_sk = sizes
            n_per_axis = n_traj // 3
            # Position + ori + SK/SD
            sigma = np.empty(self.theta_dims[idx])
            off = 0
            sigma[off:off + n_per_axis] = sigma_traj_xy; off += n_per_axis
            sigma[off:off + n_per_axis] = sigma_traj_xy; off += n_per_axis
            sigma[off:off + n_per_axis] = sigma_traj_z;  off += n_per_axis
            # Orientation weights (if present)
            ori_dim = self.ori_dims[idx]
            if ori_dim > 0:
                sigma[off:off + ori_dim] = sigma_ori;    off += ori_dim
            sigma[off:off + n_sd]       = sigma_sd;       off += n_sd
            sigma[off:off + n_sk]       = sigma_sk;       off += n_sk
            parts.append(sigma)
        return np.concatenate(parts)

    # ------------------------------------------------------------------ #
    def rollout(self, theta):
        """
        Run all DMP phases in sequence, stitching position, velocity,
        and (optionally) orientation continuously.
        Returns a single Trace covering [0, total_horizon].
        """
        all_time = []
        all_pos  = []
        all_vel  = []
        all_K    = []
        all_D    = []
        all_raw_sk = []
        all_raw_sd = []
        all_quat  = []      # quaternion trajectories
        all_omega = []      # angular velocity trajectories

        t_offset = 0.0    # global time offset for concatenation

        for idx, (dmp, ori_dmp, sizes, tdim) in enumerate(
            zip(self.dmps, self.ori_dmps, self.sizes_list, self.theta_dims)
        ):
            off = self.offsets[idx]
            theta_phase = theta[off:off + tdim]

            # Ensure DMP internal timing is correct
            dur = self.phases[idx]["duration"]
            dmp.tau = dur
            dmp.ts  = np.arange(0.0, dur + 1e-12, dmp.dt)
            dmp.T   = dmp.ts.size
            dmp.ds  = DynamicalSystems(dur)

            # --- Slice theta: [traj_xyz | ori_xyz | SD | SK] ---
            n_traj, n_sd, n_sk = sizes
            ori_dim = self.ori_dims[idx]

            # Position forcing weights
            pos_weights = theta_phase[:n_traj]
            ptr = n_traj

            # Orientation weights (if present)
            if ori_dim > 0:
                ori_weights = theta_phase[ptr:ptr + ori_dim]
                ptr += ori_dim

            # SK/SD weights (raw, for stiffness penalty)
            raw_sd = theta_phase[ptr:ptr + n_sd].copy()
            raw_sk = theta_phase[ptr + n_sd:ptr + n_sd + n_sk].copy()

            # Build position-only theta for set_theta (same layout as before)
            pos_theta = np.concatenate([pos_weights, theta_phase[ptr:ptr + n_sd + n_sk]])
            dmp.set_theta(pos_theta, sizes)
            plan = dmp.rollout_traj()

            ts    = plan["ts"] + t_offset
            y     = plan["y_des"]
            yd    = plan["yd_des"]
            K     = plan["K"]
            D     = plan["D"]

            # Pass final Q of this phase as initial Q of next phase → no jerk at boundary
            if idx + 1 < len(self.dmps):
                Q_final = plan.get("Q_final", None)
                if Q_final is not None:
                    self.dmps[idx + 1].Q_init = Q_final

            # --- Orientation rollout (if present) ---
            q_des = None
            omega = None
            if ori_dmp is not None and ori_dim > 0:
                ori_dmp.tau = dur
                ori_dmp.ts  = np.arange(0.0, dur + 1e-12, ori_dmp.dt)
                ori_dmp.T   = ori_dmp.ts.size
                ori_dmp.ds  = DynamicalSystems(dur)
                ori_dmp.set_weights(ori_weights)
                ori_plan = ori_dmp.rollout()
                q_des = ori_plan["q_des"]
                omega = ori_plan["omega"]

            # For phases after the first, drop the first timestep to
            # avoid duplicate time=boundary points.
            if idx > 0:
                ts = ts[1:]
                y  = y[1:]
                yd = yd[1:]
                K  = K[1:]
                D  = D[1:]
                if q_des is not None:
                    q_des = q_des[1:]
                    omega = omega[1:]

            all_time.append(ts)
            all_pos.append(y)
            all_vel.append(yd)
            all_K.append(K)
            all_D.append(D)
            all_raw_sk.append(raw_sk)
            all_raw_sd.append(raw_sd)
            if q_des is not None:
                all_quat.append(q_des)
                all_omega.append(omega)

            t_offset = ts[-1]

        # Stitch orientation if any phase had it
        orientation = None
        angular_velocity = None
        if self.has_orientation and len(all_quat) > 0:
            orientation = np.concatenate(all_quat, axis=0)
            angular_velocity = np.concatenate(all_omega, axis=0)

        # ── Hard obstacle projection (by construction) ─────────────────
        # Project positions outside all registered obstacle spheres BEFORE
        # building the Trace.  CGMS gains are untouched — they are computed
        # by the Cholesky ODE inside each phase's DMP rollout_traj() call
        # and do not depend on position values.
        pos_full = np.concatenate(all_pos)
        vel_full = np.concatenate(all_vel)
        pos_full, vel_full = self._projector.project(pos_full, vel_full, self.DT)

        trace = Trace(
            time=np.concatenate(all_time),
            position=pos_full,
            velocity=vel_full,
            gains={
                "K": np.concatenate(all_K),
                "D": np.concatenate(all_D),
            },
            raw_sk_weights=np.concatenate(all_raw_sk),
            raw_sd_weights=np.concatenate(all_raw_sd),
            orientation=orientation,
            angular_velocity=angular_velocity,
        )
        return trace
