# spec/compiler.py

from logic import predicates
from logic import temporal_logic
import numpy as np


class Compiler:

    def __init__(self, predicate_registry,
                 human_position=None, human_proximity_radius=None,
                 k_max_global=None):
        """
        Parameters
        ----------
        predicate_registry : dict   — name → callable
        human_position     : array (3,) or None
            If provided, enables the proximity-stiffness cost: when the
            end-effector is within *human_proximity_radius* of this point,
            tr(K) is penalised if it exceeds a low threshold.
            This is an implicit cost (like K_MAX ceiling) — no JSON change needed.
            When None (default), the cost is skipped → existing scenes unaffected.
        human_proximity_radius : float or None
            Activation radius (metres).  Only used when human_position is set.
        k_max_global : float or None
            Override the global tr(K) ceiling (default 600 N/m).
            Useful when K0 is raised (e.g., K0=500 → nominal tr(K)=1500,
            so the ceiling should be raised to ~1800).
            When None, defaults to 600 N/m (backward-compatible).
        """
        self.predicate_registry = predicate_registry
        self.human_position = (np.asarray(human_position, float)
                               if human_position is not None else None)
        self.human_proximity_radius = human_proximity_radius
        self.k_max_global = k_max_global

    def compile(self, taskspec):

        hard_clauses = []
        soft_clauses = []

        for clause in taskspec.clauses:
            # HARD and REQUIRE both use the slack-penalty path (Layer 3).
            # The geometric guarantee for HARD (Layers 1+2) is handled by
            # MultiPhaseCertifiedPolicy.setup_hard_obstacles_from_taskspec().
            if clause.modality in ("HARD", "REQUIRE"):
                hard_clauses.append(clause)
            else:
                soft_clauses.append(clause)

        def objective(trace):

            total_cost = 0.0

            # --- Hard/Require clauses with slack relaxation ---
            # Fixed SLACK_WEIGHT for ALL HARD/REQUIRE clauses regardless of clause.weight.
            # NOTE: clause.weight is intentionally NOT used here — it only applies to
            # PREFER clauses (soft cost path below). This is by design so that all
            # hard constraints are penalised equally strongly (500 × s²) and the tuned
            # optimisation balance from the working experiments is preserved.
            # The weight field on HARD/REQUIRE clauses is purely documentation/intent.
            SLACK_WEIGHT = 500.0

            for clause in hard_clauses:
                rho = self._evaluate_clause(trace, clause)
                s = max(0.0, -rho)
                total_cost += SLACK_WEIGHT * (s ** 2)

            # --- Soft clauses ---
            for clause in soft_clauses:
                rho = self._evaluate_clause(trace, clause)
                J = max(0.0, -rho)
                total_cost += clause.weight * J

            # --- Intrinsic stiffness regularizer (no JSON entry needed) ---
            # Two-layer defence:
            #   1. set_theta() clips SK/SD weights to ±SK_CLIP (Fix B) → ODE stays bounded
            #   2. This penalty acts on the RAW (pre-clip) theta stored on the trace,
            #      so PI2 sees a real cost for samples with large SK/SD weights even
            #      when the clip absorbs them.  This steers the PI2 mean away from the
            #      clip boundary honestly, preventing mean drift.
            #
            # Penalty form: λ · mean(max(0, |w| - SK_CLIP)²)
            #   - mean (not sum) → scale independent of n_bfs_slack
            #   - zero inside ±SK_CLIP, grows quadratically outside
            #   - at SK=±20 (5 over clip): penalty = 1.0 * 25 / 42 ≈ 0.6 (small, informative)
            #   - at SK=±50 (35 over clip): penalty = 1.0 * 1225 / 42 ≈ 29  (large, PI2 avoids)
            SK_CLIP      = 15.0
            STIFF_WEIGHT = 1.0
            if hasattr(trace, 'raw_sk_weights') and trace.raw_sk_weights is not None:
                w = trace.raw_sk_weights
                excess = np.maximum(0.0, np.abs(w) - SK_CLIP)
                total_cost += STIFF_WEIGHT * float(np.mean(excess**2))
            if hasattr(trace, 'raw_sd_weights') and trace.raw_sd_weights is not None:
                w = trace.raw_sd_weights
                excess = np.maximum(0.0, np.abs(w) - SK_CLIP)
                total_cost += STIFF_WEIGHT * float(np.mean(excess**2))

            # --- Hard K ceiling penalty ---
            # Safety net ONLY: prevent runaway tr(K) > 3000 N/m.
            # This is a very loose ceiling — it should NOT pull K down during
            # normal operation (K0=200, tr(K) ≈ 600 at nominal).
            # The tight per-axis human-proximity penalty handles the real
            # stiffness reduction near the human.
            # K_MAX is set very high (3000) so this only fires on runaway cases.
            K_MAX         = 3000.0   # safety ceiling only — not a shaping cost
            K_CEIL_WEIGHT = 1.0
            if trace.gains is not None and "K" in trace.gains:
                K_arr = trace.gains["K"]
                trK_arr = np.array([np.trace(K) for K in K_arr])
                excess_k = np.maximum(0.0, trK_arr - K_MAX)
                total_cost += K_CEIL_WEIGHT * float(np.mean((excess_k / K_MAX)**2))

            # --- Minimum damping regularizer (hardware safety) ---
            # The optimizer tends to zero out SD weights because low damping
            # gives faster motion that satisfies TL clauses more easily.
            # This leaves D ≈ alpha*H = 0.05*I Ns/m → zeta ≈ 0.002 (500× underdamped).
            # For Franka Panda, minimum safe damping per axis ≈ 10 Ns/m (zeta ≈ 0.35).
            # We penalize tr(D) < D_MIN_TRACE with a one-sided quadratic cost.
            #   D_MIN_TRACE = 30.0 Ns/m  (10 N·s/m per axis × 3 axes)
            #   Penalty is normalised by D_MIN_TRACE² so it is O(1) at nominal (theta=0)
            #   and reaches ~0 once tr(D) ≥ D_MIN_TRACE.
            D_MIN_TRACE = 30.0   # Ns/m  (≈ zeta 0.35 at K0=200 N/m, H=I)
            DAMP_WEIGHT = 2.0    # at nominal: 2.0 * mean((29.85/30)^2) ≈ 2.0 — same order as TL clauses
            if trace.gains is not None and "D" in trace.gains:
                D_arr = trace.gains["D"]          # (T, 3, 3)
                trD_arr = np.array([np.trace(D) for D in D_arr])
                # Normalise deficit by D_MIN_TRACE so penalty is dimensionless
                deficit_frac = np.maximum(0.0, (D_MIN_TRACE - trD_arr) / D_MIN_TRACE)
                total_cost += DAMP_WEIGHT * float(np.mean(deficit_frac**2))

            # --- Human-proximity per-axis stiffness reduction (optional) ---
            #
            # DESIGN INTENT: K should stay HIGH (≈ K0) far from human, then
            # smoothly reduce as the arm approaches, and stay LOW (≤ K_AXIS_MAX)
            # inside the proximity radius. No global K ceiling — stiffness is
            # only shaped near the human.
            #
            # Implementation:
            #   - Soft ramp zone: [r_h, 3·r_h] — linear weight 0→1 as d decreases
            #     from 3·r_h to r_h. Penalty gently starts here, so K begins
            #     dropping BEFORE entering the hard-safety radius.
            #   - Hard zone: d < r_h — full penalty weight, K_ii must be ≤ K_AXIS_MAX
            #
            # Weight function w(d):
            #   d ≥ 3·r_h  → w = 0   (no penalty, K free to be high)
            #   r_h ≤ d < 3·r_h → w = (3·r_h - d) / (2·r_h)  ∈ (0, 1]  (ramp)
            #   d < r_h    → w = 1   (full penalty)
            #
            # Pour-phase stiffness: The pour phase CGMS sub-policy resets K at
            # t=T_phase_start. If the pour happens at the human position, K must
            # ALSO stay low during pour (arm is still next to human).
            # The penalty applies to ALL timesteps within 3·r_h, regardless of phase.
            #
            # CGMS-safe: cost on trace output only. Cholesky ODE untouched.
            if self.human_position is not None and self.human_proximity_radius is not None:
                K_AXIS_MAX     = 100.0   # N/m per axis target inside proximity radius
                HUMAN_K_WEIGHT = 8.0     # strong enough to overcome HoldAtWaypoint drive
                RAMP_FACTOR    = 3.0     # ramp starts at RAMP_FACTOR × r_h

                if trace.gains is not None and "K" in trace.gains:
                    K_arr_h  = trace.gains["K"]          # (T, 3, 3)
                    pos_h    = trace.position             # (T, 3)
                    r_h      = self.human_proximity_radius
                    r_ramp   = RAMP_FACTOR * r_h

                    d_human  = np.linalg.norm(
                        pos_h - self.human_position, axis=1)  # (T,)

                    # All timesteps within the ramp zone (includes hard zone)
                    ramp_mask = d_human < r_ramp

                    if np.any(ramp_mask):
                        ramp_idx = np.where(ramp_mask)[0]
                        d_ramp   = d_human[ramp_idx]

                        # Weight: 0 at r_ramp, 1 at r_h (and below)
                        w_dist = np.clip(
                            (r_ramp - d_ramp) / (r_ramp - r_h),
                            0.0, 1.0)

                        # Per-axis excess: penalise each K_ii independently
                        per_step_cost = np.zeros(len(ramp_idx))
                        for j, t_idx in enumerate(ramp_idx):
                            K_t = K_arr_h[t_idx]
                            excess_sum = 0.0
                            for axis in range(3):
                                kii = K_t[axis, axis]
                                ex  = max(0.0, kii - K_AXIS_MAX)
                                excess_sum += (ex / K_AXIS_MAX) ** 2
                            per_step_cost[j] = w_dist[j] * excess_sum / 3.0

                        total_cost += HUMAN_K_WEIGHT * float(np.mean(per_step_cost))

            return total_cost

        return objective

    def _evaluate_clause(self, trace, clause):

        predicate_fn = self.predicate_registry[clause.predicate]
        rho_trace = predicate_fn(trace, **clause.parameters)

        if clause.operator == "eventually":
            return temporal_logic.eventually(rho_trace)

        elif clause.operator == "always":
            return temporal_logic.always(rho_trace)

        elif clause.operator == "always_during":
            t_start = clause.time_window[0]
            t_end   = clause.time_window[1]
            return temporal_logic.always_during(rho_trace, trace.time, t_start, t_end)

        elif clause.operator == "eventually_during":
            t_start = clause.time_window[0]
            t_end   = clause.time_window[1]
            return temporal_logic.eventually_during(rho_trace, trace.time, t_start, t_end)

        elif clause.operator == "until":
            # Until expects two predicates: (left, right)
            left_fn  = self.predicate_registry[clause.predicate[0]]
            right_fn = self.predicate_registry[clause.predicate[1]]
            rho_phi  = left_fn(trace, **clause.parameters["left_params"])
            rho_psi  = right_fn(trace, **clause.parameters["right_params"])
            return temporal_logic.until(rho_phi, rho_psi)

        else:
            raise NotImplementedError(
                f"Operator {clause.operator} not supported yet."
            )