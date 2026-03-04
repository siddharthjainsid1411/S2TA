"""
Quaternion utilities for orientation DMP.

Convention: q = [w, x, y, z]  (scalar-first, Hamilton convention).
All functions operate on 1-D arrays of length 4.
"""

import numpy as np


# ── Basic operations ─────────────────────────────────────────────────

def quat_normalize(q):
    """Normalize quaternion to unit norm. Returns copy."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_mul(q1, q2):
    """Hamilton quaternion product  q1 ⊗ q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate(q):
    """Quaternion conjugate (= inverse for unit quaternions)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_inv(q):
    """Quaternion inverse (for unit quaternions, same as conjugate)."""
    return quat_conjugate(q)


# ── Log / Exp maps (S³ ↔ ℝ³) ────────────────────────────────────────

def quat_log(q):
    """
    Logarithmic map: S³ → ℝ³.

    Returns the rotation vector  v ∈ ℝ³  such that  q = exp(v).
    ||v|| = half-angle of rotation (in radians).

    For the identity quaternion [1,0,0,0] returns [0,0,0].
    """
    q = quat_normalize(q)
    # Ensure w >= 0 to pick the shorter geodesic (double-cover fix)
    if q[0] < 0.0:
        q = -q
    vec = q[1:4]
    sin_half = np.linalg.norm(vec)
    if sin_half < 1e-10:
        # First-order Taylor: log(q) ≈ vec  (angle → 0)
        return vec.copy()
    half_angle = np.arctan2(sin_half, q[0])
    return (half_angle / sin_half) * vec


def quat_exp(v):
    """
    Exponential map: ℝ³ → S³.

    Given rotation vector v, returns unit quaternion q = exp(v).
    """
    v = np.asarray(v, dtype=float)
    half_angle = np.linalg.norm(v)
    if half_angle < 1e-10:
        # First-order Taylor: exp(v) ≈ [1, v]
        return quat_normalize(np.array([1.0, v[0], v[1], v[2]]))
    axis = v / half_angle
    w = np.cos(half_angle)
    xyz = np.sin(half_angle) * axis
    return np.array([w, xyz[0], xyz[1], xyz[2]])


# ── Orientation error (geodesic) ─────────────────────────────────────

def quat_error(q_current, q_goal):
    """
    Orientation error in tangent space (ℝ³).

    e = log(q_goal⁻¹ ⊗ q_current)

    Returns a 3-vector whose norm is the geodesic angle between
    q_current and q_goal.  When q_current == q_goal, returns [0,0,0].
    """
    q_err = quat_mul(quat_inv(q_goal), q_current)
    return quat_log(q_err)


def quat_distance(q1, q2):
    """
    Geodesic angular distance between two quaternions (radians, ∈ [0, π]).

    d = 2 * arccos(|⟨q1, q2⟩|)
    """
    dot = np.abs(np.dot(quat_normalize(q1), quat_normalize(q2)))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)


# ── SLERP ────────────────────────────────────────────────────────────

def quat_slerp(q0, q1, t):
    """
    Spherical linear interpolation.

    Parameters
    ----------
    q0, q1 : array-like (4,)
        Start and end quaternions.
    t : float
        Interpolation parameter ∈ [0, 1].

    Returns
    -------
    q : ndarray (4,)
        Interpolated unit quaternion.
    """
    q0 = quat_normalize(np.asarray(q0, float))
    q1 = quat_normalize(np.asarray(q1, float))
    dot = np.dot(q0, q1)
    # Pick shorter arc
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        # Very close — linear interpolation + renormalize
        return quat_normalize(q0 + t * (q1 - q0))
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    a = np.sin((1.0 - t) * theta) / sin_theta
    b = np.sin(t * theta) / sin_theta
    return quat_normalize(a * q0 + b * q1)


# ── Quaternion integration ───────────────────────────────────────────

def quat_integrate(q, omega, dt):
    """
    Integrate angular velocity ω (rad/s) for one timestep.

    q_new = exp(0.5 * ω * dt) ⊗ q     (body-frame convention)

    Parameters
    ----------
    q     : ndarray (4,)  — current orientation
    omega : ndarray (3,)  — angular velocity (rad/s)
    dt    : float         — timestep

    Returns
    -------
    q_new : ndarray (4,) — updated orientation (normalized)
    """
    delta = 0.5 * omega * dt
    q_delta = quat_exp(delta)
    return quat_normalize(quat_mul(q_delta, q))
