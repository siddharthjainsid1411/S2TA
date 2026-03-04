# logic/temporal_logic.py

import numpy as np


def smooth_min(values, k=40.0):
    """Hard min — PI² is derivative-free, so smooth approximation is unnecessary
    and introduces a systematic negative bias of -log(N)/k."""
    return float(np.min(values))


def smooth_max(values, k=40.0):
    """Hard max — PI² is derivative-free, so smooth approximation is unnecessary."""
    return float(np.max(values))


def eventually(rho, k=20.0):
    return smooth_max(rho, k)


def always(rho, k=20.0):
    return smooth_min(rho, k)


def until(rho_phi, rho_psi, k1=20.0, k2=20.0):
    T = len(rho_phi)
    values = []

    for t_prime in range(T):
        if t_prime == 0:
            min_before = np.inf
        else:
            min_before = smooth_min(rho_phi[:t_prime], k=k1)

        inner = smooth_min([rho_psi[t_prime], min_before], k=k1)
        values.append(inner)

    return smooth_max(values, k=k2)


def always_during(rho, times, t_start, t_end, k=20.0):
    """
    Always within a time window [t_start, t_end].
    Selects the portion of rho where t_start <= times <= t_end,
    then applies smooth_min (always) over that slice.

    Returns +inf (trivially satisfied) if the window is empty.
    """
    mask = (times >= t_start - 1e-9) & (times <= t_end + 1e-9)
    if not np.any(mask):
        return float('inf')
    return smooth_min(rho[mask], k)


def eventually_during(rho, times, t_start, t_end, k=20.0):
    """
    Eventually within a time window [t_start, t_end].
    Selects the portion of rho where t_start <= times <= t_end,
    then applies smooth_max (eventually) over that slice.

    Returns -inf (trivially violated) if the window is empty.
    """
    mask = (times >= t_start - 1e-9) & (times <= t_end + 1e-9)
    if not np.any(mask):
        return float('-inf')
    return smooth_max(rho[mask], k)