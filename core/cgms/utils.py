import numpy as np
"""
Helper functions
"""
def sym(M: np.ndarray) -> np.ndarray:
    """
    @brief
        Symmetrize a square matrix.

    @param M (np.ndarray)
        Input square matrix.

    @return (np.ndarray)
        Symmetrized matrix ( (M + M.T) / 2 ).
    """
    return 0.5 * (M + M.T)

def finite_diff(Y: np.ndarray, dt: float) -> np.ndarray:
    """
    @brief
        Compute finite differences along the first axis of an array.

    @param Y (np.ndarray)
        Input array. Can be 1D, 2D, or 3D.
    @param dt (float)
        Step size.

    @return (np.ndarray)
        Array of the same shape as Y, containing finite differences.
        Central differences are used for interior points.
        Forward and backward differences are used at the boundaries.
    """
    Y = np.asarray(Y, float)
    dY = np.zeros_like(Y)

    if Y.ndim == 1:
        dY[1:-1] = (Y[2:] - Y[:-2]) / (2 * dt)
        dY[0]    = (Y[1] - Y[0]) / dt
        dY[-1]   = (Y[-1] - Y[-2]) / dt

    elif Y.ndim == 2:
        dY[1:-1, :] = (Y[2:, :] - Y[:-2, :]) / (2 * dt)
        dY[0, :]    = (Y[1, :] - Y[0, :]) / dt
        dY[-1, :]   = (Y[-1, :] - Y[-2, :]) / dt

    elif Y.ndim == 3:
        dY[1:-1, :, :] = (Y[2:, :, :] - Y[:-2, :, :]) / (2 * dt)
        dY[0, :, :]    = (Y[1, :, :] - Y[0, :, :]) / dt
        dY[-1, :, :]   = (Y[-1, :, :] - Y[-2, :, :]) / dt

    else:
        raise ValueError("finite_diff: unsupported ndim")

    return dY

def lt_pack(L: np.ndarray) -> np.ndarray:
    """
    @brief
        Pack the lower-triangular entries of a 3x3 matrix into a vector.

    @param L (np.ndarray)
        Input 3x3 matrix.

    @return (np.ndarray)
        Vector [L00, L10, L11, L20, L21, L22].
    """
    return np.array([L[0, 0],
                     L[1, 0], L[1, 1],
                     L[2, 0], L[2, 1], L[2, 2]], float)

def lt_unpack(v: np.ndarray) -> np.ndarray:
    """
    @brief
        Unpack a 6-element vector into the lower-triangular part of a 3x3 matrix.

    @param v (np.ndarray)
        Input vector of length 6, in the format [L00, L10, L11, L20, L21, L22].

    @return (np.ndarray)
        3x3 lower-triangular matrix.
    """
    v = np.asarray(v, float).reshape(-1)
    if v.shape[0] != 6:
        raise ValueError("lt_unpack: input vector must have length 6")
    L = np.zeros((3, 3), float)
    L[0, 0] = v[0]
    L[1, 0], L[1, 1] = v[1], v[2]
    L[2, 0], L[2, 1], L[2, 2] = v[3], v[4], v[5]
    return L