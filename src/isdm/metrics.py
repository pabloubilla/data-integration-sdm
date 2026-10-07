import numpy as np


def harmonic_mean(a, b):
    """Row-wise harmonic mean of two arrays. NaN where both are <= 0."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, 2 * a * b / denom, np.nan)