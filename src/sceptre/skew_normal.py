"""
Skew-normal moment-matching utilities used for null calibration.
"""

from typing import Tuple

import numba as nb
import numpy as np


def fit_skew_normal_funct(y: np.ndarray) -> np.ndarray:
    """
    Match skew-normal parameters (xi, omega, alpha) to sample moments.

    Returns [xi, omega, alpha, mean, sd] to mirror the original C++ routine.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    n_doub = float(n)

    s_1 = 0.0
    s_2 = 0.0
    for i in range(n):
        s_1 += y[i]
        s_2 += y[i] * y[i]
    m_y = s_1 / n_doub
    sd_y = np.sqrt(s_2 / n_doub - m_y * m_y)

    s_3 = 0.0
    for i in range(n):
        diff = y[i] - m_y
        s_3 += diff * diff * diff
    gamma1 = s_3 / (n_doub * (sd_y * sd_y * sd_y))

    max_gamma_1 = 0.995
    if gamma1 > max_gamma_1:
        gamma1 = 0.9 * max_gamma_1

    b = np.sqrt(2.0 / np.pi)
    r = np.copysign(1.0, gamma1) * np.power(
        2.0 * np.abs(gamma1) / (4.0 - np.pi), 1.0 / 3.0
    )
    delta = r / (b * np.sqrt(1.0 + r * r))
    alpha = delta / np.sqrt(1.0 - delta * delta)
    mu_z = b * delta
    sd_z = np.sqrt(1.0 - mu_z * mu_z)
    omega = sd_y / sd_z
    xi = m_y - omega * mu_z

    return np.array([xi, omega, alpha, m_y, sd_y], dtype=np.float64)


@nb.njit
def fit_skew_normal_funct_numba(y: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated version of fit_skew_normal_funct.
    """
    n = y.size
    n_doub = float(n)

    s_1 = 0.0
    s_2 = 0.0
    for i in range(n):
        s_1 += y[i]
        s_2 += y[i] * y[i]
    m_y = s_1 / n_doub
    sd_y = np.sqrt(s_2 / n_doub - m_y * m_y)

    s_3 = 0.0
    for i in range(n):
        diff = y[i] - m_y
        s_3 += diff * diff * diff
    gamma1 = s_3 / (n_doub * (sd_y * sd_y * sd_y))

    max_gamma_1 = 0.995
    if gamma1 > max_gamma_1:
        gamma1 = 0.9 * max_gamma_1

    b = np.sqrt(2.0 / np.pi)
    r = np.copysign(1.0, gamma1) * np.power(
        2.0 * np.abs(gamma1) / (4.0 - np.pi), 1.0 / 3.0
    )
    delta = r / (b * np.sqrt(1.0 + r * r))
    alpha = delta / np.sqrt(1.0 - delta * delta)
    mu_z = b * delta
    sd_z = np.sqrt(1.0 - mu_z * mu_z)
    omega = sd_y / sd_z
    xi = m_y - omega * mu_z

    out = np.empty(5, dtype=np.float64)
    out[0] = xi
    out[1] = omega
    out[2] = alpha
    out[3] = m_y
    out[4] = sd_y
    return out
