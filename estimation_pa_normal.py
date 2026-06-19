import numpy as np
from numba import njit
from pa_loglikelihood import neg_log_likelihood_normal
from scipy.optimize import minimize_scalar

def optimize_parameters_normal(delta_bounds, network_size, k_result, m):
    res = minimize_scalar(
        lambda d: neg_log_likelihood_normal(d, network_size, k_result, m),
        bounds=delta_bounds,
        method="bounded",
        options={"xatol": 1e-6},
    )
    return float(res.x), float(res.fun)
