import numpy as np
from numba import njit
from pa_loglikelihood import neg_log_likelihood_anomaly
from scipy.optimize import minimize, minimize_scalar

def iteration_estimation_update(
    network_size, vertex_index, k_result,
    degree_v_list, edge_receive_list, m,
    beta_init, delta_init,
    max_iter=100,
    tol=1e-5,
    xatol=1e-5,   
):
    beta  = float(beta_init)
    delta = float(delta_init)

    delta_bounds = (-float(m), float(m))    # the interval of delta
    beta_bounds  = (1e-6, 10.0)             # the interval of beta

    def f(beta_, delta_):
        return neg_log_likelihood_anomaly(
            beta_, delta_, network_size, vertex_index,
            k_result, degree_v_list, edge_receive_list, m
        )

    for k in range(max_iter):
        # Step 1: optimize delta given beta
        res_d = minimize_scalar(
            lambda d: f(beta, d),
            bounds=delta_bounds,
            method="bounded",
            options={"xatol": xatol}
        )
        delta_new = float(res_d.x)

        # Step 2: optimize beta given delta_new
        res_b = minimize_scalar(
            lambda b: f(b, delta_new),
            bounds=beta_bounds,
            method="bounded",
            options={"xatol": xatol}
        )
        beta_new = float(res_b.x)

        # Early stop 
        if abs(beta_new - beta) < tol and abs(delta_new - delta) < tol:
            beta, delta = beta_new, delta_new
            break

        beta, delta = beta_new, delta_new

    neg_ll = f(beta, delta)
    return beta, delta, float(neg_ll)
