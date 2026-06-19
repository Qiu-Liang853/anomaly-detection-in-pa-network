import numpy as np
from numba import njit
from pa_loglikelihood import neg_log_likelihood_anomaly
from scipy.optimize import minimize_scalar

def objective_beta(beta, delta_fixed, network_size, vertex_index,
                   k_result, degree_list, edge_receive_list, m):
    return neg_log_likelihood_anomaly(beta, delta_fixed, network_size, vertex_index,
                                        k_result, degree_list, edge_receive_list, m)

def objective_delta(delta, beta_fixed, network_size, vertex_index,
                    k_result, degree_list, edge_receive_list, m):
    return neg_log_likelihood_anomaly(beta_fixed, delta, network_size, vertex_index,
                                        k_result, degree_list, edge_receive_list, m)

# Find beta given delta and tau
def single_estimation_beta(beta_bounds,delta_fixed, network_size, vertex_index, k_result, degree_list, edge_receive_list,m):

    res_beta = minimize_scalar(
        objective_beta,
        bounds=beta_bounds,
        args=(delta_fixed, network_size, vertex_index, k_result, degree_list, edge_receive_list,m),
        method='bounded',
        options={'xatol': 1e-6}
        )
    
    beta_hat = round(float(res_beta.x), 4)
    #print(f"Estimated beta = {beta_hat:.4f}, fun = {res_beta.fun:.4f}")
    return beta_hat

# Find delta given beta and tau
def single_estimation_delta(delta_bounds,beta_fixed, network_size, vertex_index, k_result, degree_list, edge_receive_list,m):
    res_delta = minimize_scalar(
        objective_delta,
        bounds=delta_bounds,
        args=(beta_fixed, network_size, vertex_index, k_result, degree_list, edge_receive_list,m),
        method='bounded',
        options={'xatol': 1e-6}
    )
    delta_hat = round(float(res_delta.x), 4)
    #print(f"Estimated delta = {delta_hat:.4f}, fun = {res_delta.fun:.4f}")
    return delta_hat
