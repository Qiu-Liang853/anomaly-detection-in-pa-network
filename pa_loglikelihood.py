import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def neg_log_likelihood_normal(delta, network_size, k_result, m):
    
    """
    Compute log-likelihood for PA model (undirected), where each new node enters with m edges.
    k_result: list of (receiver, degree_before_edge) for each edge, in time order.
    t enumerates edges starting at m; 
    initial vertex number should be 1.
    """
    
    L_1 = 0.0

    for t in range(m, network_size * m):
        v, degree_k = k_result[t]

        numerator = degree_k + delta

        edges_before = t
        nodes_before = edges_before // m

        denominator = nodes_before * (2 * m + delta) + edges_before % m

        if numerator <= 0.0 or denominator <= 0.0:
            return 1e10

        L_1 += np.log(numerator) - np.log(denominator)

    return -L_1
    
@njit(cache=True, fastmath=True)
def neg_log_likelihood_anomaly(beta, delta, network_size, vertex_index,
                                k_result, degree_v_list, edge_receive_list, m):
    L_1 = 0.0
    L_2 = 0.0

    # vertex_index == 1
    if vertex_index == 1:
        for t in range(m, network_size * m):
            v, degree_k = k_result[t]
            edges_before = t
            nodes_before = edges_before // m

            numerator = ((1 - edge_receive_list[t]) * (degree_k + delta) +
                         edge_receive_list[t] * (degree_v_list[t-1] + delta + nodes_before * beta))
            denominator = nodes_before * (2 * m + beta + delta) + edges_before % m

            if numerator <= 0.0 or denominator <= 0.0:
                return 1e10

            L_2 += np.log(numerator) - np.log(denominator)
    else:
        # L_1: edges before the anomaly
        for t_1 in range(m, vertex_index * m):
            v, degree_k = k_result[t_1]
            
            edges_before = t_1 # since the list start from 0, if it is 1-based, then should minus 1
            nodes_before = edges_before // m

            numerator = degree_k + delta
            denominator = nodes_before * (2 * m + delta) + edges_before % m
           # print(t_1, numerator, denominator)
            if numerator <= 0.0 or denominator <= 0.0:
                return 1e10

            L_1 += np.log(numerator) - np.log(denominator)

        # L_2: edges of the anomaly and beyond
        for t_2 in range(vertex_index * m, network_size * m):
            v, degree_k = k_result[t_2]
            edges_before = t_2
            nodes_before = edges_before // m

            numerator = ((1 - edge_receive_list[t_2]) * (degree_k + delta) +
                         edge_receive_list[t_2] * (degree_v_list[t_2-1] + delta + nodes_before * beta))
            denominator = nodes_before * (2 * m + beta + delta) + edges_before % m

            if numerator <= 0.0 or denominator <= 0.0:
                return 1e10
            #print(t_2, numerator, denominator)
            L_2 += np.log(numerator) - np.log(denominator)

    return -(L_1 + L_2)
