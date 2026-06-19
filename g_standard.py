import numpy as np
from numba import njit
from pa_core import rand_prob_node_before_anomaly, add_edge_before_anomaly

@njit
def generate_edges_before_anomaly(network_size, m, delta):
    total_edges = network_size * m
    edges = np.zeros((total_edges, 2), dtype=np.int32)
    edge_idx = 0

    # L is EXCESS list: multiplicity = deg - m
    L = np.empty(2 * total_edges, dtype=np.int32)
    L_len = 0

    degrees = np.zeros(network_size, dtype=np.int32)

    # init: node 0 with m self-loops
    for _ in range(m):
        edges[edge_idx, 0] = 0
        edges[edge_idx, 1] = 0
        edge_idx += 1
    degrees[0] = 2 * m

    # node 0 excess = (2m - m) = m  -> add m copies
    for _ in range(m):
        L[L_len] = 0
        L_len += 1

    for new_node in range(1, network_size):
        num_nodes = new_node
        for _ in range(m):
            target = add_edge_before_anomaly(L, L_len, num_nodes, new_node, m,delta)
            edges[edge_idx, 0] = new_node
            edges[edge_idx, 1] = target
            edge_idx += 1

            degrees[new_node] += 1
            degrees[target] += 1

            # target excess increases by 1
            L[L_len] = target
            L_len += 1

    return edges


def standard_PA_numba(network_size, m, delta,seed):
    
    np.random.seed(int(seed))
        
    if m < 1:
        raise ValueError(f"m must be >= 1, got m={m}")
    if delta <= -m:
        raise ValueError(f"delta must be > -m. Got delta={delta}, m={m} (delta <= -m).")

    edges = generate_edges_before_anomaly(network_size, m, delta)

    #G = nx.MultiGraph()
    #G.add_edges_from(edges)
    edges_array = np.asarray(edges, dtype=np.int32)
    #np.save(filename, edges_array)   # filename should end with ".npy"
    return edges_array
