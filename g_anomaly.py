import numpy as np
from numba import njit
from pa_core import rand_prob_node_before_anomaly, add_edge_before_anomaly

@njit
def rand_prob_node_after_anomaly(L, L_len, num_nodes, beta, tau, m, delta):
    a = tau - 1

    if beta <= 0.0 or a < 0 or a >= num_nodes:
        return rand_prob_node_before_anomaly(L, L_len, num_nodes, m, delta)

    base = float(m) + float(delta)  # delta > -m
    total_weight = float(L_len) + (base + float(beta)) * float(num_nodes)

    r = np.random.random() * total_weight

    if L_len > 0 and r < float(L_len):
        return L[np.random.randint(L_len)]

    r -= float(L_len)

    if r < base * float(num_nodes):
        return np.random.randint(num_nodes)

    return a


@njit
def add_edge_after_anomaly(L, L_len, num_nodes, new_node, beta, tau, m, delta):
    while True:
        x = rand_prob_node_after_anomaly(L, L_len, num_nodes, beta, tau, m, delta)
        if x != new_node:
            return x


# --------------------------
# Generator: before anomaly uses standard PA logic
# --------------------------
@njit
def generate_edges_with_anomaly(network_size, m, beta, tau, delta):
    init_num = 1

    total_edges = network_size * m
    edges = np.zeros((total_edges, 2), dtype=np.int32)
    edge_idx = 0

    degrees = np.zeros(network_size, dtype=np.int32)

    # Lx stores "excess stubs": node appears (deg - m) times
    # total excess stubs ≤ total_edges approximately, safe to allocate 2*total_edges
    L = np.empty(2 * total_edges, dtype=np.int32)
    L_len = 0

    # ---- init: node 0 with m self-loops ----
    # degree(0)=2m => excess = 2m - m = m => add m copies of 0 to Lx
    for _ in range(m):
        edges[edge_idx, 0] = 0
        edges[edge_idx, 1] = 0
        edge_idx += 1

    degrees[0] += 2 * m

    for _ in range(m):
        L[L_len] = 0
        L_len += 1

    # ---- before anomaly: standard PA with delta ----
    for new_node in range(init_num, tau):
        num_nodes = new_node
        for _ in range(m):
            target = add_edge_before_anomaly(L, L_len, num_nodes, new_node, m, delta)

            edges[edge_idx, 0] = new_node
            edges[edge_idx, 1] = target
            edge_idx += 1

            degrees[new_node] += 1
            degrees[target] += 1

            # target's excess increases by 1 => append once
            L[L_len] = target
            L_len += 1

        # new_node starts with degree m => excess 0 => add nothing

    # ---- after anomaly ----
    for new_node in range(tau, network_size):
        num_nodes = new_node
        for _ in range(m):
            target = add_edge_after_anomaly(L, L_len, num_nodes, new_node, beta, tau, m, delta)

            edges[edge_idx, 0] = new_node
            edges[edge_idx, 1] = target
            edge_idx += 1

            degrees[new_node] += 1
            degrees[target] += 1

            L[L_len] = target
            L_len += 1

        # new_node excess still 0 at birth

    return edges


def PA_with_anomaly_numba(network_size, m, beta, tau, delta, seed):
    
    np.random.seed(int(seed))

    if m < 1:
        raise ValueError(f"m must be >= 1, got m={m}")
    if not (1 <= tau <= network_size):
        raise ValueError(f"tau must satisfy 1 <= tau <= network_size. Got tau={tau}, network_size={network_size}")
    if delta <= -m:
        raise ValueError(f"delta must be > -m. Got delta={delta}, m={m}")
    if beta <= 0:
        raise ValueError(f"beta must be > 0. Got beta={beta}")

    edges = generate_edges_with_anomaly(network_size, m, beta, tau, delta)
    return np.asarray(edges, dtype=np.int32)
