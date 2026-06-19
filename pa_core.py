import numpy as np
from numba import njit

@njit
def rand_prob_node_before_anomaly(L, L_len, num_nodes, m,delta):
    
    base = float(m) + float(delta)         # must be > 0 (delta > -m)
    total = float(L_len) + base * float(num_nodes)
    #print("The total degree is:", total)
    r = np.random.random() * total
    if r < float(L_len):
        return L[np.random.randint(L_len)]  # degree-excess part
    else:
        return np.random.randint(num_nodes) # baseline part



@njit
def add_edge_before_anomaly(L, L_len, num_nodes, new_node, m,delta):
    """
    Returns a valid node (not equal to new_node).
    In this generator, new_node is not in the candidate set anyway,
    but we keep your "avoid self-loop" logic structure.
    """
    while True:
        random_node = rand_prob_node_before_anomaly(L, L_len, num_nodes, m,delta)
        if random_node != new_node:
            return random_node

