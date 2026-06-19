import numpy as np
from numba import njit
from scipy.special import gammaln,gamma
from collections import Counter, OrderedDict, defaultdict

def edge_degree_counting(vertex_index, edges):
    n = len(edges)
    degree_sequence = [0] * n  # Initialize degree sequence (to track degree at each timestep)
    edge_received = [0] * n    # Edge reception list: 0 means no edge received, 1 means edge received

    degree_count = 0           # Starting degree for the vertex (before any edges)

    # go through each edge and check if the vertex_index is involved
    for i in range(n):
        u, v = edges[i]
        if int(u) == vertex_index or int(v) == vertex_index:
            degree_count += 1  # Increase degree by 1 when vertex receives an edge
            edge_received[i] = 1  # Mark that an edge was received
        degree_sequence[i] = degree_count  # Record the degree for the current timestep

    return degree_sequence, edge_received

@njit(cache=True)
def edge_degree_counting_many(vertices0, edges, network_size):
    """
    vertices0: 0-based candidate vertices, shape (K,)
    edges: shape (n,2), int
    network_size: number of vertices (so we can allocate map)

    Returns:
      degree_all: int32, shape (K, n)   degree after processing edge i (same as your original)
      received_all: uint8, shape (K, n) 1 if edge i touches candidate else 0
    """
    K = vertices0.shape[0]
    n = edges.shape[0]

    # map vertex -> candidate index, -1 if not candidate
    idx_map = -np.ones(network_size, dtype=np.int32)
    for j in range(K):
        idx_map[vertices0[j]] = j

    degree_all = np.zeros((K, n), dtype=np.int32)
    received_all = np.zeros((K, n), dtype=np.uint8)
    deg = np.zeros(K, dtype=np.int32)

    for i in range(n):
        u = edges[i, 0]
        v = edges[i, 1]

        iu = -1
        iv = -1
        if 0 <= u < network_size:
            iu = idx_map[u]
        if 0 <= v < network_size:
            iv = idx_map[v]

        if iu != -1:
            deg[iu] += 1
            received_all[iu, i] = 1

        # if u and v are the same candidate, do not add twice
        if iv != -1 and iv != iu:
            deg[iv] += 1
            received_all[iv, i] = 1

        # record degrees after processing edge i (match your original)
        for j in range(K):
            degree_all[j, i] = deg[j]

    return degree_all, received_all

def expected_degree_PA_normal_all_vertices(network_size, delta, m):
    c1 = m / (2*m + delta)

    # log-gamma values (stable for big numbers)
    ln_gamma_network_c1 = gammaln(network_size + c1)
    ln_gamma_network = gammaln(network_size)

    vertex_index = np.arange(1, network_size+1)

    # c0 depends on vertex_index
    c0 = np.full(network_size, m + delta, dtype=float)
    c0[0] = 2*m + delta   # vertex 1 special case

    # log form of c2
    ln_c2 = (
        ln_gamma_network_c1 
        - gammaln(vertex_index + c1)
        + gammaln(vertex_index)
        - ln_gamma_network
    )

    c2 = np.exp(ln_c2)  # convert back from log
    return c0 * c2

def graph_series_connected_vertex_degree(edges, receiver_index=1):
    """
    Track how the degree of the receiver node evolves over time.

    Args:
        edges (list[tuple[int, int]]): List of edges (u, v)
        receiver_index (int): 0 if the receiver is the first node (u),
                              1 if the receiver is the second node (v)
    
    Returns:
        list[tuple[int, int]]: (receiver, degree_before_edge) for each edge
    """
    degree_count = defaultdict(int)
    results = []

    for edge in edges:
        u, v = edge
        receiver = edge[receiver_index]

        # Record degree before adding the edge
        results.append((receiver, degree_count[receiver]))

        # Update degrees of both endpoints
        degree_count[u] += 1
        degree_count[v] += 1

    return results

def get_top_vertices(edge_list, exp_degree, top_k_degree, top_k_ratio,delta):

    true_degree = Counter()
    for u, v in edge_list:
        true_degree[u] += 1
        true_degree[v] += 1  # undirected graph

    # Top k by degree
    if top_k_degree >0:
        top_degree_vertices = [node for node, _ in true_degree.most_common(top_k_degree)]
    else:
        top_degree_vertices = []

    # Top K by ratio
    if top_k_ratio > 0:
        ratio = {
            u: (true_degree[u] + delta) / exp_degree[u]
            for u in true_degree
            if exp_degree[u] > 0
        }
        top_ratio_vertices = [
            node for node, _ in sorted(ratio.items(), key=lambda x: x[1], reverse=True)[:top_k_ratio]
        ]
    else:
        top_ratio_vertices = []

    # remove duplicates
    vertex_list = list(set(top_degree_vertices + top_ratio_vertices))

    return vertex_list

def prepare_screening(edge_array, exp_degree, top_k_degree, top_k_ratio, delta, n=None, store_ratio=True):
    edge_array = np.asarray(edge_array, dtype=np.int64)
    exp_degree = np.asarray(exp_degree, dtype=np.float64)
    
    if n is None:
        n = int(edge_array.max()) + 1

    deg = np.bincount(edge_array[:, 0], minlength=n) + np.bincount(edge_array[:, 1], minlength=n)
    selected = []

    # Top k by degree, including all ties at k-th place
    if top_k_degree > 0:
        k = min(top_k_degree, n)
        cutoff_degree = np.partition(deg, -k)[-k]
        selected.append(np.flatnonzero(deg >= cutoff_degree))

    # Top k by ratio, including all ties at k-th place
    ratio = None
    if top_k_ratio > 0:
        ratio = np.full(n, -np.inf, dtype=np.float64)
        mask = exp_degree > 0
        ratio[mask] = (deg[mask] + delta) / exp_degree[mask]
        valid_ratio = ratio[mask]
        k = min(top_k_ratio, len(valid_ratio))
        if k > 0:
            cutoff_ratio = np.partition(valid_ratio, -k)[-k]
            selected.append(np.flatnonzero(ratio >= cutoff_ratio))

    vertex_list = np.unique(np.concatenate(selected)).tolist() if selected else []

    # degree-rank 
    maxd = int(deg.max())
    hist = np.bincount(deg, minlength=maxd + 1)
    suffix_ge = np.cumsum(hist[::-1])[::-1]  

    # optionally store ratio array 
    if not store_ratio:
        ratio = None

    return {
        "deg": deg,
        "suffix_ge": suffix_ge,
        "ratio": ratio,
        "exp_degree": exp_degree,
        "vertex_list": vertex_list,
    }


def get_vertex_ranks(v, prepared,delta):
    deg = prepared["deg"]
    suffix_ge = prepared["suffix_ge"]
    ratio = prepared["ratio"]
    exp_degree = prepared["exp_degree"]

    d0 = int(deg[v])
    maxd = len(suffix_ge) - 1
    greater = int(suffix_ge[d0 + 1]) if d0 + 1 <= maxd else 0
    degree_rank_global = 1 + greater

    # ratio + ratio rank (exact for this v)
    if exp_degree[v] > 0:
        r0 = float((d0 +delta)/ exp_degree[v])
    else:
        r0 = None

    if r0 is None:
        ratio_rank_global = None
    else:
        if ratio is None:
            # compute exact rank for this v without storing ratio array:
            valid = exp_degree > 0
            ratio_rank_global = 1 + int(np.sum(((deg[valid] +delta)/ exp_degree[valid]) > r0))
        else:
            ratio_rank_global = 1 + int(np.sum(ratio > r0))

    return  degree_rank_global,ratio_rank_global