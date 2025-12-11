import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
import pickle 
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import entropy
from k_means_constrained import KMeansConstrained

from src.distance_optimizer import optimize_cluster_split, minimize_cluster_split, centroid_dist

def partition_distance(A, B, D):
    '''
    A: indices of first partition
    B: indices of second partition
    D: distance matrix
    '''

    min_Ds = np.min(D[np.ix_(A, B)], axis = 1)
    
    return np.mean(min_Ds)


from scipy.stats import gaussian_kde
from scipy.special import rel_entr  # for KL

def KL_partition_distance(A, B, X, eps=1e-12):
    """
    Symmetric KL between the distributions of X[A] and X[B],
    estimated via KDE and Monte Carlo.

    A: indices (or boolean mask) of first partition
    B: indices (or boolean mask) of second partition
    X: data matrix of shape (n_samples, d)
    """

    # Extract partitions; shape: (nA, d), (nB, d)
    XA = np.asarray(X[A])
    XB = np.asarray(X[B])

    # Transpose for gaussian_kde: it expects (d, n_samples)
    XA_T = XA.T
    XB_T = XB.T

    kdeA = gaussian_kde(XA_T)
    kdeB = gaussian_kde(XB_T)

    # ---- KL(A || B) ≈ E_{x~A}[ log p_A(x) - log p_B(x) ] ----
    pA_on_A = kdeA(XA_T) + eps      # density of A on A's samples
    pB_on_A = kdeB(XA_T) + eps      # density of B on A's samples
    kl_AB = np.mean(np.log(pA_on_A) - np.log(pB_on_A))

    # ---- KL(B || A) ≈ E_{x~B}[ log p_B(x) - log p_A(x) ] ----
    pB_on_B = kdeB(XB_T) + eps
    pA_on_B = kdeA(XB_T) + eps
    kl_BA = np.mean(np.log(pB_on_B) - np.log(pA_on_B))

    # Symmetric version
    return 0.5 * (kl_AB + kl_BA)




def partition_sweep(D, test_frac=0.4, sample_every=100, max_iter=1000, step_fraction = 0.05, seed = 42):
    """
    D: distance matrix
    test_frac: fraction of points in A
    seed: RNG seed
    sample_every: store partition every `sample_every` successful iterations

    Returns: list of dicts with keys: 'A', 'B', 'dist_metric'
    """
    rng = np.random.default_rng(seed)
    n = len(D)
    test_size = int(n * test_frac)

    A = rng.choice(n, size=test_size, replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[A] = True
    B = np.where(~mask)[0]

    dist_metric = partition_distance(A, B, D)

    samples = [{"A": A.copy(), "B": B.copy(), "dist_metric": dist_metric}]

    it = 0
    improved = True
    while improved:

        # print(it, improved)
        if improved and (it % sample_every == 0):
            print(f'Add partition after {it} iterations with distance {dist_metric}')
            samples.append({
                "A": A.copy(),
                "B": B.copy(),
                "dist_metric": dist_metric
            })

        it += 1
        improved = False
        random_from_A = np.random.choice(A, int(step_fraction*len(A)), replace=False)
        random_from_B = np.random.choice(B, int(step_fraction*len(B)), replace=False)

        for ra in random_from_A:
            for rb in random_from_B:
                new_A = A.copy()
                new_B = B.copy()
                new_A[new_A == ra] = rb
                new_B[new_B == rb] = ra

                new_metric = partition_distance(new_A, new_B, D)
                if new_metric > dist_metric:
                    A, B = new_A, new_B
                    dist_metric = new_metric
                    improved = True
                    break
            if improved:
                break

        if it > max_iter:
            break

    return samples

def generate_pa_splits_by_sweep(
    D: np.array,
    X_pa: pd.DataFrame,
    Y_pa: pd.DataFrame,
    test_frac: float = 0.4,
    sample_every: int = 3,
    max_iter: int = 1000,
    step_fracion: int = 1,
    seed: int = 42,

):
    """
    Build spatial distance matrix from x,y, run partition_sweep,
    and return a list of splits with their distance metric.
    """
    # work with 0..n-1 positional indices
    X_pa_ = X_pa.reset_index(drop=True)
    Y_pa_ = Y_pa.reset_index(drop=True)

    # partition_sweep returns list of dicts: {'A', 'B', 'dist_metric'}
    samples = partition_sweep(D, test_frac=test_frac, sample_every=sample_every,
                              step_fraction=step_fracion, max_iter=max_iter, seed=seed)

    splits = []
    for s in samples:
        test_idx = s["A"]
        train_idx = s["B"]
        d_metric = s["dist_metric"]

        X_pa_tr = X_pa_.iloc[train_idx].reset_index(drop=True)
        X_pa_te = X_pa_.iloc[test_idx].reset_index(drop=True)
        Y_pa_tr = Y_pa_.iloc[train_idx].reset_index(drop=True)
        Y_pa_te = Y_pa_.iloc[test_idx].reset_index(drop=True)

        splits.append((X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te, d_metric))

    return splits

def random_partition(D, test_frac=0.4, test_size = 10,  rng=None):
    """
    Fast random split with size constraint.
    Returns A, B, dist_metric.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(D)
    if test_size is None:
        test_size = int(n * test_frac)

    perm = rng.permutation(n)
    A = perm[:test_size]
    B = perm[test_size:]

    dist_metric = partition_distance(A, B, D)
    return A, B, dist_metric


def splits_by_closest_swaps(
    D: np.array,
    X_pa: pd.DataFrame,
    Y_pa: pd.DataFrame,
    test_frac: float = 0.4,
    n_steps_list: list = [50],
    seed: int = 42,
    only_if_improve: bool = False,
):
    """
    Use partition_closest_swaps to generate multiple spatial splits.

    For each split:
      - start from a random partition
      - perform up to `n_steps` swaps based on closest A-B pairs
      - keep |A| ~= test_frac * n fixed

    Returns:
      List of tuples (X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te, dist_metric)
    """
    # work with 0..n-1 positional indices
    X_pa_ = X_pa.reset_index(drop=True)
    Y_pa_ = Y_pa.reset_index(drop=True)

    splits = []

    for ix, n_s in enumerate(n_steps_list):

        A, B, d_metric = partition_closest_swaps(
            D,
            test_frac=test_frac,
            n_steps=n_s,
            seed=seed + n_s + ix,
            only_if_improve=only_if_improve,
        )

        test_idx = A
        train_idx = B

        X_pa_tr = X_pa_.iloc[train_idx].reset_index(drop=True)
        X_pa_te = X_pa_.iloc[test_idx].reset_index(drop=True)
        Y_pa_tr = Y_pa_.iloc[train_idx].reset_index(drop=True)
        Y_pa_te = Y_pa_.iloc[test_idx].reset_index(drop=True)

        splits.append((X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te, d_metric))

        print(f'Generated partition with {n_s} steps with distance: {d_metric}')

    return splits


def partition_closest_swaps(
    D,
    test_frac=0.4,
    n_steps=50,
    seed=42,
    only_if_improve=False,
):
    """
    1) Random partition of {0..n-1} into A (test) and B (train).
    2) For step in 1..n_steps:
         - find closest pair (a in A, b in B)
         - find b_far in B farthest from a
         - swap a <-> b_far between A and B
         - optionally only commit swap if it improves partition_distance

    This keeps |A| and |B| fixed, pushes sets apart heuristically.

    Returns
    -------
    A, B, dist_metric
        A, B: np.ndarray of indices
        dist_metric: partition_distance(A, B, D)
    """
    rng = np.random.default_rng(seed)
    n = len(D)
    m = int(round(test_frac * n))

    # 0) random initial partition
    perm = rng.permutation(n)
    A = perm[:m]
    B = perm[m:]

    # initial distance
    dist_metric = partition_distance(A, B, D)

    for _ in range(n_steps):
        if len(A) == 0 or len(B) == 0:
            break

        # 1) find closest pair between A and B
        subD = D[np.ix_(A, B)]     # shape: (len(A), len(B))
        k = subD.argmin()
        i, j = divmod(k, subD.shape[1])
        a = A[i]
        b = B[j]

        # 2) find point in B farthest from a
        d_a_to_B = D[a, B]
        j_far = d_a_to_B.argmax()
        b_far = B[j_far]

        # if farthest is same as b, we can just skip this step
        if b_far == b:
            if only_if_improve:
                # nothing useful to do this step
                continue
            else:
                # or pick a random different B index to force movement
                if len(B) > 1:
                    choices = [idx for idx in range(len(B)) if idx != j]
                    j_far = rng.choice(choices)
                    b_far = B[j_far]
                else:
                    continue  # no alternative

        # 3) build candidate new partition with a <-> b_far swapped
        A_new = A.copy()
        B_new = B.copy()
        A_new[i] = b_far
        B_new[j_far] = a

        if only_if_improve:
            new_metric = partition_distance(A_new, B_new, D)
            if new_metric > dist_metric:
                A, B, dist_metric = A_new, B_new, new_metric
            # else: discard this swap and move to next step
        else:
            # always accept the swap; update sets, recompute metric later
            A, B = A_new, B_new

    if not only_if_improve:
        dist_metric = partition_distance(A, B, D)

    return A, B, dist_metric



def split_pa_train_test_kmeans(X_pa, Y_pa, covs, test_frac=0.3, seed=42):
    """
    Split presence–absence data into spatially distinct train/test sets.
    Falls back to random split if no 'x'/'y' columns found.
    """
    np.random.seed(seed)

    X = X_pa[covs].values 

    # Use KMeans to make spatial clusters
    n_clusters = max(10, int(1 / test_frac))  # adaptive number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    clusters = kmeans.fit_predict(X)

    # Randomly select some clusters for testing
    unique_clusters = np.unique(clusters)
    n_test = max(1, int(len(unique_clusters) * test_frac))
    test_clusters = np.random.choice(unique_clusters, n_test, replace=False)

    print(f'Divided the data into {unique_clusters} clusters, {test_clusters} used for testing')


    test_mask = np.isin(clusters, test_clusters)
    train_mask = ~test_mask

    X_tr, X_te = X_pa.iloc[train_mask], X_pa.iloc[test_mask]
    Y_tr, Y_te = Y_pa.iloc[train_mask], Y_pa.iloc[test_mask]


    return X_tr, X_te, Y_tr, Y_te



### some top down approaches
def kmeans_partition_sweep(
    D: np.ndarray,
    X: np.ndarray,
    test_frac: float = 0.4,
    n_steps_down: int = 5,
    seed: int = 42,
):
    """
    1) Run KMeans with 2 clusters on X to get an initial distant partition.
    2) Adjust the partition to match test_frac exactly, moving boundary points.
    3) Then do n_steps_down steps that gradually MIX the two sets
       (by swapping core points), so the distance gets smaller.

    Returns:
        samples: list of dicts with keys 'A', 'B', 'dist_metric',
                 from most separated to more mixed.
    """
    rng = np.random.default_rng(seed)
    n = len(D)
    target_test_size = int(round(test_frac * n))

    # --- 1) KMeans(2) to get a very separated initial split ---
    km = KMeans(n_clusters=2, random_state=seed, n_init="auto")
    labels = km.fit_predict(X)

    A = np.where(labels == 0)[0]
    B = np.where(labels == 1)[0]

    # choose which cluster is "test" to be closer to target size
    if abs(len(A) - target_test_size) > abs(len(B) - target_test_size):
        A, B = B, A  # swap so that A is closer in size to target

    # --- 2) Adjust sizes to exactly match test_frac, moving boundary points ---
    def min_dists_to_other(set1, set2):
        """For each index in set1, min distance to any index in set2."""
        if len(set1) == 0 or len(set2) == 0:
            return np.full(len(set1), np.inf)
        subD = D[np.ix_(set1, set2)]
        return subD.min(axis=1)

    # shrink A if too big: move points that are closest to B
    while len(A) > target_test_size and len(A) > 0 and len(B) > 0:
        dists = min_dists_to_other(A, B)      # boundary: small distance to B
        idx = int(np.argmin(dists))           # closest to B -> move first
        moved = A[idx]
        A = np.delete(A, idx)
        B = np.append(B, moved)

    # grow A if too small: move points from B that are closest to A
    while len(A) < target_test_size and len(B) > 0:
        dists = min_dists_to_other(B, A)      # boundary: small distance to A
        idx = int(np.argmin(dists))
        moved = B[idx]
        B = np.delete(B, idx)
        A = np.append(A, moved)

    samples = []

    def add_sample(A, B):
        m = partition_distance(A, B, D)
        samples.append({"A": A.copy(), "B": B.copy(), "dist_metric": m})

    # initial, very separated, size-constrained split
    add_sample(A, B)

    # --- 3) Move "down": progressively reduce distance by mixing core points ---
    for _ in range(n_steps_down):
        if len(A) == 0 or len(B) == 0:
            break

        # core points: far from the other side
        dA = min_dists_to_other(A, B)
        dB = min_dists_to_other(B, A)

        # if everything is inf (degenerate), stop
        if not np.any(np.isfinite(dA)) or not np.any(np.isfinite(dB)):
            break

        iA = int(np.argmax(dA))  # A point deepest inside A
        iB = int(np.argmax(dB))  # B point deepest inside B

        a_core = A[iA]
        b_core = B[iB]

        # swap them to mix sets more (sizes stay fixed)
        A_new = A.copy()
        B_new = B.copy()
        A_new[iA] = b_core
        B_new[iB] = a_core

        A, B = A_new, B_new
        add_sample(A, B)

    return samples

def splits_by_kmeans_levels(
    D: np.ndarray,
    X_pa: pd.DataFrame,
    Y_pa: pd.DataFrame,
    covs,
    test_frac: float = 0.4,
    steps_down_list=(0, 5, 10, 20),
    seed: int = 42,
):
    """
    For each value in steps_down_list, run an independent KMeans-based sweep:

      - Call kmeans_partition_sweep(...) with that n_steps_down
      - Take ONLY the last sample from that run
      - Convert to (X_tr, X_te, Y_tr, Y_te, dist_metric)

    Returns:
        List of (X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te, dist_metric),
        one per element in steps_down_list, in the same order.
    """
    # reset indices to work with 0..n-1
    X_pa_ = X_pa.reset_index(drop=True)
    Y_pa_ = Y_pa.reset_index(drop=True)

    # use spatial coordinates that match D (change columns if needed)
    coords = X_pa_[covs].to_numpy()

    splits = []

    for i, n_steps_down in enumerate(steps_down_list):
        # independent KMeans + sweep for this level
        samples = kmeans_partition_sweep(
            D,
            coords,
            test_frac=test_frac,
            n_steps_down=n_steps_down,
            seed=seed + i,  # different seed per level
        )

        # take the *last* partition from this run
        last = samples[-1]
        A, B = last["A"], last["B"]
        d_metric = last["dist_metric"]

        X_pa_tr = X_pa_.iloc[B].reset_index(drop=True)
        X_pa_te = X_pa_.iloc[A].reset_index(drop=True)
        Y_pa_tr = Y_pa_.iloc[B].reset_index(drop=True)
        Y_pa_te = Y_pa_.iloc[A].reset_index(drop=True)

        splits.append((X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te, d_metric))

        print(f'With {n_steps_down} steps down, we got {d_metric}')

    return splits


#### IDEA WITH VARYING Ks

def kmeans_size_constrained_split(
    D: np.ndarray,
    X_pa: pd.DataFrame,
    Y_pa: pd.DataFrame,
    covs,
    test_frac: float = 0.4,
    K: int = 10,
    seed: int = 42,
):
    """
    Build a spatial train/test split using KMeans with K clusters:

      1) Run KMeans with K clusters on X_pa[covs].
      2) Assign *whole clusters* to test/train to get test size as close as
         possible to test_frac * n.
      3) Fix the size constraint exactly by moving boundary points between sets
         (points closest to the opposite side).
    
    Returns:
      X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te, dist_metric
    """
    # clip K
    K = min(K, len(X_pa))

    rng = np.random.default_rng(seed)

    # Work on positional indices
    X_pa_ = X_pa.reset_index(drop=True)
    Y_pa_ = Y_pa.reset_index(drop=True)

    X = X_pa_[covs].to_numpy()
    n = len(X)
    target_test_size = int(round(test_frac * n))

    # --- 1) KMeans clustering ---

    kmeans = KMeans(n_clusters=K, random_state=seed, n_init="auto")
    labels = kmeans.fit_predict(X)
    if len(np.unique(labels)) < K:
        # split some labels randomly to ensure we have K clusters
        unique_labels = np.unique(labels)
        while len(unique_labels) < K:
            i = rng.integers(len(labels))
            labels[i] = np.max(labels) + 1
            unique_labels = np.unique(labels)
        # np.random.shuffle(labels)
    # print(f"KMeans produced {len(np.unique(labels))} clusters for K={K}")


    # clusters: C_k = { i : labels[i] == k }
    cluster_ids = np.unique(labels)
    K_eff = len(cluster_ids)

    # cluster sizes s_k
    cluster_sizes = np.array([np.sum(labels == k) for k in cluster_ids])


    # --- 2) Coarse assignment: assign whole clusters to test/train ---
    # Heuristic: sort clusters by size (descending), decide greedily
    # whether to put each cluster in test so that we keep |A| close to target.

    order = np.argsort(-cluster_sizes)  # indices into cluster_ids, descending size
    is_test_cluster = np.zeros(K_eff, dtype=bool)
    current_test_size = 0

    for idx in order:
        size_k = cluster_sizes[idx]

        # Check if assigning this cluster to test reduces the absolute size error
        err_if_assign = abs((current_test_size + size_k) - target_test_size)
        err_if_skip = abs(current_test_size - target_test_size)

        if err_if_assign < err_if_skip:
            is_test_cluster[idx] = True
            current_test_size += size_k
        # else: keep it in train

    # Map cluster assignment down to point level
    # test_mask[i] = True iff its cluster is assigned to test
    cluster_id_to_pos = {cid: i for i, cid in enumerate(cluster_ids)}
    test_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        pos = cluster_id_to_pos[labels[i]]
        if is_test_cluster[pos]:
            test_mask[i] = True

    A = np.where(test_mask)[0]
    B = np.where(~test_mask)[0]

    # --- 3) Fix size constraint exactly by moving boundary points ---
    current_test_size = len(A)
    diff = current_test_size - target_test_size

    if current_test_size == 0 or current_test_size == n:
        # degenerate, fall back to a simple random split
        perm = rng.permutation(n)
        A = perm[:target_test_size]
        B = perm[target_test_size:]

    else:
        if diff > 0:
            # A too big: move (diff) points from A to B.
            # Choose points in A that are closest to B (small min distance to B).
            if len(B) > 0:
                subD = D[np.ix_(A, B)]
                dists_A_to_B = subD.min(axis=1)
                order_A = np.argsort(dists_A_to_B)  # smallest first
                to_move = A[order_A[:diff]]

                test_mask[to_move] = False

        elif diff < 0:
            # A too small: move (-diff) points from B to A.
            if len(A) > 0 and len(B) > 0:
                subD = D[np.ix_(B, A)]
                dists_B_to_A = subD.min(axis=1)
                order_B = np.argsort(dists_B_to_A)  # smallest first
                to_move = B[order_B[:(-diff)]]

                test_mask[to_move] = True

        # Recompute A, B after adjustments
        A = np.where(test_mask)[0]
        B = np.where(~test_mask)[0]

        # In pathological cases, if |A| still not equal (e.g. empty A or B),
        # do a last resort fix by random reassignment.
        if len(A) == 0 or len(B) == 0 or len(A) != target_test_size:
            perm = rng.permutation(n)
            A = perm[:target_test_size]
            B = perm[target_test_size:]

    # Final distance metric
    dist_metric = partition_distance(A, B, D)

    # Build X/Y splits
    X_pa_tr = X_pa_.iloc[B].reset_index(drop=True)
    X_pa_te = X_pa_.iloc[A].reset_index(drop=True)
    Y_pa_tr = Y_pa_.iloc[B].reset_index(drop=True)
    Y_pa_te = Y_pa_.iloc[A].reset_index(drop=True)

    # print(
    #     f"[KMeans size-constrained] K={K}, "
    #     f"target_test_size={target_test_size}, "
    #     f"actual_test_size={len(A)}, dist_metric={dist_metric}"
    # )

    return X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te, dist_metric


def splits_by_kmeans_multik(
    D: np.ndarray,
    X_pa: pd.DataFrame,
    Y_pa: pd.DataFrame,
    covs,
    test_frac: float = 0.4,
    K_list=(2, 3, 5, 10, 20),
    seed: int = 42,
):
    """
    Generate multiple spatial splits by varying the number of KMeans clusters.

    For each K in K_list:
      - Run kmeans_size_constrained_split(...) with that K
      - Return (X_tr, X_te, Y_tr, Y_te, dist_metric)

    Returns:
      List of (X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te, dist_metric),
      one per K in K_list (same order).
    """
    splits = []

    for i, K in enumerate(K_list):
        X_tr, X_te, Y_tr, Y_te, d_metric = kmeans_size_constrained_split(
            D=D,
            X_pa=X_pa,
            Y_pa=Y_pa,
            covs=covs,
            test_frac=test_frac,
            K=K,
            seed=seed*(i+1)*2,
        )
        print(f"Generated KMeans size-constrained split with K={K}, distance={d_metric}")
        splits.append((X_tr, X_te, Y_tr, Y_te, d_metric))

    return splits

def partition_linkage_matrix(D: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Build a simple 'linkage-like' matrix for a given bipartition (A, B):

      - L[i, j] = D[i, j]  if i in A, j in B or i in B, j in A
      - L[i, j] = 0        if i, j both in A or both in B

    This emphasizes the separation induced by (A, B).
    """
    n = D.shape[0]
    L = np.zeros((n, n), dtype=float)

    # between-set distances (A x B and B x A)
    L[np.ix_(A, B)] = D[np.ix_(A, B)]
    L[np.ix_(B, A)] = D[np.ix_(B, A)]

    # diagonals already zero from zeros()
    return L



def splits_by_kmeans_convex(
    D: np.ndarray,
    X_pa: pd.DataFrame,
    Y_pa: pd.DataFrame,
    covs,
    test_frac: float = 0.4,
    K: int = 2,
    lambdas=(0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5),
    seed: int = 42,
):
    """
    """

    _, _, _, _, _, B_best, A_best  = agglomerative_size_constrained_split(
        D=D,
        D_orig=D,
        X_pa=X_pa,
        Y_pa=Y_pa,
        test_frac=test_frac,
        K=K,
        seed=seed,
        return_indices=True
    ) 
    D_max = partition_linkage_matrix(D, A_best, B_best)

    # worst splits: random split
    rng = np.random.default_rng(seed)
    A_rand, B_rand, _ = random_partition(D, test_frac=test_frac, rng=rng)


    D_min = partition_linkage_matrix(D, A_rand, B_rand)
    

    splits = []

    for i, lam in enumerate(lambdas):
        # convex combination between max and min
        # D_lam = lam * D_max + (1.0 - lam) * D_min

        # pick lambda random indices
        random_swaps = np.random.choice(D.shape[0], size=int(lam * D.shape[0]), replace=False)
        D_lam = D_max.copy()
        for idx in random_swaps:
            D_lam[idx, :] = D_min[idx, :].copy()
            D_lam[:, idx] = D_min[:, idx].copy()


        X_tr, X_te, Y_tr, Y_te, d_metric = agglomerative_size_constrained_split(
            D=D_lam,
            D_orig=D,
            X_pa=X_pa,
            Y_pa=Y_pa,
            test_frac=test_frac,
            K=K,
            seed=seed + i,
        )

        print(
            f"[convex KMeans] lambda={lam:.4f}, "
            f"K={K}, distance_metric={d_metric}"
        )

        splits.append((X_tr, X_te, Y_tr, Y_te, d_metric))

    return splits


def kmeans_size_constrained_split_convex(
    D: np.ndarray,
    X_pa: pd.DataFrame,
    Y_pa: pd.DataFrame,
    covs,
    lamb: float = 1.0,
    test_frac: float = 0.4,
    K: int = 10,
    seed: int = 42,
):
    """
    Build a spatial train/test split using KMeans with K clusters:

      1) Run KMeans with K clusters on X_pa[covs].
      2) Assign *whole clusters* to test/train to get test size as close as
         possible to test_frac * n.
      3) Fix the size constraint exactly by moving boundary points between sets
         (points closest to the opposite side).
    
    Returns:
      X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te, dist_metric
    """
    # clip K
    K = min(K, len(X_pa))

    rng = np.random.default_rng(seed)

    # Work on positional indices
    X_pa_ = X_pa.reset_index(drop=True)
    Y_pa_ = Y_pa.reset_index(drop=True)

    X = X_pa_[covs].to_numpy().copy()
    X = lamb * X + (1 - lamb) * (1/(X+1e-6))  
    n = len(X)
    target_test_size = int(round(test_frac * n))

    # --- 1) KMeans clustering ---

    kmeans = KMeans(n_clusters=K, random_state=seed, n_init="auto")
    labels = kmeans.fit_predict(X)
    if len(np.unique(labels)) < K:
        # split some labels randomly to ensure we have K clusters
        unique_labels = np.unique(labels)
        while len(unique_labels) < K:
            i = rng.integers(len(labels))
            labels[i] = np.max(labels) + 1
            unique_labels = np.unique(labels)
        # np.random.shuffle(labels)
    # print(f"KMeans produced {len(np.unique(labels))} clusters for K={K}")


    # clusters: C_k = { i : labels[i] == k }
    cluster_ids = np.unique(labels)
    K_eff = len(cluster_ids)

    # cluster sizes s_k
    cluster_sizes = np.array([np.sum(labels == k) for k in cluster_ids])
    print('Cluster sizes:', cluster_sizes)


    # --- 2) Coarse assignment: assign whole clusters to test/train ---
    # Heuristic: sort clusters by size (descending), decide greedily
    # whether to put each cluster in test so that we keep |A| close to target.

    order = np.argsort(-cluster_sizes)  # indices into cluster_ids, descending size
    is_test_cluster = np.zeros(K_eff, dtype=bool)
    current_test_size = 0

    for idx in order:
        size_k = cluster_sizes[idx]

        # Check if assigning this cluster to test reduces the absolute size error
        err_if_assign = abs((current_test_size + size_k) - target_test_size)
        err_if_skip = abs(current_test_size - target_test_size)

        if err_if_assign < err_if_skip:
            is_test_cluster[idx] = True
            current_test_size += size_k
        # else: keep it in train

    # Map cluster assignment down to point level
    # test_mask[i] = True iff its cluster is assigned to test
    cluster_id_to_pos = {cid: i for i, cid in enumerate(cluster_ids)}
    test_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        pos = cluster_id_to_pos[labels[i]]
        if is_test_cluster[pos]:
            test_mask[i] = True

    A = np.where(test_mask)[0]
    B = np.where(~test_mask)[0]

    # --- 3) Fix size constraint exactly by moving boundary points ---
    current_test_size = len(A)
    diff = current_test_size - target_test_size

    if current_test_size == 0 or current_test_size == n:
        # degenerate, fall back to a simple random split
        perm = rng.permutation(n)
        A = perm[:target_test_size]
        B = perm[target_test_size:]

    else:
        if diff > 0:
            # A too big: move (diff) points from A to B.
            # Choose points in A that are closest to B (small min distance to B).
            if len(B) > 0:
                subD = D[np.ix_(A, B)]
                dists_A_to_B = subD.min(axis=1)
                order_A = np.argsort(dists_A_to_B)  # smallest first
                to_move = A[order_A[:diff]]

                test_mask[to_move] = False

        elif diff < 0:
            # A too small: move (-diff) points from B to A.
            if len(A) > 0 and len(B) > 0:
                subD = D[np.ix_(B, A)]
                dists_B_to_A = subD.min(axis=1)
                order_B = np.argsort(dists_B_to_A)  # smallest first
                to_move = B[order_B[:(-diff)]]

                test_mask[to_move] = True

        # Recompute A, B after adjustments
        A = np.where(test_mask)[0]
        B = np.where(~test_mask)[0]

        # In pathological cases, if |A| still not equal (e.g. empty A or B),
        # do a last resort fix by random reassignment.
        if len(A) == 0 or len(B) == 0 or len(A) != target_test_size:
            perm = rng.permutation(n)
            A = perm[:target_test_size]
            B = perm[target_test_size:]

    # Final distance metric
    dist_metric = partition_distance(A, B, D)

    # Build X/Y splits
    X_pa_tr = X_pa_.iloc[B].reset_index(drop=True)
    X_pa_te = X_pa_.iloc[A].reset_index(drop=True)
    Y_pa_tr = Y_pa_.iloc[B].reset_index(drop=True)
    Y_pa_te = Y_pa_.iloc[A].reset_index(drop=True)

    # print(
    #     f"[KMeans size-constrained] K={K}, "
    #     f"target_test_size={target_test_size}, "
    #     f"actual_test_size={len(A)}, dist_metric={dist_metric}"
    # )

    return X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te, dist_metric

def agglomerative_size_constrained_split(
    D: np.ndarray,
    D_orig: np.ndarray,
    X_pa: pd.DataFrame,
    Y_pa: pd.DataFrame,
    test_frac: float = 0.4,
    K: int = 2,
    linkage: str = "average",
    seed: int = 42,
    return_indices: bool = False,
):
    """
    Size-constrained split using Agglomerative Clustering on a distance matrix D.

    Steps:
      1) Run AgglomerativeClustering with K clusters on D (precomputed distance).
      2) Assign *whole clusters* to test/train to get |test| close to
         test_frac * n (greedy by cluster size).
      3) Fix the size constraint exactly by moving boundary points
         (those closest to the opposite side) using D.
      4) Return X/Y train/test splits and the partition_distance(A, B, D).

    Parameters
    ----------
    D : np.ndarray (n x n)
        Distance matrix.
    X_pa, Y_pa : pd.DataFrame
        Features and targets (same order as D).
    test_frac : float
        Fraction of observations in the test set.
    K : int
        Number of clusters for agglomerative clustering.
    linkage : {"average", "complete", "single", "ward"}
        Linkage type for agglomerative clustering (note: "ward" ignores metric).
    seed : int
        RNG seed (used only for fallback / tie-breaking).
    return_indices : bool
        If True, also returns (train_idx, test_idx) arrays of indices 0..n-1.

    Returns
    -------
    X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te, dist_metric
    (and optionally train_idx, test_idx if return_indices=True)
    """
    rng = np.random.default_rng(seed)

    # clip K so it is not larger than n
    K = min(K, len(X_pa))

    # Work on positional indices
    X_pa_ = X_pa.reset_index(drop=True)
    Y_pa_ = Y_pa.reset_index(drop=True)

    n = len(X_pa_)
    target_test_size = int(round(test_frac * n))

    # --- 1) Agglomerative clustering on the distance matrix D ---
    # We use metric='precomputed' so it uses D directly.
    agg = AgglomerativeClustering(
        n_clusters=K,
        metric="precomputed",
        linkage=linkage,
    )

    labels = agg.fit_predict(D)

    # In principle, AgglomerativeClustering should give exactly K labels,
    # but just in case, handle any degeneracy lightly.
    cluster_ids = np.unique(labels)
    K_eff = len(cluster_ids)

    # cluster sizes s_k
    cluster_sizes = np.array([np.sum(labels == k) for k in cluster_ids])

    # --- 2) Coarse assignment: assign whole clusters to test/train ---
    order = np.argsort(-cluster_sizes)  # indices into cluster_ids, descending size
    is_test_cluster = np.zeros(K_eff, dtype=bool)
    current_test_size = 0

    for idx in order:
        size_k = cluster_sizes[idx]
        err_if_assign = abs((current_test_size + size_k) - target_test_size)
        err_if_skip = abs(current_test_size - target_test_size)

        if err_if_assign < err_if_skip:
            is_test_cluster[idx] = True
            current_test_size += size_k

    # Map cluster assignment down to point level
    cluster_id_to_pos = {cid: i for i, cid in enumerate(cluster_ids)}
    test_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        pos = cluster_id_to_pos[labels[i]]
        if is_test_cluster[pos]:
            test_mask[i] = True

    A = np.where(test_mask)[0]   # test indices
    B = np.where(~test_mask)[0]  # train indices

    # --- 3) Fix size constraint exactly by moving boundary points ---
    current_test_size = len(A)
    diff = current_test_size - target_test_size

    if current_test_size == 0 or current_test_size == n:
        # degenerate, fall back to a simple random split
        perm = rng.permutation(n)
        A = perm[:target_test_size]
        B = perm[target_test_size:]
    else:
        if diff > 0 and len(B) > 0:
            # A too big: move (diff) points from A to B (closest to B)
            subD = D[np.ix_(A, B)]
            dists_A_to_B = subD.min(axis=1)
            order_A = np.argsort(dists_A_to_B)  # smallest first
            to_move = A[order_A[:diff]]
            test_mask[to_move] = False

        elif diff < 0 and len(A) > 0:
            # A too small: move (-diff) points from B to A (closest to A)
            subD = D[np.ix_(B, A)]
            dists_B_to_A = subD.min(axis=1)
            order_B = np.argsort(dists_B_to_A)  # smallest first
            to_move = B[order_B[:(-diff)]]
            test_mask[to_move] = True

        # recompute A, B after adjustments
        A = np.where(test_mask)[0]
        B = np.where(~test_mask)[0]

        # last-resort fix if something is still pathological
        if len(A) == 0 or len(B) == 0 or len(A) != target_test_size:
            perm = rng.permutation(n)
            A = perm[:target_test_size]
            B = perm[target_test_size:]

    # Final distance metric with the original distance matrix
    dist_metric = partition_distance(A, B, D_orig)

    # Build X/Y splits
    X_pa_tr = X_pa_.iloc[B].reset_index(drop=True)
    X_pa_te = X_pa_.iloc[A].reset_index(drop=True)
    Y_pa_tr = Y_pa_.iloc[B].reset_index(drop=True)
    Y_pa_te = Y_pa_.iloc[A].reset_index(drop=True)

    if return_indices:
        return X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te, dist_metric, B, A
    else:
        return X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te, dist_metric




def splits_along_partition_path(
    D: np.ndarray,
    X_pa: pd.DataFrame,
    Y_pa: pd.DataFrame, 
    covs,
    test_frac: float = 0.4,
    K: int = 2,
    n_partitions: int = 10,
):
    """
    Generate a sequence of spatial splits following a path between two
    bipartitions (A_start,B_start) and (A_end,B_end).

    For each intermediate (A_t,B_t) we build:
      (X_tr, X_te, Y_tr, Y_te, dist_metric)

    Returns:
      List of (X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te, dist_metric),
      one per step along the path.
    """
    X_pa_ = X_pa.reset_index(drop=True)
    Y_pa_ = Y_pa.reset_index(drop=True)

    X_pa_np = X_pa_[covs].to_numpy()
    
    _, _, _, _, _, B_best, A_best  = agglomerative_size_constrained_split(
        D=D,
        D_orig=D,
        X_pa=X_pa,
        Y_pa=Y_pa,
        test_frac=test_frac,
        K=K,
        seed=132,
        return_indices=True
    ) 
    # max_distance = partition_distance(A_best, B_best, D)
    max_distance = KL_partition_distance(A_best, B_best, X_pa_np)
    # D_max = partition_linkage_matrix(D, A_best, B_best)

    # worst splits: random split
    # rng = np.random.default_rng(seed)
    test_size = len(A_best)
    A_rand, B_rand, _ = random_partition(D,test_size=test_size)
    # min_distance = partition_distance(A_rand, B_rand, D)
    min_distance = KL_partition_distance(A_rand, B_rand, X_pa_np)

    objective_distances = np.linspace(max_distance, min_distance, n_partitions)
    print(f"Max distance: {max_distance}, Min distance: {min_distance}")
    print(f"Objective distances along path: {objective_distances}")

    # check sizes match
    print(f"Start partition size: |A|={len(A_rand)}, |B|={len(B_rand)}")
    print(f"End partition size:   |A|={len(A_best)}, |B|={len(B_best)}")

    partitions = path_between_partitions(A_best, B_best, A_rand, B_rand)

    n_total_partitions = len(partitions)
    evaluate_every = max(1, n_total_partitions // (n_partitions * 30))
    print(evaluate_every)

    splits = []
    current_ix = 0
    for step, (A_t, B_t) in enumerate(partitions):
        if step % evaluate_every != 0 and step != n_total_partitions - 1 and step != 0:
            continue
        # if step % 10 == 0:
        # print(f"Processing path step {step}/{len(partitions)-1}...")
        # d_metric = partition_distance(A_t, B_t, D)
        d_metric = KL_partition_distance(A_t, B_t, X_pa_np)
        if d_metric <= objective_distances[current_ix]:
            # move to next target distance
            current_ix +=1

            X_pa_tr = X_pa_.iloc[B_t].reset_index(drop=True)
            X_pa_te = X_pa_.iloc[A_t].reset_index(drop=True)
            Y_pa_tr = Y_pa_.iloc[B_t].reset_index(drop=True)
            Y_pa_te = Y_pa_.iloc[A_t].reset_index(drop=True)

            splits.append((X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te, d_metric))
            print(f"Path step {step}: dist_metric={d_metric}")
        if current_ix >= n_partitions:
            break

    return splits

def path_between_partitions(A_start: np.ndarray,
                            B_start: np.ndarray,
                            A_end: np.ndarray,
                            B_end: np.ndarray):
    """
    Build a list of bipartitions (A_t, B_t) forming a path from
    (A_start,B_start) to (A_end,B_end) by gradually swapping membership
    of points so that |A_t| stays constant.

    Assumes:
      - A_start, B_start, A_end, B_end are disjoint pairs whose unions are
        the same set {0..n-1}
      - |A_start| == |A_end|

    Returns:
      partitions: list of (A_t, B_t), t = 0..m
                  where (A_0,B_0) = (A_start,B_start),
                        (A_m,B_m) = (A_end,B_end).
    """
    A_start = np.array(A_start, dtype=int)
    B_start = np.array(B_start, dtype=int)
    A_end   = np.array(A_end,   dtype=int)
    B_end   = np.array(B_end,   dtype=int)

    # elements that must leave A_start and enter B_start
    S_from = np.setdiff1d(A_start, A_end)
    # elements that must enter A to match A_end
    S_to   = np.setdiff1d(A_end, A_start)

    assert len(S_from) == len(S_to), "Test sizes of start/end must match."

    A_curr = set(A_start.tolist())
    B_curr = set(B_start.tolist())

    partitions = []
    partitions.append((np.array(sorted(A_curr)), np.array(sorted(B_curr))))

    # Gradually swap membership along the difference sets
    for r, a in zip(S_from, S_to):
        # r: remove from A, add to B
        # a: remove from B, add to A
        if r in A_curr:
            A_curr.remove(r)
            B_curr.add(r)
        if a in B_curr:
            B_curr.remove(a)
            A_curr.add(a)

        A_t = np.array(sorted(A_curr))
        B_t = np.array(sorted(B_curr))
        partitions.append((A_t, B_t))

    return partitions



def partition_sweep_one_on_one(
    X_pa: pd.DataFrame,
    Y_pa: pd.DataFrame,
    covs_cluster: list,
    covs_distance: list,
    K_clusters: int = 10,
    select_subset: int = 20,
):

    D = pairwise_distances(
        X_pa[covs_distance].to_numpy(),
        metric='euclidean'
    )

    # generate K clusters with size constrained
    n_samples = len(X_pa)
    avg = n_samples / K_clusters
    min_size = np.floor(avg)
    max_size = np.ceil(avg)

    print(f"Generating {K_clusters} clusters with sizes in [{min_size}, {max_size}]")
    kmeans = KMeansConstrained(
        n_clusters=K_clusters,
        size_min=min_size,
        size_max=max_size,
        random_state=42,
        n_init=10,
    )

    X_coords = X_pa[covs_cluster].to_numpy()
    labels = kmeans.fit_predict(X_coords)

    unique_labels = np.unique(labels)

    splits = []

    print('Calculating all one-on-one cluster splits...')
    for test_ix in unique_labels:
        for train_ix in unique_labels:
            if test_ix == train_ix:
                continue

            test_mask = (labels == test_ix)
            train_mask = (labels == train_ix)

            A = np.where(test_mask)[0]
            B = np.where(train_mask)[0]

            X_pa_tr = X_pa.iloc[B].reset_index(drop=True)
            X_pa_te = X_pa.iloc[A].reset_index(drop=True)
            Y_pa_tr = Y_pa.iloc[B].reset_index(drop=True)
            Y_pa_te = Y_pa.iloc[A].reset_index(drop=True)

            d_metric = partition_distance(A, B, D)

            splits.append((X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te, d_metric))
            # print(f"Cluster pair (test={test_ix}, train={train_ix}): distance={d_metric}")
    
    # Select `select_at_random` splits so that distances are as uniform as possible
    n_splits = len(splits)
    if (select_subset is None) or (select_subset >= n_splits):
        # Just take all splits
        selected_splits = splits
    else:
        # Sort splits by distance
        splits_sorted = sorted(splits, key=lambda s: s[4])  # s[4] is d_metric

        # Pick approximately uniformly spaced indices in [0, n_splits-1]
        idx = np.linspace(0, n_splits - 1, num=select_subset)
        idx = np.round(idx).astype(int)
        idx = np.unique(idx)  # just in case of rounding collisions

        selected_splits = [splits_sorted[i] for i in idx]
    
    for s in selected_splits:
        print(f"Selected split with distance={s[4]}")

    return selected_splits


def partition_sweep_optimal(
    D: np.ndarray,
    X_pa: pd.DataFrame,
    Y_pa: pd.DataFrame,
    covs_cluster: list,
    covs_distance: list,
    test_frac: float = 0.4,
    K: int = 20 ,
    n_partitions: int = 10,
):
# generate K clusters with size constrained
    n_samples = len(X_pa)
    avg = n_samples / K
    min_size = np.floor(avg)
    max_size = np.ceil(avg)

    print(f"Generating {K} clusters with sizes in [{min_size}, {max_size}]")
    kmeans = KMeansConstrained(
        n_clusters=K,
        size_min=min_size,
        size_max=max_size,
        random_state=42,
        n_init=10,
    )

    X_coords = X_pa[covs_cluster].to_numpy()
    labels = kmeans.fit_predict(X_coords)

    # check this after, and change for Mahalanobis
    D_centroids = centroid_dist(pd.DataFrame({
        'x': X_coords[:, 0],
        'y': X_coords[:, 1],
        'cluster': labels
    }))

    k_for_optimization = int(K*test_frac)
    print(D_centroids)
    print(f"Optimizing over k={k_for_optimization} clusters...")
    max_cluster_assignment, max_distance = optimize_cluster_split(
        D_centroids,
        k_for_optimization
    )
    min_cluster_assignment, min_distance = minimize_cluster_split(
        D_centroids,
        k_for_optimization
    )   
    print(f"Max distance: {max_distance}, Min distance: {min_distance}")
    objective_distances = np.linspace(max_distance, min_distance, n_partitions)
    print(f"Objective distances along path: {objective_distances}")

    splits = []
    for i, dist in enumerate(objective_distances):
        print(f"Processing partition {i+1}/{n_partitions} with target distance {dist}...")
        
        if i == 0:
            cluster_assignment, final_distance = max_cluster_assignment, max_distance
        elif i == n_partitions - 1:
            cluster_assignment, final_distance = min_cluster_assignment, min_distance
        else: 
            cluster_assignment, final_distance = minimize_cluster_split(
                D_centroids,
                k_for_optimization,
                target_min_distance=dist)
            print(f'Target {dist}, achieved {final_distance}')

        test_clusters = [i for i, v in enumerate(cluster_assignment) if v > 0.5]
        train_clusters = [i for i, v in enumerate(cluster_assignment) if v <= 0.5]

        test_mask = np.isin(labels, test_clusters)
        train_mask = np.isin(labels, train_clusters)
        A = np.where(test_mask)[0]
        B = np.where(train_mask)[0]
        X_pa_tr = X_pa.iloc[B].reset_index(drop=True)
        X_pa_te = X_pa.iloc[A].reset_index(drop=True)
        Y_pa_tr = Y_pa.iloc[B].reset_index(drop=True)
        Y_pa_te = Y_pa.iloc[A].reset_index(drop=True)
        
        # # this should be the real metric, for now we will use 
        # # the centroid distance as a proxy
        # d_metric = partition_distance(A, B, D) 

        d_metric = final_distance

        print(f"Partition {i+1}: distance={d_metric}, |A|={len(A)}, |B|={len(B)}")
        # yield (X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te, d_metric) # version with yield
        splits.append((X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te, d_metric))
    return splits


def partition_sweep_ranges(
    X_pa: pd.DataFrame,
    Y_pa: pd.DataFrame,
    covs_cluster: list,
    covs_distance: list,
    K_clusters: int = 50,
    select_subset: int = 20,
    train_proportion = .4
):

    D = pairwise_distances(
        X_pa[covs_distance].to_numpy(),
        metric='mahalanobis'
    )

    # generate K clusters with size constrained
    n_samples = len(X_pa)
    if n_samples / K_clusters < 10:
        K_clusters = n_samples // 10
        if select_subset > K_clusters:
            select_subset = K_clusters

        print(f"Adjusted K_clusters to {K_clusters} due to small sample size.")
    avg = n_samples / K_clusters
    min_size = np.floor(avg)
    max_size = np.ceil(avg)

    print(f"Generating {K_clusters} clusters with sizes in [{min_size}, {max_size}]")
    kmeans = KMeansConstrained(
        n_clusters=K_clusters,
        size_min=min_size,
        size_max=max_size,
        random_state=42,
        n_init=10,
    )

    X_coords = X_pa[covs_cluster].to_numpy()
    labels = kmeans.fit_predict(X_coords)

    # centroids calculation using covs_distance
    centroids = np.array([X_pa[covs_distance].iloc[labels == i].mean(axis=0) for i in range(K_clusters)])
    # distance between centroids
    D_centroids = pairwise_distances(
        centroids,
        metric='mahalanobis'
    )

    unique_labels = np.unique(labels)

    splits = []
    split_type_list = []

    # select subset of unique labels to reduce number of splits (these are for test, at random)
    selected_labels = np.random.choice(unique_labels, size=select_subset, replace=False)

    clusters_in_train = int(K_clusters * train_proportion)
    print(f"Using {clusters_in_train} clusters in train set.")

    for test_ix in selected_labels:
        # select the closests clusters to test_ix to form train set using D_centroids
        dists_to_test = D_centroids[test_ix].copy()
        dists_to_test[test_ix] = np.inf  # ignore self-distance

        # one with the closests clusters_in_train, one with the middle clusters_in_train, one with the farthest clusters_in_train
        for option in ['closest', 'middle', 'farthest']:
            if option == 'closest':
                train_ixs = np.argsort(dists_to_test)[:clusters_in_train]
            elif option == 'middle':
                sorted_ixs = np.argsort(dists_to_test)
                start_ix = (K_clusters - clusters_in_train) // 2
                train_ixs = sorted_ixs[start_ix:start_ix + clusters_in_train]
            elif option == 'farthest':
                sorted_ixs = np.argsort(dists_to_test)[::-1]
                train_ixs = sorted_ixs[:clusters_in_train]
            
            # remove test_ix from train_ixs if present
            train_ixs = train_ixs[train_ixs != test_ix]

            test_mask = (labels == test_ix)
            train_mask = np.isin(labels, train_ixs)
            A = np.where(test_mask)[0]
            B = np.where(train_mask)[0]
            X_pa_tr = X_pa.iloc[B].reset_index(drop=True)
            X_pa_te = X_pa.iloc[A].reset_index(drop=True)
            Y_pa_tr = Y_pa.iloc[B].reset_index(drop=True)
            Y_pa_te = Y_pa.iloc[A].reset_index(drop=True)
            d_metric = partition_distance(A, B, D)
            splits.append((X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te, d_metric))
            split_type_list.append((option, test_ix))
            print(f"Cluster pair (test={test_ix}, train={train_ixs}): distance={d_metric}")





    return splits, split_type_list
  
    
if __name__ == '__main__':

    D = np.array([
        [0, 1, 4, 7, 9],
        [1, 0, 3, 8, 6],
        [4, 3, 0, 2, 5],
        [7, 8, 2, 0, 3],
        [9, 6, 5, 3, 0]
    ], dtype=float)

    partition_sweep(D)




