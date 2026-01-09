import numpy as np
import cvxpy as cp
import geopandas as gpd
import pandas as pd
from k_means_constrained import KMeansConstrained
import matplotlib.pyplot as plt


# ============================
#   OPTIMIZATION FUNCTIONS
# ============================

def optimize_cluster_split(D: np.ndarray, K: int):
    """
    Maximize the separation between test clusters and train clusters.

    D: (n x n) distance matrix between cluster centroids.
    K: number of clusters to assign to the TEST set.

    Returns
    -------
    x_opt : np.ndarray
        Binary vector of length n, 1 = test cluster, 0 = train cluster.
    obj_value : float
        Optimal objective value (sum of min distances to nearest train).
    """
    n = len(D)
    M = float(D.max())  # tight big-M helps efficiency

    x = cp.Variable(n, boolean=True)   # 1 = test cluster
    w = cp.Variable(n)                 # min distance to a train cluster

    constraints = []

    # Pick exactly K test clusters
    constraints.append(cp.sum(x) == K)

    # If cluster i is train (x[i] = 0) => w[i] = 0 (upper bound)
    for i in range(n):
        constraints.append(w[i] <= M * x[i])
        constraints.append(w[i] >= 0)

    # Ensure w[i] is the minimum distance from test cluster i to any TRAIN cluster
    # For train cluster j (x[j] = 0): w[i] <= D[i,j]
    # For test cluster j (x[j] = 1): w[i] <= D[i,j] + M (loose upper bound)
    for i in range(n):
        for j in range(n):
            constraints.append(w[i] <= D[i, j] + M * x[j])

    objective = cp.Maximize(cp.sum(w))

    problem = cp.Problem(objective, constraints)
    obj_value = problem.solve(solver=cp.SCIPY)

    x_opt = np.round(x.value).astype(int)

    print("[optimize_cluster_split]")
    print("x (test clusters):", x_opt)
    print("w (min distances):", w.value)
    print("objective:", obj_value)

    return x_opt, obj_value


def minimize_cluster_split(
        D: np.ndarray,
        K: int,
        target_min_distance = None,
        verbose: bool = True
):
    """
    Minimize (or match) the total distance between test clusters and their assigned train cluster.

    D: (n x n) distance matrix between cluster centroids.
    K: number of clusters to assign to TEST.
    target_min_distance:
        - If None: minimize total distance sum(w).
        - Else : minimize |sum(w) - target_min_distance|.

    Returns
    -------
    x_opt : np.ndarray
        Binary vector of length n, 1 = test cluster, 0 = train cluster.
    final_distance : float
        sum(w) at the optimum (total distance).
    """
    n_vars = len(D)
    x = cp.Variable(n_vars, boolean=True)                   # 1 = test
    z = cp.Variable((n_vars, n_vars), boolean=True)         # assignment test i -> train j
    w = cp.Variable(n_vars)                                 # distance for each i

    constraints = []

    # Choose exactly K test clusters
    constraints += [cp.sum(x) == K]

    # Distances nonnegative
    constraints += [w >= 0]

    for i in range(n_vars):
        # If i is test (x[i]=1) → assign to exactly one train
        # If i is train (x[i]=0) → assign to no one
        constraints += [cp.sum(z[i, :]) == x[i]]

        for j in range(n_vars):
            # Can only assign i to j if j is TRAIN (x[j]=0)
            constraints += [z[i, j] <= 1 - x[j]]

        # w[i] is the distance to its assigned train
        constraints += [w[i] == cp.sum(cp.multiply(z[i, :], D[i, :]))]

    if target_min_distance is None:
        objective = cp.Minimize(cp.sum(w))
    else:
        # Minimize |target_min_distance - sum(w)| using an auxiliary variable t
        t = cp.Variable()  # nonnegative deviation
        constraints += [
            t >= cp.sum(w) - target_min_distance,
            t >= target_min_distance - cp.sum(w),
        ]
        objective = cp.Minimize(t)

    problem = cp.Problem(objective, constraints)
    solution_val = problem.solve(solver=cp.SCIPY)

    # final distance actually obtained
    final_distance = float(np.sum(w.value))

    x_opt = np.round(x.value).astype(int)

    if verbose:
        print("[minimize_cluster_split]")
        print("x (test clusters):", x_opt)
        print("w (distances):", w.value)
        print("Solver objective:", solution_val)
        print("Final_distance (sum w):", final_distance)

    return x_opt, final_distance


# ============================
#   HELPER FUNCTIONS
# ============================

def centroid_dist(df_xy: pd.DataFrame) -> np.ndarray:
    """
    Compute pairwise Euclidean distance matrix between cluster centroids.
    df_xy must have columns: 'x', 'y', 'cluster'
    """
    centroids = df_xy.groupby("cluster")[["x", "y"]].mean().to_numpy()
    diff = centroids[:, None, :] - centroids[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=2))
    return D


def plot_split_map(gdf_map: gpd.GeoDataFrame, df_pa: pd.DataFrame, title: str = ""):
    """
    Plot the train/test split for presence-absence points.
    df_pa must contain columns 'x', 'y', 'set' (values 'train' / 'test').
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    gdf_map.boundary.plot(ax=ax, color='gray', linewidth=1)

    gdf_points_pa = gpd.GeoDataFrame(
        df_pa,
        geometry=gpd.points_from_xy(df_pa.x, df_pa.y),
        crs=gdf_map.crs
    )
    gdf_points_pa.plot(column='set', ax=ax, legend=True, markersize=8, cmap='Set1')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def assignment_to_sets(cluster_assignment: np.ndarray) -> dict:
    """
    Convert binary cluster assignment vector to lists of test/train cluster indices.
    """
    test_clusters = [i for i, v in enumerate(cluster_assignment) if v > 0.5]
    train_clusters = [i for i, v in enumerate(cluster_assignment) if v <= 0.5]
    return {"test": test_clusters, "train": train_clusters}


# ============================
#   MAIN + ALPHA EXPERIMENT
# ============================

if __name__ == "__main__":

    region = 'swi'  # example region code
    group = ''
    group_ = '_' if group != '' else ''

    # Borders (vector layer)
    gdf_map = gpd.read_file(f'data/raw/NCEAS/Borders/{region}.gpkg')

    # Records - I assume these are CSVs, so use pandas
    df_po = pd.read_csv(f'data/processed/NCEAS/Records/train_po/{region.upper()}train_po{group_}{group}.csv')
    df_pa = pd.read_csv(f'data/raw/NCEAS/Records/test_pa/{region.upper()}test_pa{group_}{group}.csv')
    df_env = pd.read_csv(f'data/raw/NCEAS/Records/test_env/{region.upper()}test_env{group_}{group}.csv')

    # Ensure coordinates are floats
    df_pa[['x', 'y']] = df_pa[['x', 'y']].astype(float)

    # -------------------------------------------------
    # 1. K-means constrained clustering on PA points
    # -------------------------------------------------
    n_samples = len(df_pa)
    n_clusters = 20
    avg = n_samples / n_clusters
    min_size = int(np.floor(avg))
    max_size = int(np.ceil(avg))

    # Scale coordinates to [0,1] for clustering
    df_xy_scaled = df_pa[['x', 'y']].copy()
    df_xy_scaled['x'] = (df_xy_scaled['x'] - df_xy_scaled['x'].min()) / (df_xy_scaled['x'].max() - df_xy_scaled['x'].min())
    df_xy_scaled['y'] = (df_xy_scaled['y'] - df_xy_scaled['y'].min()) / (df_xy_scaled['y'].max() - df_xy_scaled['y'].min())

    print('Optimizing KMeansConstrained clustering...')
    kmeans = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=min_size,
        size_max=max_size,
        random_state=0
    )

    df_xy_scaled['cluster'] = kmeans.fit_predict(df_xy_scaled[['x', 'y']])
    print('Done clustering!')

    # Attach cluster index back to df_pa
    df_pa = df_pa.merge(df_xy_scaled[['cluster']], left_index=True, right_index=True)

    # Distance matrix between cluster centroids (in scaled space)
    D = centroid_dist(df_xy_scaled.assign(cluster=df_xy_scaled['cluster']))

    # -------------------------------------------------
    # 2. Get extreme solutions (max.sep and min.dist)
    # -------------------------------------------------
    K_test = 5  # number of test clusters

    cluster_assignment_max, sol_max = optimize_cluster_split(D, K_test)
    cluster_assignment_min, total_min_dist = minimize_cluster_split(D, K_test)

    # -------------------------------------------------
    # 3. Alpha experiment between min and max
    #    target = alpha * sol_max + (1 - alpha) * total_min_dist
    # -------------------------------------------------
    alphas = np.linspace(0.0, 1.0, 5)



    for alpha in alphas:
        if alpha == 0.0:
            cluster_assignment_alpha = cluster_assignment_min
            final_distance = total_min_dist
            target_distance = total_min_dist
        elif alpha == 1.0:
            cluster_assignment_alpha = cluster_assignment_max
            final_distance = sol_max
            target_distance = sol_max
        else:
            target_distance = alpha * sol_max + (1.0 - alpha) * total_min_dist

            cluster_assignment_alpha, final_distance = minimize_cluster_split(
                D,
                K_test,
                target_min_distance=target_distance,
                verbose=False
            )

        print(f'For alpha = {alpha:.2f}: target_distance = {target_distance:.2f}, final_distance = {final_distance:.2f}')

        sets = assignment_to_sets(cluster_assignment_alpha)
        test_clusters = set(sets["test"])

        df_pa_alpha = df_pa.copy()
        df_pa_alpha['set'] = df_pa_alpha['cluster'].apply(
            lambda c: 'test' if c in test_clusters else 'train'
        )

        title = f"Alpha = {alpha:.2f} | target = {target_distance:.2f} | final = {final_distance:.2f}"
        print(title)
        plot_split_map(gdf_map, df_pa_alpha, title=title)


### QUICK BACKUP OF distance_optimizer.py ###
# import numpy as np
# import cvxpy as cp
# import geopandas as gpd
# import pandas as pd
# from k_means_constrained import KMeansConstrained
# # PULP


# # def optimize_cluster_split(
# #         D: np.array, # distance matrix between every component (i,j)
# #         K: int # number of clusters to test
# # ):
# #     n_vars = len(D)
# #     x = cp.Variable(n_vars, boolean=True)
# #     z = cp.Variable(shape = (n_vars, n_vars), boolean=True)
# #     w = cp.Variable(shape = n_vars)

# #     constraints = []
# #     constraints += [cp.sum(x) == K]
# #     for i in range(n_vars):
# #         # w is the min distance 
# #         constraints += [ w[i] <= cp.multiply(z[i,:],D[i,:]) + 1e6*(1 - z[i,:]) ]
# #         for j in range(n_vars):
# #             constraints += [2*z[i,j] <= x[i] + 1-x[j]]
# #             constraints += [z[i,j] >= x[i] - x[j]]
# #     objective = cp.sum(w)
# #     solution = cp.Problem(cp.Maximize(objective), constraints).solve(solver=cp.SCIPY)
    
# #     print('x')
# #     print(x.value)
# #     print('z')
# #     print(z.value)
# #     print('w')
# #     print(w.value)
# #     print('solution')
# #     print(solution)


# #     return x.value


# # # print('Running optimizer MIP')
# # optimize_cluster_split(D, 5)
# # # print('Done!')


# def optimize_cluster_split(D: np.ndarray, K: int):
#     n = len(D)
#     M = float(D.max())  # tight big-M helps efficiency

#     x = cp.Variable(n, boolean=True)   # 1 = test cluster
#     w = cp.Variable(n)                 # min distance to a train cluster

#     constraints = []
    
#     # pick exactly K test clusters
#     constraints.append(cp.sum(x) == K)
    
#     # If not test → w = 0
#     for i in range(n):
#         constraints.append(w[i] <= M * x[i])

#     # min-distance constraints:
#     # w[i] <= D[i,j]  for every train j
#     for i in range(n):
#         for j in range(n):
#             constraints.append(w[i] <= D[i,j] + M * x[j])
    
#     # maximize separation
#     objective = cp.Maximize(cp.sum(w))

#     sol = cp.Problem(objective, constraints).solve(solver=cp.SCIPY)

#     print("x (test clusters):", x.value)
#     print("w (min distances):", w.value)
#     print("objective:", sol)

#     return x.value, sol



# def minimize_cluster_split(
#         D: np.array,  # distance matrix between every component (i,j)
#         K: int,        # number of clusters to test
#         target_min_distance: float | None = None,
#         verbose: bool = True
# ):
#     n_vars = len(D)
#     x = cp.Variable(n_vars, boolean=True)            # 1 = test
#     z = cp.Variable(shape=(n_vars, n_vars), boolean=True)  # assignment test i -> train j
#     w = cp.Variable(shape=n_vars)                   # distance for each i

#     constraints = []

#     # choose exactly K test clusters
#     constraints += [cp.sum(x) == K]

#     # distances are nonnegative
#     constraints += [w >= 0]

#     for i in range(n_vars):
#         # if i is test (x[i]=1) → assign to exactly one train
#         # if i is train (x[i]=0) → assign to no one
#         constraints += [cp.sum(z[i, :]) == x[i]]

    

#         for j in range(n_vars):
#             # can only assign i to j if j is TRAIN (x[j]=0)
#             constraints += [z[i, j] <= 1 - x[j]]

#         # w[i] is the distance to its assigned train
#         constraints += [w[i] == cp.sum(cp.multiply(z[i, :], D[i, :]))]

#     if target_min_distance is None:
#         objective = cp.Minimize(cp.sum(w))
#     else:
#         # minimize |target_min_distance - sum(w)| in a linear way
#         t = cp.Variable()  # nonnegative deviation
#         constraints += [
#             t >= cp.sum(w) - target_min_distance,
#             t >= target_min_distance - cp.sum(w),
#         ]
#         objective = cp.Minimize(t)

#     solution = cp.Problem(objective, constraints).solve(solver=cp.SCIPY)

#     # final distance
#     final_distance = sum(w.value)

#     if verbose:
#         print("x (test clusters):", x.value)
#         print("w (min distances):", w.value)
#         print('Solution:', solution)
#         print("Final_distance:", final_distance)

#     return x.value, final_distance


# # cluster_assignment_1, sol1 = optimize_cluster_split(D, 5)
# # cluster_assignment_2, sol2 = minimize_cluster_split(D, 5)
# # target_distance = (sol1*0.3 + sol2*0.7) 
# # print('target_distance:', target_distance)
# # cluster_assignment_3 = minimize_cluster_split(D, 5, target_min_distance=target_distance)

# # # alpha slider between 0 and 1 with 20 steps
# # alpha = np.linspace(0, 1, 20)
# # for a in alpha:
# #     target_distance = a * sol1 + (1 - a) * sol2
    
# #     cluster_assignment, final_distance = minimize_cluster_split(D, 5, target_min_distance=target_distance, verbose=False)

# #     print(f'alpha: {a}, target_distance: {target_distance}, final_distance: {final_distance}')

if __name__ == "__main__":

    region = 'swi'  # example region code
    group = ''
    group_ = '_' if group != '' else ''
    gdf_map = gpd.read_file(f'data/raw/NCEAS/Borders/{region}.gpkg')  # path for a .gpkg

    # read data\raw\NCEAS\Records\train_po\AWTtrain_po.csv with region variable
    df_po = gpd.read_file(f'data/processed/NCEAS/Records/train_po/{region.upper()}train_po{group_}{group}.csv')


    # data\raw\NCEAS\Records\test_pa\AWTtest_pa_bird.csv
    df_pa = gpd.read_file(f'data/raw/NCEAS/Records/test_pa/{region.upper()}test_pa{group_}{group}.csv')
    df_env = pd.read_csv(f'data/raw/NCEAS/Records/test_env/{region.upper()}test_env{group_}{group}.csv')
    df_pa[['x','y']] = df_pa[['x','y']].astype('float')

    n_samples = len(df_pa)
    n_clusters = 20
    avg = n_samples / n_clusters
    min_size = np.floor(avg)
    max_size = np.ceil(avg)


    df_xy_scaled = df_pa[['x','y']].copy()
    df_xy_scaled['x'] = (df_xy_scaled['x'] - df_xy_scaled['x'].min()) / (df_xy_scaled['x'].max() - df_xy_scaled['x'].min())
    df_xy_scaled['y'] = (df_xy_scaled['y'] - df_xy_scaled['y'].min()) / (df_xy_scaled['y'].max() - df_xy_scaled['y'].min())
    print('Optimizing KmeansConstrained clustering...')
    kmeans = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=min_size,
        size_max=max_size,
        random_state=0
    )

    df_xy_scaled['cluster'] = kmeans.fit_predict(df_xy_scaled[['x','y']])
    print('Done!')

    df_pa = df_pa.merge(df_xy_scaled[['cluster']], left_index=True, right_index=True)


    def centroid_dist(df):
        c = df.groupby("cluster")[["x","y"]].mean().to_numpy()
        return np.sqrt(((c[:,None]-c)**2).sum(2))

    D = centroid_dist(df_xy_scaled)

    # print('Running optimizer MIP')
    # # cluster_assignment = optimize_cluster_split(D, 5)
    # # cluster_assignment = minimize_cluster_split(D, 5)
    # cluster_assignment = optimize_cluster_split_slider(D, 5, alpha=.3)[0]
    # print('Done!')

    cluster_assignment_1 = optimize_cluster_split(D, 5)
    cluster_assignment_2 = minimize_cluster_split(D, 5)[0]
    cluster_assignment = None

    # assign clusters to test/train based on optimizer result (cluster_assignment is a binary array)
    # and 'cluster' is a column with a cluster index
    test_clusters = [i for i in range(len(cluster_assignment)) if cluster_assignment[i] > 0.5]
    train_clusters = [i for i in range(len(cluster_assignment)) if cluster_assignment[i] <= 0.5]
    df_pa['set'] = df_pa['cluster'].apply(lambda x: 'test' if x in test_clusters else 'train')

    # plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_map.boundary.plot(ax=ax, color='gray', linewidth=1)
    gdf_points_pa = gpd.GeoDataFrame(
        df_pa,
        geometry=gpd.points_from_xy(df_pa.x, df_pa.y),
        crs=str(gdf_map.crs)  # Set CRS (important!)
    )
    gdf_points_pa.plot(column='set', ax=ax, legend=True, markersize=8, cmap='Set1')
    plt.show()