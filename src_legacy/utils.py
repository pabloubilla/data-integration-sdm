import numpy as np
from sklearn.cluster import KMeans
# from k_means_constrained import KMeansConstrained
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
import pickle



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


def split_pa_train_test_spatially(X_pa, Y_pa, test_frac=0.3, seed=42, K: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    """
    Split presence–absence data into spatially distinct train/test sets.
    Falls back to random split if no 'x'/'y' columns found.
    """
    np.random.seed(seed)

    # --- Spatially aware split ---
    if {'x', 'y'}.issubset(X_pa.columns):
        coords = X_pa[['x', 'y']].values
        coords = (coords - np.mean(coords,axis=0))/np.std(coords,axis=0)

        # Use KMeans to make spatial clusters
        n_clusters = max(K, int(1 / test_frac))  # adaptive number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
        clusters = kmeans.fit_predict(coords)

        # Randomly select some clusters for testing
        unique_clusters = np.unique(clusters)
        n_test = max(1, int(len(unique_clusters) * test_frac))
        test_clusters = np.random.choice(unique_clusters, n_test, replace=False)

        print(f'Divided the data into {unique_clusters} clusters, {test_clusters} used for testing')


        test_mask = np.isin(clusters, test_clusters)
        train_mask = ~test_mask

        X_tr, X_te = X_pa.iloc[train_mask], X_pa.iloc[test_mask]
        Y_tr, Y_te = Y_pa.iloc[train_mask], Y_pa.iloc[test_mask]
    else:
        # --- Fallback to random split ---
        X_tr, X_te, Y_tr, Y_te = train_test_split(
            X_pa, Y_pa, test_size=test_frac, random_state=seed, stratify=Y_pa
        )

    return X_tr, X_te, Y_tr, Y_te

def safe_reindex_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        df = df.copy()
        for c in missing:
            df[c] = 0
    return df[cols]


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    covariates: List[str],
    output_path: Optional[str] = None,
    verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler().fit(X_train[covariates])
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[covariates] = scaler.transform(X_train[covariates])
    X_test_scaled[covariates] = scaler.transform(X_test[covariates])
    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(scaler, f)
    if verbose:
        print("\nFeature scaling summary:")
        for col in covariates:
            print(f"{col}: Train mean={X_train_scaled[col].mean():.4f}, Train std={X_train_scaled[col].std():.4f}, "
                  f"Test mean={X_test_scaled[col].mean():.4f}, Test std={X_test_scaled[col].std():.4f}")
    return X_train_scaled, X_test_scaled, scaler