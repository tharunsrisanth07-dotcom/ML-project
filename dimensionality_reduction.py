"""
Dimensionality Reduction Module
================================
Applies:
- Truncated SVD on TF-IDF (sparse-compatible)
- PCA on dense embeddings (GloVe, DistilBERT)
Analyzes explained variance and its effect on performance.
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import config


def apply_truncated_svd(X_train, X_test, n_components):
    """
    Apply Truncated SVD to sparse TF-IDF features.

    Parameters
    ----------
    X_train, X_test : sparse matrix
        TF-IDF feature matrices.
    n_components : int
        Number of SVD components.

    Returns
    -------
    tuple
        (X_train_reduced, X_test_reduced, svd_model)
    """
    svd = TruncatedSVD(n_components=n_components, random_state=config.RANDOM_SEED)
    X_train_reduced = svd.fit_transform(X_train)
    X_test_reduced = svd.transform(X_test)
    return X_train_reduced, X_test_reduced, svd


def apply_pca(X_train, X_test, n_components):
    """
    Apply PCA to dense embedding features.

    Parameters
    ----------
    X_train, X_test : ndarray
        Dense feature matrices.
    n_components : int
        Number of PCA components.

    Returns
    -------
    tuple
        (X_train_reduced, X_test_reduced, pca_model)
    """
    pca = PCA(n_components=n_components, random_state=config.RANDOM_SEED)
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)
    return X_train_reduced, X_test_reduced, pca


def analyze_dimensionality_reduction(features_dict, y_train, y_test):
    """
    Comprehensive dimensionality reduction analysis across all feature types.

    For each feature type, applies reduction at multiple component levels,
    records explained variance, and evaluates a Logistic Regression probe.

    Parameters
    ----------
    features_dict : dict
        {feature_name: {'train': X_train, 'test': X_test, 'is_sparse': bool}}
    y_train, y_test : array-like
        Labels.

    Returns
    -------
    dict
        {feature_name: {
            'components': list,
            'explained_variances': list of arrays,
            'cumulative_variances': list,
            'accuracies': list,
        }}
    """
    print("=" * 60)
    print("STAGE: Dimensionality Reduction Analysis")
    print("=" * 60)

    results = {}

    for feat_name, feat_data in tqdm(features_dict.items(), desc="  Dim. reduction"):
        X_train = feat_data['train']
        X_test = feat_data['test']
        is_sparse = feat_data.get('is_sparse', False)

        # Determine components list based on feature type
        if feat_name == 'TF-IDF':
            components_list = config.SVD_COMPONENTS
        elif feat_name == 'GloVe':
            components_list = config.PCA_COMPONENTS_GLOVE
        else:
            components_list = config.PCA_COMPONENTS_EMBEDDING

        # Filter components that exceed the valid range for the reducer.
        # TruncatedSVD needs fewer components than the sparse feature dimension;
        # PCA needs no more than min(n_samples, n_features).
        max_dim = X_train.shape[1]
        max_components = min(X_train.shape[0], max_dim)
        if is_sparse:
            max_components = max(1, max_dim - 1)
        components_list = [c for c in components_list if 1 <= c <= max_components]

        print(f"\n  Feature: {feat_name} (dim={max_dim})")
        if not components_list:
            print("    Skipping: no valid component counts for this feature matrix.")
            results[feat_name] = {
                'components': [],
                'explained_variances': [],
                'cumulative_variances': [],
                'accuracies': [],
            }
            continue

        explained_variances = []
        cumulative_vars = []
        accuracies = []

        for n_comp in components_list:
            # Apply reduction
            if is_sparse:
                X_tr_red, X_te_red, model = apply_truncated_svd(X_train, X_test, n_comp)
            else:
                X_tr_red, X_te_red, model = apply_pca(X_train, X_test, n_comp)

            # Record explained variance
            ev = model.explained_variance_ratio_
            explained_variances.append(ev)
            cum_var = np.sum(ev)
            cumulative_vars.append(cum_var)

            # Quick LR probe to measure performance at this dimensionality
            lr = LogisticRegression(
                C=config.LR_C, max_iter=config.LR_MAX_ITER,
                solver='lbfgs',
                random_state=config.RANDOM_SEED,
            )
            lr.fit(X_tr_red, y_train)
            acc = accuracy_score(y_test, lr.predict(X_te_red))
            accuracies.append(acc)

            print(f"    n_components={n_comp:>4d} | "
                  f"Explained Var: {cum_var:.4f} | "
                  f"LR Accuracy: {acc:.4f}")

        results[feat_name] = {
            'components': components_list,
            'explained_variances': explained_variances,
            'cumulative_variances': cumulative_vars,
            'accuracies': accuracies,
        }

    # Also compute full PCA/SVD for scree plot (on the best embedding)
    # We'll add the full eigenvalue spectrum for the largest feature set
    print()
    return results
