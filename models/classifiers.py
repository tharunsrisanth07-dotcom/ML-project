"""
Classical ML Classifiers Module
================================
Implements all required models:
- Logistic Regression
- Linear SVM
- KNN
- KMeans (unsupervised baseline)
"""

import time
import numpy as np
from scipy.sparse import issparse
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import config


def _build_models():
    """Create dictionary of all supervised model instances."""
    return {
        'Logistic Regression': LogisticRegression(
            C=config.LR_C,
            max_iter=config.LR_MAX_ITER,
            solver='lbfgs',
            random_state=config.RANDOM_SEED,
        ),
        'Linear SVM': LinearSVC(
            C=config.SVM_C,
            max_iter=config.SVM_MAX_ITER,
            random_state=config.RANDOM_SEED,
            dual='auto',
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=config.KNN_NEIGHBORS,
            metric='cosine',
            algorithm='brute',
            n_jobs=-1,
        ),
    }


def _optimal_cluster_mapping(y_true, y_pred, n_classes):
    """
    Find optimal mapping from cluster IDs to true class labels
    using the Hungarian algorithm.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Cluster assignments.
    n_classes : int
        Number of classes/clusters.

    Returns
    -------
    dict
        Mapping from cluster ID to class label.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return mapping


def train_and_evaluate_all(X_train, X_test, y_train, y_test, feature_name=""):
    """
    Train all models on given features and return predictions + metadata.

    Parameters
    ----------
    X_train, X_test : array-like or sparse matrix
        Feature matrices for train and test sets.
    y_train, y_test : array-like
        Labels for train and test sets.
    feature_name : str
        Name of the feature representation (for display).

    Returns
    -------
    dict
        {model_name: {'y_pred': array, 'train_time': float, 'model': object}}
    """
    results = {}

    # ---- Supervised Models ----
    models = _build_models()

    for name, model in tqdm(models.items(), desc=f"  Training models [{feature_name}]"):
        print(f"    Training {name}...")

        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        try:
            y_pred = model.predict(X_test)
        except ValueError as exc:
            if name == 'KNN' and issparse(X_train):
                print(f"      Sparse KNN prediction failed ({exc}); retrying with dense arrays.")
                y_pred = model.fit(X_train.toarray(), y_train).predict(X_test.toarray())
            else:
                raise

        results[name] = {
            'y_pred': y_pred,
            'train_time': train_time,
            'model': model,
        }
        print(f"      Done in {train_time:.2f}s")

    # ---- KMeans (Unsupervised) ----
    print(f"    Training KMeans...")
    start = time.time()
    kmeans = KMeans(
        n_clusters=config.KMEANS_N_CLUSTERS,
        n_init=config.KMEANS_N_INIT,
        max_iter=config.KMEANS_MAX_ITER,
        random_state=config.RANDOM_SEED,
    )
    kmeans.fit(X_train)
    train_time = time.time() - start

    # Predict cluster assignments on test set
    cluster_preds = kmeans.predict(X_test)

    # Map clusters to class labels using training data
    train_clusters = kmeans.predict(X_train)
    mapping = _optimal_cluster_mapping(y_train, train_clusters, config.NUM_CLASSES)
    y_pred_kmeans = np.array([mapping.get(c, 0) for c in cluster_preds])

    results['KMeans'] = {
        'y_pred': y_pred_kmeans,
        'train_time': train_time,
        'model': kmeans,
        'cluster_labels': cluster_preds,
        'cluster_mapping': mapping,
    }
    print(f"      KMeans done in {train_time:.2f}s | Cluster mapping: {mapping}")

    return results
