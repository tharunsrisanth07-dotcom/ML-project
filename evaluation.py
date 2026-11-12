"""
Evaluation Module
==================
Computes all required metrics and builds comparison tables.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, cohen_kappa_score,
    adjusted_rand_score, normalized_mutual_info_score,
)
from tabulate import tabulate

import config


def compute_metrics(y_true, y_pred):
    """
    Compute all classification metrics for a single model.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    dict
        Dictionary of computed metrics.
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        'per_class_f1': f1_score(y_true, y_pred, average=None, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
    }


def compute_clustering_metrics(y_true, cluster_labels):
    """
    Compute clustering-specific metrics for KMeans.

    Parameters
    ----------
    y_true : array-like
        True labels.
    cluster_labels : array-like
        Raw cluster assignments (before mapping).

    Returns
    -------
    dict
        ARI, NMI, and purity.
    """
    ari = adjusted_rand_score(y_true, cluster_labels)
    nmi = normalized_mutual_info_score(y_true, cluster_labels)

    # Purity
    cm = confusion_matrix(y_true, cluster_labels)
    purity = np.sum(np.amax(cm, axis=0)) / np.sum(cm)

    return {'ARI': ari, 'NMI': nmi, 'Purity': purity}


def evaluate_all_results(all_results, y_test):
    """
    Evaluate all model results across all feature types.

    Parameters
    ----------
    all_results : dict
        {feature_name: {model_name: {'y_pred': array, 'train_time': float, ...}}}
    y_test : array-like
        True labels.

    Returns
    -------
    tuple
        (metrics_df: pd.DataFrame, detailed_metrics: dict)
    """
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    rows = []
    detailed = {}

    for feat_name, model_results in all_results.items():
        detailed[feat_name] = {}

        for model_name, result in model_results.items():
            y_pred = result['y_pred']
            metrics = compute_metrics(y_test, y_pred)
            metrics['train_time'] = result['train_time']

            # Add clustering metrics for KMeans
            if model_name == 'KMeans' and 'cluster_labels' in result:
                clust_metrics = compute_clustering_metrics(y_test, result['cluster_labels'])
                metrics.update(clust_metrics)

            detailed[feat_name][model_name] = metrics

            rows.append({
                'Feature': feat_name,
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Macro F1': metrics['macro_f1'],
                'Precision': metrics['macro_precision'],
                'Recall': metrics['macro_recall'],
                'Kappa': metrics['cohen_kappa'],
                'Time (s)': metrics['train_time'],
            })

    df = pd.DataFrame(rows)

    # Print summary table
    print("\n" + "=" * 90)
    print("MASTER COMPARISON TABLE")
    print("=" * 90)
    print(tabulate(df, headers='keys', tablefmt='grid',
                   floatfmt=('.0f', '', '', '.4f', '.4f', '.4f', '.4f', '.4f', '.2f'),
                   showindex=False))
    print()

    # Print per-feature results
    for feat_name in all_results:
        print(f"\n--- {feat_name} ---")
        feat_df = df[df['Feature'] == feat_name]
        print(tabulate(feat_df, headers='keys', tablefmt='grid',
                       floatfmt=('.0f', '', '', '.4f', '.4f', '.4f', '.4f', '.4f', '.2f'),
                       showindex=False))

    # Print KMeans clustering metrics
    print("\n--- KMeans Clustering Metrics ---")
    km_rows = []
    for feat_name, model_metrics in detailed.items():
        if 'KMeans' in model_metrics and 'ARI' in model_metrics['KMeans']:
            km = model_metrics['KMeans']
            km_rows.append({
                'Feature': feat_name,
                'ARI': km['ARI'],
                'NMI': km['NMI'],
                'Purity': km['Purity'],
            })
    if km_rows:
        print(tabulate(pd.DataFrame(km_rows), headers='keys', tablefmt='grid',
                       floatfmt=('.0f', '', '.4f', '.4f', '.4f'), showindex=False))

    # Save results to CSV
    df.to_csv(f'{config.RESULTS_DIR}/comparison_results.csv', index=False)
    if km_rows:
        pd.DataFrame(km_rows).to_csv(
            f'{config.RESULTS_DIR}/kmeans_clustering_metrics.csv', index=False)

    print(f"\n  [OK] Results saved to {config.RESULTS_DIR}/")

    # Display DataFrames inline for Colab/Jupyter
    try:
        from IPython.display import display, HTML
        print("\nMaster Comparison Table:")
        styled = df.style.background_gradient(
            subset=['Accuracy', 'Macro F1'], cmap='YlGn'
        ).format({
            'Accuracy': '{:.4f}', 'Macro F1': '{:.4f}',
            'Precision': '{:.4f}', 'Recall': '{:.4f}',
            'Kappa': '{:.4f}', 'Time (s)': '{:.2f}',
        }).set_caption("Model x Feature Comparison")
        display(styled)

        if km_rows:
            print("\nKMeans Clustering Metrics:")
            km_styled = pd.DataFrame(km_rows).style.background_gradient(
                cmap='YlGn'
            ).format({
                'ARI': '{:.4f}', 'NMI': '{:.4f}', 'Purity': '{:.4f}',
            }).set_caption("KMeans Clustering Quality")
            display(km_styled)
    except ImportError:
        pass  # Not in a notebook environment, tables already printed above

    return df, detailed
