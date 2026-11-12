"""
Visualization Module
=====================
Publication-quality plots for all stages of the project:
- EDA plots (handled in data_loader.py)
- t-SNE embeddings (sparse + contextual)
- Dimensionality reduction analysis
- Model comparison charts
- Confusion matrices
- KMeans cluster analysis
"""

import os
import inspect
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import issparse
from tqdm import tqdm

import config


# ================================================================
# Apply Global Style
# ================================================================
plt.rcParams.update(config.PLOT_STYLE)


def _is_notebook():
    """Check if running inside an interactive notebook (Jupyter / Colab)."""
    try:
        import IPython
        shell = IPython.get_ipython()
        return shell is not None and 'TerminalInteractiveShell' not in str(type(shell))
    except Exception:
        return False


def _save_plot(fig, filename):
    """Save figure to disk and display inline (for Colab/Jupyter)."""
    path = os.path.join(config.PLOTS_DIR, filename)
    fig.savefig(path, dpi=config.PLOT_DPI, bbox_inches='tight', facecolor='white')
    if _is_notebook():
        plt.show()
    plt.close(fig)
    print(f"    [OK] Saved: {filename}")


# ================================================================
# t-SNE Visualizations
# ================================================================
def _compute_tsne(X, perplexity=30, n_iter=1000):
    """Compute 2D t-SNE embedding."""
    n_samples = X.shape[0]
    if n_samples <= 3:
        raise ValueError("t-SNE needs more than 3 samples.")
    perplexity = min(perplexity, max(2, (n_samples - 1) // 3))

    # For sparse matrices, reduce dimensionality first with SVD
    if issparse(X):
        n_comp = min(50, X.shape[0] - 1, X.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=config.RANDOM_SEED)
        X = svd.fit_transform(X)

    # If dense and high-dimensional, truncate for speed
    if X.shape[1] > 100:
        from sklearn.decomposition import PCA
        n_comp = min(50, X.shape[0] - 1, X.shape[1])
        pca = PCA(n_components=n_comp, random_state=config.RANDOM_SEED)
        X = pca.fit_transform(X)

    tsne_kwargs = {
        'n_components': 2,
        'perplexity': perplexity,
        'random_state': config.RANDOM_SEED,
        'init': 'pca',
        'learning_rate': 'auto',
    }
    if 'max_iter' in inspect.signature(TSNE).parameters:
        tsne_kwargs['max_iter'] = n_iter
    else:
        tsne_kwargs['n_iter'] = n_iter

    tsne = TSNE(**tsne_kwargs)
    return tsne.fit_transform(X)


def plot_tsne_single(X, y, title, filename, perplexity=30):
    """
    Create a single t-SNE scatter plot colored by class.

    Parameters
    ----------
    X : array-like or sparse matrix
        Feature matrix.
    y : array-like
        Class labels.
    title : str
        Plot title.
    filename : str
        Output filename.
    """
    print(f"    Computing t-SNE for {title}...")
    X_2d = _compute_tsne(X, perplexity=perplexity)

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)

    for i, cls_name in enumerate(config.CLASS_NAMES):
        mask = y == i
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=config.CLASS_COLORS[i],
            label=cls_name,
            alpha=0.6,
            s=15,
            edgecolors='white',
            linewidth=0.3,
        )

    ax.set_title(f't-SNE Visualization - {title}', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=13)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=13)
    ax.legend(fontsize=12, loc='best', framealpha=0.9, markerscale=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    _save_plot(fig, filename)
    return X_2d


def plot_tsne_comparison(tsne_results, y):
    """
    Side-by-side t-SNE comparison of all feature types.

    Parameters
    ----------
    tsne_results : dict
        {feature_name: X_2d_array}
    y : array-like
        Class labels.
    """
    n_features = len(tsne_results)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (feat_name, X_2d) in enumerate(tsne_results.items()):
        ax = axes[idx]
        for i, cls_name in enumerate(config.CLASS_NAMES):
            mask = y == i
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=config.CLASS_COLORS[i],
                label=cls_name,
                alpha=0.55,
                s=12,
                edgecolors='white',
                linewidth=0.2,
            )
        ax.set_title(feat_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE 1', fontsize=11)
        ax.set_ylabel('t-SNE 2', fontsize=11)
        ax.legend(fontsize=9, loc='best', framealpha=0.8, markerscale=1.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused axes
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('t-SNE Comparison: Sparse vs Contextual Features',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_plot(fig, '05_tsne_comparison.png')


def generate_all_tsne_plots(features_dict, y):
    """
    Generate individual and comparison t-SNE plots for all feature types.

    Parameters
    ----------
    features_dict : dict
        {feature_name: X_data (array or sparse)}
    y : array-like
        Labels.

    Returns
    -------
    dict
        {feature_name: X_2d} t-SNE coordinates for reuse.
    """
    print("\n  Generating t-SNE visualizations...")
    tsne_results = {}
    plot_idx = 6

    for feat_name, X in tqdm(features_dict.items(), desc="  t-SNE plots"):
        filename = f'{plot_idx:02d}_tsne_{feat_name.lower().replace(" ", "_").replace("-", "_")}.png'
        X_2d = plot_tsne_single(X, y, feat_name, filename)
        tsne_results[feat_name] = X_2d
        plot_idx += 1

    # Comparison plot
    plot_tsne_comparison(tsne_results, y)

    return tsne_results


# ================================================================
# Dimensionality Reduction Plots
# ================================================================
def plot_explained_variance(dr_results):
    """
    Plot cumulative explained variance curves for all feature types.

    Parameters
    ----------
    dr_results : dict
        Output from dimensionality_reduction.analyze_dimensionality_reduction().
    """
    fig, axes = plt.subplots(1, 2, figsize=config.FIGURE_SIZE_WIDE)

    # Separate sparse (SVD) and dense (PCA) results
    sparse_feats = {k: v for k, v in dr_results.items() if k == 'TF-IDF'}
    dense_feats = {k: v for k, v in dr_results.items() if k != 'TF-IDF'}

    # Plot sparse (SVD)
    ax = axes[0]
    for feat_name, data in sparse_feats.items():
        ax.plot(data['components'], data['cumulative_variances'],
                'o-', linewidth=2.5, markersize=8, label=feat_name,
                color=config.FEATURE_COLORS.get(feat_name, '#333'))
        # Annotate points
        for x, y_val in zip(data['components'], data['cumulative_variances']):
            ax.annotate(f'{y_val:.3f}', (x, y_val), textcoords="offset points",
                        xytext=(0, 12), ha='center', fontsize=9, fontweight='bold')

    ax.set_title('Truncated SVD - Cumulative Explained Variance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance Ratio')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='95% threshold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Plot dense (PCA)
    ax = axes[1]
    colors = ['#FFA07A', '#45B7D1', '#98D8C8']
    for idx, (feat_name, data) in enumerate(dense_feats.items()):
        color = config.FEATURE_COLORS.get(feat_name, colors[idx % len(colors)])
        ax.plot(data['components'], data['cumulative_variances'],
                'o-', linewidth=2.5, markersize=8, label=feat_name, color=color)
        for x, y_val in zip(data['components'], data['cumulative_variances']):
            ax.annotate(f'{y_val:.3f}', (x, y_val), textcoords="offset points",
                        xytext=(0, 12), ha='center', fontsize=9, fontweight='bold')

    ax.set_title('PCA - Cumulative Explained Variance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance Ratio')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    _save_plot(fig, '11_explained_variance.png')


def plot_scree(dr_results):
    """
    Scree plot showing individual explained variance ratios.

    Parameters
    ----------
    dr_results : dict
        Output from dimensionality_reduction analysis.
    """
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_SMALL)

    for feat_name, data in dr_results.items():
        # Use the highest-component reduction for the scree plot
        if data['explained_variances']:
            ev = data['explained_variances'][-1]  # Largest component count
            ax.plot(range(1, len(ev) + 1), ev, linewidth=1.5, alpha=0.8,
                    label=f'{feat_name} ({len(ev)} comp)',
                    color=config.FEATURE_COLORS.get(feat_name, '#333'))

    ax.set_title('Scree Plot - Individual Explained Variance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Component Index')
    ax.set_ylabel('Explained Variance Ratio')
    ax.legend(fontsize=10)
    ax.set_xlim(0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    _save_plot(fig, '12_scree_plot.png')


def plot_performance_vs_components(dr_results):
    """
    Plot accuracy vs number of components for each feature type.

    Parameters
    ----------
    dr_results : dict
        Output from dimensionality_reduction analysis.
    """
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)

    for feat_name, data in dr_results.items():
        color = config.FEATURE_COLORS.get(feat_name, '#333')
        ax.plot(data['components'], data['accuracies'],
                'o-', linewidth=2.5, markersize=9, label=feat_name, color=color)
        # Annotate
        for x, y_val in zip(data['components'], data['accuracies']):
            ax.annotate(f'{y_val:.3f}', (x, y_val), textcoords="offset points",
                        xytext=(0, 12), ha='center', fontsize=9, fontweight='bold')

    ax.set_title('LR Accuracy vs. Number of Components (Dim. Reduction)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Test Accuracy')
    ax.legend(fontsize=11, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    _save_plot(fig, '13_performance_vs_components.png')


# ================================================================
# Model Comparison Plots
# ================================================================
def plot_model_comparison_bars(metrics_df):
    """
    Grouped bar charts for Accuracy and Macro F1 across models and features.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Comparison DataFrame with columns: Feature, Model, Accuracy, Macro F1.
    """
    for metric, plot_num in [('Accuracy', 14), ('Macro F1', 15)]:
        fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_WIDE)

        pivot = metrics_df.pivot(index='Model', columns='Feature', values=metric)
        # Reorder models
        model_order = [m for m in config.MODEL_NAMES if m in pivot.index]
        pivot = pivot.reindex(model_order)

        x = np.arange(len(model_order))
        n_features = len(pivot.columns)
        width = 0.8 / n_features

        for i, feat in enumerate(pivot.columns):
            color = config.FEATURE_COLORS.get(feat, f'C{i}')
            bars = ax.bar(x + i * width - (n_features - 1) * width / 2,
                          pivot[feat].values, width, label=feat,
                          color=color, edgecolor='white', linewidth=0.8, alpha=0.9)
            # Annotate values
            for bar in bars:
                if not np.isnan(bar.get_height()):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f'{bar.get_height():.3f}', ha='center', va='bottom',
                            fontsize=7, fontweight='bold', rotation=45)

        ax.set_title(f'{metric} Comparison Across Models and Features',
                     fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Model', fontsize=13)
        ax.set_ylabel(metric, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(model_order, fontsize=11)
        ax.legend(fontsize=10, loc='lower right', framealpha=0.9)
        ax.set_ylim(0, min(1.0, pivot.max().max() * 1.15))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        _save_plot(fig, f'{plot_num}_comparison_{metric.lower().replace(" ", "_")}.png')


def plot_heatmap_results(metrics_df):
    """
    Heatmap of Model x Feature to Accuracy.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Comparison DataFrame.
    """
    fig, axes = plt.subplots(1, 2, figsize=config.FIGURE_SIZE_WIDE)

    for idx, metric in enumerate(['Accuracy', 'Macro F1']):
        ax = axes[idx]
        pivot = metrics_df.pivot(index='Model', columns='Feature', values=metric)
        model_order = [m for m in config.MODEL_NAMES if m in pivot.index]
        pivot = pivot.reindex(model_order)

        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd',
                    linewidths=1, linecolor='white', ax=ax,
                    cbar_kws={'shrink': 0.8},
                    annot_kws={'size': 11, 'fontweight': 'bold'})
        ax.set_title(f'{metric} Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    _save_plot(fig, '16_heatmap_results.png')


def plot_confusion_matrices(detailed_metrics, y_test, feature_names=None):
    """
    Grid of confusion matrices for each feature type's best model.

    Parameters
    ----------
    detailed_metrics : dict
        {feature_name: {model_name: metrics_dict}}
    y_test : array-like
        True labels.
    feature_names : list, optional
        Feature names to include.
    """
    if feature_names is None:
        feature_names = list(detailed_metrics.keys())

    n_features = len(feature_names)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_features == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, feat_name in enumerate(feature_names):
        model_metrics = detailed_metrics[feat_name]

        # Find best model by accuracy
        best_model = max(model_metrics.keys(),
                         key=lambda m: model_metrics[m]['accuracy'])
        cm = model_metrics[best_model]['confusion_matrix']

        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        ax = axes[idx]
        sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                    xticklabels=config.CLASS_NAMES,
                    yticklabels=config.CLASS_NAMES,
                    linewidths=0.5, linecolor='white', ax=ax,
                    annot_kws={'size': 10})
        ax.set_title(f'{feat_name}\n(Best: {best_model})',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Normalized Confusion Matrices (Best Model per Feature)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_plot(fig, '17_confusion_matrices.png')


def plot_training_time(metrics_df):
    """
    Bar chart comparing training time across models and features.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Must contain 'Time (s)' column.
    """
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)

    pivot = metrics_df.pivot(index='Model', columns='Feature', values='Time (s)')
    model_order = [m for m in config.MODEL_NAMES if m in pivot.index]
    pivot = pivot.reindex(model_order)

    pivot.plot(kind='bar', ax=ax, edgecolor='white', linewidth=0.8, alpha=0.9,
               color=[config.FEATURE_COLORS.get(c, f'C{i}')
                      for i, c in enumerate(pivot.columns)])

    ax.set_title('Training Time Comparison', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Model', fontsize=13)
    ax.set_ylabel('Time (seconds)', fontsize=13)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax.tick_params(axis='x', rotation=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    _save_plot(fig, '18_training_time.png')


def plot_per_class_f1(detailed_metrics):
    """
    Box plot of per-class F1 scores across models for each feature type.

    Parameters
    ----------
    detailed_metrics : dict
        {feature_name: {model_name: {'per_class_f1': array}}}
    """
    fig, axes = plt.subplots(1, len(detailed_metrics),
                             figsize=(6 * len(detailed_metrics), 6),
                             sharey=True)
    if len(detailed_metrics) == 1:
        axes = [axes]

    for idx, (feat_name, model_metrics) in enumerate(detailed_metrics.items()):
        ax = axes[idx]
        data = []
        labels = []
        for model_name in config.MODEL_NAMES:
            if model_name in model_metrics:
                f1_per_class = model_metrics[model_name]['per_class_f1']
                data.append(f1_per_class)
                labels.append(model_name)

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
        for patch, model_name in zip(bp['boxes'], labels):
            patch.set_facecolor(config.MODEL_COLORS.get(model_name, '#ccc'))
            patch.set_alpha(0.75)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)

        ax.set_title(feat_name, fontsize=13, fontweight='bold')
        ax.set_ylabel('F1 Score' if idx == 0 else '')
        ax.tick_params(axis='x', rotation=30)
        ax.set_ylim(0, 1.05)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Per-Class F1 Score Distribution by Model',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_plot(fig, '19_per_class_f1_boxplot.png')


def plot_radar_chart(detailed_metrics, feature_name, models=None):
    """
    Radar/spider chart for multi-metric comparison of models on one feature type.

    Parameters
    ----------
    detailed_metrics : dict
        {feature_name: {model_name: metrics_dict}}
    feature_name : str
        Feature to show.
    models : list, optional
        Models to include.
    """
    if models is None:
        models = [m for m in config.MODEL_NAMES if m in detailed_metrics.get(feature_name, {})]

    metric_keys = ['accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'cohen_kappa']
    metric_labels = ['Accuracy', 'Macro F1', 'Precision', 'Recall', 'Cohen Kappa']

    n_metrics = len(metric_keys)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model_name in models:
        if model_name not in detailed_metrics.get(feature_name, {}):
            continue
        metrics = detailed_metrics[feature_name][model_name]
        values = [metrics.get(k, 0) for k in metric_keys]
        values += values[:1]

        color = config.MODEL_COLORS.get(model_name, '#333')
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(f'Multi-Metric Radar - {feature_name}',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    _save_plot(fig, f'20_radar_{feature_name.lower().replace(" ", "_").replace("-", "_")}.png')


def plot_kmeans_clusters(tsne_coords, cluster_labels, y_true, feature_name):
    """
    t-SNE colored by KMeans cluster (shapes for true labels).

    Parameters
    ----------
    tsne_coords : ndarray
        2D t-SNE coordinates.
    cluster_labels : array-like
        KMeans cluster assignments.
    y_true : array-like
        True class labels.
    feature_name : str
        Feature name for title.
    """
    fig, axes = plt.subplots(1, 2, figsize=config.FIGURE_SIZE_WIDE)

    # Left: colored by cluster
    cluster_colors = ['#FF6B6B', '#4ECDC4', '#FFA07A', '#98D8C8']
    for c in range(config.NUM_CLASSES):
        mask = cluster_labels == c
        axes[0].scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                        c=cluster_colors[c], label=f'Cluster {c}',
                        alpha=0.6, s=15, edgecolors='white', linewidth=0.3)
    axes[0].set_title(f'KMeans Clusters - {feature_name}', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Right: colored by true labels
    for i, cls in enumerate(config.CLASS_NAMES):
        mask = y_true == i
        axes[1].scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                        c=config.CLASS_COLORS[i], label=cls,
                        alpha=0.6, s=15, edgecolors='white', linewidth=0.3)
    axes[1].set_title(f'True Labels - {feature_name}', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.suptitle('KMeans Clusters vs True Labels', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_plot(fig, f'21_kmeans_{feature_name.lower().replace(" ", "_").replace("-", "_")}.png')
