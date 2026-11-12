"""
Main pipeline orchestrator.

Runs the complete NLP text classification project:
1. Load and sample the AG News dataset
2. Exploratory data analysis
3. Text preprocessing
4. Sparse feature extraction (BoW and TF-IDF)
5. Frozen embedding extraction (DistilBERT and GloVe)
6. Classical ML model training
7. Dimensionality reduction analysis
8. Evaluation and visualisation
"""

import time
import warnings

from tqdm import tqdm

import config
import visualization as viz
from data_loader import load_and_sample_dataset, run_eda, split_dataset
from dimensionality_reduction import analyze_dimensionality_reduction
from evaluation import evaluate_all_results
from feature_extraction.contextual_embeddings import DistilBERTExtractor, GloVeExtractor
from feature_extraction.sparse_features import SparseFeatureExtractor
from models.classifiers import train_and_evaluate_all
from preprocessing import TextPreprocessor


warnings.filterwarnings("ignore")


def main():
    """Execute the full machine learning pipeline."""
    total_start = time.time()

    print("=" * 60)
    print("Statistical Machine Learning - Course Project")
    print("Text Classification: Sparse Features vs Contextual Embeddings")
    print("Dataset: AG News (8,000 samples, 4 classes)")
    print("=" * 60)
    print()

    # ============================================================
    # STAGE 1: Load and Sample Dataset
    # ============================================================
    df = load_and_sample_dataset()

    # ============================================================
    # STAGE 2: Exploratory Data Analysis
    # ============================================================
    run_eda(df)

    # ============================================================
    # STAGE 3: Train/Test Split
    # ============================================================
    print("=" * 60)
    print("STAGE 3: Train/Test Split")
    print("=" * 60)
    X_train_raw, X_test_raw, y_train, y_test = split_dataset(df)

    # ============================================================
    # STAGE 4: Text Preprocessing
    # ============================================================
    print("=" * 60)
    print("STAGE 4: Text Preprocessing")
    print("=" * 60)
    preprocessor = TextPreprocessor()
    X_train_clean = preprocessor.preprocess(X_train_raw)
    X_test_clean = preprocessor.preprocess(X_test_raw)
    print(f"  [OK] Preprocessed {len(X_train_clean)} train + {len(X_test_clean)} test texts")
    print()

    # ============================================================
    # STAGE 5: Sparse Feature Extraction (Part A)
    # ============================================================
    print("=" * 60)
    print("STAGE 5: Sparse Feature Extraction (Part A)")
    print("=" * 60)
    sparse_extractor = SparseFeatureExtractor()
    sparse_train = sparse_extractor.fit_transform(X_train_clean)
    sparse_test = sparse_extractor.transform(X_test_clean)
    print()

    # ============================================================
    # STAGE 6: Contextual Embedding Extraction (Part B)
    # ============================================================
    print("=" * 60)
    print("STAGE 6: Contextual Embedding Extraction (Part B)")
    print("=" * 60)

    print("\n  [DistilBERT]")
    distilbert = DistilBERTExtractor()
    distilbert_train = distilbert.extract_embeddings(X_train_raw, cache_prefix="distilbert_train")
    distilbert_test = distilbert.extract_embeddings(X_test_raw, cache_prefix="distilbert_test")

    print("\n  [GloVe]")
    glove = GloVeExtractor()
    glove_train = glove.extract_embeddings(X_train_clean, cache_prefix="glove_train")
    glove_test = glove.extract_embeddings(X_test_clean, cache_prefix="glove_test")
    print()

    # ============================================================
    # Assemble feature sets
    # ============================================================
    all_features = {
        "BoW": {"train": sparse_train["bow"], "test": sparse_test["bow"]},
        "TF-IDF": {"train": sparse_train["tfidf"], "test": sparse_test["tfidf"]},
        "GloVe": {"train": glove_train["glove"], "test": glove_test["glove"]},
        "DistilBERT-CLS": {
            "train": distilbert_train["distilbert_cls"],
            "test": distilbert_test["distilbert_cls"],
        },
        "DistilBERT-Mean": {
            "train": distilbert_train["distilbert_mean"],
            "test": distilbert_test["distilbert_mean"],
        },
    }

    print(f"  Feature representations prepared: {list(all_features.keys())}")
    for name, data in all_features.items():
        print(f"    {name}: train={data['train'].shape}, test={data['test'].shape}")
    print()

    # ============================================================
    # STAGE 7: Train Models on All Feature Types (Part C)
    # ============================================================
    print("=" * 60)
    print("STAGE 7: Model Training (Part C)")
    print("=" * 60)

    all_results = {}
    for feat_name, feat_data in tqdm(all_features.items(), desc="Feature types"):
        print(f"\n  === Feature: {feat_name} ===")
        results = train_and_evaluate_all(
            feat_data["train"],
            feat_data["test"],
            y_train,
            y_test,
            feature_name=feat_name,
        )
        all_results[feat_name] = results
    print()

    # ============================================================
    # STAGE 8: Dimensionality Reduction (Part D)
    # ============================================================
    print("=" * 60)
    print("STAGE 8: Dimensionality Reduction (Part D)")
    print("=" * 60)

    dr_features = {
        "TF-IDF": {"train": sparse_train["tfidf"], "test": sparse_test["tfidf"], "is_sparse": True},
        "GloVe": {"train": glove_train["glove"], "test": glove_test["glove"], "is_sparse": False},
        "DistilBERT-CLS": {
            "train": distilbert_train["distilbert_cls"],
            "test": distilbert_test["distilbert_cls"],
            "is_sparse": False,
        },
        "DistilBERT-Mean": {
            "train": distilbert_train["distilbert_mean"],
            "test": distilbert_test["distilbert_mean"],
            "is_sparse": False,
        },
    }

    dr_results = analyze_dimensionality_reduction(dr_features, y_train, y_test)
    print()

    # ============================================================
    # STAGE 9: Evaluation
    # ============================================================
    metrics_df, detailed_metrics = evaluate_all_results(all_results, y_test)

    # ============================================================
    # STAGE 10: Visualisations
    # ============================================================
    print("\n" + "=" * 60)
    print("STAGE 10: Generating Visualisations")
    print("=" * 60)

    tsne_features = {
        "BoW": sparse_test["bow"],
        "TF-IDF": sparse_test["tfidf"],
        "GloVe": glove_test["glove"],
        "DistilBERT-CLS": distilbert_test["distilbert_cls"],
        "DistilBERT-Mean": distilbert_test["distilbert_mean"],
    }
    tsne_results = viz.generate_all_tsne_plots(tsne_features, y_test)

    print("\n  Generating dimensionality reduction plots...")
    viz.plot_explained_variance(dr_results)
    viz.plot_scree(dr_results)
    viz.plot_performance_vs_components(dr_results)

    print("\n  Generating model comparison plots...")
    viz.plot_model_comparison_bars(metrics_df)
    viz.plot_heatmap_results(metrics_df)
    viz.plot_training_time(metrics_df)
    viz.plot_per_class_f1(detailed_metrics)

    print("\n  Generating confusion matrices...")
    viz.plot_confusion_matrices(detailed_metrics, y_test)

    print("\n  Generating radar charts...")
    for feat_name in ["TF-IDF", "DistilBERT-CLS"]:
        if feat_name in detailed_metrics:
            viz.plot_radar_chart(detailed_metrics, feat_name)

    print("\n  Generating KMeans cluster plots...")
    for feat_name in ["TF-IDF", "DistilBERT-CLS"]:
        if feat_name in all_results and feat_name in tsne_results:
            kmeans_result = all_results[feat_name].get("KMeans", {})
            if "cluster_labels" in kmeans_result:
                viz.plot_kmeans_clusters(
                    tsne_results[feat_name],
                    kmeans_result["cluster_labels"],
                    y_test,
                    feat_name,
                )

    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Total execution time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"  Plots saved to:      {config.PLOTS_DIR}")
    print(f"  Results saved to:    {config.RESULTS_DIR}")
    print(f"  Cached embeddings:   {config.CACHE_DIR}")

    print("\n  Top-3 Configurations by Accuracy:")
    top3 = metrics_df.nlargest(3, "Accuracy")[["Feature", "Model", "Accuracy", "Macro F1"]]
    for _, row in top3.iterrows():
        print(
            f"    {row['Feature']} + {row['Model']}: "
            f"Acc={row['Accuracy']:.4f}, F1={row['Macro F1']:.4f}"
        )

    print()
    return metrics_df, detailed_metrics


if __name__ == "__main__":
    main()
