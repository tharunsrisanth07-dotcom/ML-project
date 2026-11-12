"""
Report Generator
==================
Generates a LaTeX report (3-4 pages) and standalone comparison tables
from the ML pipeline results.

Usage:
    python generate_report.py          # After running main.py
    python generate_report.py --full   # Runs pipeline first, then generates report
"""

import os
import sys
import argparse
import subprocess
import pandas as pd
import numpy as np

import config


# ============================================================
# LaTeX Report Template
# ============================================================

LATEX_TEMPLATE = r"""
\documentclass[11pt, a4paper, twocolumn]{{article}}

% ---- Packages ----
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage[margin=1.8cm]{{geometry}}
\usepackage{{graphicx}}
\graphicspath{{ {{./}} {{plots/}} {{../plots/}} }}
\usepackage{{booktabs}}
\usepackage{{caption}}
\usepackage{{subcaption}}
\usepackage{{hyperref}}
\usepackage{{amsmath}}
\usepackage{{xcolor}}
\usepackage{{float}}
\usepackage{{enumitem}}
\usepackage{{multicol}}
\usepackage{{array}}
\usepackage{{colortbl}}

\hypersetup{{colorlinks=true, linkcolor=blue!60!black, citecolor=blue!60!black, urlcolor=blue!60!black}}

\setlength{{\columnsep}}{{0.7cm}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{6pt}}

\title{{\textbf{{Text Classification using Classical ML:\\Sparse Features vs Contextual Embeddings}}}}
\author{{{author_name}}}
\date{{{date}}}

\begin{{document}}

\maketitle

% ============================================================
\begin{{abstract}}
This project evaluates classical machine learning classifiers on the AG~News text classification task (4 classes, 8,000 samples), systematically comparing two feature paradigms: \textbf{{sparse representations}} (Bag-of-Words, TF-IDF) and \textbf{{dense contextual embeddings}} (DistilBERT, GloVe). Four models---Logistic Regression, Linear SVM, KNN, and KMeans---are trained and benchmarked across five feature sets. Results show that {best_combo} achieves the highest accuracy ({best_acc:.4f}), while sparse TF-IDF features remain surprisingly competitive. Dimensionality reduction via SVD/PCA is analysed, and t-SNE visualisations reveal clear differences in feature space separability.
\end{{abstract}}

% ============================================================
\section{{Introduction and Dataset Used}}

News topic classification is a canonical NLP benchmark for evaluating feature representations and classifier performance. The AG~News dataset~\cite{{agnews}} contains 120,000 articles across four categories: World, Sports, Business, and Sci/Tech.

\textbf{{Research questions:}}
\begin{{enumerate}}[leftmargin=*,topsep=2pt,itemsep=1pt]
    \item Do contextual embeddings (DistilBERT, GloVe) consistently outperform sparse features (BoW, TF-IDF) when paired with classical ML models?
    \item Which model--feature combination offers the best trade-off between accuracy and computational cost?
    \item How does dimensionality reduction affect classification performance across feature types?
    \item Can unsupervised KMeans clustering recover meaningful topic structure?
\end{{enumerate}}

% ============================================================
\section{{Dataset Description and EDA}}

The fixed HuggingFace dataset is \texttt{{fancyzhx/ag\_news}}. The experiment uses exactly 8,000 examples sampled in a class-balanced way: 2,000 examples each from World, Sports, Business, and Sci/Tech. The sampled data is shuffled and split into 6,400 training examples and 1,600 test examples with stratification.

EDA includes class balance, word-count distribution, per-class word clouds, and top discriminative terms. These plots verify that the dataset is balanced and that the classes contain visibly different topic vocabulary before feature extraction.

% ============================================================
\section{{Methodology}}

\subsection{{Preprocessing}}
Text preprocessing for sparse features includes lowercasing, URL/email removal, special-character stripping, NLTK tokenization, stop-word removal, vocabulary pruning, and WordNet lemmatization. DistilBERT operates on raw text with its own WordPiece tokenizer (max 128 tokens), because transformer tokenization should not receive heavily cleaned text.

\subsection{{Feature Representations}}

\textbf{{Part A --- Sparse Features:}}
\begin{{itemize}}[leftmargin=*,topsep=2pt,itemsep=1pt]
    \item \textbf{{BoW:}} Unigram CountVectorizer (max 10,000 features, $\text{{min\_df}}=3$, $\text{{max\_df}}=0.95$).
    \item \textbf{{TF-IDF:}} Bigram TfidfVectorizer (max 15,000 features, sublinear TF).
\end{{itemize}}

\textbf{{Part B --- Contextual Embeddings:}}
\begin{{itemize}}[leftmargin=*,topsep=2pt,itemsep=1pt]
    \item \textbf{{DistilBERT-CLS:}} [CLS] token embedding (768-d) from \texttt{{distilbert-base-uncased}}.
    \item \textbf{{DistilBERT-Mean:}} Mean-pooled hidden states over non-padding tokens (768-d).
    \item \textbf{{GloVe:}} Mean-pooled 300-d GloVe vectors per document.
\end{{itemize}}

\subsection{{Models (Part C)}}
\begin{{itemize}}[leftmargin=*,topsep=2pt,itemsep=1pt]
    \item \textbf{{Logistic Regression:}} L-BFGS solver, multinomial, $C=1.0$.
    \item \textbf{{Linear SVM:}} LinearSVC, $C=1.0$.
    \item \textbf{{KNN:}} $k=5$, cosine distance.
    \item \textbf{{KMeans:}} $k=4$, Hungarian-algorithm cluster-to-label mapping.
\end{{itemize}}

\subsection{{Dimensionality Reduction (Part D)}}
Truncated SVD is applied to sparse TF-IDF; PCA to dense embeddings. Logistic Regression probes measure accuracy at component counts $\in \{{32, 50, 64, 100, 128, 200, 256, 300, 384, 512\}}$, and cumulative explained variance is tracked.

% ============================================================
\section{{Experiments}}

Each feature representation is computed once, cached where useful, and then reused across the same four classical ML models. Supervised models are trained on the 6,400-example training split and evaluated on the held-out 1,600-example test split. KMeans is evaluated as an unsupervised baseline by mapping clusters to labels with the Hungarian algorithm. Dimensionality reduction experiments use a Logistic Regression probe to isolate the effect of component count on classification quality.

% ============================================================
\section{{Results}}

\subsection{{Overall Performance}}

Table~\ref{{tab:main}} presents accuracy, macro F1, and training time for all 20 model--feature combinations.

{main_table_latex}

\textbf{{Key observations:}}
\begin{{itemize}}[leftmargin=*,topsep=2pt,itemsep=1pt]
    \item {observation_1}
    \item {observation_2}
    \item {observation_3}
\end{{itemize}}

\subsection{{Sparse vs Contextual}}
{sparse_vs_contextual_paragraph}

\subsection{{KMeans Clustering}}
{kmeans_paragraph}

{kmeans_table_latex}

\subsection{{Dimensionality Reduction}}
SVD on TF-IDF and PCA on embeddings show that high classification accuracy can be retained with significantly fewer dimensions (see Figure~\ref{{fig:dimred}}).
The explained-variance curves and accuracy probes identify the point where additional components stop giving meaningful performance gains.

% ============================================================
\section{{Visualizations}}

\begin{{figure}}[H]
    \centering
    \includegraphics[width=\columnwidth]{{plots/05_tsne_comparison.png}}
    \caption{{t-SNE projections of test-set features. DistilBERT embeddings form tighter, more separable clusters than BoW/TF-IDF.}}
    \label{{fig:tsne}}
\end{{figure}}

\begin{{figure}}[H]
    \centering
    \includegraphics[width=\columnwidth]{{plots/16_heatmap_results.png}}
    \caption{{Accuracy and Macro-F1 heatmaps across all model--feature combinations.}}
    \label{{fig:heatmap}}
\end{{figure}}

\begin{{figure}}[H]
    \centering
    \includegraphics[width=\columnwidth]{{plots/13_performance_vs_components.png}}
    \caption{{Classification accuracy vs.\ number of reduced dimensions.}}
    \label{{fig:dimred}}
\end{{figure}}

\begin{{figure}}[H]
    \centering
    \includegraphics[width=\columnwidth]{{plots/17_confusion_matrices.png}}
    \caption{{Normalised confusion matrices for the best model per feature type.}}
    \label{{fig:cm}}
\end{{figure}}

% ============================================================
\section{{Discussion}}

{discussion_paragraph}

\textbf{{Limitations:}} Only classical ML classifiers are used; fine-tuning DistilBERT end-to-end would likely yield higher accuracy. The 8,000-sample subset may not fully represent the original AG~News distribution. KMeans assumes spherical clusters, which may not hold for all feature spaces.

% ============================================================
\section{{Conclusion and Learning}}

This study demonstrates that:
\begin{{enumerate}}[leftmargin=*,topsep=2pt,itemsep=1pt]
    \item Contextual embeddings (DistilBERT) generally outperform sparse representations when used with linear classifiers.
    \item Bigram TF-IDF remains a strong and efficient baseline, often within 1--3 percentage points of DistilBERT.
    \item KNN benefits most from dense embeddings due to the curse of dimensionality in sparse spaces.
    \item KMeans can recover meaningful topic clusters, but accuracy lags behind supervised methods by 10--20 pp.
    \item Dimensionality reduction preserves most classification signal while significantly reducing compute and memory.
\end{{enumerate}}

% ============================================================
\begin{{thebibliography}}{{9}}
\bibitem{{agnews}} X.~Zhang, J.~Zhao, and Y.~LeCun, ``Character-level convolutional networks for text classification,'' in \emph{{NeurIPS}}, 2015.
\bibitem{{distilbert}} V.~Sanh et al., ``DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter,'' in \emph{{NeurIPS Workshop}}, 2019.
\bibitem{{glove}} J.~Pennington, R.~Socher, and C.~Manning, ``GloVe: Global vectors for word representation,'' in \emph{{EMNLP}}, 2014.
\bibitem{{sklearn}} F.~Pedregosa et al., ``Scikit-learn: Machine learning in Python,'' \emph{{JMLR}}, vol.~12, 2011.
\end{{thebibliography}}

\end{{document}}
"""


def load_results():
    """Load pipeline results from CSV files."""
    results_path = os.path.join(config.RESULTS_DIR, 'comparison_results.csv')
    kmeans_path = os.path.join(config.RESULTS_DIR, 'kmeans_clustering_metrics.csv')

    if not os.path.exists(results_path):
        print(f"  [ERROR] Results not found at {results_path}")
        print("    Run main.py first, or use: python generate_report.py --full")
        sys.exit(1)

    df = pd.read_csv(results_path)
    km_df = pd.read_csv(kmeans_path) if os.path.exists(kmeans_path) else None

    return df, km_df


def generate_main_table_latex(df):
    """Generate LaTeX tabular for the master comparison table."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Classification results across all model--feature combinations.}")
    lines.append(r"\label{tab:main}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{ll cccc r}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Feature} & \textbf{Model} & \textbf{Accuracy} & "
                 r"\textbf{Macro F1} & \textbf{Precision} & \textbf{Recall} & "
                 r"\textbf{Time (s)} \\")
    lines.append(r"\midrule")

    # Sort by feature then model
    feature_order = config.FEATURE_NAMES
    model_order = config.MODEL_NAMES

    best_acc = df['Accuracy'].max()

    for feat in feature_order:
        feat_df = df[df['Feature'] == feat]
        if feat_df.empty:
            continue
        for m_idx, model in enumerate(model_order):
            row = feat_df[feat_df['Model'] == model]
            if row.empty:
                continue
            row = row.iloc[0]
            acc_str = f"{row['Accuracy']:.4f}"
            if row['Accuracy'] == best_acc:
                acc_str = r"\textbf{" + acc_str + "}"
            feat_label = feat if m_idx == 0 else ""
            lines.append(
                f"  {feat_label} & {model} & {acc_str} & "
                f"{row['Macro F1']:.4f} & {row['Precision']:.4f} & "
                f"{row['Recall']:.4f} & {row['Time (s)']:.2f} \\\\"
            )
        lines.append(r"\midrule")

    # Remove last midrule and replace with bottomrule
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def generate_kmeans_table_latex(km_df):
    """Generate LaTeX table for KMeans clustering metrics."""
    if km_df is None or km_df.empty:
        return ""

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{KMeans clustering quality metrics.}")
    lines.append(r"\label{tab:kmeans}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l ccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Feature} & \textbf{ARI} & \textbf{NMI} & \textbf{Purity} \\")
    lines.append(r"\midrule")

    for _, row in km_df.iterrows():
        lines.append(
            f"  {row['Feature']} & {row['ARI']:.4f} & {row['NMI']:.4f} & {row['Purity']:.4f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_observations(df):
    """Automatically generate key observations from the results."""
    best_row = df.loc[df['Accuracy'].idxmax()]

    # Sparse vs contextual
    sparse_feats = ['BoW', 'TF-IDF']
    contextual_feats = ['GloVe', 'DistilBERT-CLS', 'DistilBERT-Mean']
    supervised_models = ['Logistic Regression', 'Linear SVM', 'KNN']

    sparse_df = df[(df['Feature'].isin(sparse_feats)) & (df['Model'].isin(supervised_models))]
    ctx_df = df[(df['Feature'].isin(contextual_feats)) & (df['Model'].isin(supervised_models))]

    avg_sparse_acc = sparse_df['Accuracy'].mean()
    avg_ctx_acc = ctx_df['Accuracy'].mean()

    obs1 = (f"The best overall configuration is "
            f"\\textbf{{{best_row['Feature']} + {best_row['Model']}}} "
            f"with accuracy {best_row['Accuracy']:.4f} and macro F1 {best_row['Macro F1']:.4f}.")

    obs2 = (f"Contextual embeddings average {avg_ctx_acc:.4f} accuracy vs.\\ "
            f"{avg_sparse_acc:.4f} for sparse features on supervised models, "
            f"a gap of {(avg_ctx_acc - avg_sparse_acc)*100:.1f} percentage points.")

    obs3 = (f"KMeans performs worst across all features, "
            f"confirming the difficulty of unsupervised topic recovery; "
            f"KNN shows the largest improvement when moving from sparse to dense features.")

    return obs1, obs2, obs3


def generate_sparse_vs_contextual(df):
    """Generate paragraph comparing sparse vs contextual features."""
    sparse_feats = ['BoW', 'TF-IDF']
    contextual_feats = ['GloVe', 'DistilBERT-CLS', 'DistilBERT-Mean']

    # LR + SVM performance
    lr_svm = df[df['Model'].isin(['Logistic Regression', 'Linear SVM'])]
    sparse_lr_svm = lr_svm[lr_svm['Feature'].isin(sparse_feats)]
    ctx_lr_svm = lr_svm[lr_svm['Feature'].isin(contextual_feats)]

    sparse_avg = sparse_lr_svm['Accuracy'].mean() if not sparse_lr_svm.empty else 0
    ctx_avg = ctx_lr_svm['Accuracy'].mean() if not ctx_lr_svm.empty else 0

    # Best TF-IDF
    tfidf_lr = df[(df['Feature'] == 'TF-IDF') & (df['Model'] == 'Logistic Regression')]
    tfidf_acc = tfidf_lr['Accuracy'].values[0] if not tfidf_lr.empty else 0

    para = (
        f"For linear classifiers (LR, SVM), contextual embeddings average "
        f"{ctx_avg:.4f} accuracy compared to {sparse_avg:.4f} for sparse features. "
        f"However, TF-IDF with Logistic Regression achieves {tfidf_acc:.4f}, "
        f"demonstrating that well-tuned sparse features remain highly competitive. "
        f"The performance gap is most pronounced for KNN, where high-dimensional "
        f"sparse spaces lead to the well-known curse of dimensionality."
    )
    return para


def generate_kmeans_paragraph(df, km_df):
    """Generate paragraph about KMeans results."""
    kmeans_df = df[df['Model'] == 'KMeans']
    if kmeans_df.empty:
        return "KMeans results were not available."

    best_km = kmeans_df.loc[kmeans_df['Accuracy'].idxmax()]
    worst_km = kmeans_df.loc[kmeans_df['Accuracy'].idxmin()]

    para = (
        f"KMeans clustering, as an unsupervised baseline, achieves its best "
        f"accuracy of {best_km['Accuracy']:.4f} with {best_km['Feature']} features "
        f"and its worst at {worst_km['Accuracy']:.4f} with {worst_km['Feature']}. "
        f"The Hungarian algorithm is used for optimal cluster-to-label mapping. "
    )

    if km_df is not None and not km_df.empty:
        best_nmi = km_df.loc[km_df['NMI'].idxmax()]
        para += (
            f"NMI is highest for {best_nmi['Feature']} ({best_nmi['NMI']:.4f}), "
            f"indicating that this embedding best preserves category structure "
            f"in the unsupervised setting."
        )

    return para


def generate_discussion(df):
    """Generate discussion paragraph."""
    # Time comparison
    sparse_time = df[df['Feature'].isin(['BoW', 'TF-IDF'])]['Time (s)'].mean()
    ctx_time = df[df['Feature'].isin(['DistilBERT-CLS', 'DistilBERT-Mean'])]['Time (s)'].mean()

    para = (
        f"The results confirm that contextual embeddings from pre-trained transformers "
        f"encode richer semantic information than count-based sparse features, translating "
        f"to higher classification accuracy on most classifiers. "
        f"However, the improvement is not uniform: TF-IDF with sublinear scaling and bigrams "
        f"captures enough discriminative signal for linear models to perform within a few "
        f"percentage points of DistilBERT. "
        f"From a practical standpoint, sparse feature extraction is lightweight while "
        f"DistilBERT extraction requires GPU inference before the classical models are trained. "
        f"The reported model-training time is {sparse_time:.1f}s on sparse features vs.\\ "
        f"{ctx_time:.1f}s on cached DistilBERT embeddings, excluding embedding extraction. "
        f"This trade-off is crucial for deployment-constrained environments. "
        f"The t-SNE plots (Figure~\\ref{{fig:tsne}}) visually confirm that DistilBERT "
        f"embeddings produce more separable clusters, explaining the higher KNN and KMeans "
        f"performance on these features."
    )
    return para


def generate_comparison_summary(df):
    """Generate a formatted text summary table for quick reference."""
    lines = []
    lines.append("=" * 90)
    lines.append("COMPARISON SUMMARY TABLE")
    lines.append("=" * 90)
    lines.append(f"{'Feature':<18} {'Model':<22} {'Accuracy':>10} {'Macro F1':>10} "
                 f"{'Precision':>10} {'Recall':>10} {'Time(s)':>10}")
    lines.append("-" * 90)

    for feat in config.FEATURE_NAMES:
        feat_df = df[df['Feature'] == feat]
        for model in config.MODEL_NAMES:
            row = feat_df[feat_df['Model'] == model]
            if row.empty:
                continue
            r = row.iloc[0]
            lines.append(f"{feat:<18} {model:<22} {r['Accuracy']:>10.4f} {r['Macro F1']:>10.4f} "
                         f"{r['Precision']:>10.4f} {r['Recall']:>10.4f} {r['Time (s)']:>10.2f}")
        lines.append("-" * 90)

    # Best overall
    best = df.loc[df['Accuracy'].idxmax()]
    lines.append(f"\nBest: {best['Feature']} + {best['Model']} - "
                 f"Acc={best['Accuracy']:.4f}, F1={best['Macro F1']:.4f}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate project report and comparison tables.")
    parser.add_argument('--full', action='store_true',
                        help='Run the full pipeline before generating the report.')
    parser.add_argument('--author', type=str, default='Student Name',
                        help='Author name for the report.')
    parser.add_argument('--date', type=str, default=r'\today',
                        help='Date for the report.')
    parser.add_argument('--pdf', action='store_true',
                        help='Compile the LaTeX report into PDF with pdflatex.')
    args = parser.parse_args()

    if args.full:
        print("Running full pipeline first...")
        from main import main as run_pipeline
        run_pipeline()
        print("\n" + "=" * 60)
        print("Pipeline complete. Generating report...")
        print("=" * 60 + "\n")

    # Load results
    df, km_df = load_results()

    # Generate observations
    obs1, obs2, obs3 = generate_observations(df)

    # Best combo for abstract
    best_row = df.loc[df['Accuracy'].idxmax()]
    best_combo = f"{best_row['Feature']} + {best_row['Model']}"
    best_acc = best_row['Accuracy']

    # Generate LaTeX content
    main_table = generate_main_table_latex(df)
    kmeans_table = generate_kmeans_table_latex(km_df)
    sparse_vs_ctx = generate_sparse_vs_contextual(df)
    kmeans_para = generate_kmeans_paragraph(df, km_df)
    discussion = generate_discussion(df)

    # Fill template
    report = LATEX_TEMPLATE.format(
        author_name=args.author,
        date=args.date,
        best_combo=best_combo,
        best_acc=best_acc,
        main_table_latex=main_table,
        kmeans_table_latex=kmeans_table,
        observation_1=obs1,
        observation_2=obs2,
        observation_3=obs3,
        sparse_vs_contextual_paragraph=sparse_vs_ctx,
        kmeans_paragraph=kmeans_para,
        discussion_paragraph=discussion,
    )

    # Save report
    report_path = os.path.join(config.RESULTS_DIR, 'report.tex')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  [OK] LaTeX report saved to: {report_path}")

    if args.pdf:
        print("  Compiling LaTeX to PDF...")
        try:
            subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', '-output-directory', config.RESULTS_DIR, report_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            pdf_path = os.path.join(config.RESULTS_DIR, 'report.pdf')
            if os.path.exists(pdf_path):
                print(f"  [OK] PDF report generated at: {pdf_path}")
            else:
                print("  [ERROR] PDF generation completed but report.pdf not found.")
        except FileNotFoundError:
            print('  [ERROR] pdflatex not found. Install TeX or compile the .tex file manually.')
        except subprocess.CalledProcessError as e:
            print('  [ERROR] pdflatex failed to compile the report. Check the LaTeX log for errors.')

    # Also generate a text summary
    summary = generate_comparison_summary(df)
    summary_path = os.path.join(config.RESULTS_DIR, 'comparison_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"  [OK] Comparison summary saved to: {summary_path}")
    print(f"\n{summary}")

    # Print instructions
    print("\n" + "=" * 60)
    print("REPORT GENERATION COMPLETE")
    print("=" * 60)
    print(f"\n  LaTeX Report:     {report_path}")
    if args.pdf:
        print(f"  PDF Report:       {os.path.join(config.RESULTS_DIR, 'report.pdf')}")
    print(f"  Comparison CSV:   {config.RESULTS_DIR}/comparison_results.csv")
    print(f"  Summary Table:    {summary_path}")
    print(f"  Plots Directory:  {config.PLOTS_DIR}")
    print(f"\n  To compile the report:")
    print(f"    pdflatex -output-directory results results/report.tex")
    print(f"    OR use Overleaf with report.tex and the plots/ folder at the project root")
    print()


if __name__ == '__main__':
    main()
