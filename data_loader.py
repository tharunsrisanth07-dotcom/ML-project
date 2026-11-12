"""
Data Loading and Exploratory Data Analysis Module
==================================================
- Loads AG News from HuggingFace
- Stratified sampling to 8,000 samples
- Train/test split
- Comprehensive EDA with visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import WordCloud

import config


_WORD_PATTERN = re.compile(r"[a-z]{3,}")


def _is_notebook():
    """Check if running inside an interactive notebook (Jupyter / Colab)."""
    try:
        import IPython
        shell = IPython.get_ipython()
        return shell is not None and 'TerminalInteractiveShell' not in str(type(shell))
    except Exception:
        return False


def load_and_sample_dataset():
    """
    Load AG News from HuggingFace and create a stratified sample of 8,000 rows.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['text', 'label', 'class_name'].
    """
    print("=" * 60)
    print("STAGE 1: Loading AG News Dataset")
    print("=" * 60)

    # Load full dataset (train + test)
    dataset = load_dataset(config.DATASET_NAME)
    train_df = dataset['train'].to_pandas()
    test_df = dataset['test'].to_pandas()
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    print(f"  Full dataset size: {len(full_df):,}")
    print(f"  Classes: {full_df['label'].nunique()}")

    # Stratified sampling: 2,000 per class = 8,000 total
    samples_per_class = config.NUM_SAMPLES // config.NUM_CLASSES
    sampled_dfs = []
    for label in range(config.NUM_CLASSES):
        class_df = full_df[full_df['label'] == label]
        sampled = class_df.sample(n=samples_per_class, random_state=config.RANDOM_SEED)
        sampled_dfs.append(sampled)

    df = pd.concat(sampled_dfs, ignore_index=True)
    df = df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)

    # Add class name column
    df['class_name'] = df['label'].map(config.CLASS_LABEL_MAP)

    print(f"  Sampled dataset size: {len(df):,}")
    print(f"  Samples per class: {samples_per_class:,}")
    print()

    return df


def split_dataset(df):
    """
    Perform stratified train/test split (80/20).

    Parameters
    ----------
    df : pd.DataFrame
        Full sampled DataFrame.

    Returns
    -------
    tuple
        (X_train_texts, X_test_texts, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].tolist(),
        df['label'].values,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=df['label'].values
    )

    print(f"  Train set size: {len(X_train):,}")
    print(f"  Test set size:  {len(X_test):,}")
    print(f"  Train class distribution: {dict(Counter(y_train))}")
    print(f"  Test class distribution:  {dict(Counter(y_test))}")
    print()

    return X_train, X_test, y_train, y_test


def run_eda(df):
    """
    Perform comprehensive Exploratory Data Analysis and generate plots.

    Parameters
    ----------
    df : pd.DataFrame
        The sampled dataset with columns ['text', 'label', 'class_name'].
    """
    print("=" * 60)
    print("STAGE 2: Exploratory Data Analysis")
    print("=" * 60)

    plt.rcParams.update(config.PLOT_STYLE)

    # Add text length column
    df['text_length'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))

    # ---- Print Summary Statistics ----
    print("\n  Dataset Summary Statistics:")
    print("  " + "-" * 50)
    stats = df.groupby('class_name').agg(
        count=('text', 'count'),
        avg_char_length=('text_length', 'mean'),
        avg_word_count=('word_count', 'mean'),
        min_length=('text_length', 'min'),
        max_length=('text_length', 'max'),
    ).round(1)
    print(stats.to_string())
    print()

    # ---- Plot 1: Class Distribution ----
    _plot_class_distribution(df)

    # ---- Plot 2: Text Length Distribution ----
    _plot_text_length_distribution(df)

    # ---- Plot 3: Word Clouds ----
    _plot_word_clouds(df)

    # ---- Plot 4: Top Words per Class ----
    _plot_top_words(df)

    print("  [OK] All EDA plots saved to:", config.PLOTS_DIR)
    print()


def _plot_class_distribution(df):
    """Bar chart showing class distribution with counts annotated."""
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_SMALL)

    counts = df['class_name'].value_counts().reindex(config.CLASS_NAMES)
    bars = ax.bar(counts.index, counts.values, color=config.CLASS_COLORS,
                  edgecolor='white', linewidth=1.5, width=0.6)

    # Annotate bars
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=13)

    ax.set_title('Class Distribution in AG News (8,000 Samples)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('News Category', fontsize=13)
    ax.set_ylabel('Number of Samples', fontsize=13)
    ax.set_ylim(0, max(counts.values) * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/01_class_distribution.png', dpi=config.PLOT_DPI, bbox_inches='tight')
    if _is_notebook():
        plt.show()
    plt.close()


def _plot_text_length_distribution(df):
    """Overlaid histograms of text length per class + box plot."""
    fig, axes = plt.subplots(1, 2, figsize=config.FIGURE_SIZE_WIDE)

    # Histogram
    for i, cls in enumerate(config.CLASS_NAMES):
        subset = df[df['class_name'] == cls]['word_count']
        axes[0].hist(subset, bins=40, alpha=0.55, label=cls,
                     color=config.CLASS_COLORS[i], edgecolor='white', linewidth=0.5)

    axes[0].set_title('Word Count Distribution by Class', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Word Count')
    axes[0].set_ylabel('Frequency')
    axes[0].legend(framealpha=0.9)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Box plot
    box_data = [df[df['class_name'] == cls]['word_count'].values for cls in config.CLASS_NAMES]
    bp = axes[1].boxplot(box_data, labels=config.CLASS_NAMES, patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], config.CLASS_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)

    axes[1].set_title('Word Count Box Plot by Class', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Word Count')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/02_text_length_distribution.png', dpi=config.PLOT_DPI, bbox_inches='tight')
    if _is_notebook():
        plt.show()
    plt.close()


def _plot_word_clouds(df):
    """2x2 grid of word clouds, one per class."""
    fig, axes = plt.subplots(2, 2, figsize=config.FIGURE_SIZE_GRID)
    axes = axes.flatten()

    for i, cls in enumerate(config.CLASS_NAMES):
        text = ' '.join(df[df['class_name'] == cls]['text'].tolist())
        wc = WordCloud(
            width=800, height=500,
            background_color='white',
            colormap='viridis',
            max_words=100,
            random_state=config.RANDOM_SEED,
            contour_width=1,
            contour_color=config.CLASS_COLORS[i],
        ).generate(text)
        axes[i].imshow(wc, interpolation='bilinear')
        axes[i].set_title(f'{cls}', fontsize=16, fontweight='bold',
                          color=config.CLASS_COLORS[i], pad=10)
        axes[i].axis('off')

    plt.suptitle('Word Clouds by News Category', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/03_word_clouds.png', dpi=config.PLOT_DPI, bbox_inches='tight')
    if _is_notebook():
        plt.show()
    plt.close()


def _plot_top_words(df):
    """Horizontal bar chart of top-20 words per class."""
    fig, axes = plt.subplots(2, 2, figsize=config.FIGURE_SIZE_GRID)
    axes = axes.flatten()

    for i, cls in enumerate(config.CLASS_NAMES):
        texts = df[df['class_name'] == cls]['text'].tolist()
        all_words = [
            word
            for word in _WORD_PATTERN.findall(' '.join(texts).lower())
            if word not in ENGLISH_STOP_WORDS
        ]
        word_counts = Counter(all_words).most_common(20)
        if word_counts:
            words, counts = zip(*word_counts)
        else:
            words, counts = (), ()

        axes[i].barh(range(len(words)), counts, color=config.CLASS_COLORS[i],
                     edgecolor='white', linewidth=0.5, alpha=0.85)
        axes[i].set_yticks(range(len(words)))
        axes[i].set_yticklabels(words, fontsize=10)
        axes[i].invert_yaxis()
        axes[i].set_title(f'Top 20 Words - {cls}', fontsize=13, fontweight='bold')
        axes[i].set_xlabel('Frequency')
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)

    plt.suptitle('Most Frequent Words by Category', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/04_top_words_per_class.png', dpi=config.PLOT_DPI, bbox_inches='tight')
    if _is_notebook():
        plt.show()
    plt.close()
