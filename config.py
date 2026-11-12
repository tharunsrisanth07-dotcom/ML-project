"""
Configuration file for the Statistical Machine Learning Project.
Contains all hyperparameters, paths, and constants.
"""

import os

# ============================================================
# Random Seed
# ============================================================
RANDOM_SEED = 42

# ============================================================
# Dataset Configuration
# ============================================================
DATASET_NAME = "fancyzhx/ag_news"
NUM_SAMPLES = 8000          # Total samples to use
TEST_SIZE = 0.2             # 80/20 train-test split
NUM_CLASSES = 4
CLASS_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']
CLASS_LABEL_MAP = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}

# ============================================================
# Text Preprocessing
# ============================================================
MIN_DF = 3                  # Minimum document frequency
MAX_DF = 0.95               # Maximum document frequency ratio

# ============================================================
# Sparse Feature Configuration
# ============================================================
BOW_MAX_FEATURES = 10000
TFIDF_MAX_FEATURES = 15000
TFIDF_NGRAM_RANGE = (1, 2)  # Bigram TF-IDF

# ============================================================
# DistilBERT Configuration
# ============================================================
DISTILBERT_MODEL = 'distilbert-base-uncased'
EMBEDDING_BATCH_SIZE = 64   # Batch size for GPU inference
MAX_SEQ_LENGTH = 128

# ============================================================
# GloVe Configuration
# ============================================================
GLOVE_DIM = 300
GLOVE_URL = 'https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip'
GLOVE_BACKUP_URL = 'https://nlp.stanford.edu/data/glove.6B.zip'
GLOVE_FILENAME = 'glove.6B.300d.txt'

# ============================================================
# Dimensionality Reduction
# ============================================================
SVD_COMPONENTS = [50, 100, 200, 300]
PCA_COMPONENTS_EMBEDDING = [32, 64, 128, 256, 384, 512]
PCA_COMPONENTS_GLOVE = [32, 64, 128, 200]

# ============================================================
# Model Hyperparameters
# ============================================================
LR_C = 1.0
LR_MAX_ITER = 2000
SVM_C = 1.0
SVM_MAX_ITER = 5000
KNN_NEIGHBORS = 5
KMEANS_N_CLUSTERS = NUM_CLASSES
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 500

# ============================================================
# Paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
GLOVE_DIR = os.path.join(CACHE_DIR, 'glove')

# Create directories
for _dir in [PLOTS_DIR, RESULTS_DIR, CACHE_DIR, GLOVE_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ============================================================
# Visualization Configuration
# ============================================================
PLOT_DPI = 300
FIGURE_SIZE = (12, 8)
FIGURE_SIZE_SMALL = (8, 6)
FIGURE_SIZE_WIDE = (16, 8)
FIGURE_SIZE_GRID = (16, 12)

CLASS_COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6']
CLASS_COLOR_MAP = dict(zip(CLASS_NAMES, CLASS_COLORS))

FEATURE_NAMES = ['BoW', 'TF-IDF', 'GloVe', 'DistilBERT-CLS', 'DistilBERT-Mean']
FEATURE_COLORS = {
    'BoW': '#FF6B6B',
    'TF-IDF': '#4ECDC4',
    'GloVe': '#FFA07A',
    'DistilBERT-CLS': '#45B7D1',
    'DistilBERT-Mean': '#98D8C8',
}

MODEL_NAMES = ['Logistic Regression', 'Linear SVM', 'KNN', 'KMeans']
MODEL_COLORS = {
    'Logistic Regression': '#FF6B6B',
    'Linear SVM': '#4ECDC4',
    'KNN': '#96CEB4',
    'KMeans': '#DDA0DD',
}

# ============================================================
# Plot Style Settings
# ============================================================
PLOT_STYLE = {
    'figure.facecolor': 'white',
    'axes.facecolor': '#F8F9FA',
    'axes.edgecolor': '#DEE2E6',
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.color': '#E9ECEF',
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'legend.framealpha': 0.9,
    'figure.dpi': 100,
    'savefig.dpi': PLOT_DPI,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
}
