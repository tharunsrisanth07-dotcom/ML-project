# Statistical Machine Learning Course Project

## Text Classification using Classical ML: Sparse Features vs Contextual Embeddings

This project compares classical machine learning models on AG News multi-class text classification using:

- Sparse features: unigram Bag-of-Words and unigram+bigram TF-IDF
- Frozen pretrained features: DistilBERT CLS and mean-pooled embeddings
- Bonus dense features: mean-pooled GloVe embeddings
- Classical models: Logistic Regression, Linear SVM, KNN, and KMeans
- Dimensionality reduction: Truncated SVD for TF-IDF, PCA for dense embeddings

## Dataset

Dataset: `fancyzhx/ag_news` from Hugging Face

- Total project sample: 8,000 rows
- Classes: World, Sports, Business, Sci/Tech
- Sampling: 2,000 examples per class
- Split: 80 percent train, 20 percent test

## Project Structure

```text
ML project/
|-- config.py
|-- data_loader.py
|-- preprocessing.py
|-- dimensionality_reduction.py
|-- evaluation.py
|-- visualization.py
|-- main.py
|-- generate_report.py
|-- requirements.txt
|-- StatML_Colab.ipynb
|-- feature_extraction/
|   |-- sparse_features.py
|   |-- contextual_embeddings.py
|-- models/
|   |-- classifiers.py
|-- plots/
|-- results/
|-- cache/
```

## Recommended: Run in Google Colab GPU

1. Open Google Colab.
2. Go to `Runtime > Change runtime type`.
3. Select `T4 GPU` or any available GPU, then click `Save`.
4. In a code cell, clone the repository directly:
   ```bash
   !git clone https://github.com/tharunsrisanth07-dotcom/ML-project.git
   %cd ML-project
   ```
   *(Or upload `StatML_Colab.ipynb` and follow the prompt).*
5. Run notebook cells from top to bottom.

The notebook displays plots inline and also saves all PNG files into `plots/`.

## Colab Script Run

If you prefer running the full script instead of notebook cells:

```python
!pip install -q -r requirements.txt
!python main.py
!python generate_report.py --author "Your Name"
```

To display saved plots after running `main.py`:

```python
from IPython.display import Image, display
import glob

for path in sorted(glob.glob("plots/*.png")):
    print(path)
    display(Image(filename=path))
```

## Local Run

```bash
pip install -r requirements.txt
python main.py
python generate_report.py --author "Your Name"
```

Local CPU runs are possible, but DistilBERT embedding extraction will be much slower than Colab GPU.

## Outputs

| Output | Location |
|---|---|
| EDA and comparison plots | `plots/*.png` |
| Main results table | `results/comparison_results.csv` |
| KMeans clustering metrics | `results/kmeans_clustering_metrics.csv` |
| Text summary | `results/comparison_summary.txt` |
| LaTeX report | `results/report.tex` |
| Cached embeddings | `cache/*.npy` |

## Generated Plots

| Filename | Description |
|---|---|
| `01_class_distribution.png` | Class balance |
| `02_text_length_distribution.png` | Word count histograms and box plots |
| `03_word_clouds.png` | Word clouds per class |
| `04_top_words_per_class.png` | Top words per class after stop-word filtering |
| `05_tsne_comparison.png` | t-SNE comparison across feature spaces |
| `06_tsne_*.png` to `10_tsne_*.png` | Individual t-SNE plots |
| `11_explained_variance.png` | SVD/PCA cumulative explained variance |
| `12_scree_plot.png` | Scree plot |
| `13_performance_vs_components.png` | Accuracy vs reduced dimensions |
| `14_comparison_accuracy.png` | Accuracy bar chart |
| `15_comparison_macro_f1.png` | Macro F1 bar chart |
| `16_heatmap_results.png` | Accuracy and Macro F1 heatmaps |
| `17_confusion_matrices.png` | Normalized confusion matrices |
| `18_training_time.png` | Training time comparison |
| `19_per_class_f1_boxplot.png` | Per-class F1 distribution |
| `20_radar_*.png` | Multi-metric radar charts |
| `21_kmeans_*.png` | KMeans clusters vs true labels |

## Report Sections

The generated report follows the required format:

- Introduction and Dataset used
- Dataset Description and EDA
- Methodology
- Experiments
- Results
- Discussion
- Conclusion and Learning
