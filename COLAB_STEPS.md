# Google Colab GPU Steps

## 1. Prepare Colab

1. Open https://colab.research.google.com/.
2. Upload `StatML_Colab.ipynb`.
3. Open `Runtime > Change runtime type`.
4. Set `Hardware accelerator` to `T4 GPU` or any available GPU.
5. Click `Save`.

## 2. Set Up The Project

### Option A: Clone Directly from GitHub (Recommended)

In a new code cell in Colab, run:

```bash
!git clone https://github.com/tharunsrisanth07-dotcom/ML-project.git
%cd ML-project
```

### Option B: Upload Zip File

Run the notebook upload cell. When Colab asks for a file, upload a zip of the whole `ML-project` folder.

To make the zip on Windows:

1. Go to the folder that contains `ML_pro`.
2. Right-click the folder.
3. Choose `Send to > Compressed (zipped) folder`.
4. Upload that zip in Colab.

## 3. Install Dependencies

Run the dependency cell:

```python
!pip install -q -r requirements.txt
```

Optional PDF support:

```python
!apt-get update -y >/dev/null
!apt-get install -y -qq texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra >/dev/null
```

## 4. Run The Pipeline

Recommended: run every notebook cell from top to bottom. This shows plots inline.

Alternative full-script run:

```python
!python main.py
```

## 5. See Plots

In the notebook, plots display automatically after each visualization cell. To display all saved plots again:

```python
from IPython.display import Image, display
import glob

for path in sorted(glob.glob("plots/*.png")):
    print(path)
    display(Image(filename=path))
```

## 6. See Results

```python
import pandas as pd

results = pd.read_csv("results/comparison_results.csv")
display(results.sort_values("Accuracy", ascending=False))

kmeans = pd.read_csv("results/kmeans_clustering_metrics.csv")
display(kmeans)
```

## 7. Generate Report

```python
!python generate_report.py --author "Your Name"
```

If TeX is installed and you want a PDF:

```python
!python generate_report.py --author "Your Name" --pdf
```

## 8. Download Outputs

The final notebook cell creates `ML_Project_Complete.zip`, containing:

- source files
- `plots/`
- `results/`
- `requirements.txt`
- `README.md`
- `COLAB_STEPS.md`
