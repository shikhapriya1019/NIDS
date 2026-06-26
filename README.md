# NIDS Supervised Pipeline — Step-by-Step Guide

**Author:** Shikha Priya | **GH103517** | **GISMA University**

## STEP 1 — DOWNLOAD THE ZIP FILE

* Click the ZIP file link provided above to download it.
* Save it on your Desktop.
* Right-click the downloaded ZIP file and select **"Extract Here"** or **"Extract All"**.
* The extracted folder will be named **`nids_project`**.

---

## STEP 2 — INSTALL PYTHON (If Not Already Installed)

* Open your browser and go to: https://www.python.org/downloads/
* Click the **"Download Python 3.11"** button.
* Install Python.
* **IMPORTANT:** Make sure to check the **"Add Python to PATH"** checkbox during installation.
* Wait for the installation to complete.

### Verify Python Installation

1. Open **Command Prompt** (Windows) by searching for **cmd**.
2. Type:

```bash
python --version
```

You should see output similar to:

```text
Python 3.11.x
```

---

## STEP 3 — NAVIGATE TO THE PROJECT FOLDER

### Windows (Command Prompt)

```bash
cd Desktop\nids_project
```

### Mac/Linux (Terminal)

```bash
cd Desktop/nids_project
```

---

## STEP 4 — CREATE A VIRTUAL ENVIRONMENT

### Windows

```bash
python -m venv nids_env
nids_env\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv nids_env
source nids_env/bin/activate
```

After activation, your terminal should display something like:

```text
(nids_env) C:\Desktop\nids_project>
```

---

## STEP 5 — INSTALL REQUIRED PACKAGES

Run:

```bash
pip install -r requirements.txt
```

The following packages will be installed automatically:

* scikit-learn
* xgboost
* pandas
* numpy
* matplotlib
* seaborn
* joblib

This usually takes **2–3 minutes** and requires an internet connection.

---

## STEP 6 — TEST THE PIPELINE USING MOCK DATA

Before using real datasets, test the pipeline with generated data.

Run:

```bash
python run_pipeline.py --generate-data
```

This will:

* ✅ Generate mock data
* ✅ Train all six supervised models
* ✅ Perform cross-dataset evaluation
* ✅ Generate plots
* ✅ Save `metrics.json`

Estimated runtime: **2–5 minutes**

---

## STEP 7 — RUN THE PIPELINE WITH REAL DATA

### Download the Datasets

**CICIDS2017**

https://www.unb.ca/cic/datasets/ids-2017.html

**UNSW-NB15**

https://research.unsw.edu.au/projects/unsw-nb15-dataset

**CSE-CIC-IDS2018**

https://www.unb.ca/cic/datasets/ids-2018.html

### Rename the CSV Files and Place Them in:

```text
data/raw/
```

Use the following filenames:

```text
cicids2017_train.csv
cicids2017_test.csv
unswnb15_test.csv
cse_cic_ids2018_test.csv
```

Run the pipeline:

```bash
python run_pipeline.py --data-dir data/raw --skip-knn
```

---

## STEP 8 — VIEW THE RESULTS

After execution, the following files will be generated:

```text
output/
    metrics.json                  ← Performance metrics (F1, ROC-AUC, etc.)

plots/
    roc_cicids2017.png            ← ROC Curve
    roc_unswnb15.png              ← ROC Curve
    roc_cse_cic_ids2018.png       ← ROC Curve
    pr_cicids2017.png             ← Precision-Recall Curve
    f1_comparison.png             ← Model comparison bar chart
    pca_datasets.png              ← Dataset distribution (PCA)
    scores_*.png                  ← Score distribution plots
```

---

## STEP 9 — UPLOAD THE PROJECT TO GITHUB

1. Log in to **github.com**.
2. Click **"New Repository"**.
3. Repository name:

```text
cross-dataset-nids-supervised
```

4. Select **Public**.
5. **Do NOT** add a README (leave it unchecked).
6. Click **"Create Repository"**.
7. Select **"uploading an existing file"**.
8. Drag and drop the entire **nids_project** folder contents.
9. Commit message:

```text
Initial commit supervised NIDS pipeline
```

10. Click **"Commit changes"**.

---

# COMMON ERRORS AND FIXES

### Error

```text
ModuleNotFoundError: No module named 'xgboost'
```

**Fix**

```bash
pip install xgboost
```

---

### Error

```text
ModuleNotFoundError: No module named 'sklearn'
```

**Fix**

```bash
pip install scikit-learn
```

---

### Error

```text
FileNotFoundError: cicids2017_train.csv
```

**Fix**

Place the required CSV files inside the `data/raw/` folder.

---

### Error

```text
python is not recognized
```

**Fix**

Reinstall Python and ensure that **"Add Python to PATH"** is checked during installation.

---

### Error

```text
pip is not recognized
```

**Fix**

```bash
python -m pip install -r requirements.txt
```

---

# QUICK REFERENCE — ALL COMMANDS

## Activate Virtual Environment

### Windows

```bash
nids_env\Scripts\activate
```

### Mac/Linux

```bash
source nids_env/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Test with Mock Data

```bash
python run_pipeline.py --generate-data
```

---

## Run with Real Data (Fast Mode — Skip KNN)

```bash
python run_pipeline.py --data-dir data/raw --skip-knn
```

---

## Run with Real Data (All Models)

```bash
python run_pipeline.py --data-dir data/raw
```

---

## Custom Configuration

```bash
python run_pipeline.py --data-dir data/raw --cv-folds 5 --n-iter 20 --k-features 20
```

---

# SUPERVISOR REQUIREMENTS — ALL COVERED

✅ Supervised Learning — 6 classifiers

✅ Data Preprocessing — Cleaning, alignment, and scaling

✅ Feature Selection — SelectKBest (Top 20 features)

✅ Cross-Validation — Stratified 5-Fold Cross Validation

✅ Hyperparameter Tuning — GridSearchCV / RandomizedSearchCV

✅ Performance Evaluation — Precision, Recall, F1-Score, ROC-AUC, PR-AUC, MCC

✅ Cross-Dataset Testing — UNSW-NB15 and CSE-CIC-IDS2018

✅ Plots & Visualization — ROC Curves, Precision-Recall Curves, F1 Comparison Bar Chart, PCA Visualization, and Score Distribution Plots
