# NIDS Supervised Pipeline — Step by Step Guide
## Author: Shikha Priya | GH103517 | GISMA University

---

# STEP 1 — ZIP DOWNLOAD KARO

1. Upar ZIP file ka link hai — click karke download karo
2. Apne Desktop pe save karo
3. Right click → "Extract Here" ya "Extract All"
4. Folder name hoga: nids_project

---

# STEP 2 — PYTHON INSTALL KARO (Agar nahi hai)

1. Browser me jao: https://www.python.org/downloads/
2. "Download Python 3.11" button click karo
3. Install karo — IMPORTANT: "Add Python to PATH" checkbox ZAROOR tick karo
4. Install complete hone do

Check karo Python install hua ya nahi:
- Windows: Start → search "cmd" → Command Prompt kholo
- Type karo:  python --version
- Output aana chahiye:  Python 3.11.x

---

# STEP 3 — FOLDER ME JAO

Windows (Command Prompt me):
    cd Desktop\nids_project

Mac/Linux (Terminal me):
    cd Desktop/nids_project

---

# STEP 4 — VIRTUAL ENVIRONMENT BANAO

Windows:
    python -m venv nids_env
    nids_env\Scripts\activate

Mac/Linux:
    python3 -m venv nids_env
    source nids_env/bin/activate

Activate hone ke baad terminal me dikhega:
    (nids_env) C:\Desktop\nids_project>

---

# STEP 5 — PACKAGES INSTALL KARO

    pip install -r requirements.txt

Iske baad automatically install hoga:
- scikit-learn
- xgboost
- pandas
- numpy
- matplotlib
- seaborn
- joblib

2-3 minute lagenge. Internet chahiye.

---

# STEP 6 — PEHLE TEST KARO (Mock Data Se)

Real datasets ki zarurat nahi — pehle test karo:

    python run_pipeline.py --generate-data

Yeh kya karega:
✅ Mock data generate karega
✅ Saare 6 models train karega
✅ Cross-dataset evaluation karega
✅ Plots banayega
✅ metrics.json save karega

Total time: 2-5 minute

---

# STEP 7 — REAL DATA SE CHALAO

## Dataset Download Links:

CICIDS2017:
    https://www.unb.ca/cic/datasets/ids-2017.html

UNSW-NB15:
    https://research.unsw.edu.au/projects/unsw-nb15-dataset

CSE-CIC-IDS2018:
    https://www.unb.ca/cic/datasets/ids-2018.html

## CSV Files Ko Rename Karo aur data/raw/ Me Rakho:

    cicids2017_train.csv
    cicids2017_test.csv
    unswnb15_test.csv
    cse_cic_ids2018_test.csv

## Run Karo:

    python run_pipeline.py --data-dir data/raw --skip-knn

---

# STEP 8 — RESULTS DEKHO

Run hone ke baad yeh files ban jaati hain:

output/
    metrics.json                 ← Saare numbers (F1, ROC-AUC, etc.)
    plots/
        roc_cicids2017.png       ← ROC curve
        roc_unswnb15.png         ← ROC curve
        roc_cse_cic_ids2018.png  ← ROC curve
        pr_cicids2017.png        ← Precision-Recall curve
        f1_comparison.png        ← Model comparison bar chart
        pca_datasets.png         ← Dataset distribution
        scores_*.png             ← Score distributions

---

# STEP 9 — GITHUB PE UPLOAD KARO

1. github.com → Login karo
2. "New Repository" click karo
3. Name: cross-dataset-nids-supervised
4. Public select karo
5. README add karo — UNCHECK karo
6. "Create repository" click karo
7. "uploading an existing file" click karo
8. nids_project folder ki SAARI files drag & drop karo
9. Commit message: Initial commit supervised NIDS pipeline
10. "Commit changes" click karo

---

# COMMON ERRORS AUR FIX:

ERROR: ModuleNotFoundError: No module named 'xgboost'
FIX:   pip install xgboost

ERROR: ModuleNotFoundError: No module named 'sklearn'
FIX:   pip install scikit-learn

ERROR: FileNotFoundError: cicids2017_train.csv
FIX:   CSV files data/raw/ folder me rakho

ERROR: python not recognized
FIX:   Python reinstall karo — "Add to PATH" tick karo

ERROR: pip not recognized
FIX:   python -m pip install -r requirements.txt

---

# QUICK REFERENCE — ALL COMMANDS:

# Virtual env activate
nids_env\Scripts\activate          (Windows)
source nids_env/bin/activate       (Mac/Linux)

# Install
pip install -r requirements.txt

# Mock data test
python run_pipeline.py --generate-data

# Real data (fast - no KNN)
python run_pipeline.py --data-dir data/raw --skip-knn

# Real data (all models)
python run_pipeline.py --data-dir data/raw

# Custom settings
python run_pipeline.py --data-dir data/raw --cv-folds 5 --n-iter 20 --k-features 20

---

# SUPERVISOR KE REQUIREMENTS — SAB COVER:

✅ Supervised Learning        — 6 classifiers
✅ Data Preprocessing         — cleaning, alignment, scaling
✅ Feature Selection          — SelectKBest (top 20 features)
✅ Cross-Validation           — Stratified 5-Fold CV
✅ Hyperparameter Tuning      — GridSearchCV / RandomizedSearchCV
✅ Performance Evaluation     — Precision, Recall, F1, ROC-AUC, PR-AUC, MCC
✅ Cross-Dataset Testing      — UNSW-NB15 + CSE-CIC-IDS2018
✅ Plots & Visualisation      — ROC, PR, F1 bar, PCA, score distribution

---


