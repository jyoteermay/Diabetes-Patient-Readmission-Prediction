# Hospital Readmission Prediction — Diabetes 130-US Hospitals

Predicting early (within 30 days) hospital readmission for diabetic patients using machine learning.

---

## Dataset

**Source:** [UCI ML Repository — Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

- **100,000+** hospital encounters from 130 US hospitals
- **50 features** including patient demographics, diagnoses, medications, and lab results
- **Target:** Binary — was the patient readmitted within 30 days? (`1 = Yes`, `0 = No`)
- **Class imbalance:** ~11% positive (readmitted within 30 days)

---

## Project Structure

```
diabetes_readmission_project/
├── diabetes_readmission.ipynb   # Main notebook (run top-to-bottom)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

> **Note:** The dataset (`diabetic_data.csv`) is not included. Download it from the UCI link above and set `DATA_PATH` in the notebook accordingly.

---

## Notebook Outline

| Section | Description |
|---------|-------------|
| 1. Imports | All libraries in one place |
| 2. Data Loading | Load CSV, initial shape check |
| 3. Data Cleaning | Handle `?` placeholders, cast ID columns |
| 4. Missing Value Analysis | Drop high-missing columns, fill/remove remaining |
| 5. Feature Engineering (pre-EDA) | Target encoding, medication chi-square selection, ICD-9 grouping, admission/discharge mapping |
| 6. Statistical Tests | Chi-square, t-test, ANOVA, Pearson correlation |
| 7. EDA | Class distribution, numerical distributions, correlation heatmap, categorical countplots, box plots |
| 8. Feature Engineering (post-EDA) | Age midpoint, elderly flag, utilisation features, interaction terms |
| 9. Modelling Pipeline | Train/val/test split, log-transform, OHE, impute → scale → SMOTE, GridSearchCV |
| 10. Results & Evaluation | Model comparison table, threshold optimisation, test metrics, ROC curve, feature importance |

---

## Methodology

### Imbalance Handling

| Model | Training Data | Imbalance Strategy |
|-------|-------------|-------------------|
| Random Forest | Real (imbalanced) | `class_weight="balanced_subsample"` |
| XGBoost | Real (imbalanced) | `scale_pos_weight` (~8×) |
| LightGBM | Real (imbalanced) | `is_unbalance=True` |
| Logistic Regression | SMOTE-balanced | None (data already balanced) |

Tree models are trained on real data because they overfit SMOTE's synthetic interpolations. Logistic Regression benefits from SMOTE because balanced data directly shifts the linear decision boundary.

### Preprocessing Pipeline (No Leakage)
```
Imputer  →  fit on train only
Scaler   →  fit on real train only (before SMOTE)
SMOTE    →  applied in scaled space, train only
```

### Threshold Strategy
Two thresholds are tuned on the validation set:
- **Max-F1 threshold** — balances precision and recall
- **Max-Recall threshold** (precision ≥ 0.30) — for clinical use where missing a readmission is costlier than a false alarm

---

## Results

| Metric | Value (F1-optimised threshold) |
|--------|-------------------------------|
| Best Model | XGBoost |
| ROC AUC | ~0.67 |
| PR AUC | ~0.22 |
| Recall | ~0.46 |
| Precision | ~0.20 |
| F1-score | ~0.28 |

ROC AUC of 0.67 is above the logistic regression baseline (~0.62–0.65) and reflects the genuine difficulty of this task — the signal-to-noise ratio is low and class imbalance is severe.

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/diabetes_readmission_project.git
cd diabetes_readmission_project

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
# https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
# Place diabetic_data.csv in the project root and update DATA_PATH in the notebook

# 5. Launch Jupyter
jupyter notebook diabetes_readmission.ipynb
```

---

## Requirements

See `requirements.txt`. Key libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `imbalanced-learn`, `matplotlib`, `seaborn`.

---

## Author

Built as a capstone project for the Great Learning ML programme.
