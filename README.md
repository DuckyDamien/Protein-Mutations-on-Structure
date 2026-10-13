# Protein Variant-Effect ML Practice Project

This project implements a medical-biology-style machine learning workflow on the `FECA_ECOLI_Tsuboyama_2023_2D1U` deep mutational scanning dataset. The main task is binary classification of whether a single amino-acid substitution is stabilizing, followed by a regression extension that predicts continuous stability effect size (`ddG_ML_float`).

## What the script does

- Resolves `substitutions_raw_DMS.zip` from `PROTEIN_VARIANT_ZIP_PATH`, the project folder, the parent folder, or the user's Downloads folder
- Filters to single mutants only
- Parses mutation strings such as `Q1E` into wild-type residue, position, and mutant residue
- Engineers amino-acid chemistry and local sequence-context features
- Trains:
  - majority-class baseline classifier
  - logistic regression classifier
  - random forest classifier
  - XGBoost classifier
  - LightGBM classifier
  - stacking classifier meta-learner
  - dummy regressor
  - random forest regressor
  - XGBoost regressor
  - LightGBM regressor
  - stacking regressor meta-learner
- Selects the best classifier on validation F1 and ROC-AUC
- Selects the best regressor on validation MAE and RMSE
- Evaluates on a held-out test split
- Generates validation-set permutation importance and SHAP-based interpretation for tree models
- Saves figures, metrics, reports, and fitted models

## Architecture walkthrough

The project is organized as a data-to-artifact pipeline:

1. `load_dataset()` resolves the ZIP archive portably, then reads the FECA DMS CSV member.
2. `build_modeling_table()` filters to single mutants, parses mutation strings into wild-type residue, position, and mutant residue, creates the classification and regression targets, and runs defensive data checks.
3. Feature builders combine biophysical amino-acid deltas, local sequence windows, amino-acid counts, and relative sequence position.
4. `split_dataset()` creates train, validation, and test partitions. Training fits models, validation chooses models and powers interpretability, and test is reserved for final performance reporting.
5. Each model builder wraps preprocessing and estimation in a scikit-learn `Pipeline`, so categorical encoding, imputation, and scaling are learned only from the training fold.
6. The model zoo compares baselines, linear models, tree ensembles, gradient boosting, and stacking ensembles for classification and regression.
7. The artifact layer writes metrics, figures, fitted models, and a markdown report under `protein_variant_ml_outputs/`.

This architecture keeps the biological feature logic separate from modeling, keeps preprocessing inside train-only pipelines, and preserves the test set as the final holdout.

## Libraries used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `joblib`
- `xgboost`
- `lightgbm`
- `shap`

## How to run

```powershell
python .\protein_variant_ml_project.py
```

Optional settings:

```powershell
$env:PROTEIN_VARIANT_ZIP_PATH = "C:\path\to\substitutions_raw_DMS.zip"
$env:PROTEIN_VARIANT_N_JOBS = "-1"
python .\protein_variant_ml_project.py
```

Outputs are written to:

```text
protein_variant_ml_outputs/
```

## Current model lineup

Classification models:

- `majority_baseline`
- `logistic_regression`
- `random_forest`
- `xgboost`
- `lightgbm`
- `stacking_classifier`

Regression models:

- `dummy_regressor`
- `random_forest_regressor`
- `xgboost_regressor`
- `lightgbm_regressor`
- `stacking_regressor`

## Current results snapshot

- Best classifier: `lightgbm`
- Best regressor: `stacking_regressor`
- SHAP is used for tree-based interpretation on the validation split

## Research roadmap

- Replace random splits with position-grouped validation, such as `GroupKFold` or a group-aware train/validation/test split using residue position as the group.
- Add 3D structural features from an AlphaFold or PDB structure, including SASA, RSA, secondary structure, and distance to binding-site residues.
- Add Optuna or `RandomizedSearchCV` around the existing pipelines to tune tree depth, learning rates, regularization, and ensemble size without leaking preprocessing state.

## Why this is useful for medical biology practice

The project teaches several skills that transfer well to molecular and medical bioinformatics:

- variant effect prediction
- biologically informed feature engineering
- classification versus regression framing
- leakage-aware modeling
- interpretable ML for mutation biology

It is still important to keep the task scoped correctly: this dataset measures protein stability, not patient outcomes, so the project is a training ground for molecular ML rather than a direct clinical model.
