from __future__ import annotations

import json
import math
import os
import re
import warnings
import zipfile
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import seaborn as sns
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import StackingClassifier, StackingRegressor
from xgboost import XGBClassifier, XGBRegressor


RANDOM_STATE = 42
WINDOW_SIZE = 2
N_JOBS = int(os.getenv("PROTEIN_VARIANT_N_JOBS", "-1"))
DEFAULT_ZIP_CANDIDATES = [
    Path.cwd() / "substitutions_raw_DMS.zip",
    Path.cwd().parent / "substitutions_raw_DMS.zip",
    Path.home() / "Downloads" / "substitutions_raw_DMS.zip",
]
ZIP_MEMBER = "substitutions_raw_DMS/FECA_ECOLI_Tsuboyama_2023_2D1U.csv"
OUTPUT_DIR = Path.cwd() / "protein_variant_ml_outputs"
MODEL_DIR = OUTPUT_DIR / "models"
FIG_DIR = OUTPUT_DIR / "figures"


AA_PROPERTIES: Dict[str, Dict[str, float | str | bool]] = {
    "A": {"hydrophobicity": 1.8, "volume": 88.6, "charge": 0, "polarity": "nonpolar", "aromatic": False},
    "R": {"hydrophobicity": -4.5, "volume": 173.4, "charge": 1, "polarity": "polar", "aromatic": False},
    "N": {"hydrophobicity": -3.5, "volume": 114.1, "charge": 0, "polarity": "polar", "aromatic": False},
    "D": {"hydrophobicity": -3.5, "volume": 111.1, "charge": -1, "polarity": "polar", "aromatic": False},
    "C": {"hydrophobicity": 2.5, "volume": 108.5, "charge": 0, "polarity": "polar", "aromatic": False},
    "Q": {"hydrophobicity": -3.5, "volume": 143.8, "charge": 0, "polarity": "polar", "aromatic": False},
    "E": {"hydrophobicity": -3.5, "volume": 138.4, "charge": -1, "polarity": "polar", "aromatic": False},
    "G": {"hydrophobicity": -0.4, "volume": 60.1, "charge": 0, "polarity": "nonpolar", "aromatic": False},
    "H": {"hydrophobicity": -3.2, "volume": 153.2, "charge": 0.5, "polarity": "polar", "aromatic": True},
    "I": {"hydrophobicity": 4.5, "volume": 166.7, "charge": 0, "polarity": "nonpolar", "aromatic": False},
    "L": {"hydrophobicity": 3.8, "volume": 166.7, "charge": 0, "polarity": "nonpolar", "aromatic": False},
    "K": {"hydrophobicity": -3.9, "volume": 168.6, "charge": 1, "polarity": "polar", "aromatic": False},
    "M": {"hydrophobicity": 1.9, "volume": 162.9, "charge": 0, "polarity": "nonpolar", "aromatic": False},
    "F": {"hydrophobicity": 2.8, "volume": 189.9, "charge": 0, "polarity": "nonpolar", "aromatic": True},
    "P": {"hydrophobicity": -1.6, "volume": 112.7, "charge": 0, "polarity": "nonpolar", "aromatic": False},
    "S": {"hydrophobicity": -0.8, "volume": 89.0, "charge": 0, "polarity": "polar", "aromatic": False},
    "T": {"hydrophobicity": -0.7, "volume": 116.1, "charge": 0, "polarity": "polar", "aromatic": False},
    "W": {"hydrophobicity": -0.9, "volume": 227.8, "charge": 0, "polarity": "nonpolar", "aromatic": True},
    "Y": {"hydrophobicity": -1.3, "volume": 193.6, "charge": 0, "polarity": "polar", "aromatic": True},
    "V": {"hydrophobicity": 4.2, "volume": 140.0, "charge": 0, "polarity": "nonpolar", "aromatic": False},
}

CONSERVATIVE_GROUPS = [
    set("AVLIM"),
    set("FYW"),
    set("STNQ"),
    set("KRH"),
    set("DE"),
    set("GP"),
    set("C"),
]


@dataclass
class SplitData:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def load_dataset() -> pd.DataFrame:
    zip_path = resolve_zip_path()
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(ZIP_MEMBER) as raw_file:
            return pd.read_csv(TextIOWrapper(raw_file, encoding="utf-8"))


def resolve_zip_path() -> Path:
    env_path = os.getenv("PROTEIN_VARIANT_ZIP_PATH")
    candidates = [Path(env_path).expanduser()] if env_path else []
    candidates.extend(DEFAULT_ZIP_CANDIDATES)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not find substitutions_raw_DMS.zip. Set PROTEIN_VARIANT_ZIP_PATH "
        f"or place the archive in one of these locations:\n{searched}"
    )


def parse_mutation(mut_type: str) -> Tuple[str, int, str]:
    match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mut_type)
    if not match:
        raise ValueError(f"Unexpected mutation format: {mut_type}")
    wt_aa, pos, mut_aa = match.groups()
    return wt_aa, int(pos), mut_aa


def is_conservative_change(wt_aa: str, mut_aa: str) -> bool:
    for group in CONSERVATIVE_GROUPS:
        if wt_aa in group and mut_aa in group:
            return True
    return False


def add_sequence_window_features(frame: pd.DataFrame, window_size: int = WINDOW_SIZE) -> pd.DataFrame:
    offsets = list(range(-window_size, window_size + 1))
    aa_keys = sorted(AA_PROPERTIES)

    for offset in offsets:
        frame[f"context_{offset:+d}"] = [
            seq[pos - 1 + offset] if 0 <= pos - 1 + offset < len(seq) else "PAD"
            for seq, pos in zip(frame["aa_seq"], frame["position"])
        ]

    for aa in aa_keys:
        frame[f"window_count_{aa}"] = [
            sum(1 for idx in range(pos - 1 - window_size, pos + window_size) if 0 <= idx < len(seq) and seq[idx] == aa)
            for seq, pos in zip(frame["aa_seq"], frame["position"])
        ]

    return frame


def build_modeling_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["mut_type", "aa_seq", "Stabilizing_mut", "ddG_ML_float", "dG_ML_float", "domain_length"]
    df = raw_df.loc[raw_df["mut_class"] == "single", cols].copy()
    df["Stabilizing_mut"] = df["Stabilizing_mut"].replace("-", np.nan)
    df["ddG_ML_float"] = pd.to_numeric(df["ddG_ML_float"], errors="coerce")
    df["dG_ML_float"] = pd.to_numeric(df["dG_ML_float"], errors="coerce")
    df["domain_length"] = pd.to_numeric(df["domain_length"], errors="coerce")
    df = df.dropna(subset=["Stabilizing_mut", "ddG_ML_float", "dG_ML_float", "aa_seq", "mut_type", "domain_length"]).reset_index(drop=True)

    parsed = df["mut_type"].apply(parse_mutation)
    df["wt_aa"] = parsed.apply(lambda item: item[0])
    df["position"] = parsed.apply(lambda item: item[1])
    df["mut_aa"] = parsed.apply(lambda item: item[2])

    df["label"] = df["Stabilizing_mut"].astype(str).str.strip().str.lower().map({"true": 1, "false": 0})
    df["substitution_pair"] = df["wt_aa"] + "->" + df["mut_aa"]

    df["wt_hydrophobicity"] = df["wt_aa"].map(lambda aa: AA_PROPERTIES[aa]["hydrophobicity"])
    df["mut_hydrophobicity"] = df["mut_aa"].map(lambda aa: AA_PROPERTIES[aa]["hydrophobicity"])
    df["delta_hydrophobicity"] = df["mut_hydrophobicity"] - df["wt_hydrophobicity"]

    df["wt_volume"] = df["wt_aa"].map(lambda aa: AA_PROPERTIES[aa]["volume"])
    df["mut_volume"] = df["mut_aa"].map(lambda aa: AA_PROPERTIES[aa]["volume"])
    df["delta_volume"] = df["mut_volume"] - df["wt_volume"]

    df["wt_charge"] = df["wt_aa"].map(lambda aa: AA_PROPERTIES[aa]["charge"])
    df["mut_charge"] = df["mut_aa"].map(lambda aa: AA_PROPERTIES[aa]["charge"])
    df["delta_charge"] = df["mut_charge"] - df["wt_charge"]

    df["wt_polarity"] = df["wt_aa"].map(lambda aa: AA_PROPERTIES[aa]["polarity"])
    df["mut_polarity"] = df["mut_aa"].map(lambda aa: AA_PROPERTIES[aa]["polarity"])
    df["polarity_change"] = np.where(df["wt_polarity"] == df["mut_polarity"], "same", "changed")

    df["wt_aromatic"] = df["wt_aa"].map(lambda aa: AA_PROPERTIES[aa]["aromatic"]).astype(int)
    df["mut_aromatic"] = df["mut_aa"].map(lambda aa: AA_PROPERTIES[aa]["aromatic"]).astype(int)
    df["aromatic_change"] = df["mut_aromatic"] - df["wt_aromatic"]
    df["conservative_change"] = [is_conservative_change(wt, mut) for wt, mut in zip(df["wt_aa"], df["mut_aa"])]
    df["position_frac"] = df["position"] / df["domain_length"]

    df = add_sequence_window_features(df)

    checks = {
        "single_mutants_only": bool((~df["mut_type"].str.contains(":")).all()),
        "no_missing_labels": bool(df["label"].notna().all()),
        "position_in_bounds": bool(((df["position"] >= 1) & (df["position"] <= df["aa_seq"].str.len())).all()),
    }
    df.attrs["data_checks"] = checks
    return df


def split_dataset(df: pd.DataFrame) -> SplitData:
    train_valid, test = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    train, valid = train_test_split(
        train_valid,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=train_valid["label"],
    )
    return SplitData(train=train.reset_index(drop=True), valid=valid.reset_index(drop=True), test=test.reset_index(drop=True))


def get_feature_lists(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    categorical = [
        "wt_aa",
        "mut_aa",
        "substitution_pair",
        "wt_polarity",
        "mut_polarity",
        "polarity_change",
        "conservative_change",
    ] + [col for col in df.columns if col.startswith("context_")]
    numeric = [
        "position",
        "domain_length",
        "position_frac",
        "wt_hydrophobicity",
        "mut_hydrophobicity",
        "delta_hydrophobicity",
        "wt_volume",
        "mut_volume",
        "delta_volume",
        "wt_charge",
        "mut_charge",
        "delta_charge",
        "wt_aromatic",
        "mut_aromatic",
        "aromatic_change",
    ] + [col for col in df.columns if col.startswith("window_count_")]
    return categorical, numeric


def build_logistic_pipeline(categorical: Iterable[str], numeric: Iterable[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical)),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric),
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )


def build_forest_classifier(categorical: Iterable[str], numeric: Iterable[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical)),
            ("num", SimpleImputer(strategy="median"), list(numeric)),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                ),
            ),
        ]
    )


def build_forest_regressor(categorical: Iterable[str], numeric: Iterable[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical)),
            ("num", SimpleImputer(strategy="median"), list(numeric)),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                ),
            ),
        ]
    )


def build_xgboost_classifier(categorical: Iterable[str], numeric: Iterable[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical)),
            ("num", SimpleImputer(strategy="median"), list(numeric)),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                    eval_metric="logloss",
                ),
            ),
        ]
    )


def build_lightgbm_classifier(categorical: Iterable[str], numeric: Iterable[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical)),
            ("num", SimpleImputer(strategy="median"), list(numeric)),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                    force_col_wise=True,
                    verbosity=-1,
                ),
            ),
        ]
    )


def build_xgboost_regressor(categorical: Iterable[str], numeric: Iterable[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical)),
            ("num", SimpleImputer(strategy="median"), list(numeric)),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBRegressor(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                ),
            ),
        ]
    )


def build_lightgbm_regressor(categorical: Iterable[str], numeric: Iterable[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical)),
            ("num", SimpleImputer(strategy="median"), list(numeric)),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LGBMRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                    force_col_wise=True,
                    verbosity=-1,
                ),
            ),
        ]
    )


def build_stacking_classifier(categorical: Iterable[str], numeric: Iterable[str]) -> StackingClassifier:
    estimators = [
        ("logistic", build_logistic_pipeline(categorical, numeric)),
        ("random_forest", build_forest_classifier(categorical, numeric)),
        ("xgboost", build_xgboost_classifier(categorical, numeric)),
        ("lightgbm", build_lightgbm_classifier(categorical, numeric)),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
        cv=5,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=N_JOBS,
    )


def build_stacking_regressor(categorical: Iterable[str], numeric: Iterable[str]) -> StackingRegressor:
    estimators = [
        ("random_forest", build_forest_regressor(categorical, numeric)),
        ("xgboost", build_xgboost_regressor(categorical, numeric)),
        ("lightgbm", build_lightgbm_regressor(categorical, numeric)),
    ]
    return StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(alphas=(0.1, 1.0, 10.0)),
        cv=5,
        n_jobs=N_JOBS,
        passthrough=False,
    )


def classification_metrics(model: Pipeline | DummyClassifier, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds = model.predict(X)
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X)[:, 1]
        else:
            scores = preds.astype(float)
    return {
        "f1": f1_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "roc_auc": roc_auc_score(y, scores),
        "pr_auc": average_precision_score(y, scores),
    }


def regression_metrics(model: Pipeline | DummyRegressor, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds = model.predict(X)
    return {
        "mae": mean_absolute_error(y, preds),
        "rmse": math.sqrt(mean_squared_error(y, preds)),
    }


def ensure_dirs() -> None:
    for path in [OUTPUT_DIR, MODEL_DIR, FIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def plot_classification_figures(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=axes[0], colorbar=False)
    axes[0].set_title("Confusion Matrix")
    RocCurveDisplay.from_predictions(y_test, probs, ax=axes[1])
    axes[1].set_title("ROC Curve")
    PrecisionRecallDisplay.from_predictions(y_test, probs, ax=axes[2])
    axes[2].set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "classification_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def get_logistic_feature_importance(model: Pipeline, top_n: int = 15) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    coeffs = model.named_steps["model"].coef_[0]
    frame = pd.DataFrame({"feature": feature_names, "coefficient": coeffs})
    frame["abs_coefficient"] = frame["coefficient"].abs()
    return frame.sort_values("abs_coefficient", ascending=False).head(top_n)


def get_permutation_importance_table(model: Pipeline, X: pd.DataFrame, y: pd.Series, top_n: int = 15) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = permutation_importance(model, X, y, n_repeats=20, random_state=RANDOM_STATE, scoring="f1", n_jobs=N_JOBS)
    frame = pd.DataFrame({"feature": X.columns, "importance_mean": result.importances_mean, "importance_std": result.importances_std})
    return frame.sort_values("importance_mean", ascending=False).head(top_n)


def plot_feature_importance(importance_df: pd.DataFrame, value_col: str, output_name: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df, y="feature", x=value_col, ax=ax, color="#3b82f6")
    ax.set_title(title)
    ax.set_xlabel(value_col.replace("_", " ").title())
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIG_DIR / output_name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def get_shap_importance_table(model: Pipeline, X: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    transformed = model.named_steps["preprocessor"].transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    estimator = model.named_steps["model"]
    explainer = shap.TreeExplainer(estimator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(transformed)
    if isinstance(shap_values, list):
        shap_array = np.asarray(shap_values[-1])
    else:
        shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        if shap_array.shape[1] == len(feature_names):
            shap_array = shap_array[:, :, -1]
        elif shap_array.shape[2] == len(feature_names):
            shap_array = shap_array[:, -1, :]
        else:
            raise ValueError(f"Unexpected SHAP array shape: {shap_array.shape}")
    if shap_array.ndim != 2:
        raise ValueError(f"Expected 2D SHAP array, got shape {shap_array.shape}")
    mean_abs_shap = np.abs(shap_array).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
    )


def create_biological_examples(df: pd.DataFrame, probs: np.ndarray, pred_labels: np.ndarray) -> Dict[str, List[Dict[str, float | str]]]:
    scored = df.copy()
    scored["pred_prob_stabilizing"] = probs
    scored["pred_label"] = pred_labels

    stabilizing = (
        scored.loc[scored["pred_label"] == 1, ["mut_type", "substitution_pair", "delta_hydrophobicity", "delta_charge", "pred_prob_stabilizing", "ddG_ML_float"]]
        .sort_values("pred_prob_stabilizing", ascending=False)
        .head(5)
    )
    destabilizing = (
        scored.loc[scored["pred_label"] == 0, ["mut_type", "substitution_pair", "delta_hydrophobicity", "delta_charge", "pred_prob_stabilizing", "ddG_ML_float"]]
        .sort_values("pred_prob_stabilizing", ascending=True)
        .head(5)
    )
    return {
        "high_confidence_stabilizing": stabilizing.to_dict(orient="records"),
        "high_confidence_destabilizing": destabilizing.to_dict(orient="records"),
    }


def write_markdown_report(
    classification_results: Dict[str, Dict[str, float]],
    regression_results: Dict[str, Dict[str, float]],
    data_checks: Dict[str, bool],
    duplicate_cross_split: bool,
    logistic_importance: pd.DataFrame,
    forest_importance: pd.DataFrame,
    shap_importance: pd.DataFrame | None,
    biological_examples: Dict[str, List[Dict[str, float | str]]],
    best_model_name: str,
) -> None:
    report_path = OUTPUT_DIR / "README_variant_effect_project.md"
    top_logistic = dataframe_to_markdown(logistic_importance[["feature", "coefficient"]])
    top_forest = dataframe_to_markdown(forest_importance)
    top_shap = dataframe_to_markdown(shap_importance) if shap_importance is not None else "_SHAP not generated for the selected model._"

    lines = [
        "# Variant-Effect ML Project",
        "",
        "## Biological Question",
        "",
        "Can we predict whether a single amino-acid substitution in `FECA_ECOLI_Tsuboyama_2023_2D1U` is stabilizing using only mutation identity, amino-acid chemistry, and local sequence context?",
        "",
        "## Dataset Scope",
        "",
        "- Source: `FECA_ECOLI_Tsuboyama_2023_2D1U.csv` inside `substitutions_raw_DMS.zip`",
        "- Modeling subset: single mutants only",
        "- Classification label: `Stabilizing_mut`",
        "- Regression target: `ddG_ML_float`",
        "",
        "## Data Checks",
        "",
        *(f"- {name.replace('_', ' ').title()}: {'PASS' if passed else 'FAIL'}" for name, passed in data_checks.items()),
        f"- Duplicate mutation rows crossing splits: {'YES' if duplicate_cross_split else 'NO'}",
        "",
        "## Classification Results",
        "",
        dataframe_to_markdown(pd.DataFrame(classification_results).T.round(4)),
        "",
        f"Best validation model: **{best_model_name}**",
        "",
        "## Regression Results",
        "",
        dataframe_to_markdown(pd.DataFrame(regression_results).T.round(4)),
        "",
        "## Interpretability",
        "",
        "### Logistic Regression Top Coefficients",
        "",
        top_logistic,
        "",
        "### Validation Permutation Importance",
        "",
        top_forest,
        "",
        "### SHAP Importance",
        "",
        top_shap,
        "",
        "## Biological Interpretation Notes",
        "",
        "- Mutation identity and local sequence context both carry predictive signal, which is consistent with real variant-effect workflows in molecular biology.",
        "- Large physicochemical shifts such as strong charge changes or hydrophobicity shifts are often enriched among destabilizing predictions.",
        "- This project is useful practice for medical biology pipelines, but it is not a clinical pathogenicity model because the target comes from experimental protein stability rather than patient outcomes.",
        "",
        "## Example Predictions",
        "",
        "```json",
        json.dumps(biological_examples, indent=2),
        "```",
        "",
        "## Generated Artifacts",
        "",
        "- `figures/classification_summary.png`",
        "- `figures/logistic_feature_importance.png`",
        "- `figures/validation_permutation_importance.png`",
        "- `figures/shap_feature_importance.png`",
        "- `models/best_classifier.joblib`",
        "- `models/regressor.joblib`",
        "- `metrics_summary.json`",
        "",
        "## Next Extensions",
        "",
        "- Add double-mutant rows to study epistasis.",
        "- Compare position-only versus chemistry-aware features.",
        "- Add position-grouped cross-validation to estimate out-of-position generalization.",
        "- Add 3D structural features such as SASA, RSA, secondary structure, and binding-site distance.",
        "- Add Optuna or RandomizedSearchCV inside the existing preprocessing pipelines.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    render_df = df.copy()
    render_df = render_df.fillna("")
    headers = [str(col) for col in render_df.columns]
    rows = [[str(value) for value in row] for row in render_df.to_numpy().tolist()]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


def main() -> None:
    ensure_dirs()
    raw_df = load_dataset()
    modeling_df = build_modeling_table(raw_df)
    splits = split_dataset(modeling_df)

    duplicate_membership: Dict[str, set[str]] = {}
    for split_name, frame in [("train", splits.train), ("valid", splits.valid), ("test", splits.test)]:
        for mut in frame["mut_type"]:
            duplicate_membership.setdefault(mut, set()).add(split_name)
    duplicate_cross_split = any(len(splits_seen) > 1 for splits_seen in duplicate_membership.values())

    categorical_features, numeric_features = get_feature_lists(modeling_df)
    feature_columns = categorical_features + numeric_features

    X_train = splits.train[feature_columns]
    y_train = splits.train["label"]
    X_valid = splits.valid[feature_columns]
    y_valid = splits.valid["label"]
    X_test = splits.test[feature_columns]
    y_test = splits.test["label"]

    classifiers = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": build_logistic_pipeline(categorical_features, numeric_features),
        "random_forest": build_forest_classifier(categorical_features, numeric_features),
        "xgboost": build_xgboost_classifier(categorical_features, numeric_features),
        "lightgbm": build_lightgbm_classifier(categorical_features, numeric_features),
        "stacking_classifier": build_stacking_classifier(categorical_features, numeric_features),
    }

    classification_results: Dict[str, Dict[str, float]] = {}
    fitted_classifiers: Dict[str, Pipeline | DummyClassifier] = {}
    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        fitted_classifiers[name] = model
        metrics = classification_metrics(model, X_valid, y_valid)
        classification_results[f"{name}_valid"] = metrics

    best_model_name = max(
        ["logistic_regression", "random_forest", "xgboost", "lightgbm", "stacking_classifier"],
        key=lambda name: (classification_results[f"{name}_valid"]["f1"], classification_results[f"{name}_valid"]["roc_auc"]),
    )
    best_model = fitted_classifiers[best_model_name]
    classification_results[f"{best_model_name}_test"] = classification_metrics(best_model, X_test, y_test)
    classification_results["majority_baseline_test"] = classification_metrics(fitted_classifiers["majority_baseline"], X_test, y_test)

    regression_models = {
        "dummy_regressor": DummyRegressor(strategy="mean"),
        "random_forest_regressor": build_forest_regressor(categorical_features, numeric_features),
        "xgboost_regressor": build_xgboost_regressor(categorical_features, numeric_features),
        "lightgbm_regressor": build_lightgbm_regressor(categorical_features, numeric_features),
        "stacking_regressor": build_stacking_regressor(categorical_features, numeric_features),
    }
    regression_results: Dict[str, Dict[str, float]] = {}
    y_train_reg = splits.train["ddG_ML_float"]
    y_valid_reg = splits.valid["ddG_ML_float"]
    y_test_reg = splits.test["ddG_ML_float"]

    fitted_regressors: Dict[str, Pipeline | DummyRegressor] = {}
    for name, model in regression_models.items():
        model.fit(X_train, y_train_reg)
        fitted_regressors[name] = model
        regression_results[f"{name}_valid"] = regression_metrics(model, X_valid, y_valid_reg)

    best_regressor_name = min(
        ["random_forest_regressor", "xgboost_regressor", "lightgbm_regressor", "stacking_regressor"],
        key=lambda name: (regression_results[f"{name}_valid"]["mae"], regression_results[f"{name}_valid"]["rmse"]),
    )
    regression_results[f"{best_regressor_name}_test"] = regression_metrics(fitted_regressors[best_regressor_name], X_test, y_test_reg)
    regression_results["dummy_regressor_test"] = regression_metrics(fitted_regressors["dummy_regressor"], X_test, y_test_reg)

    plot_classification_figures(best_model, X_test, y_test)

    logistic_importance = get_logistic_feature_importance(fitted_classifiers["logistic_regression"])
    plot_feature_importance(logistic_importance.head(15), "coefficient", "logistic_feature_importance.png", "Logistic Regression Coefficients")

    forest_importance = get_permutation_importance_table(best_model, X_valid, y_valid)
    plot_feature_importance(forest_importance, "importance_mean", "validation_permutation_importance.png", "Validation Permutation Importance")

    shap_importance = None
    if best_model_name in {"random_forest", "xgboost", "lightgbm"}:
        shap_importance = get_shap_importance_table(best_model, X_valid)
        plot_feature_importance(shap_importance, "mean_abs_shap", "shap_feature_importance.png", "SHAP Feature Importance")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probs = best_model.predict_proba(X_test)[:, 1]
        preds = best_model.predict(X_test)
    biological_examples = create_biological_examples(splits.test, probs, preds)

    joblib.dump(best_model, MODEL_DIR / "best_classifier.joblib")
    joblib.dump(fitted_regressors[best_regressor_name], MODEL_DIR / "regressor.joblib")

    metrics_summary = {
        "classification": classification_results,
        "regression": regression_results,
        "data_checks": modeling_df.attrs["data_checks"],
        "best_model_name": best_model_name,
        "best_regressor_name": best_regressor_name,
    }
    (OUTPUT_DIR / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")

    write_markdown_report(
        classification_results=classification_results,
        regression_results=regression_results,
        data_checks=modeling_df.attrs["data_checks"],
        duplicate_cross_split=duplicate_cross_split,
        logistic_importance=logistic_importance,
        forest_importance=forest_importance,
        shap_importance=shap_importance,
        biological_examples=biological_examples,
        best_model_name=best_model_name,
    )

    print(json.dumps(metrics_summary, indent=2))


if __name__ == "__main__":
    main()
