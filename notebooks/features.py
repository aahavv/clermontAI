"""
features.py — portable feature engineering for ClermontAI.

Mirrors Phase 11 of the EDA/FE notebook exactly, but split into:

    fit_feature_pipeline(X_raw, y)  -> (state, X_fe)   # learns + transforms train
    apply_feature_pipeline(X_raw, state) -> X_fe       # transforms anything else

`state` is a plain dict holding every learned object. It is the ONLY thing
that has to travel to serving time alongside the model + calibrator.

Phases 1-10 of the notebook were pure analysis (plots, AUC tables, candidate
pool exploration) and produced no transformation of the saved matrices, so
nothing from them needs to be reproduced here. Every transform that actually
touched X_train/X_test lives in Phase 11 (cells 60-80), and all of it is below.

Order of operations is identical to the notebook:
    1. DAYS_EMPLOYED sentinel  -> flag + NaN
    2. EXT_SOURCE_* missing flags
    3. engineered ratios (+inf/-inf -> NaN)
    4. Yeo-Johnson on the money columns
    5. drop no-signal categoricals
    6. bucket rare categories -> 'Other'
    7. target-encode categoricals
    8. median imputation + final column order
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, TargetEncoder
from sklearn.impute import SimpleImputer

# ----------------------------------------------------------------------------
# Constants the notebook hardcoded (these are knowledge, not learned state).
# ----------------------------------------------------------------------------
SENTINEL = 365243                       # DAYS_EMPLOYED stand-in for "never employed"
EXT_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
SKEW_CANDIDATES = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "CREDIT_GOODS_RATIO", "DEBT_SERVICE_BURDEN", "ANNUITY_CREDIT_RATIO",
]
DROP_CATS = ["WEEKDAY_APPR_PROCESS_START", "FLAG_OWN_REALTY", "NAME_TYPE_SUITE"]
MIN_COUNT = 100                         # rare-category cutoff


# ============================================================================
# STATELESS PIECES — pure formulas, need nothing from training.
# Used identically by fit and apply.
# ============================================================================
def _add_sentinel_flag(X):
    X = X.copy()
    X["DAYS_EMPLOYED_was_sentinel"] = (X["DAYS_EMPLOYED"] == SENTINEL).astype(int)
    X.loc[X["DAYS_EMPLOYED"] == SENTINEL, "DAYS_EMPLOYED"] = np.nan
    return X


def _add_ext_missing_flags(X):
    X = X.copy()
    for col in EXT_COLS:
        if col in X.columns:
            X[f"{col}_was_missing"] = X[col].isna().astype(int)
    return X


def _add_ratios(X):
    X = X.copy()
    g   = X.get("AMT_GOODS_PRICE")
    c   = X.get("AMT_CREDIT")
    a   = X.get("AMT_ANNUITY")
    inc = X.get("AMT_INCOME_TOTAL")

    if c is not None and g is not None:
        X["CREDIT_GOODS_GAP"]   = c - g
        X["CREDIT_GOODS_RATIO"] = c / g
    if a is not None and inc is not None:
        X["DEBT_SERVICE_BURDEN"] = a / inc
    if a is not None and c is not None:
        X["ANNUITY_CREDIT_RATIO"] = a / c

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    return X


def _stateless_block(X):
    """Everything that needs no learned state, in notebook order."""
    X = _add_sentinel_flag(X)
    X = _add_ext_missing_flags(X)
    X = _add_ratios(X)
    return X


# ============================================================================
# FIT — learn every stateful object once, on the training split.
# ============================================================================
def fit_feature_pipeline(X_raw, y):
    """
    Returns (state, X_fe).

    state keys:
        drop_cats        list[str]            columns removed before encoding
        skew_cols        list[str]            columns Yeo-Johnson was fit on
        power_transformer  fitted PowerTransformer (holds the per-col lambdas)
        frequent_levels  dict[str, set]       per-cat whitelist of levels to KEEP
        cat_cols         list[str]            categoricals that got target-encoded
        target_encoder   fitted TargetEncoder (holds per-category rates)
        imputer          fitted SimpleImputer (holds per-column medians)
        feature_order    list[str]            final column order fed to the model
    """
    state = {}
    X = X_raw.copy()

    # ---- 1-3. stateless block (sentinel, ext flags, ratios) ----
    X = _stateless_block(X)

    # ---- 5. drop the no-signal categoricals (note: before bucket/encode) ----
    state["drop_cats"] = [c for c in DROP_CATS if c in X.columns]
    X = X.drop(columns=state["drop_cats"])

    # ---- 4. Yeo-Johnson on the money columns (fit lambdas on TRAIN) ----
    # NB: in the notebook this ran before the drop, but order is independent —
    # the skew cols and drop cols are disjoint sets, so result is identical.
    state["skew_cols"] = [c for c in SKEW_CANDIDATES if c in X.columns]
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    pt.fit(X[state["skew_cols"]])
    X[state["skew_cols"]] = pt.transform(X[state["skew_cols"]])
    state["power_transformer"] = pt

    # ---- 6. bucket rare categories -> 'Other' (whitelist learned on TRAIN) ----
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    state["frequent_levels"] = {}
    for col in cat_cols:
        vc = X[col].value_counts()
        state["frequent_levels"][col] = set(vc[vc >= MIN_COUNT].index)
    X = _apply_rare(X, state["frequent_levels"])

    # ---- 7. target-encode categoricals (cross-fitted on TRAIN) ----
    state["cat_cols"] = X.select_dtypes(include=["object"]).columns.tolist()
    te = TargetEncoder(target_type="binary", smooth="auto", random_state=42)
    # fit_transform uses internal cross-fitting so train rows don't see own target
    X[state["cat_cols"]] = te.fit_transform(X[state["cat_cols"]], y)
    state["target_encoder"] = te

    # ---- 8. median imputation + freeze column order ----
    feature_cols = X.columns.tolist()
    imp = SimpleImputer(strategy="median")
    X_arr = imp.fit_transform(X[feature_cols])
    state["imputer"] = imp
    state["feature_order"] = feature_cols

    X = pd.DataFrame(X_arr, columns=feature_cols, index=X.index)
    return state, X


# ============================================================================
# APPLY — read from state only. No .fit(), no .median(), no learning.
# ============================================================================
def apply_feature_pipeline(X_raw, state):
    X = X_raw.copy()

    # 1-3. stateless block
    X = _stateless_block(X)

    # 5. drop the same categoricals
    X = X.drop(columns=[c for c in state["drop_cats"] if c in X.columns])

    # 4. Yeo-Johnson with the TRAINED lambdas
    X[state["skew_cols"]] = state["power_transformer"].transform(X[state["skew_cols"]])

    # 6. bucket rare via the TRAIN whitelist (unseen levels -> 'Other')
    X = _apply_rare(X, state["frequent_levels"])

    # 7. target-encode with the TRAINED rates (unseen cats -> learned default)
    X[state["cat_cols"]] = state["target_encoder"].transform(X[state["cat_cols"]])

    # 8. impute with TRAIN medians, then enforce exact column set + order
    #    Add any column the model expects but this input lacks (filled by the
    #    imputer's stored median), and drop anything extra.
    for col in state["feature_order"]:
        if col not in X.columns:
            X[col] = np.nan
    X = X[state["feature_order"]]
    X_arr = state["imputer"].transform(X)
    X = pd.DataFrame(X_arr, columns=state["feature_order"], index=X.index)
    return X


# ----------------------------------------------------------------------------
# shared helper
# ----------------------------------------------------------------------------
def _apply_rare(X, frequent_levels):
    X = X.copy()
    for col, keep in frequent_levels.items():
        if col in X.columns:
            X[col] = X[col].where(X[col].isin(keep), other="Other")
    return X
