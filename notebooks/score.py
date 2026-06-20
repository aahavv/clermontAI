r"""
score.py — the one place that wires the four artifacts together.

Location:  C:\clermontAI\notebooks\score.py   (next to features.py)
Artifacts: C:\clermontAI\models\            (the four files below)

Loads the four files once and exposes a single function:
    score(raw_dataframe) -> calibrated default probabilities

Your dashboard / API / notebook just does:
    from score import score
    probs = score(applicants_df)

Running `python score.py` directly does a built-in self-test that needs NO
csv file — it scores one dummy applicant so you get a success signal.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import apply_feature_pipeline   # features.py sits next to this file

# ---- locate the models folder relative to this file ----
HERE = Path(__file__).resolve().parent           # ...\clermontAI\notebooks
MODELS = HERE.parent / "models"                   # ...\clermontAI\models

# ---- load the four artifacts ONCE at import time ----
_model      = joblib.load(MODELS / "lgbm_tuned.joblib")
_calibrator = joblib.load(MODELS / "calibrator.joblib")
_state      = joblib.load(MODELS / "transformers.joblib")
_contract   = json.loads((MODELS / "model_contract.json").read_text())

_FEATURE_ORDER     = _contract["feature_order"]
_RAW_INPUT_COLUMNS = _contract["raw_input_columns"]


def score(raw: pd.DataFrame) -> np.ndarray:
    """
    raw: DataFrame of applicants in ORIGINAL column format (same columns as
         application_train.csv minus TARGET/SK_ID_CURR). One row or many.
    returns: 1D numpy array of calibrated default probabilities (0..1).
    """
    raw = raw.drop(columns=[c for c in ["SK_ID_CURR", "TARGET"] if c in raw.columns])

    X = apply_feature_pipeline(raw, _state)   # translate raw -> 125 model features
    X = X[_FEATURE_ORDER]                      # exact order the model expects

    raw_log_odds = _model.predict(X, raw_score=True)            # model -> log-odds
    calibrated = _calibrator.predict_proba(                     # Platt -> probability
        raw_log_odds.reshape(-1, 1))[:, 1]
    return calibrated


# ---------------------------------------------------------------------------
# self-test: needs no CSV. Builds one dummy applicant and scores it.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Artifacts loaded OK from:", MODELS)
    print(f"  model expects {len(_FEATURE_ORDER)} features")
    print(f"  contract version {_contract.get('version')} "
          f"({_contract.get('date','')[:10]})")

    dummy = {c: np.nan for c in _RAW_INPUT_COLUMNS}
    dummy.update({
        "AMT_INCOME_TOTAL": 135000.0,
        "AMT_CREDIT":       512000.0,
        "AMT_ANNUITY":      24700.0,
        "AMT_GOODS_PRICE":  450000.0,
        "DAYS_BIRTH":       -15000,
        "DAYS_EMPLOYED":    365243,
        "EXT_SOURCE_2":     0.42,
        "EXT_SOURCE_3":     0.31,
    })
    prob = score(pd.DataFrame([dummy]))[0]
    print(f"\nSELF-TEST OK -> dummy applicant default probability: {prob:.1%}")
    print("score() is working. Import it with:  from score import score")
