"""
api.py — FastAPI serving layer for ClermontAI credit risk model.

Endpoints:
    GET  /health        — liveness check
    GET  /contract      — model metadata (version, ROC-AUC, feature count)
    POST /score         — score one applicant, returns default probability + risk tier
    POST /score/batch   — score a list of applicants
    POST /explain       — score + top-N SHAP feature contributions

Run with:
    cd C:\\clermontAI
    venv_new\\Scripts\\python.exe -m uvicorn app.api:app --reload --port 8000
"""

import json
import sys
import os
from pathlib import Path

import shap
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator
from typing import Optional, List

# Add notebooks/ to path so we can import score + features
NOTEBOOKS = Path(__file__).resolve().parent.parent / "notebooks"
sys.path.insert(0, str(NOTEBOOKS))

from score import score, _contract, _RAW_INPUT_COLUMNS, _model, _state  # noqa: E402
from features import apply_feature_pipeline  # noqa: E402

# TreeExplainer with tree_path_dependent (default) only supports model_output="raw".
# We compute log-odds SHAP values and convert each to a probability delta at display time.
_explainer = shap.TreeExplainer(_model)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ClermontAI Credit Risk API",
    description="Credit default probability scoring for thin-file borrowers.",
    version=_contract.get("version", "1.0.0"),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ApplicantIn(BaseModel):
    """
    All fields are optional — missing values are imputed by the pipeline.
    Categorical fields accept the raw string values from the Home Credit dataset.
    Numeric DAYS_* fields are negative integers (days before application).
    """
    # Categorical
    NAME_CONTRACT_TYPE: Optional[str] = None    # "Cash loans" | "Revolving loans"
    CODE_GENDER: Optional[str] = None           # "M" | "F"
    FLAG_OWN_CAR: Optional[str] = None          # "Y" | "N"
    FLAG_OWN_REALTY: Optional[str] = None       # "Y" | "N"
    NAME_TYPE_SUITE: Optional[str] = None
    NAME_INCOME_TYPE: Optional[str] = None
    NAME_EDUCATION_TYPE: Optional[str] = None
    NAME_FAMILY_STATUS: Optional[str] = None
    NAME_HOUSING_TYPE: Optional[str] = None
    OCCUPATION_TYPE: Optional[str] = None
    ORGANIZATION_TYPE: Optional[str] = None
    WEEKDAY_APPR_PROCESS_START: Optional[str] = None
    FONDKAPREMONT_MODE: Optional[str] = None
    HOUSETYPE_MODE: Optional[str] = None
    WALLSMATERIAL_MODE: Optional[str] = None
    EMERGENCYSTATE_MODE: Optional[str] = None

    # Financial
    CNT_CHILDREN: Optional[float] = None
    AMT_INCOME_TOTAL: Optional[float] = None
    AMT_CREDIT: Optional[float] = None
    AMT_ANNUITY: Optional[float] = None
    AMT_GOODS_PRICE: Optional[float] = None
    CNT_FAM_MEMBERS: Optional[float] = None

    # Days (negative integers — days before loan application)
    DAYS_BIRTH: Optional[float] = None
    DAYS_EMPLOYED: Optional[float] = None       # 365243 = never employed sentinel
    DAYS_REGISTRATION: Optional[float] = None
    DAYS_ID_PUBLISH: Optional[float] = None
    DAYS_LAST_PHONE_CHANGE: Optional[float] = None
    OWN_CAR_AGE: Optional[float] = None

    # External credit bureau scores (0–1 scale, strongest predictors)
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: Optional[float] = None
    EXT_SOURCE_3: Optional[float] = None

    # Flags
    FLAG_MOBIL: Optional[float] = None
    FLAG_EMP_PHONE: Optional[float] = None
    FLAG_WORK_PHONE: Optional[float] = None
    FLAG_CONT_MOBILE: Optional[float] = None
    FLAG_PHONE: Optional[float] = None
    FLAG_EMAIL: Optional[float] = None

    # Region
    REGION_POPULATION_RELATIVE: Optional[float] = None
    REGION_RATING_CLIENT: Optional[float] = None
    REGION_RATING_CLIENT_W_CITY: Optional[float] = None
    HOUR_APPR_PROCESS_START: Optional[float] = None
    REG_REGION_NOT_LIVE_REGION: Optional[float] = None
    REG_REGION_NOT_WORK_REGION: Optional[float] = None
    LIVE_REGION_NOT_WORK_REGION: Optional[float] = None
    REG_CITY_NOT_LIVE_CITY: Optional[float] = None
    REG_CITY_NOT_WORK_CITY: Optional[float] = None
    LIVE_CITY_NOT_WORK_CITY: Optional[float] = None

    # Social circle observations
    OBS_30_CNT_SOCIAL_CIRCLE: Optional[float] = None
    DEF_30_CNT_SOCIAL_CIRCLE: Optional[float] = None
    OBS_60_CNT_SOCIAL_CIRCLE: Optional[float] = None
    DEF_60_CNT_SOCIAL_CIRCLE: Optional[float] = None

    # Bureau inquiry counts
    AMT_REQ_CREDIT_BUREAU_HOUR: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_DAY: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_WEEK: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_MON: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_QRT: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_YEAR: Optional[float] = None

    # Document flags (FLAG_DOCUMENT_2 through FLAG_DOCUMENT_21)
    FLAG_DOCUMENT_2: Optional[float] = None
    FLAG_DOCUMENT_3: Optional[float] = None
    FLAG_DOCUMENT_4: Optional[float] = None
    FLAG_DOCUMENT_5: Optional[float] = None
    FLAG_DOCUMENT_6: Optional[float] = None
    FLAG_DOCUMENT_7: Optional[float] = None
    FLAG_DOCUMENT_8: Optional[float] = None
    FLAG_DOCUMENT_9: Optional[float] = None
    FLAG_DOCUMENT_10: Optional[float] = None
    FLAG_DOCUMENT_11: Optional[float] = None
    FLAG_DOCUMENT_12: Optional[float] = None
    FLAG_DOCUMENT_13: Optional[float] = None
    FLAG_DOCUMENT_14: Optional[float] = None
    FLAG_DOCUMENT_15: Optional[float] = None
    FLAG_DOCUMENT_16: Optional[float] = None
    FLAG_DOCUMENT_17: Optional[float] = None
    FLAG_DOCUMENT_18: Optional[float] = None
    FLAG_DOCUMENT_19: Optional[float] = None
    FLAG_DOCUMENT_20: Optional[float] = None
    FLAG_DOCUMENT_21: Optional[float] = None

    # Building info (AVG / MODE / MEDI sets — optional, most are NaN in production)
    APARTMENTS_AVG: Optional[float] = None
    BASEMENTAREA_AVG: Optional[float] = None
    YEARS_BEGINEXPLUATATION_AVG: Optional[float] = None
    YEARS_BUILD_AVG: Optional[float] = None
    COMMONAREA_AVG: Optional[float] = None
    ELEVATORS_AVG: Optional[float] = None
    ENTRANCES_AVG: Optional[float] = None
    FLOORSMAX_AVG: Optional[float] = None
    FLOORSMIN_AVG: Optional[float] = None
    LANDAREA_AVG: Optional[float] = None
    LIVINGAPARTMENTS_AVG: Optional[float] = None
    LIVINGAREA_AVG: Optional[float] = None
    NONLIVINGAPARTMENTS_AVG: Optional[float] = None
    NONLIVINGAREA_AVG: Optional[float] = None
    APARTMENTS_MODE: Optional[float] = None
    BASEMENTAREA_MODE: Optional[float] = None
    YEARS_BEGINEXPLUATATION_MODE: Optional[float] = None
    YEARS_BUILD_MODE: Optional[float] = None
    COMMONAREA_MODE: Optional[float] = None
    ELEVATORS_MODE: Optional[float] = None
    ENTRANCES_MODE: Optional[float] = None
    FLOORSMAX_MODE: Optional[float] = None
    FLOORSMIN_MODE: Optional[float] = None
    LANDAREA_MODE: Optional[float] = None
    LIVINGAPARTMENTS_MODE: Optional[float] = None
    LIVINGAREA_MODE: Optional[float] = None
    NONLIVINGAPARTMENTS_MODE: Optional[float] = None
    NONLIVINGAREA_MODE: Optional[float] = None
    APARTMENTS_MEDI: Optional[float] = None
    BASEMENTAREA_MEDI: Optional[float] = None
    YEARS_BEGINEXPLUATATION_MEDI: Optional[float] = None
    YEARS_BUILD_MEDI: Optional[float] = None
    COMMONAREA_MEDI: Optional[float] = None
    ELEVATORS_MEDI: Optional[float] = None
    ENTRANCES_MEDI: Optional[float] = None
    FLOORSMAX_MEDI: Optional[float] = None
    FLOORSMIN_MEDI: Optional[float] = None
    LANDAREA_MEDI: Optional[float] = None
    LIVINGAPARTMENTS_MEDI: Optional[float] = None
    LIVINGAREA_MEDI: Optional[float] = None
    NONLIVINGAPARTMENTS_MEDI: Optional[float] = None
    NONLIVINGAREA_MEDI: Optional[float] = None
    TOTALAREA_MODE: Optional[float] = None

    model_config = {"extra": "ignore"}


def _applicant_to_row(applicant: ApplicantIn) -> dict:
    row = {c: np.nan for c in _RAW_INPUT_COLUMNS}
    supplied = {k: v for k, v in applicant.model_dump().items() if v is not None}
    row.update(supplied)
    return row


class ScoreResponse(BaseModel):
    default_probability: float
    risk_tier: str      # "LOW" | "MEDIUM" | "HIGH"
    decision: str       # "APPROVE" | "REVIEW" | "DECLINE"


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float   # positive = pushes toward default, negative = away from default
    raw_value: float    # the actual feature value fed to the model


class ExplainResponse(BaseModel):
    default_probability: float
    risk_tier: str
    decision: str
    base_probability: float           # model's average prediction (baseline)
    top_features: List[FeatureContribution]  # sorted by abs(shap_value) desc


def _tier(prob: float) -> tuple[str, str]:
    if prob < 0.15:
        return "LOW", "APPROVE"
    elif prob < 0.40:
        return "MEDIUM", "REVIEW"
    else:
        return "HIGH", "DECLINE"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/contract")
def contract():
    return {
        "version": _contract.get("version"),
        "date": _contract.get("date", "")[:10],
        "validation_roc_auc": _contract.get("validation_roc_auc"),
        "feature_count": len(_contract.get("feature_order", [])),
        "calibrator": _contract.get("calibrator"),
    }


@app.post("/score", response_model=ScoreResponse)
def score_one(applicant: ApplicantIn):
    try:
        row = _applicant_to_row(applicant)
        prob = float(score(pd.DataFrame([row]))[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    tier, decision = _tier(prob)
    return ScoreResponse(
        default_probability=round(prob, 4),
        risk_tier=tier,
        decision=decision,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain_one(applicant: ApplicantIn, top_n: int = 10):
    """Score one applicant and return the top_n SHAP feature contributions."""
    try:
        row = _applicant_to_row(applicant)
        df_raw = pd.DataFrame([row])

        # Run through the feature pipeline to get the engineered feature matrix
        X_fe = apply_feature_pipeline(df_raw, _state)
        X_fe = X_fe[_contract["feature_order"]]

        # Score via the calibrated path
        raw_log_odds = _model.predict(X_fe, raw_score=True)
        from score import _calibrator
        prob = float(_calibrator.predict_proba(raw_log_odds.reshape(-1, 1))[0, 1])

        # SHAP values (TreeExplainer returns log-odds contributions for LightGBM)
        shap_vals = _explainer.shap_values(X_fe)
        # For binary LightGBM, shap_values returns a single array (log-odds space)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # positive class
        contributions = shap_vals[0]  # single row

        # True base rate = actual training set default rate (stored in contract).
        # We avoid sigmoid(expected_value) because sigmoid(mean(log_odds)) != mean(sigmoid(log_odds))
        # due to Jensen's inequality, which inflates the displayed base rate (e.g. 10% vs true 8.07%).
        base_log_odds = _explainer.expected_value
        if isinstance(base_log_odds, (list, np.ndarray)):
            base_log_odds = float(base_log_odds[1])
        else:
            base_log_odds = float(base_log_odds)

        base_prob = _contract.get("train_default_rate", 0.080729)

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        # Convert each SHAP log-odds contribution to a probability delta:
        # delta_i = sigmoid(base_log_odds + shap_i) - sigmoid(base_log_odds)
        contributions = np.array([sigmoid(base_log_odds + s) - sigmoid(base_log_odds)
                                   for s in contributions])

        features = _contract["feature_order"]
        raw_vals = X_fe.iloc[0].tolist()

        # Rank by absolute SHAP value, take top_n
        ranked = sorted(
            zip(features, contributions, raw_vals),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:top_n]

        tier, decision = _tier(prob)
        return ExplainResponse(
            default_probability=round(prob, 4),
            risk_tier=tier,
            decision=decision,
            base_probability=round(base_prob, 4),
            top_features=[
                FeatureContribution(
                    feature=f,
                    shap_value=round(float(s), 4),
                    raw_value=round(float(v), 4),
                )
                for f, s, v in ranked
            ],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/score/batch", response_model=List[ScoreResponse])
def score_batch(applicants: List[ApplicantIn]):
    if not applicants:
        return []
    if len(applicants) > 500:
        raise HTTPException(status_code=400, detail="Batch limit is 500 applicants.")
    try:
        rows = [_applicant_to_row(a) for a in applicants]
        probs = score(pd.DataFrame(rows))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    results = []
    for prob in probs:
        tier, decision = _tier(float(prob))
        results.append(ScoreResponse(
            default_probability=round(float(prob), 4),
            risk_tier=tier,
            decision=decision,
        ))
    return results
