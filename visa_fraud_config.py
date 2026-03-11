"""
visa_fraud/config.py
────────────────────
Central configuration for the Visa Fraud Detection System.
Mirrors real Visa ML platform environment configs.
"""
from dataclasses import dataclass, field
from typing import List, Dict

# ── Model Registry ──────────────────────────────────────────
MODEL_VERSION   = "2.4.1"
MODEL_NAME      = "visa_fraud_detector"
EXPERIMENT_NAME = "fraud_detection_v2"

# ── Feature Groups (mirrors Visa Feature Store) ──────────────
AMOUNT_FEATURES = [
    "amount",
    "amount_log1p",
    "amount_zscore",           # z-score vs cardholder 30-day mean
    "amount_vs_merchant_avg",  # ratio vs merchant category avg
    "amount_percentile",       # percentile within cardholder history
]

VELOCITY_FEATURES = [
    "txn_count_1h",
    "txn_count_6h",
    "txn_count_24h",
    "txn_count_7d",
    "amount_sum_1h",
    "amount_sum_24h",
    "unique_merchants_24h",
    "unique_countries_7d",
    "declined_count_24h",
    "velocity_score",           # composite velocity index
]

BEHAVIORAL_FEATURES = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_night",                 # 11pm–5am
    "is_holiday",
    "days_since_last_txn",
    "avg_txn_hour",             # cardholder typical transaction hour
    "hour_deviation",           # abs(hour - avg_txn_hour)
    "typical_merchant_category",# 1 if matches cardholder's most frequent category
]

GEOGRAPHIC_FEATURES = [
    "is_domestic",
    "is_high_risk_country",
    "country_change_flag",      # different country from last txn
    "distance_from_home_km",
    "distance_from_last_txn_km",
    "impossible_travel_flag",   # distance/time implies impossible travel
    "ip_country_mismatch",      # billing country vs IP geolocation
]

CARD_FEATURES = [
    "card_present",
    "is_emv_chip",
    "is_contactless",
    "is_3ds_authenticated",
    "card_age_days",
    "days_since_last_change",   # days since PIN/address change
    "is_new_device",
    "device_fingerprint_match",
    "billing_address_change_flag",
]

MERCHANT_FEATURES = [
    "merchant_risk_score",      # merchant's historical fraud rate
    "merchant_age_days",
    "merchant_avg_ticket",
    "merchant_fraud_rate_30d",
    "is_high_risk_mcc",         # high-risk merchant category code
    "merchant_country_risk",
    "is_recurring",
    "is_card_not_present",
]

NETWORK_FEATURES = [
    "shared_device_fraud_rate",  # fraud rate of device across all cardholders
    "shared_ip_fraud_rate",      # fraud rate of IP across cardholders
    "merchant_network_fraud_rate",
    "bin_fraud_rate_7d",         # fraud rate for card BIN (first 6 digits)
]

ALL_FEATURES = (
    AMOUNT_FEATURES
    + VELOCITY_FEATURES
    + BEHAVIORAL_FEATURES
    + GEOGRAPHIC_FEATURES
    + CARD_FEATURES
    + MERCHANT_FEATURES
    + NETWORK_FEATURES
)

# ── Model Hyperparameters ────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators":       600,
    "max_depth":          7,
    "learning_rate":      0.03,
    "subsample":          0.8,
    "colsample_bytree":   0.75,
    "colsample_bylevel":  0.75,
    "min_child_weight":   5,
    "reg_alpha":          0.1,
    "reg_lambda":         1.0,
    "gamma":              0.1,
    "scale_pos_weight":   49,    # 2% fraud rate → 98/2
    "eval_metric":        "aucpr",
    "use_label_encoder":  False,
    "random_state":       42,
    "n_jobs":             -1,
    "tree_method":        "hist",
}

LGBM_PARAMS = {
    "n_estimators":       600,
    "max_depth":          7,
    "learning_rate":      0.03,
    "num_leaves":         63,
    "subsample":          0.8,
    "colsample_bytree":   0.75,
    "min_child_samples":  20,
    "reg_alpha":          0.1,
    "reg_lambda":         1.0,
    "class_weight":       "balanced",
    "random_state":       42,
    "n_jobs":             -1,
    "verbose":            -1,
}

# ── Thresholds & Business Rules ──────────────────────────────
# Visa uses tiered thresholds matching business impact
DECISION_THRESHOLDS = {
    "auto_approve":  0.05,   # < 5% → approve without friction
    "soft_decline":  0.40,   # 40–70% → step-up auth (3DS, OTP)
    "hard_decline":  0.70,   # > 70% → decline transaction
    "high_risk":     0.85,   # > 85% → flag for immediate review + card block
}

# False Negative cost >> False Positive cost in fraud
# $1 fraud loss ≈ $3 operational cost → set recall target
PRECISION_RECALL_TARGETS = {
    "min_precision": 0.40,   # at least 40% of flagged are actually fraud
    "min_recall":    0.80,   # catch at least 80% of all fraud
}

# ── Monitoring ───────────────────────────────────────────────
DRIFT_THRESHOLDS = {
    "psi_warning":   0.10,   # Population Stability Index
    "psi_critical":  0.25,
    "fraud_rate_delta_pct": 20.0,  # % change in fraud rate triggers alert
    "auc_drop":      0.03,   # AUC drop that triggers retraining
}
