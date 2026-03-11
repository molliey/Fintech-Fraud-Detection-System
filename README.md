This repo provides a reference pipeline for transaction fraud detection: synthetic data generation, feature engineering, XGBoost/LightGBM training, ensemble blending, cost-driven thresholding, SHAP-based explainability (optional), and production monitoring (PSI, prediction logs).

## Business Objectives

- Detect fraud with high recall while controlling operational cost from false positives.
- Key KPIs: ROC-AUC, Recall (target ≥ 0.80), Average Precision (AP), False Alert Rate, and business cost as defined by `FraudCostMatrix`.
- Stakeholders: Risk Ops, Product, Engineering, Compliance.

## Architecture (high level)

Transaction Source → Feature Engineering → Models (XGB/LGBM) → Ensemble → Decision & Actions
                                        ↓
                                   Explainability
                                        ↓
                                    Monitoring

Components:
- Data generator & pipeline: `VisaTransactionSimulator`, `FeatureEngineeringPipeline` (`visa_fraud_data_pipeline.py`).
- Models: `XGBoostFraudModel`, `LightGBMFraudModel`, ensemble `FraudEnsemble` (`visa_fraud_model.py`).
- Evaluation & Cost: `FraudCostMatrix`, `ModelEvaluator`.
- Explainability: `FraudExplainer` (SHAP, optional).
- Monitoring & Reporting: `ProductionMonitor`, `FraudModelVisualizer` (`visa_fraud_main.py`).

## Quick Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scikit-learn xgboost lightgbm matplotlib seaborn
# Optional: shap for explanations
pip install shap
```

## CLI / API Usage

- Real-time scoring: after training, call `FraudEnsemble.score_transaction(X_single)` to get `{'fraud_score','decision','reason','risk_band'}`.
- Threshold optimization: `FraudCostMatrix.find_optimal_threshold(y_true, y_prob)` returns cost-optimal threshold and metrics.

## Outputs

- Model report PNG via `FraudModelVisualizer` (default `./outputs/visa_fraud_detection_report.png`).
- Monitoring report via `ProductionMonitor.generate_monitoring_report()` (dict).

## Acceptance Criteria

- Example targets: ROC-AUC ≥ 0.90, Recall ≥ 0.80, Precision ≥ 0.40, and lower total business cost vs baseline. Adjust targets per business needs.




