"""
visa_fraud/visualization.py + main pipeline
─────────────────────────────────────────────
Production-quality charts used in Visa model review boards.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from pathlib import Path
import logging
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

VISA_BLUE   = "#1A1F71"
VISA_GOLD   = "#F7A800"
VISA_RED    = "#E31837"
VISA_GREEN  = "#009473"
PALETTE     = [VISA_BLUE, VISA_GOLD, VISA_RED, VISA_GREEN, "#00A3E0", "#6B2D8B"]
plt.style.use("seaborn-v0_8-whitegrid")


class FraudModelVisualizer:

    def __init__(self, output_dir: str = "/mnt/user-data/outputs"):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def full_model_report(
        self,
        y_test: np.ndarray,
        ensemble_prob: np.ndarray,
        xgb_prob: np.ndarray,
        lgbm_prob: np.ndarray,
        threshold: float,
        feature_importance: pd.Series,
        feature_names: list,
    ) -> str:

        fig = plt.figure(figsize=(26, 22))
        fig.patch.set_facecolor("#F5F7FA")
        gs = gridspec.GridSpec(
            3, 3, figure=fig,
            hspace=0.40, wspace=0.35,
            top=0.93, bottom=0.06,
            left=0.06, right=0.97,
        )
        fig.suptitle(
            "VISA FRAUD DETECTION — MODEL PERFORMANCE REPORT",
            fontsize=18, fontweight="bold", color=VISA_BLUE, y=0.97,
        )

        # ── 1: ROC Curve ──────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        for probs, label, color in [
            (ensemble_prob, "Ensemble", VISA_BLUE),
            (xgb_prob,      "XGBoost",  VISA_GOLD),
            (lgbm_prob,     "LightGBM", VISA_RED),
        ]:
            fpr, tpr, _ = roc_curve(y_test, probs)
            auc = roc_auc_score(y_test, probs)
            ax1.plot(fpr, tpr, color=color, lw=2.2, label=f"{label}  AUC={auc:.4f}")
        ax1.plot([0,1],[0,1], "k--", lw=1, alpha=0.5, label="Random")
        ax1.fill_between(fpr, tpr, alpha=0.05, color=VISA_BLUE)
        ax1.set_xlabel("False Positive Rate", fontsize=10)
        ax1.set_ylabel("True Positive Rate", fontsize=10)
        ax1.set_title("ROC Curve", fontsize=12, fontweight="bold", color=VISA_BLUE)
        ax1.legend(fontsize=8)
        ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.02)

        # ── 2: Precision-Recall Curve ─────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        for probs, label, color in [
            (ensemble_prob, "Ensemble", VISA_BLUE),
            (xgb_prob,      "XGBoost",  VISA_GOLD),
            (lgbm_prob,     "LightGBM", VISA_RED),
        ]:
            prec, rec, _ = precision_recall_curve(y_test, probs)
            ap = average_precision_score(y_test, probs)
            ax2.plot(rec, prec, color=color, lw=2.2, label=f"{label}  AP={ap:.4f}")
        baseline = y_test.mean()
        ax2.axhline(baseline, color="gray", linestyle="--", lw=1, label=f"Baseline ({baseline:.3f})")
        ax2.axvline(0.80, color=VISA_GREEN, linestyle=":", lw=1.5, alpha=0.8, label="Recall=0.80 target")
        ax2.set_xlabel("Recall", fontsize=10)
        ax2.set_ylabel("Precision", fontsize=10)
        ax2.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold", color=VISA_BLUE)
        ax2.legend(fontsize=8)
        ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)

        # ── 3: Fraud Score Distribution ───────────────────────
        ax3 = fig.add_subplot(gs[0, 2])
        legit_scores = ensemble_prob[y_test == 0]
        fraud_scores = ensemble_prob[y_test == 1]
        bins = np.linspace(0, 1, 60)
        ax3.hist(legit_scores, bins=bins, color=VISA_BLUE, alpha=0.65,
                 label=f"Legitimate (n={len(legit_scores):,})", density=True)
        ax3.hist(fraud_scores, bins=bins, color=VISA_RED, alpha=0.65,
                 label=f"Fraud (n={len(fraud_scores):,})", density=True)
        ax3.axvline(threshold, color=VISA_GOLD, lw=2.5, linestyle="--",
                    label=f"Threshold={threshold:.3f}")
        ax3.set_xlabel("Fraud Score", fontsize=10)
        ax3.set_ylabel("Density", fontsize=10)
        ax3.set_title("Score Distribution by Class", fontsize=12, fontweight="bold", color=VISA_BLUE)
        ax3.legend(fontsize=8)
        ax3.set_yscale("log")

        # ── 4: Threshold vs Precision/Recall ──────────────────
        ax4 = fig.add_subplot(gs[1, 0])
        thresholds_arr = np.linspace(0.01, 0.99, 200)
        precisions_at_t, recalls_at_t, f1s = [], [], []
        for t in thresholds_arr:
            yp = (ensemble_prob >= t).astype(int)
            tp = ((yp == 1) & (y_test == 1)).sum()
            fp = ((yp == 1) & (y_test == 0)).sum()
            fn = ((yp == 0) & (y_test == 1)).sum()
            p = tp / (tp + fp + 1e-9)
            r = tp / (tp + fn + 1e-9)
            precisions_at_t.append(p)
            recalls_at_t.append(r)
            f1s.append(2*p*r/(p+r+1e-9))
        ax4.plot(thresholds_arr, precisions_at_t, color=VISA_BLUE, lw=2, label="Precision")
        ax4.plot(thresholds_arr, recalls_at_t,    color=VISA_RED,  lw=2, label="Recall")
        ax4.plot(thresholds_arr, f1s,              color=VISA_GREEN,lw=2, label="F1")
        ax4.axvline(threshold, color=VISA_GOLD, lw=2, linestyle="--", label=f"Optimal={threshold:.3f}")
        ax4.set_xlabel("Decision Threshold", fontsize=10)
        ax4.set_ylabel("Score", fontsize=10)
        ax4.set_title("Metrics vs Threshold", fontsize=12, fontweight="bold", color=VISA_BLUE)
        ax4.legend(fontsize=8)
        ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)

        # ── 5: Feature Importance (Top 20) ────────────────────
        ax5 = fig.add_subplot(gs[1, 1:])
        top_fi = feature_importance.head(20)
        colors_fi = [VISA_RED if v > top_fi.quantile(0.75)
                     else VISA_GOLD if v > top_fi.quantile(0.50)
                     else VISA_BLUE for v in top_fi.values]
        bars = ax5.barh(top_fi.index[::-1], top_fi.values[::-1],
                        color=colors_fi[::-1], alpha=0.88, edgecolor="white", height=0.7)
        for bar, val in zip(bars, top_fi.values[::-1]):
            ax5.text(val + top_fi.max() * 0.01, bar.get_y() + bar.get_height()/2,
                     f"{val:.4f}", va="center", fontsize=7.5)
        ax5.set_xlabel("Mean Feature Importance", fontsize=10)
        ax5.set_title("Top 20 Feature Importances (XGBoost)", fontsize=12,
                      fontweight="bold", color=VISA_BLUE)
        ax5.tick_params(axis="y", labelsize=8)

        # ── 6: Confusion Matrix ────────────────────────────────
        ax6 = fig.add_subplot(gs[2, 0])
        y_pred = (ensemble_prob >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        im = ax6.imshow(cm, cmap="Blues", aspect="auto")
        ax6.set_xticks([0, 1]); ax6.set_yticks([0, 1])
        ax6.set_xticklabels(["Predicted\nLegit", "Predicted\nFraud"], fontsize=9)
        ax6.set_yticklabels(["Actual\nLegit", "Actual\nFraud"], fontsize=9)
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax6.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                         fontsize=14, fontweight="bold", color=color)
        ax6.set_title("Confusion Matrix", fontsize=12, fontweight="bold", color=VISA_BLUE)
        plt.colorbar(im, ax=ax6, fraction=0.046)

        # ── 7: Cumulative Fraud Captured (Lift Curve) ─────────
        ax7 = fig.add_subplot(gs[2, 1])
        sorted_idx  = np.argsort(ensemble_prob)[::-1]
        sorted_labels = y_test[sorted_idx]
        cumulative_fraud = np.cumsum(sorted_labels) / y_test.sum()
        cumulative_pop   = np.arange(1, len(y_test)+1) / len(y_test)
        ax7.plot(cumulative_pop * 100, cumulative_fraud * 100,
                 color=VISA_BLUE, lw=2.5, label="Ensemble Model")
        ax7.plot([0, 100], [0, 100], "k--", lw=1, alpha=0.5, label="Random")
        # Mark key recall points
        for recall_target in [0.5, 0.8, 0.9]:
            idx = np.searchsorted(cumulative_fraud, recall_target)
            if idx < len(cumulative_pop):
                pop_pct = cumulative_pop[idx] * 100
                ax7.axvline(pop_pct, color=VISA_GOLD, lw=1, linestyle=":", alpha=0.8)
                ax7.text(pop_pct + 0.5, recall_target * 100 - 5,
                         f"{recall_target*100:.0f}% recall\n@ top {pop_pct:.1f}%",
                         fontsize=7.5, color=VISA_GOLD)
        ax7.fill_between(cumulative_pop * 100, cumulative_fraud * 100,
                         cumulative_pop * 100, alpha=0.08, color=VISA_BLUE)
        ax7.set_xlabel("% Transactions Reviewed (by risk score)", fontsize=10)
        ax7.set_ylabel("% Fraud Captured", fontsize=10)
        ax7.set_title("Cumulative Lift / Gain Curve", fontsize=12, fontweight="bold", color=VISA_BLUE)
        ax7.legend(fontsize=9)
        ax7.set_xlim(0, 100); ax7.set_ylim(0, 102)

        # ── 8: Business Cost vs Threshold ─────────────────────
        ax8 = fig.add_subplot(gs[2, 2])
        fn_cost, fp_cost = 175.0, 53.0
        costs_at_t = []
        for t in thresholds_arr:
            yp = (ensemble_prob >= t).astype(int)
            tp_c = ((yp==1)&(y_test==1)).sum()
            fp_c = ((yp==1)&(y_test==0)).sum()
            fn_c = ((yp==0)&(y_test==1)).sum()
            costs_at_t.append(fp_c * fp_cost + fn_c * fn_cost + tp_c * 5.0)
        ax8.plot(thresholds_arr, [c/1000 for c in costs_at_t],
                 color=VISA_RED, lw=2.5)
        best_t_idx = int(np.argmin(costs_at_t))
        ax8.axvline(thresholds_arr[best_t_idx], color=VISA_GOLD, lw=2.5,
                    linestyle="--", label=f"Min cost @ {thresholds_arr[best_t_idx]:.3f}")
        ax8.fill_between(thresholds_arr, [c/1000 for c in costs_at_t],
                         min(costs_at_t)/1000, alpha=0.10, color=VISA_RED)
        ax8.set_xlabel("Decision Threshold", fontsize=10)
        ax8.set_ylabel("Total Business Cost ($K)", fontsize=10)
        ax8.set_title("Business Cost vs Threshold", fontsize=12, fontweight="bold", color=VISA_BLUE)
        ax8.legend(fontsize=9)

        out_path = self.out / "visa_fraud_detection_report.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"Report saved → {out_path}")
        return str(out_path)


# ══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════
def run_full_pipeline():
    import sys
    sys.path.insert(0, "/home/claude")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    from sklearn.model_selection import train_test_split

    from visa_fraud.data_pipeline import (
        VisaTransactionSimulator,
        FeatureEngineeringPipeline,
    )
    from visa_fraud.model import (
        FraudEnsemble,
        ModelEvaluator,
        FraudCostMatrix,
        ProductionMonitor,
    )

    print("\n" + "="*65)
    print("  VISA FRAUD DETECTION SYSTEM — FULL PIPELINE")
    print("="*65)

    # ── Step 1: Generate realistic transaction data ──────────
    print("\n[1/7] Generating Transaction Data...")
    sim = VisaTransactionSimulator(n_cardholders=3_000, seed=42)
    df  = sim.generate_transactions(
        n_transactions=80_000,
        fraud_rate=0.0172,
        days=90,
    )

    # ── Step 2: Feature Engineering ─────────────────────────
    print("\n[2/7] Engineering Features...")
    fe_pipe = FeatureEngineeringPipeline()
    X, y, feature_names = fe_pipe.fit_transform(df)

    # Train / validation / test split
    # Mirroring real Visa setup: time-based split (not random)
    # Here we simulate it with random stratified split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15, random_state=42, stratify=y_trainval
    )
    logger.info(
        f"Train: {len(y_train):,} | Val: {len(y_val):,} | Test: {len(y_test):,} | "
        f"Fraud in test: {y_test.sum():,} ({y_test.mean():.3%})"
    )

    # ── Step 3: Train Ensemble ───────────────────────────────
    print("\n[3/7] Training XGBoost + LightGBM Ensemble...")
    ensemble = FraudEnsemble()
    train_results = ensemble.train(X_train, y_train, X_val, y_val, feature_names)

    print(f"\n  XGB OOF AUC:   {train_results['xgb_oof_auc']:.5f}")
    print(f"  LGBM OOF AUC:  {train_results['lgbm_oof_auc']:.5f}")
    print(f"  Ensemble Val AUC: {train_results['ensemble_val_roc_auc']:.5f}")
    print(f"  Optimal threshold: {train_results['threshold_info']['cost_optimal_threshold']:.4f}")

    # ── Step 4: Evaluate on Hold-out Test Set ───────────────
    print("\n[4/7] Evaluating on Test Set...")
    evaluator = ModelEvaluator(FraudCostMatrix())
    ensemble_prob = ensemble.predict_proba(X_test)
    xgb_prob      = ensemble.xgb_model.predict_proba(X_test)
    lgbm_prob     = ensemble.lgbm_model.predict_proba(X_test)
    threshold     = train_results["threshold_info"]["cost_optimal_threshold"]

    test_results = evaluator.full_evaluation(
        y_test, ensemble_prob, threshold, model_name="Visa Fraud Ensemble v2.4"
    )

    # ── Step 5: SHAP Explainability ──────────────────────────
    print("\n[5/7] Computing SHAP Explainability...")
    try:
        import shap
        from visa_fraud.model import FraudExplainer
        explainer = FraudExplainer(ensemble.xgb_model, feature_names)
        explainer.fit(X_train[:500])

        # Explain a high-risk transaction
        high_risk_idx = np.argmax(ensemble_prob)
        explanation = explainer.explain_transaction(X_test[high_risk_idx])
        print(f"\n  Top risk factors for highest-scored transaction:")
        for feat, val in list(explanation.get("top_risk_factors", {}).items())[:5]:
            print(f"    {feat:40s}  SHAP={val:+.4f}")
    except Exception as e:
        logger.warning(f"SHAP explanation skipped: {e}")
        explanation = {}

    # ── Step 6: Feature Importance ──────────────────────────
    print("\n[6/7] Computing Feature Importance...")
    fi = ensemble.xgb_model.feature_importances_
    print(f"\n  Top 10 Features (XGBoost gain importance):")
    for i, (feat, val) in enumerate(fi.head(10).items(), 1):
        print(f"    {i:2d}. {feat:40s}  {val:.5f}")

    # ── Step 7: Visualization ────────────────────────────────
    print("\n[7/7] Generating Model Report Dashboard...")
    viz = FraudModelVisualizer()
    report_path = viz.full_model_report(
        y_test=y_test,
        ensemble_prob=ensemble_prob,
        xgb_prob=xgb_prob,
        lgbm_prob=lgbm_prob,
        threshold=threshold,
        feature_importance=fi,
        feature_names=feature_names,
    )

    # ── Demo: Real-time transaction scoring ─────────────────
    print("\n" + "-"*50)
    print("DEMO: Real-time Transaction Scoring")
    print("-"*50)
    demo_indices = {
        "Low-risk (legit grocery)":  np.where((y_test == 0) & (ensemble_prob < 0.05))[0],
        "High-risk (flagged fraud)":  np.where((y_test == 1) & (ensemble_prob > 0.70))[0],
        "Borderline (step-up auth)":  np.where((ensemble_prob >= 0.15) & (ensemble_prob <= 0.45))[0],
    }
    for label, idxs in demo_indices.items():
        if len(idxs) > 0:
            result = ensemble.score_transaction(X_test[idxs[0]])
            actual = "FRAUD" if y_test[idxs[0]] else "LEGIT"
            print(f"\n  [{label}]")
            print(f"    Actual:      {actual}")
            print(f"    Fraud Score: {result['fraud_score']:.6f}")
            print(f"    Decision:    {result['decision']}")
            print(f"    Risk Band:   {result['risk_band']}")

    # ── Monitoring demo ──────────────────────────────────────
    print("\n" + "-"*50)
    print("DEMO: Production Monitoring")
    print("-"*50)
    monitor = ProductionMonitor()
    for prob, actual in zip(ensemble_prob[:500], y_test[:500]):
        score_result = ensemble.score_transaction(
            X_test[np.random.randint(len(X_test))]
        )
        monitor.log_prediction(
            fraud_score=float(prob),
            decision=score_result["decision"],
            actual_label=int(actual),
        )
    mon_report = monitor.generate_monitoring_report()
    print(f"\n  Predictions logged:     {mon_report['total_predictions']}")
    print(f"  Avg fraud score:        {mon_report['avg_fraud_score']:.4f}")
    print(f"  High-risk rate:         {mon_report['high_risk_rate_pct']:.2f}%")
    print(f"  Score P50/P95/P99:      "
          f"{mon_report['score_p50']:.4f} / "
          f"{mon_report['score_p95']:.4f} / "
          f"{mon_report['score_p99']:.4f}")
    if "labeled_auc" in mon_report:
        print(f"  Labeled AUC:            {mon_report['labeled_auc']:.4f}")

    # ── Final summary ────────────────────────────────────────
    print("\n" + "="*65)
    print("  FINAL RESULTS SUMMARY")
    print("="*65)
    print(f"  ROC-AUC:              {test_results['roc_auc']:.5f}")
    print(f"  Avg Precision:        {test_results['avg_precision']:.5f}")
    print(f"  Recall (fraud caught):{test_results['recall']:.5f}  ({test_results['fraud_detection_rate']:.1f}%)")
    print(f"  Precision:            {test_results['precision']:.5f}")
    print(f"  F2 Score:             {test_results['f2_score']:.5f}")
    print(f"  False Alert Rate:     {test_results['false_alert_rate_pct']:.2f}%")
    print(f"  Total Business Cost:  ${test_results['total_cost_usd']:,.2f}")
    print(f"  Fraud caught:         {test_results['true_positives']:,} / {test_results['n_fraud']:,}")
    print(f"  Missed fraud:         {test_results['false_negatives']:,}")
    print(f"  Wrong declines:       {test_results['false_positives']:,}")
    print(f"\n  Dashboard:            {report_path}")
    print("="*65)
    print("  ✅ Visa Fraud Detection Pipeline Complete")
    print("="*65)

    return ensemble, test_results


if __name__ == "__main__":
    run_full_pipeline()
