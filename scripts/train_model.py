"""
Model training script for ResistAI.
Trains Random Forest + XGBoost multi-label classifiers,
evaluates performance, and saves the best model.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score, classification_report
)
from xgboost import XGBClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def load_processed_data():
    """Load processed train/test data."""
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    Y_train = pd.read_csv("data/processed/Y_train.csv")
    Y_test = pd.read_csv("data/processed/Y_test.csv")
    
    feature_names = list(X_train.columns)
    antibiotic_names = list(Y_train.columns)
    
    print(f"Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Testing:  {X_test.shape[0]} samples")
    print(f"Antibiotics: {len(antibiotic_names)}")
    
    return X_train, X_test, Y_train, Y_test, feature_names, antibiotic_names


def train_random_forest(X_train, Y_train):
    """Train a multi-output Random Forest classifier."""
    print("\n  Training Random Forest...")
    rf = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    )
    rf.fit(X_train, Y_train)
    return rf


def train_xgboost(X_train, Y_train):
    """Train a multi-output XGBoost classifier with per-antibiotic class balancing."""
    print("  Training XGBoost (with class balancing)...")
    
    # Compute per-antibiotic scale_pos_weight
    estimators = []
    for i, col in enumerate(Y_train.columns):
        n_neg = (Y_train.iloc[:, i] == 0).sum()
        n_pos = (Y_train.iloc[:, i] == 1).sum()
        spw = max(n_neg / max(n_pos, 1), 1.0)
        
        est = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            random_state=42,
            eval_metric='logloss',
            verbosity=0,
            n_jobs=-1
        )
        estimators.append(est)
    
    # Train each estimator on its corresponding antibiotic
    xgb = MultiOutputClassifier(XGBClassifier())  # Placeholder
    xgb.estimators_ = []
    
    for i, (est, col) in enumerate(zip(estimators, Y_train.columns)):
        est.fit(X_train, Y_train.iloc[:, i])
        xgb.estimators_.append(est)
    
    # Patch predict/predict_proba to work like MultiOutputClassifier
    xgb.classes_ = [np.array([0, 1]) for _ in Y_train.columns]
    
    return xgb


def evaluate_model(model, X_test, Y_test, antibiotic_names, model_name="Model"):
    """Evaluate a multi-label model comprehensively."""
    Y_pred = model.predict(X_test)
    
    # Per-antibiotic metrics
    results = {}
    f1_scores = []
    auc_scores = []
    acc_scores = []
    
    print(f"\n  {'─' * 60}")
    print(f"  {model_name} — Per-Antibiotic Performance")
    print(f"  {'─' * 60}")
    print(f"  {'Antibiotic':<35} {'F1':>6} {'AUC':>6} {'Acc':>6}")
    print(f"  {'─' * 60}")
    
    for i, abx in enumerate(antibiotic_names):
        f1 = f1_score(Y_test.iloc[:, i], Y_pred[:, i], zero_division=0)
        acc = accuracy_score(Y_test.iloc[:, i], Y_pred[:, i])
        
        try:
            if len(Y_test.iloc[:, i].unique()) > 1:
                y_proba = model.predict_proba(X_test)
                proba = y_proba[i][:, 1] if y_proba[i].shape[1] > 1 else y_proba[i][:, 0]
                auc = roc_auc_score(Y_test.iloc[:, i], proba)
            else:
                auc = 0.5
        except Exception:
            auc = 0.5
        
        f1_scores.append(f1)
        auc_scores.append(auc)
        acc_scores.append(acc)
        
        print(f"  {abx:<35} {f1:.3f}  {auc:.3f}  {acc:.3f}")
        
        results[abx] = {"f1": f1, "auc": auc, "accuracy": acc}
    
    # Overall metrics
    overall_f1 = np.mean(f1_scores)
    overall_auc = np.mean(auc_scores)
    overall_acc = np.mean(acc_scores)
    
    print(f"  {'─' * 60}")
    print(f"  {'OVERALL (macro avg)':<35} {overall_f1:.3f}  {overall_auc:.3f}  {overall_acc:.3f}")
    print(f"  {'─' * 60}")
    
    return {
        "per_antibiotic": results,
        "overall_f1": overall_f1,
        "overall_auc": overall_auc,
        "overall_accuracy": overall_acc,
    }


def test_held_out_isolates(model, X_test, Y_test, antibiotic_names, n=3):
    """Test the model on n held-out isolates and show predictions vs truth."""
    print(f"\n  Testing on {n} held-out isolates:")
    print(f"  {'─' * 50}")
    
    indices = X_test.sample(n, random_state=42).index
    
    for idx in indices:
        X_sample = X_test.loc[[idx]]
        y_true = Y_test.loc[idx]
        y_pred = model.predict(X_sample)[0]
        
        print(f"\n  Isolate (index {idx}):")
        mismatches = 0
        for i, abx in enumerate(antibiotic_names):
            true_val = "R" if y_true.iloc[i] == 1 else "S"
            pred_val = "R" if y_pred[i] == 1 else "S"
            match = "✅" if true_val == pred_val else "❌"
            if true_val != pred_val:
                mismatches += 1
            print(f"    {abx:<35} True: {true_val}  Pred: {pred_val}  {match}")
        
        accuracy = 1 - (mismatches / len(antibiotic_names))
        print(f"    Isolate accuracy: {accuracy:.1%} ({mismatches} mismatches)")


def run_training():
    """Full training pipeline."""
    print("=" * 60)
    print("ResistAI Model Training")
    print("=" * 60)
    
    # Load data
    X_train, X_test, Y_train, Y_test, feature_names, antibiotic_names = load_processed_data()
    
    # Train models
    print("\n[Training Models]")
    rf_model = train_random_forest(X_train, Y_train)
    xgb_model = train_xgboost(X_train, Y_train)
    
    # Evaluate
    print("\n[Evaluating Models]")
    rf_results = evaluate_model(rf_model, X_test, Y_test, antibiotic_names, "Random Forest")
    xgb_results = evaluate_model(xgb_model, X_test, Y_test, antibiotic_names, "XGBoost")
    
    # Select best model
    print("\n[Model Selection]")
    if xgb_results["overall_f1"] >= rf_results["overall_f1"]:
        best_model = xgb_model
        best_name = "XGBoost"
        best_results = xgb_results
    else:
        best_model = rf_model
        best_name = "Random Forest"
        best_results = rf_results
    
    print(f"  🏆 Best model: {best_name}")
    print(f"     F1: {best_results['overall_f1']:.3f} | AUC: {best_results['overall_auc']:.3f} | Acc: {best_results['overall_accuracy']:.3f}")
    
    # Test on held-out isolates
    print(f"\n[Held-Out Isolate Testing — {best_name}]")
    test_held_out_isolates(best_model, X_test, Y_test, antibiotic_names)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    
    model_metadata = {
        "feature_names": feature_names,
        "antibiotic_names": antibiotic_names,
        "model_name": best_name,
        "metrics": best_results,
    }
    
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(model_metadata, "models/model_metadata.pkl")
    
    # Also save both models
    joblib.dump(rf_model, "models/random_forest.pkl")
    joblib.dump(xgb_model, "models/xgboost.pkl")
    
    # Save metrics summary
    metrics_df = pd.DataFrame({
        abx: {
            "RF_F1": rf_results["per_antibiotic"][abx]["f1"],
            "RF_AUC": rf_results["per_antibiotic"][abx]["auc"],
            "XGB_F1": xgb_results["per_antibiotic"][abx]["f1"],
            "XGB_AUC": xgb_results["per_antibiotic"][abx]["auc"],
        }
        for abx in antibiotic_names
    }).T
    metrics_df.to_csv("models/metrics_summary.csv")
    
    print(f"\n  Saved: models/best_model.pkl ({best_name})")
    print(f"  Saved: models/model_metadata.pkl")
    print(f"  Saved: models/metrics_summary.csv")
    
    print("\n" + "=" * 60)
    print("✅ Model training complete!")
    print("=" * 60)
    
    return best_model, model_metadata


if __name__ == "__main__":
    run_training()
