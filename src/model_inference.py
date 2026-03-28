"""
Model inference module for ResistAI.
Loads a trained model and provides prediction functions.
"""

import joblib
import numpy as np
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best_model.pkl")
METADATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "model_metadata.pkl")


def load_model(model_path=None, metadata_path=None):
    """Load the trained model and metadata."""
    if model_path is None:
        model_path = MODEL_PATH
    if metadata_path is None:
        metadata_path = METADATA_PATH
    
    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    return model, metadata


def predict_resistance(model, metadata, input_features):
    """
    Predict resistance across all antibiotics for a single isolate.
    
    Args:
        model: trained multi-output classifier
        metadata: dict with feature_names, antibiotic_names, encoders
        input_features: dict of feature values for the isolate
    
    Returns:
        predictions: dict {antibiotic: {"label": str, "probability": float, "encoded": int}}
    """
    feature_names = metadata["feature_names"]
    antibiotic_names = metadata["antibiotic_names"]
    
    # Build feature vector
    X = pd.DataFrame([input_features])[feature_names]
    
    # Get predictions
    y_pred = model.predict(X)
    
    # Get probabilities if available
    try:
        y_proba = model.predict_proba(X)
    except AttributeError:
        y_proba = None
    
    results = {}
    label_map = {0: "Susceptible", 1: "Resistant"}
    
    for i, abx in enumerate(antibiotic_names):
        pred = int(y_pred[0][i]) if hasattr(y_pred[0], '__len__') else int(y_pred[0])
        
        if y_proba is not None and i < len(y_proba):
            # Multi-output: each element is the proba array for one output
            prob_resistant = float(y_proba[i][0][1]) if y_proba[i][0].shape[0] > 1 else float(y_proba[i][0][0])
        else:
            prob_resistant = float(pred)
        
        results[abx] = {
            "label": label_map.get(pred, "Unknown"),
            "probability": prob_resistant,
            "encoded": pred,
        }
    
    return results


def predict_from_ast_results(model, metadata, organism_encoded, age, gender_encoded,
                              specimen_encoded, department_encoded, ast_results):
    """
    Predict resistance from user-provided AST (Antibiotic Susceptibility Test) results.
    
    Args:
        ast_results: dict {antibiotic_name: encoded_value} where S=0, I=1, R=2
    """
    antibiotic_names = metadata["antibiotic_names"]
    feature_names = metadata["feature_names"]
    
    # Build the feature vector
    features = {}
    features["Age"] = age
    features["Gender_encoded"] = gender_encoded
    features["Organism_encoded"] = organism_encoded
    features["Specimen_encoded"] = specimen_encoded
    features["Department_encoded"] = department_encoded
    
    # Encode AST results
    encoded_cols = [f"{abx}_encoded" for abx in antibiotic_names]
    for abx in antibiotic_names:
        col = f"{abx}_encoded"
        features[col] = ast_results.get(abx, -1)  # -1 for unknown
    
    # Compute derived features
    resistance_count = sum(1 for v in ast_results.values() if v == 2)
    susceptible_count = sum(1 for v in ast_results.values() if v == 0)
    total = len(antibiotic_names)
    
    features["Resistance_Count"] = resistance_count
    features["Susceptible_Count"] = susceptible_count
    features["MDR_Flag"] = 1 if resistance_count >= 3 else 0
    features["Resistance_Ratio"] = resistance_count / total if total > 0 else 0
    
    # Add Kaggle enrichment features (use defaults if not available)
    for col in feature_names:
        if col.startswith("Kaggle_") and col not in features:
            features[col] = 0.5
    
    return predict_resistance(model, metadata, features)
