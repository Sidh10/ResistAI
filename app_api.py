from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import sys
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.recommender import rank_antibiotics, get_recommendation, ANTIBIOTIC_CLASSES

app = FastAPI(title="ResistAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def read_root():
    return FileResponse("frontend/index.html")

# Load model
model_path = os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")
meta_path = os.path.join(os.path.dirname(__file__), "models", "model_metadata.pkl")
model = joblib.load(model_path)
metadata = joblib.load(meta_path)

FEATURE_NAMES = metadata["feature_names"]
ANTIBIOTIC_NAMES = metadata["antibiotic_names"]

ORGANISMS = [
    "Escherichia coli", "Enterobacteria spp.", "Klebsiella pneumoniae",
    "Proteus mirabilis", "Citrobacter spp.", "Morganella morganii",
    "Serratia marcescens", "Pseudomonas aeruginosa", "Acinetobacter baumannii",
    "Staphylococcus aureus", "Enterococcus spp.",
    "Streptococcus pneumoniae", "Unknown"
]
GENDERS = ["Female", "Male", "Unknown"]

class PredictRequest(BaseModel):
    organism: str
    age: int
    gender: str
    diabetes: bool
    hypertension: bool
    hospital_before: bool
    infection_freq: int
    known_ast: dict = {}  # { abx_name: value } where value=0(S), 1(I), 2(R)

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        organism_encoded = ORGANISMS.index(req.organism)
    except ValueError:
        organism_encoded = ORGANISMS.index("Unknown")
        
    try:
        gender_encoded = GENDERS.index(req.gender)
    except ValueError:
        gender_encoded = GENDERS.index("Unknown")
        
    features = {
        "Age": req.age,
        "Gender_encoded": gender_encoded,
        "Organism_encoded": organism_encoded,
        "Source_encoded": 0,
        "Diabetes": 1 if req.diabetes else 0,
        "Hypertension": 1 if req.hypertension else 0,
        "Hospital_before": 1 if req.hospital_before else 0,
        "Infection_Freq": req.infection_freq
    }
    
    X_input = pd.DataFrame([features])
    for col in FEATURE_NAMES:
        if col not in X_input.columns:
            X_input[col] = 0
    X_input = X_input[FEATURE_NAMES]
    
    y_pred = model.predict(X_input)
    try:
        y_proba = model.predict_proba(X_input)
    except AttributeError:
        y_proba = None
        
    predictions = {}
    for i, abx in enumerate(ANTIBIOTIC_NAMES):
        pred = int(y_pred[0][i])
        if y_proba is not None and i < len(y_proba):
            try:
                prob_r = float(y_proba[i][0][1]) if y_proba[i][0].shape[0] > 1 else float(pred)
            except:
                prob_r = float(pred)
        else:
            prob_r = float(pred)
            
        label = "Resistant" if pred == 1 else "Susceptible"
        source = "predicted"
        
        known = req.known_ast.get(abx, -1)
        if known != -1:
            if known == 2:
                label = "Resistant"
                prob_r = 1.0
            elif known == 1:
                label = "Intermediate"
                prob_r = 0.5
            else:
                label = "Susceptible"
                prob_r = 0.0
            source = "known"
            
        predictions[abx] = {
            "label": label,
            "probability": prob_r,
            "encoded": 1 if label == "Resistant" else 0,
            "source": source
        }
        
    ranked = rank_antibiotics(predictions)
    recommendations, primary, warnings = get_recommendation(ranked)
    
    # Optional SHAP calculation could be added here, returning mock/top ones for now
    shap_data = [
        {"feature": "Prior Hospitalization", "importance": 0.20},
        {"feature": "Age", "importance": 0.12},
        {"feature": "Infection Frequency", "importance": 0.08},
        {"feature": "Diabetes", "importance": -0.04}
    ]
    
    return {
        "predictions": predictions,
        "ranked": ranked,
        "primary": primary,
        "warnings": warnings,
        "shap_summary": shap_data
    }
