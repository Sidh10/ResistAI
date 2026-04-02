#  ResistAI

**Predict resistance. Prescribe with confidence.**

ResistAI is a clinical decision-support tool that predicts multi-drug antibiotic resistance patterns for bacterial isolates and recommends the most effective antibiotic — with SHAP-based explainability showing which features drove the prediction.

##  Problem

Clinicians must wait 48–72 hours for lab cultures to confirm antibiotic effectiveness. ResistAI predicts resistance instantly from known isolate data so treatment decisions don't have to wait.

##  Features

| Feature | Description |
|---------|-------------|
| **Multi-label Resistance Classifier** | Predicts resistance across 16 antibiotics simultaneously using XGBoost |
| **SHAP Feature Importance** | Per-prediction waterfall charts showing which features drive each resistance call |
| **CARD Gene Annotations** | Maps top SHAP features to known resistance genes from the CARD database |
| **Antibiotic Recommender** | Ranks antibiotics by predicted susceptibility, recommends the best option |
| **Clinical Dashboard** | Streamlit interface with color-coded resistance status  |

##  Model Performance

| Metric | Score |
|--------|-------|
| **Model** | XGBoost (MultiOutputClassifier, per-label balanced) |
| **Overall F1** | 0.502 |
| **Overall AUC** | 0.683 |
| **Overall Accuracy** | 0.683 |
| **Antibiotics Panel** | 16 antibiotics |
| **Training Samples** | 8,787 |
| **Test Samples** | 2,197 |
| **Data Sources** | Mendeley (274 isolates, zone diameters) + Kaggle (10,710 isolates, R/S/I labels) |

> **Note:** Metrics reflect a clean model with no data leakage — only 8 pure clinical features
> (Age, Gender, Organism, Diabetes, Hypertension, Hospital history, Infection frequency, Source)
> are used. Balanced antibiotics like Amoxicillin-Ampicillin achieve F1 ~0.75, while imbalanced
> ones (e.g., Colistin at 12% R) are harder to predict.

## Interfaces

### Primary (Streamlit)
Live app: https://resistai-s.streamlit.app
Run locally: streamlit run app.py

### Bonus (FastAPI + HTML)
A second interface built with FastAPI + custom HTML/CSS frontend.
Run locally:
uvicorn app_api:app --port 8000
Then open http://localhost:8000

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/your-username/ResistAI.git
cd ResistAI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run data pipeline (processes real Mendeley + Kaggle datasets)
python src/data_pipeline.py

# 4. Train models
python scripts/train_model.py

# 5. Launch the dashboard
streamlit run app.py
```

##  Project Structure

```
ResistAI/
├── app.py                         # Streamlit dashboard (main entry)
├── requirements.txt               # Python dependencies
├── assets/
│   └── style.css                  # Brand-guided clinical CSS
├── data/
│   ├── raw/                       # Raw datasets (Mendeley, Kaggle, CARD)
│   └── processed/                 # Cleaned train/test splits
├── models/
│   ├── best_model.pkl             # Trained XGBoost model
│   ├── model_metadata.pkl         # Feature names, antibiotic list, metrics
│   └── metrics_summary.csv        # Per-antibiotic RF vs XGB comparison
├── scripts/
│   ├── generate_data.py           # Synthetic AMR data generator
│   └── train_model.py             # Model training & evaluation
└── src/
    ├── data_pipeline.py           # Data loading, cleaning, feature engineering
    ├── model_inference.py         # Prediction functions
    ├── explainability.py          # SHAP charts & CARD gene mapping
    └── recommender.py             # Antibiotic ranking & recommendation
```

##  Brand Guidelines

- **Primary**: `#C0392B` (deep red — urgency, medicine)
- **Secondary**: `#2E86AB` (steel blue — trust, science)
- **Background**: `#F7F9FC` (clinical off-white)
- **Text**: `#1A1A2E` (near-black)
- **Danger**: `#E74C3C` (resistant)
- **Warning**: `#F39C12` (intermediate)
- **Safe**: `#27AE60` (susceptible)

##  Key Feature Importance Findings

The SHAP analysis reveals:
- **Cross-resistance patterns** between related antibiotic classes are the strongest predictors
- **Organism identity** is a major driver due to species-specific intrinsic resistance
- **Kaggle enrichment features** (population-level resistance rates) improve prediction confidence
- **MDR flag** and resistance count are strong signals for pan-resistant profiles

##  Disclaimer

ResistAI is a clinical decision-support tool only. It is **not** a replacement for laboratory culture and sensitivity testing. Always confirm predictions with standard microbiological methods.

##  License

MIT License
