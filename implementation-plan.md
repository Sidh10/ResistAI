# ResistAI Implementation Plan

## Goal Description
Build a clinical decision-support tool (ResistAI) that predicts multi-drug resistance patterns for bacterial isolates using known susceptibility data, explains predictions via SHAP feature importance, and recommends the most effective antibiotic. The MVP will be built using Python, scikit-learn, and Streamlit, and deployed on Streamlit Cloud.

## User Review Required
> [!IMPORTANT]  
> Please review this implementation plan before we proceed.
> 1. **Data Assets:** Are the Mendeley AMR dataset, Kaggle multi-resistance dataset, and CARD database already downloaded in your local environment, or do we need to build automated download/fetch scripts?
> 2. **Model Workflow:** The plan assumes we will train the ML model locally via Jupyter Notebooks and export a pre-trained `.pkl` artifact to be loaded by the Streamlit app. Is this approach approved?

## Proposed Changes

### 1. Data & ML Pipeline
#### [NEW] `data/` - Directory for raw and processed datasets (to be gitignored).
#### [NEW] `notebooks/01_data_cleaning.ipynb` - Data preprocessing and missing value imputation.
#### [NEW] `notebooks/02_model_training.ipynb` - Training Random Forest/XGBoost multi-label classifiers and exporting the model (`model.pkl`).
#### [NEW] `notebooks/03_shap_analysis.ipynb` - SHAP baseline generation and CARD mapping exploration.

### 2. Streamlit Application Core
#### [NEW] `app.py` - Main Streamlit interface and application layout structure.
#### [NEW] `src/model_inference.py` - Functions for loading the model and predicting drug resistance.
#### [NEW] `src/recommender.py` - Logic to rank antibiotics based on predictions and confidence scores.
#### [NEW] `src/explainability.py` - SHAP horizontal bar chart generation & CARD gene annotation mapping.
#### [NEW] `assets/style.css` - Custom CSS injecting brand guidelines (Inter/JetBrains Mono fonts, 8px rounded corners, clinical color palette).

### 3. Project Configuration
#### [NEW] `requirements.txt` - Python dependencies (streamlit, scikit-learn, xgboost, shap, pandas, etc.).
#### [NEW] `README.md` - Documentation with metrics, setup steps, and demo GIF.

## Verification Plan

### Automated Tests
- Basic data validation checks ensuring missing susceptibility inputs are handled without failing.
- Unit tests for the recommendation ranking algorithm to ensure predicted susceptible antibiotics (green) rank higher than resistant ones (red).

### Manual Verification (Demo Path)
1. **Interface Launch:** Start the Streamlit app locally (`streamlit run app.py`) and verify styling matches clinical guidelines.
2. **Data Input:** Input a known bacterial isolate's test results into the UI.
3. **Prediction Output:** Confirm the multi-label classifier displays resistance predictions across the full antibiotic panel.
4. **Explainability View:** Check that an ordered SHAP horizontal bar chart and CARD annotation appear alongside every prediction.
5. **Recommendation Check:** Verify the "Next antibiotic" recommendation ranks the most effective option cleanly.
