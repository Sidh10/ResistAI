# ResistAI Task Plan

## 1. Environment & Data Setup
- [ ] 1.1 Set up Python virtual environment and `requirements.txt`.
- [ ] 1.2 Initialize Jupyter Notebook for EDA.
- [ ] 1.3 Download and clean Mendeley AMR and Kaggle multi-resistance datasets.
- [ ] 1.4 Handle missing susceptibility values gracefully.

## 2. Model Training & Evaluation
- [ ] 2.1 Train Random Forest / XGBoost multi-label classifier.
- [ ] 2.2 Evaluate model performance metrics.
- [ ] 2.3 Integrate SHAP for feature importance extraction.
- [ ] 2.4 Map top SHAP features to CARD gene annotations.

## 3. Recommender & Logic Layer
- [ ] 3.1 Develop recommendation logic to rank effective antibiotics.
- [ ] 3.2 Structure output to include prediction confidence scores.

## 4. Frontend Development (Streamlit)
- [ ] 4.1 Scaffold Streamlit app (`app.py`).
- [ ] 4.2 Apply clinical brand guidelines (Colors: #C0392B, #2E86AB, #F7F9FC bg; Fonts: Inter, JetBrains Mono).
- [ ] 4.3 Build prediction interface for known isolate test results input.
- [ ] 4.4 Render horizontal SHAP bar charts and color-coded resistance status (red/amber/green system).
- [ ] 4.5 Display ranked antibiotic recommendations.

## 5. Documentation & Deployment
- [ ] 5.1 Update README with performance metrics, feature importance findings, and setup instructions.
- [ ] 5.2 Record a demo GIF for the README.
- [ ] 5.3 Deploy to Streamlit Cloud.
