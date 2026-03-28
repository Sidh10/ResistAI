# ResistAI Progress Tracker

## Phase 1: Data Engineering ✅
- [x] Initialize Python environment and `requirements.txt`
- [x] Process primary (Mendeley AMR — Dataset.xlsx, zone diameters) & secondary (Kaggle — Bacteria_dataset_Multiresictance.csv)
- [x] Process CARD database for gene annotations
- [x] Handle missing susceptibility values gracefully (mode imputation per organism)
- [x] CLSI breakpoint conversion for Mendeley zone diameters → R/I/S
- [x] Organism name normalization (typo fixes in Kaggle data)

## Phase 2: ML Modeling ✅
- [x] Train Multi-label Random Forest / XGBoost classifier (with class balancing)
- [x] Extract SHAP feature importance values (per-estimator TreeExplainer)
- [x] Export trained model artifact (models/best_model.pkl)
- [x] Document model performance metrics (F1: 0.502, AUC: 0.683, Acc: 0.683)
- [x] Eliminated data leakage (removed antibiotic-encoded features from X)

## Phase 3: Clinical Logic ✅
- [x] Map top SHAP features to CARD gene annotations
- [x] Build "Next antibiotic" ranking logic using confidence scores
- [x] Known AST override — clinician-provided results replace model predictions

## Phase 4: Streamlit Interface ✅
- [x] Apply clinical dashboard aesthetic (Colors, 8px rounded corners)
- [x] Build prediction input interface (sidebar with clinical history + AST dropdowns)
- [x] Render resistance statuses (Red/Amber/Green color system)
- [x] Show Known vs Predicted badges on resistance cards
- [x] Render SHAP horizontal bar charts alongside predictions
- [x] Display antibiotic recommendations ranked list

## Phase 5: Demo & Deployment
- [x] Verify end-to-end path (input → predict → SHAP explanation → recommendation)
- [x] Verify predictions vary across isolates (not all identical)
- [ ] Record demo GIF
- [x] Finalize README setup instructions
- [ ] Deploy to Streamlit Cloud
