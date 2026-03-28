# ResistAI

## What this is
A clinical decision-support tool that predicts multi-drug resistance patterns
for bacterial isolates and recommends the most effective antibiotic — with
explainable AI showing which features drove the prediction.

## The problem
Clinicians must wait 48–72 hours for lab cultures to confirm antibiotic
effectiveness; ResistAI predicts resistance instantly from known isolate data
so treatment decisions don't have to wait.

## Target user
A clinical microbiologist or ICU physician with a bacterial isolate's known
susceptibility test results who needs an immediate, explainable treatment
recommendation.

## Core features (MVP only)
1. Multi-label resistance classifier — predicts resistance across all antibiotics
   simultaneously using the primary + secondary datasets
2. SHAP feature importance — identifies which bacterial features most strongly
   drive each resistance prediction, shown as a ranked visual
3. CARD gene annotation overlay — maps top SHAP features to known resistance
   genes from the CARD database for biological grounding
4. "Next antibiotic" recommender — given a resistance profile, ranks antibiotics
   by predicted effectiveness and suggests the best next option
5. Prediction interface — input a strain's known test results, receive resistance
   predictions + recommendation + SHAP explanation in one view

## What this is NOT
- Not a diagnostic tool — it predicts resistance, not disease or infection type
- Not a replacement for lab culture — it augments, not replaces, clinical judgment
- Not a single-label classifier — always predict across the full antibiotic panel
- Do not add features not listed above without asking

## Tech stack
- ML: Python, scikit-learn (Random Forest + XGBoost), SHAP, RDKit (optional)
- Data: Mendeley AMR dataset (primary), Kaggle multi-resistance dataset (secondary),
  CARD database (gene annotations, optional enrichment)
- Frontend: Streamlit (fastest path to a working demo interface)
- Deployment: Streamlit Cloud (free, no config needed)
- Notebook: Jupyter (for model training, EDA, and visualizations)

## Quality rules
- Every feature must work end-to-end before moving to the next
- SHAP explanations must be visible in the demo interface, not just the notebook
- Handle missing susceptibility values gracefully (common in real AMR datasets)
- Push to GitHub after each completed feature with a clear commit message
- README must include: model performance metrics, feature importance findings,
  setup instructions, and a recorded demo GIF

## North star
At demo time: judge inputs a bacterial isolate's known test results, model
predicts resistance across all antibiotics, SHAP chart explains why, and the
interface recommends the single best antibiotic to try next.