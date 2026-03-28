"""
ResistAI — Clinical Decision-Support Dashboard
Predicts multi-drug antibiotic resistance patterns, explains predictions
via SHAP, and recommends the most effective antibiotic.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.recommender import rank_antibiotics, get_recommendation, format_recommendation_table, ANTIBIOTIC_CLASSES
from src.explainability import (
    compute_shap_values, generate_waterfall_chart,
    generate_global_importance, get_card_annotation, fig_to_base64
)

# ══════════════════════════════════════════════════════════════════════════════
#  Page Config
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ResistAI — Predict Resistance. Prescribe with Confidence.",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Load Model & Data
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    """Load the trained model and metadata."""
    model_path = os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")
    meta_path = os.path.join(os.path.dirname(__file__), "models", "model_metadata.pkl")
    model = joblib.load(model_path)
    metadata = joblib.load(meta_path)
    return model, metadata


@st.cache_resource
def load_training_data():
    """Load training data for SHAP background."""
    X_train = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "processed", "X_train.csv"))
    return X_train


@st.cache_resource
def load_card_data():
    """Load CARD gene annotation database."""
    card_path = os.path.join(os.path.dirname(__file__), "data", "raw", "card_genes.csv")
    if os.path.exists(card_path):
        return pd.read_csv(card_path)
    return None


model, metadata = load_model()
X_train = load_training_data()
card_df = load_card_data()

FEATURE_NAMES = metadata["feature_names"]
ANTIBIOTIC_NAMES = metadata["antibiotic_names"]
MODEL_NAME = metadata.get("model_name", "XGBoost")
MODEL_METRICS = metadata.get("metrics", {})

# ── Real data organism & feature mappings ──
ORGANISMS = [
    "Escherichia coli", "Enterobacteria spp.", "Klebsiella pneumoniae",
    "Proteus mirabilis", "Citrobacter spp.", "Morganella morganii",
    "Serratia marcescens", "Pseudomonas aeruginosa", "Acinetobacter baumannii",
    "Staphylococcus aureus", "Enterococcus spp.",
    "Streptococcus pneumoniae", "Unknown"
]
GENDERS = ["Female", "Male", "Unknown"]
YES_NO = ["No", "Yes"]

RESISTANCE_OPTIONS = {"Susceptible (S)": 0, "Intermediate (I)": 1, "Resistant (R)": 2, "Unknown / Not Tested": -1}


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar — Isolate Input
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 15px 0;">
        <h1 style="color: #C0392B; margin-bottom: 0; font-size: 2rem;">🧬 ResistAI</h1>
        <p style="color: #8899aa; font-size: 0.9rem; margin-top: 4px;">
            Predict resistance. Prescribe with confidence.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    with st.expander("ℹ️ **How to use**", expanded=True):
        st.markdown("""
        **Step 1:** Select organism and fill clinical history  
        **Step 2:** Enter any known AST results (optional)  
        **Step 3:** Click Predict to see full resistance panel
        """)
    
    st.markdown("---")
    st.markdown("### 🏥 Patient & Isolate Info")
    
    organism = st.selectbox("**Organism**", ORGANISMS, index=0)
    organism_encoded = ORGANISMS.index(organism)
    
    col_a, col_b = st.columns(2)
    with col_a:
        age = st.number_input("**Age**", min_value=0, max_value=120, value=55, step=1)
    with col_b:
        gender = st.selectbox("**Gender**", GENDERS)
    gender_encoded = GENDERS.index(gender)
    
    st.markdown("---")
    st.markdown("### 📋 Clinical History")
    
    col_c, col_d = st.columns(2)
    with col_c:
        diabetes = st.selectbox("**Diabetes**", YES_NO, index=0)
        diabetes_flag = 1 if diabetes == "Yes" else 0
    with col_d:
        hypertension = st.selectbox("**Hypertension**", YES_NO, index=0)
        hypertension_flag = 1 if hypertension == "Yes" else 0
    
    col_e, col_f = st.columns(2)
    with col_e:
        hospital_before = st.selectbox("**Prior Hospitalization**", YES_NO, index=0)
        hospital_flag = 1 if hospital_before == "Yes" else 0
    with col_f:
        infection_freq = st.number_input("**Infection Frequency**", min_value=0, max_value=10, value=0, step=1)
    
    st.markdown("---")
    st.markdown("### 🧪 Known AST Results")
    st.markdown(
        '<p style="color: #8899aa; font-size: 0.8rem;">Set known susceptibility values. '
        'Leave as "Unknown" if not tested.</p>',
        unsafe_allow_html=True
    )
    
    ast_results = {}
    for abx in ANTIBIOTIC_NAMES:
        val = st.selectbox(
            f"**{abx}**",
            list(RESISTANCE_OPTIONS.keys()),
            index=3,  # Default to Unknown
            key=f"ast_{abx}"
        )
        ast_results[abx] = RESISTANCE_OPTIONS[val]
    
    st.markdown("---")
    predict_clicked = st.button("🔬 **Predict Resistance**", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Main Panel — Header
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="padding: 10px 0 5px 0;">
    <h1 style="color: #1A1A2E; margin-bottom: 2px; font-size: 1.8rem;">
        🧬 ResistAI <span style="color: #C0392B;">Clinical Dashboard</span>
    </h1>
    <p style="color: #6b7280; font-size: 0.95rem; margin-top: 0;">
        Multi-drug resistance prediction with explainable AI — powered by {model_name}
    </p>
</div>
""".format(model_name=MODEL_NAME), unsafe_allow_html=True)

# ── Model metrics bar ──
if MODEL_METRICS:
    mcols = st.columns(5)
    with mcols[0]:
        st.metric("Model", MODEL_NAME)
    with mcols[1]:
        st.metric("Antibiotics", f"{len(ANTIBIOTIC_NAMES)}")
    with mcols[2]:
        st.metric("Training samples", "10,984")
    with mcols[3]:
        st.metric("Overall F1", f"{MODEL_METRICS.get('overall_f1', 0):.3f}")
    with mcols[4]:
        st.metric("Overall AUC", f"{MODEL_METRICS.get('overall_auc', 0):.3f}")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
#  Prediction Logic
# ══════════════════════════════════════════════════════════════════════════════

if predict_clicked:
    with st.spinner("🔬 Running resistance prediction..."):
        
        # ── Build feature vector (8 pure clinical features) ──
        features = {}
        features["Age"] = age
        features["Gender_encoded"] = gender_encoded
        features["Organism_encoded"] = organism_encoded
        features["Source_encoded"] = 0  # Runtime input treated as Kaggle-like
        features["Diabetes"] = diabetes_flag
        features["Hypertension"] = hypertension_flag
        features["Hospital_before"] = hospital_flag
        features["Infection_Freq"] = infection_freq
        
        # Ensure feature order matches model training
        X_input = pd.DataFrame([features])
        # Reorder and fill missing columns
        for col in FEATURE_NAMES:
            if col not in X_input.columns:
                X_input[col] = 0
        X_input = X_input[FEATURE_NAMES]
        
        # ── Predict ──
        y_pred = model.predict(X_input)
        
        # Get probabilities
        try:
            y_proba = model.predict_proba(X_input)
        except AttributeError:
            y_proba = None
        
        # Build predictions dict
        predictions = {}
        for i, abx in enumerate(ANTIBIOTIC_NAMES):
            pred = int(y_pred[0][i])
            if y_proba is not None and i < len(y_proba):
                try:
                    prob_r = float(y_proba[i][0][1]) if y_proba[i][0].shape[0] > 1 else float(pred)
                except (IndexError, AttributeError):
                    prob_r = float(pred)
            else:
                prob_r = float(pred)
            
            label = "Resistant" if pred == 1 else "Susceptible"
            source = "predicted"
            
            # Override with known AST result if clinician provided one
            known = ast_results.get(abx, -1)
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
                "source": source,
            }
        
        # ── Rank & Recommend ──
        ranked = rank_antibiotics(predictions)
        recommendations, primary, warnings = get_recommendation(ranked)
        
        # ── SHAP Values ──
        try:
            import shap
            # MultiOutputClassifier wraps individual estimators — compute SHAP per estimator
            shap_values = []
            for est in model.estimators_:
                explainer = shap.TreeExplainer(est)
                sv = explainer.shap_values(X_input)
                shap_values.append(sv)
            shap_computed = True
        except Exception as e:
            shap_computed = False
            shap_error = str(e)
    
    # ══════════════════════════════════════════════════════════════════════════
    #  Results Display — 3 Columns
    # ══════════════════════════════════════════════════════════════════════════
    
    # ── Warnings ──
    for w in warnings:
        if "All antibiotics" in w:
            st.markdown(f'<div class="danger-box">{w}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warning-box">{w}</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1.2, 1.3, 1])
    
    # ────────────────────────────────────────────────────────────────────────
    #  Column 1: Resistance Prediction Table
    # ────────────────────────────────────────────────────────────────────────
    with col1:
        st.markdown("### 🧫 Resistance Predictions")
        st.markdown(f'<p style="color: #6b7280; font-size: 0.85rem;">Organism: <strong>{organism}</strong></p>', unsafe_allow_html=True)
        
        for abx in ANTIBIOTIC_NAMES:
            pred = predictions[abx]
            color = "#E74C3C" if pred["label"] == "Resistant" else "#F39C12" if pred["label"] == "Intermediate" else "#27AE60"
            bg = "#FDEDEC" if pred["label"] == "Resistant" else "#FEF9E7" if pred["label"] == "Intermediate" else "#EAFAF1"
            emoji = "🔴" if pred["label"] == "Resistant" else "🟡" if pred["label"] == "Intermediate" else "🟢"
            prob_pct = f"{pred['probability']:.0%}"
            source_badge = '<span style="font-size:0.65rem; color:#2E86AB; background:#E8F4FD; padding:1px 5px; border-radius:3px; margin-left:4px;">✓ Known</span>' if pred.get("source") == "known" else '<span style="font-size:0.65rem; color:#888; background:#f0f0f0; padding:1px 5px; border-radius:3px; margin-left:4px;">Predicted</span>'
            
            abx_class = ANTIBIOTIC_CLASSES.get(abx, "")
            
            st.markdown(f"""
            <div style="display: flex; align-items: center; justify-content: space-between; 
                        padding: 8px 14px; margin: 4px 0; border-radius: 8px; 
                        background-color: {bg}; border-left: 4px solid {color};">
                <div>
                    <span style="font-weight: 600; color: #1A1A2E; font-size: 0.92rem;">{abx}</span>
                    <span style="color: #888; font-size: 0.75rem; margin-left: 6px;">{abx_class}</span>
                    {source_badge}
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; 
                                 color: {color}; font-weight: 600;">{prob_pct}</span>
                    <span style="font-size: 0.9rem;">{emoji}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Summary stats
        r_count = sum(1 for p in predictions.values() if p["label"] == "Resistant")
        s_count = sum(1 for p in predictions.values() if p["label"] == "Susceptible")
        st.markdown(f"""
        <div style="margin-top: 12px; padding: 10px; background: white; border-radius: 8px; 
                    box-shadow: 0 1px 3px rgba(0,0,0,0.06);">
            <span style="color: #E74C3C; font-weight: 700;">🔴 {r_count} Resistant</span> &nbsp;|&nbsp;
            <span style="color: #27AE60; font-weight: 700;">🟢 {s_count} Susceptible</span>
        </div>
        """, unsafe_allow_html=True)
    
    # ────────────────────────────────────────────────────────────────────────
    #  Column 2: SHAP Explainability
    # ────────────────────────────────────────────────────────────────────────
    with col2:
        st.markdown("### 📊 SHAP Explanation")
        
        if shap_computed:
            # Select which antibiotic to explain
            explain_abx = st.selectbox(
                "Explain prediction for:",
                ANTIBIOTIC_NAMES,
                index=0,
                key="explain_select"
            )
            abx_idx = ANTIBIOTIC_NAMES.index(explain_abx)
            
            # Generate waterfall chart
            try:
                fig = generate_waterfall_chart(
                    shap_values, FEATURE_NAMES, 0, abx_idx,
                    explain_abx, X_input, top_n=10
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as e:
                st.warning(f"Could not generate SHAP chart: {e}")
            
            # CARD gene annotations for top features
            st.markdown("#### 🧬 CARD Gene Annotations")
            
            if isinstance(shap_values, list) and abx_idx < len(shap_values):
                sv = shap_values[abx_idx]
                # Handle XGBClassifier list-of-arrays-per-class format
                if isinstance(sv, list):
                    sv = sv[1] if len(sv) > 1 else sv[0]
                if isinstance(sv, np.ndarray):
                    if len(sv.shape) == 3:
                        vals = sv[0, :, 1]
                    elif len(sv.shape) == 2:
                        vals = sv[0, :]
                    elif len(sv.shape) == 1:
                        vals = sv
                    else:
                        vals = np.zeros(len(FEATURE_NAMES))
                else:
                    vals = np.zeros(len(FEATURE_NAMES))
            else:
                vals = np.zeros(len(FEATURE_NAMES))
            
            top_idx = np.argsort(np.abs(vals))[-5:][::-1]
            
            for idx in top_idx:
                feat = FEATURE_NAMES[idx]
                card_genes = get_card_annotation(feat)
                clean_feat = feat.replace("_encoded", "").replace("_", " ")
                gene_str = ", ".join(card_genes[:2])
                direction = "↑ drives resistance" if vals[idx] > 0 else "↓ reduces resistance"
                
                st.markdown(f"""
                <div style="padding: 6px 12px; margin: 3px 0; border-radius: 6px; 
                            background: white; border-left: 3px solid #2E86AB;
                            box-shadow: 0 1px 2px rgba(0,0,0,0.04);">
                    <strong style="color: #1A1A2E;">{clean_feat}</strong>
                    <span style="color: #666; font-size: 0.8rem;"> — {direction}</span><br/>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #2E86AB;">
                        Genes: {gene_str}
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("SHAP computation failed. Predictions still shown.")
    
    # ────────────────────────────────────────────────────────────────────────
    #  Column 3: Antibiotic Recommendation
    # ────────────────────────────────────────────────────────────────────────
    with col3:
        st.markdown("### 💊 Recommendation")
        
        if primary:
            rec_color = "#27AE60" if primary["status"] == "Susceptible" else "#F39C12" if primary["status"] == "Intermediate" else "#E74C3C"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #E8F4FD 0%, #d4edfc 100%);
                        padding: 18px 20px; border-radius: 8px; margin-bottom: 12px;
                        border: 2px solid #2E86AB;">
                <p style="color: #2E86AB; font-size: 0.8rem; font-weight: 600; margin: 0; text-transform: uppercase; letter-spacing: 0.5px;">
                    🏆 BEST RECOMMENDATION
                </p>
                <h2 style="color: #1A1A2E; margin: 6px 0 4px 0; font-size: 1.4rem;">
                    {primary['antibiotic']}
                </h2>
                <p style="color: #666; font-size: 0.85rem; margin: 0;">
                    {primary['class']}
                </p>
                <div style="margin-top: 8px;">
                    <span style="background: {rec_color}; color: white; padding: 4px 14px; 
                                 border-radius: 20px; font-weight: 600; font-size: 0.82rem;">
                        {primary['emoji']} {primary['status']}
                    </span>
                    <span style="font-family: 'JetBrains Mono', monospace; color: #1A1A2E; 
                                 font-weight: 600; margin-left: 10px;">
                        {primary['susceptibility_score']:.0%} susceptible
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Full ranking
        st.markdown("#### Ranked Antibiotics")
        
        for i, abx in enumerate(ranked):
            if abx["status"] == "Susceptible":
                bar_color = "#27AE60"
                bg_color = "#EAFAF1"
            elif abx["status"] == "Intermediate":
                bar_color = "#F39C12"
                bg_color = "#FEF9E7"
            else:
                bar_color = "#E74C3C"
                bg_color = "#FDEDEC"
            
            pct = abx["susceptibility_score"] * 100
            
            st.markdown(f"""
            <div style="padding: 6px 10px; margin: 3px 0; border-radius: 6px; 
                        background: {bg_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.82rem; font-weight: 500; color: #1A1A2E;">
                        {i+1}. {abx['antibiotic']}
                    </span>
                    <span style="font-size: 0.85rem;">{abx['emoji']}</span>
                </div>
                <div style="background: #e5e7eb; border-radius: 4px; height: 6px; margin-top: 4px;">
                    <div style="background: {bar_color}; width: {pct}%; height: 100%; 
                                border-radius: 4px;"></div>
                </div>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #666;">
                    {pct:.0f}% susceptible
                </span>
            </div>
            """, unsafe_allow_html=True)
    
    # ── Global Feature Importance (below the 3 columns) ──
    st.markdown("---")
    with st.expander("📈 Global Feature Importance (All Antibiotics)", expanded=False):
        if shap_computed:
            try:
                fig_global = generate_global_importance(
                    shap_values, FEATURE_NAMES, ANTIBIOTIC_NAMES, top_n=15
                )
                st.pyplot(fig_global, use_container_width=True)
                plt.close(fig_global)
            except Exception as e:
                st.warning(f"Could not generate global importance chart: {e}")

else:
    # ── Landing state — no prediction yet ──
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px;">
        <h2 style="color: #2E86AB; font-size: 1.6rem;">Welcome to ResistAI</h2>
        <p style="color: #6b7280; font-size: 1.05rem; max-width: 600px; margin: 8px auto 24px;">
            Enter a bacterial isolate's details and known susceptibility test results 
            in the sidebar, then click <strong>🔬 Predict Resistance</strong>.
        </p>
        <div style="display: flex; gap: 30px; justify-content: center; flex-wrap: wrap; margin-top: 30px;">
            <div style="background: white; padding: 24px; border-radius: 8px; width: 200px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.06); text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 8px;">🧫</div>
                <h4 style="margin: 0; color: #1A1A2E;">Predict</h4>
                <p style="color: #888; font-size: 0.82rem; margin: 4px 0 0;">
                    Multi-label resistance across 16 antibiotics
                </p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 8px; width: 200px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.06); text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 8px;">📊</div>
                <h4 style="margin: 0; color: #1A1A2E;">Explain</h4>
                <p style="color: #888; font-size: 0.82rem; margin: 4px 0 0;">
                    SHAP charts show which features drive each prediction
                </p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 8px; width: 200px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.06); text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 8px;">💊</div>
                <h4 style="margin: 0; color: #1A1A2E;">Recommend</h4>
                <p style="color: #888; font-size: 0.82rem; margin: 4px 0 0;">
                    Ranked antibiotics by predicted susceptibility
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ──
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; font-size: 0.78rem; padding: 8px 0;">
    <strong>ResistAI</strong> — Predict resistance. Prescribe with confidence.<br/>
    ⚠️ Clinical decision-support tool only. Not a replacement for laboratory culture and sensitivity testing.
</div>
""", unsafe_allow_html=True)
