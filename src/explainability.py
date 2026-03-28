"""
SHAP Explainability module for ResistAI.
Generates per-prediction and global feature importance visualizations.
Maps top features to CARD gene annotations.
"""

import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import io
import base64


# ── CARD Gene Mapping ──────────────────────────────────────────────────────────

CARD_GENE_MAP = {
    "Amikacin": ["aac(6')-Ib", "armA", "rmtB"],
    "Amoxicillin-Ampicillin": ["blaTEM-1", "blaSHV-1", "blaOXA-1"],
    "Amoxicillin-Clavulanate": ["blaTEM-1", "blaOXA-1"],
    "Cefazolin": ["blaTEM-1", "blaSHV-1"],
    "Cefoxitin": ["blaCMY-2", "ampC"],
    "Ceftazidime": ["blaCTX-M-15", "blaSHV-12", "blaNDM-1"],
    "Ceftriaxone-Cefotaxime": ["blaCTX-M-15", "blaCMY-2"],
    "Chloramphenicol": ["catA1", "cmlA", "floR"],
    "Ciprofloxacin": ["qnrA", "qnrB", "aac(6')-Ib-cr"],
    "Colistin": ["mcr-1", "mcr-2", "pmrA mutation"],
    "Gentamicin": ["aac(3)-IIa", "ant(2'')-Ia"],
    "Imipenem": ["blaKPC-2", "blaNDM-1", "blaOXA-48"],
    "Nalidixic Acid": ["gyrA mutation", "qnrS"],
    "Nitrofurantoin": ["nfsA mutation", "nfsB mutation"],
    "Ofloxacin": ["qnrA", "gyrA mutation", "parC mutation"],
    "Trimethoprim-Sulfamethoxazole": ["dfrA1", "sul1", "sul2"],
    # Feature-level mappings
    "Organism_encoded": ["Species-specific intrinsic resistance"],
    "Age": ["Age-associated resistance patterns"],
    "Gender_encoded": ["No known gene association"],
    "Source_encoded": ["Dataset source indicator"],
    "Diabetes": ["Infection susceptibility factor"],
    "Hypertension": ["Comorbidity indicator"],
    "Hospital_before": ["Healthcare-associated infection risk"],
    "Infection_Freq": ["Recurrence-associated resistance selection"],
}


def get_card_annotation(feature_name):
    """Map a feature to its CARD gene annotations."""
    return CARD_GENE_MAP.get(feature_name, ["No known gene mapping"])


def compute_shap_values(model, X_train, X_explain=None, max_samples=100):
    """
    Compute SHAP values for the model.
    
    Args:
        model: trained model (supports multi-output)
        X_train: training data for background (sampled)
        X_explain: samples to explain (default: X_train subset)
        max_samples: max background samples for SHAP
    
    Returns:
        shap_values: SHAP values object
        explainer: SHAP TreeExplainer
    """
    # Use a subset for background
    if len(X_train) > max_samples:
        bg = X_train.sample(max_samples, random_state=42)
    else:
        bg = X_train
    
    explainer = shap.TreeExplainer(model)
    
    if X_explain is None:
        X_explain = bg
    
    shap_values = explainer.shap_values(X_explain)
    
    return shap_values, explainer


def generate_waterfall_chart(shap_values, feature_names, sample_idx, antibiotic_idx,
                              antibiotic_name, X_sample, top_n=10):
    """
    Generate a SHAP waterfall chart for a single prediction.
    Returns a matplotlib figure.
    """
    # Brand colors
    RESIST_RED = "#C0392B"
    SAFE_GREEN = "#27AE60"
    BG_COLOR = "#F7F9FC"
    TEXT_COLOR = "#1A1A2E"
    
    # Get SHAP values for this sample and antibiotic
    # shap_values is a list of per-estimator SHAP outputs
    if isinstance(shap_values, list):
        sv = shap_values[antibiotic_idx]
        # XGBClassifier returns list of [class_0, class_1] or a 2D array
        if isinstance(sv, list):
            # List of arrays per class — take class 1 (resistant)
            sv = sv[1] if len(sv) > 1 else sv[0]
        if isinstance(sv, np.ndarray):
            if len(sv.shape) == 3:
                vals = sv[sample_idx, :, 1]
            elif len(sv.shape) == 2:
                vals = sv[sample_idx, :]
            elif len(sv.shape) == 1:
                vals = sv
            else:
                vals = np.zeros(len(feature_names))
        else:
            vals = np.zeros(len(feature_names))
    else:
        vals = shap_values[sample_idx, :]
    
    # Get top features by absolute SHAP value
    abs_vals = np.abs(vals)
    top_indices = np.argsort(abs_vals)[-top_n:][::-1]
    
    top_features = [feature_names[i] for i in top_indices]
    top_vals = [vals[i] for i in top_indices]
    top_x_vals = [float(X_sample.iloc[sample_idx, i]) for i in top_indices]
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    colors = [RESIST_RED if v > 0 else SAFE_GREEN for v in top_vals]
    
    y_pos = np.arange(len(top_features))
    bars = ax.barh(y_pos, top_vals, color=colors, edgecolor='white', linewidth=0.5, height=0.65)
    
    # Labels
    clean_names = [f.replace("_encoded", "").replace("_", " ") for f in top_features]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(clean_names, fontsize=10, color=TEXT_COLOR)
    ax.invert_yaxis()
    
    ax.set_xlabel("SHAP Value (impact on resistance prediction)", fontsize=10, color=TEXT_COLOR)
    ax.set_title(f"Feature Importance — {antibiotic_name}", fontsize=13, 
                 fontweight='bold', color=TEXT_COLOR, pad=12)
    
    # Add value annotations
    for i, (bar, val, feat) in enumerate(zip(bars, top_vals, top_features)):
        x_pos = bar.get_width()
        card_genes = get_card_annotation(feat)
        gene_str = card_genes[0] if card_genes else ""
        ax.annotate(f"{val:+.3f}", xy=(x_pos, bar.get_y() + bar.get_height()/2),
                   fontsize=8, color=TEXT_COLOR, va='center',
                   ha='left' if val > 0 else 'right',
                   xytext=(5 if val > 0 else -5, 0), textcoords='offset points')
    
    ax.axvline(x=0, color=TEXT_COLOR, linewidth=0.8, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(TEXT_COLOR)
    ax.spines['left'].set_color(TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    
    plt.tight_layout()
    return fig


def generate_global_importance(shap_values, feature_names, antibiotic_names, top_n=15):
    """
    Generate a global feature importance bar chart across all antibiotics.
    Returns a matplotlib figure.
    """
    BG_COLOR = "#F7F9FC"
    TEXT_COLOR = "#1A1A2E"
    BLUE = "#2E86AB"
    
    # Average absolute SHAP values across all antibiotics and samples
    if isinstance(shap_values, list):
        all_vals = []
        for sv in shap_values:
            # Handle list-of-arrays per class from XGBClassifier
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            if isinstance(sv, np.ndarray):
                if len(sv.shape) == 3:
                    all_vals.append(np.abs(sv[:, :, 1]).mean(axis=0))
                elif len(sv.shape) == 2:
                    all_vals.append(np.abs(sv).mean(axis=0))
                elif len(sv.shape) == 1:
                    all_vals.append(np.abs(sv))
        if all_vals:
            mean_importance = np.mean(all_vals, axis=0)
        else:
            mean_importance = np.zeros(len(feature_names))
    else:
        mean_importance = np.abs(shap_values).mean(axis=0)
    
    # Top features
    top_idx = np.argsort(mean_importance)[-top_n:][::-1]
    top_features = [feature_names[i] for i in top_idx]
    top_importance = [mean_importance[i] for i in top_idx]
    
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    y_pos = np.arange(len(top_features))
    bars = ax.barh(y_pos, top_importance, color=BLUE, edgecolor='white', 
                   linewidth=0.5, height=0.65, alpha=0.85)
    
    clean_names = [f.replace("_encoded", "").replace("_", " ") for f in top_features]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(clean_names, fontsize=10, color=TEXT_COLOR)
    ax.invert_yaxis()
    
    ax.set_xlabel("Mean |SHAP Value| (global importance)", fontsize=10, color=TEXT_COLOR)
    ax.set_title("Global Feature Importance Across All Antibiotics", fontsize=13,
                 fontweight='bold', color=TEXT_COLOR, pad=12)
    
    # Add CARD annotations for top features
    for i, (bar, feat) in enumerate(zip(bars, top_features)):
        card_genes = get_card_annotation(feat)
        gene_str = ", ".join(card_genes[:2])
        ax.annotate(f"  {gene_str}", xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                   fontsize=7, color="#666", va='center', ha='left', style='italic')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(TEXT_COLOR)
    ax.spines['left'].set_color(TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    
    plt.tight_layout()
    return fig


def fig_to_base64(fig):
    """Convert a matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64
