"""
Antibiotic Recommender for ResistAI.
Given a resistance prediction profile, ranks antibiotics by
predicted susceptibility probability and recommends the best option.
"""

import pandas as pd
import numpy as np


# Brand color codes for resistance status
COLOR_MAP = {
    "Susceptible": "#27AE60",    # Green — safe
    "Intermediate": "#F39C12",   # Amber — warning
    "Resistant": "#E74C3C",      # Red — danger
    "Unknown": "#95A5A6",        # Gray — unknown
}

STATUS_EMOJI = {
    "Susceptible": "🟢",
    "Intermediate": "🟡",
    "Resistant": "🔴",
    "Unknown": "⚪",
}


# Antibiotic class groupings (for diversity in recommendations)
ANTIBIOTIC_CLASSES = {
    "Amikacin": "Aminoglycosides",
    "Amoxicillin-Ampicillin": "Penicillins",
    "Amoxicillin-Clavulanate": "Penicillin Combinations",
    "Cefazolin": "Cephalosporins (1st gen)",
    "Cefoxitin": "Cephalosporins (2nd gen)",
    "Ceftazidime": "Cephalosporins (3rd gen)",
    "Ceftriaxone-Cefotaxime": "Cephalosporins (3rd gen)",
    "Chloramphenicol": "Amphenicols",
    "Ciprofloxacin": "Fluoroquinolones",
    "Colistin": "Polymyxins",
    "Gentamicin": "Aminoglycosides",
    "Imipenem": "Carbapenems",
    "Nalidixic Acid": "Quinolones",
    "Nitrofurantoin": "Nitrofurans",
    "Ofloxacin": "Fluoroquinolones",
    "Trimethoprim-Sulfamethoxazole": "Folate Inhibitors",
}


def rank_antibiotics(predictions):
    """
    Rank antibiotics by predicted susceptibility (best = lowest resistance probability).
    
    Args:
        predictions: dict from model_inference.predict_resistance
            {abx_name: {"label": str, "probability": float, "encoded": int}}
    
    Returns:
        ranked: list of dicts, sorted by susceptibility (best first)
    """
    ranked = []
    for abx, pred in predictions.items():
        susceptibility_score = 1.0 - pred["probability"]  # Higher = more susceptible
        
        ranked.append({
            "antibiotic": abx,
            "class": ANTIBIOTIC_CLASSES.get(abx, "Other"),
            "status": pred["label"],
            "resistance_probability": pred["probability"],
            "susceptibility_score": susceptibility_score,
            "confidence": abs(pred["probability"] - 0.5) * 2,  # 0-1 confidence
            "color": COLOR_MAP.get(pred["label"], COLOR_MAP["Unknown"]),
            "emoji": STATUS_EMOJI.get(pred["label"], STATUS_EMOJI["Unknown"]),
        })
    
    # Sort by susceptibility score (highest first = most likely to work)
    ranked.sort(key=lambda x: x["susceptibility_score"], reverse=True)
    
    return ranked


def get_recommendation(ranked_antibiotics, top_n=3):
    """
    Get the top-N recommended antibiotics.
    Prioritizes susceptible antibiotics, then intermediate.
    For ties, prefers narrow-spectrum agents.
    
    Returns:
        recommendations: list of top-N best antibiotics
        primary: the single best recommendation
        warnings: list of warning strings
    """
    susceptible = [a for a in ranked_antibiotics if a["status"] == "Susceptible"]
    intermediate = [a for a in ranked_antibiotics if a["status"] == "Intermediate"]
    resistant = [a for a in ranked_antibiotics if a["status"] == "Resistant"]
    
    warnings = []
    
    # Check for pan-resistance
    if len(susceptible) == 0 and len(intermediate) == 0:
        warnings.append("⚠️ All antibiotics show predicted resistance. Consult infectious disease specialist immediately.")
    elif len(susceptible) == 0:
        warnings.append("⚠️ No clearly susceptible antibiotics found. Consider intermediate options with caution.")
    
    # MDR warning
    if len(resistant) >= 3:
        warnings.append(f"⚠️ Multi-drug resistant profile detected ({len(resistant)} antibiotics resistant).")
    
    # Get top recommendations
    recommendations = ranked_antibiotics[:top_n]
    primary = ranked_antibiotics[0] if ranked_antibiotics else None
    
    return recommendations, primary, warnings


def format_recommendation_table(ranked_antibiotics):
    """
    Format the ranking as a DataFrame for display.
    """
    rows = []
    for i, abx in enumerate(ranked_antibiotics):
        rows.append({
            "Rank": i + 1,
            "Antibiotic": abx["antibiotic"],
            "Class": abx["class"],
            "Status": f"{abx['emoji']} {abx['status']}",
            "Susceptibility": f"{abx['susceptibility_score']:.1%}",
            "Confidence": f"{abx['confidence']:.1%}",
        })
    
    return pd.DataFrame(rows)
