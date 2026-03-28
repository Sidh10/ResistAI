"""
Data processing pipeline for ResistAI — REAL DATA VERSION.
Loads Dataset.xlsx (Mendeley) and Bacteria_dataset_Multiresictance.csv (Kaggle),
handles real-world messiness, and outputs clean train/test splits.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
#  CLSI Breakpoints (mm zone diameters → R/I/S)
#  Source: CLSI M100 standard for Enterobacterales
# ══════════════════════════════════════════════════════════════════════════════

CLSI_BREAKPOINTS = {
    # (Susceptible >=, Intermediate range, Resistant <=)
    "IMIPENEM":      {"S": 23, "R": 19},
    "CEFTAZIDIME":   {"S": 21, "R": 17},
    "GENTAMICIN":    {"S": 15, "R": 12},
    "AUGMENTIN":     {"S": 18, "R": 13},  # Amoxicillin-Clavulanate
    "CIPROFLOXACIN": {"S": 21, "R": 15},
}

# ── Antibiotic name mapping (Kaggle abbreviation → full name) ──────────────
KAGGLE_ABX_MAP = {
    "AMX/AMP":           "Amoxicillin-Ampicillin",
    "AMC":               "Amoxicillin-Clavulanate",
    "CZ":                "Cefazolin",
    "FOX":               "Cefoxitin",
    "CTX/CRO":           "Ceftriaxone-Cefotaxime",
    "IPM":               "Imipenem",
    "GEN":               "Gentamicin",
    "AN":                "Amikacin",
    "Acide nalidixique":  "Nalidixic Acid",
    "ofx":               "Ofloxacin",
    "CIP":               "Ciprofloxacin",
    "C":                 "Chloramphenicol",
    "Co-trimoxazole":    "Trimethoprim-Sulfamethoxazole",
    "Furanes":           "Nitrofurantoin",
    "colistine":         "Colistin",
}

# Reverse map for Mendeley → Kaggle alignment
MENDELEY_TO_COMMON = {
    "IMIPENEM":      "Imipenem",
    "CEFTAZIDIME":   "Ceftazidime",
    "GENTAMICIN":    "Gentamicin",
    "AUGMENTIN":     "Amoxicillin-Clavulanate",
    "CIPROFLOXACIN": "Ciprofloxacin",
}

# Organism name normalization (fix typos in Kaggle)
ORGANISM_FIXES = {
    "E.coi":                   "Escherichia coli",
    "E.cli":                   "Escherichia coli",
    "E. coli":                 "Escherichia coli",
    "Escherichia coli":        "Escherichia coli",
    "Klbsiella pneumoniae":    "Klebsiella pneumoniae",
    "Klebsiella pneumoniae":   "Klebsiella pneumoniae",
    "Enter.bacteria spp.":     "Enterobacteria spp.",
    "Enteobacteria spp.":      "Enterobacteria spp.",
    "Enterobacteria spp.":     "Enterobacteria spp.",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Label encoding
# ══════════════════════════════════════════════════════════════════════════════

LABEL_MAP = {"Susceptible": 0, "Intermediate": 1, "Resistant": 2}


def normalize_resistance_label(val):
    """Normalize the messy Kaggle resistance labels to S/I/R."""
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    if val in ("R", "r", "Resistant"):
        return "Resistant"
    elif val in ("S", "s", "Susceptible"):
        return "Susceptible"
    elif val in ("I", "i", "Intermediate"):
        return "Intermediate"
    elif val in ("?", "missing", "error", "unknown", "--"):
        return np.nan
    else:
        return np.nan


def zone_to_label(zone_mm, antibiotic):
    """Convert inhibition zone diameter (mm) to R/I/S using CLSI breakpoints."""
    bp = CLSI_BREAKPOINTS.get(antibiotic)
    if bp is None or pd.isna(zone_mm):
        return np.nan
    zone_mm = float(zone_mm)
    if zone_mm >= bp["S"]:
        return "Susceptible"
    elif zone_mm <= bp["R"]:
        return "Resistant"
    else:
        return "Intermediate"


# ══════════════════════════════════════════════════════════════════════════════
#  Loading functions
# ══════════════════════════════════════════════════════════════════════════════

def load_mendeley(path="data/raw/Dataset.xlsx"):
    """Load and process Mendeley dataset (zone diameters → R/I/S)."""
    df = pd.read_excel(path)
    print(f"[Mendeley] Loaded {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"  Columns: {list(df.columns)}")
    
    # Convert zone diameters to R/I/S labels
    for col in ["IMIPENEM", "CEFTAZIDIME", "GENTAMICIN", "AUGMENTIN", "CIPROFLOXACIN"]:
        common_name = MENDELEY_TO_COMMON[col]
        df[common_name] = df[col].apply(lambda x: zone_to_label(x, col))
    
    # Add source and Location as features
    df["Source"] = "Mendeley"
    
    # Extract location components (site-type, e.g., IFE = site, T/C/S = type)
    df["Site"] = df["Location"].str.split("-").str[0]
    df["Sample_Type"] = df["Location"].str.split("-").str[1]
    
    # Print conversion stats
    converted_cols = list(MENDELEY_TO_COMMON.values())
    for col in converted_cols:
        dist = df[col].value_counts().to_dict()
        print(f"  {col}: {dist}")
    
    return df


def load_kaggle(path="data/raw/Bacteria_dataset_Multiresictance.csv"):
    """Load and process Kaggle multi-resistance dataset."""
    df = pd.read_csv(path)
    print(f"[Kaggle]   Loaded {df.shape[0]} rows × {df.shape[1]} cols")
    
    # ── Extract organism (remove sample ID prefix) ──
    df["Organism_Raw"] = df["Souches"].str.replace(r'^S\d+\s*', '', regex=True)
    
    # Normalize organism names (fix typos)
    df["Organism"] = df["Organism_Raw"].map(
        lambda x: ORGANISM_FIXES.get(x, x) if pd.notna(x) else x
    )
    
    # ── Parse age/gender ──
    df["Age"] = df["age/gender"].str.extract(r'(\d+)').astype(float)
    df["Gender"] = df["age/gender"].str.extract(r'/([MF])')
    
    # ── Normalize antibiotic labels ──
    kaggle_abx_cols = list(KAGGLE_ABX_MAP.keys())
    for col in kaggle_abx_cols:
        common_name = KAGGLE_ABX_MAP[col]
        df[common_name] = df[col].apply(normalize_resistance_label)
    
    # ── Clean comorbidity features ──
    df["Diabetes_flag"] = df["Diabetes"].map({"True": 1, "No": 0}).fillna(-1).astype(int)
    df["Hypertension_flag"] = df["Hypertension"].map({"Yes": 1, "No": 0}).fillna(-1).astype(int)
    df["Hospital_before_flag"] = df["Hospital_before"].map({"Yes": 1, "No": 0}).fillna(-1).astype(int)
    
    # Clean Infection_Freq
    df["Infection_Freq_clean"] = pd.to_numeric(df["Infection_Freq"], errors="coerce").fillna(-1).astype(int)
    
    df["Source"] = "Kaggle"
    
    # Print organism stats
    print(f"  Organisms (after normalization): {df['Organism'].nunique()} unique")
    print(f"  Top 5: {df['Organism'].value_counts().head(5).to_dict()}")
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def find_shared_antibiotics(mendeley_df, kaggle_df):
    """Find antibiotics present in both datasets."""
    mendeley_abx = set(MENDELEY_TO_COMMON.values())
    kaggle_abx = set(KAGGLE_ABX_MAP.values())
    shared = mendeley_abx & kaggle_abx
    all_abx = mendeley_abx | kaggle_abx
    
    print(f"\n[Antibiotic Alignment]")
    print(f"  Mendeley: {len(mendeley_abx)} antibiotics")
    print(f"  Kaggle:   {len(kaggle_abx)} antibiotics")
    print(f"  Shared:   {len(shared)} → {sorted(shared)}")
    print(f"  Total:    {len(all_abx)} unique antibiotics")
    
    return sorted(all_abx), sorted(shared)


def build_unified_dataset(mendeley_df, kaggle_df, all_antibiotics):
    """
    Build a unified dataset from both sources.
    Uses Kaggle as primary (larger, richer features).
    Mendeley adds zone-converted labels for shared antibiotics.
    """
    print(f"\n[Building Unified Dataset]")
    
    # ── Kaggle rows (primary) ──
    kaggle_records = []
    for _, row in kaggle_df.iterrows():
        record = {
            "Organism": row.get("Organism", "Unknown"),
            "Age": row.get("Age", np.nan),
            "Gender": row.get("Gender", "Unknown"),
            "Diabetes": row.get("Diabetes_flag", -1),
            "Hypertension": row.get("Hypertension_flag", -1),
            "Hospital_before": row.get("Hospital_before_flag", -1),
            "Infection_Freq": row.get("Infection_Freq_clean", -1),
            "Source": "Kaggle",
        }
        for abx in all_antibiotics:
            record[abx] = row.get(abx, np.nan)
        kaggle_records.append(record)
    
    # ── Mendeley rows (supplement; no patient demographics) ──
    mendeley_records = []
    for _, row in mendeley_df.iterrows():
        record = {
            "Organism": "Unknown",  # Mendeley doesn't specify organism
            "Age": np.nan,
            "Gender": "Unknown",
            "Diabetes": -1,
            "Hypertension": -1,
            "Hospital_before": -1,
            "Infection_Freq": -1,
            "Source": "Mendeley",
        }
        for abx in all_antibiotics:
            record[abx] = row.get(abx, np.nan)
        mendeley_records.append(record)
    
    combined = pd.DataFrame(kaggle_records + mendeley_records)
    print(f"  Combined: {combined.shape[0]} rows × {combined.shape[1]} cols")
    print(f"  Kaggle: {len(kaggle_records)}, Mendeley: {len(mendeley_records)}")
    
    return combined


def preprocess_unified(df, all_antibiotics):
    """
    Full preprocessing: encode labels, handle missing, engineer features.
    """
    print(f"\n[Preprocessing]")
    df = df.copy()
    
    # ── 1. Handle missing antibiotic values (mode imputation per organism) ──
    missing_before = df[all_antibiotics].isna().sum().sum()
    
    for col in all_antibiotics:
        for org in df["Organism"].unique():
            mask = (df["Organism"] == org) & df[col].isna()
            if mask.sum() > 0:
                mode_vals = df.loc[df["Organism"] == org, col].mode()
                if len(mode_vals) > 0:
                    df.loc[mask, col] = mode_vals.iloc[0]
    
    # Fill remaining NaN (organisms with all-NaN for an antibiotic)
    for col in all_antibiotics:
        mode_all = df[col].mode()
        if len(mode_all) > 0:
            df[col] = df[col].fillna(mode_all.iloc[0])
        else:
            df[col] = df[col].fillna("Susceptible")
    
    missing_after = df[all_antibiotics].isna().sum().sum()
    print(f"  Missing antibiotic values: {missing_before} → {missing_after}")
    
    # ── 2. Encode resistance labels ──
    for col in all_antibiotics:
        df[f"{col}_encoded"] = df[col].map(LABEL_MAP).fillna(0).astype(int)
    
    # ── 3. Encode categorical features ──
    le_org = LabelEncoder()
    df["Organism_encoded"] = le_org.fit_transform(df["Organism"].fillna("Unknown"))
    
    le_gender = LabelEncoder()
    df["Gender_encoded"] = le_gender.fit_transform(df["Gender"].fillna("Unknown"))
    
    le_source = LabelEncoder()
    df["Source_encoded"] = le_source.fit_transform(df["Source"])
    
    # ── 4. Impute Age ──
    median_age = df["Age"].median()
    df["Age"] = df["Age"].fillna(median_age)
    
    # ── 5. Engineer resistance profile features ──
    encoded_cols = [f"{c}_encoded" for c in all_antibiotics]
    df["Resistance_Count"] = (df[encoded_cols] == 2).sum(axis=1)
    df["Susceptible_Count"] = (df[encoded_cols] == 0).sum(axis=1)
    df["Intermediate_Count"] = (df[encoded_cols] == 1).sum(axis=1)
    df["MDR_Flag"] = (df["Resistance_Count"] >= 3).astype(int)
    df["Resistance_Ratio"] = df["Resistance_Count"] / len(all_antibiotics)
    
    encoders = {"organism": le_org, "gender": le_gender, "source": le_source}
    
    return df, encoders


def build_feature_matrix(df, all_antibiotics):
    """Build X (features) and Y (multi-label binary targets).
    
    IMPORTANT: antibiotic encoded columns and derived aggregate features
    (Resistance_Count, MDR_Flag, etc.) are NOT included as features
    because they are computed from the targets — indirect leakage.
    Only pure clinical/demographic features are used.
    """
    feature_cols = [
        "Age", "Gender_encoded", "Organism_encoded", "Source_encoded",
        "Diabetes", "Hypertension", "Hospital_before", "Infection_Freq",
    ]
    
    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    X = df[feature_cols].copy()
    
    # Y: binary multi-label (Resistant=1, else=0)
    Y = pd.DataFrame()
    for abx in all_antibiotics:
        Y[abx] = (df[f"{abx}_encoded"] == 2).astype(int)
    
    print(f"\n[Feature Matrix]")
    print(f"  X: {X.shape} (pure clinical features — no target leakage)")
    print(f"  Features: {feature_cols}")
    print(f"  Y: {Y.shape}")
    
    return X, Y, feature_cols, all_antibiotics


def run_pipeline():
    """Execute the full data pipeline."""
    print("=" * 65)
    print("ResistAI Data Pipeline — REAL DATA")
    print("=" * 65)
    
    # Step 1: Load
    print("\n[Step 1] Loading datasets...")
    mendeley_df = load_mendeley()
    kaggle_df = load_kaggle()
    
    # Step 2: Find shared antibiotics
    all_antibiotics, shared = find_shared_antibiotics(mendeley_df, kaggle_df)
    
    # Step 3: Build unified dataset
    combined = build_unified_dataset(mendeley_df, kaggle_df, all_antibiotics)
    
    # Step 4: Preprocess
    processed, encoders = preprocess_unified(combined, all_antibiotics)
    
    # Step 5: Build features
    X, Y, feature_cols, abx_names = build_feature_matrix(processed, all_antibiotics)
    
    # Step 6: Train/test split
    print(f"\n[Train/Test Split]")
    
    # Stratify on a column that has enough representation
    strat_col = Y.iloc[:, 0]  # First antibiotic
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=strat_col
    )
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    
    # Step 7: Verify no data leakage
    train_idx = set(X_train.index)
    test_idx = set(X_test.index)
    assert train_idx.isdisjoint(test_idx), "❌ DATA LEAKAGE!"
    print(f"  ✅ No data leakage")
    
    # Step 8: Class distribution
    print(f"\n[Class Distribution (train set)]")
    for col in abx_names:
        dist = Y_train[col].value_counts().to_dict()
        r_pct = dist.get(1, 0) / len(Y_train) * 100
        s_pct = dist.get(0, 0) / len(Y_train) * 100
        imbalance = "⚠️ IMBALANCED" if r_pct > 85 or s_pct > 85 else ""
        print(f"  {col:35s} R={dist.get(1,0):5d} ({r_pct:5.1f}%)  S={dist.get(0,0):5d} ({s_pct:5.1f}%)  {imbalance}")
    
    # Step 9: Save
    os.makedirs("data/processed", exist_ok=True)
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    Y_train.to_csv("data/processed/Y_train.csv", index=False)
    Y_test.to_csv("data/processed/Y_test.csv", index=False)
    processed.to_csv("data/processed/full_processed.csv", index=False)
    pd.Series(feature_cols).to_csv("data/processed/feature_names.csv", index=False)
    pd.Series(abx_names).to_csv("data/processed/antibiotic_names.csv", index=False)
    
    print(f"\n  Saved all processed files to data/processed/")
    print(f"\n{'='*65}")
    print(f"✅ Real data pipeline complete!")
    print(f"{'='*65}")
    
    return X_train, X_test, Y_train, Y_test, feature_cols, abx_names, encoders


if __name__ == "__main__":
    run_pipeline()
