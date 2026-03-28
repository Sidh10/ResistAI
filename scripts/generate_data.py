"""
Generate realistic synthetic AMR (Antimicrobial Resistance) datasets
mirroring the structure of Mendeley AMR and Kaggle multi-resistance datasets.

This script creates two CSV files:
1. data/raw/mendeley_amr.csv — Primary dataset with isolate-level AST results
2. data/raw/kaggle_resistance.csv — Secondary dataset with multi-drug resistance profiles
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────────────

ORGANISMS = [
    "Escherichia coli", "Klebsiella pneumoniae", "Staphylococcus aureus",
    "Pseudomonas aeruginosa", "Acinetobacter baumannii", "Enterococcus faecalis",
    "Enterococcus faecium", "Streptococcus pneumoniae", "Proteus mirabilis",
    "Serratia marcescens"
]

ANTIBIOTICS = [
    "Amoxicillin", "Ampicillin", "Ciprofloxacin", "Levofloxacin",
    "Gentamicin", "Amikacin", "Ceftriaxone", "Cefepime",
    "Meropenem", "Imipenem", "Trimethoprim-Sulfamethoxazole",
    "Piperacillin-Tazobactam", "Tetracycline", "Doxycycline",
    "Colistin", "Vancomycin"
]

SPECIMEN_TYPES = ["Blood", "Urine", "Sputum", "Wound", "CSF", "BAL"]
GENDERS = ["Male", "Female"]
DEPARTMENTS = ["ICU", "Medicine", "Surgery", "Pediatrics", "Emergency"]

# Known resistance patterns (biologically plausible)
# E.coli: typically resistant to Ampicillin, Amoxicillin; susceptible to Carbapenems
# MRSA: resistant to many beta-lactams; susceptible to Vancomycin
# Pseudomonas: intrinsically resistant to many; susceptible to antipseudomonal agents
RESISTANCE_PROFILES = {
    "Escherichia coli": {
        "Amoxicillin": 0.65, "Ampicillin": 0.70, "Ciprofloxacin": 0.45,
        "Levofloxacin": 0.40, "Gentamicin": 0.30, "Amikacin": 0.10,
        "Ceftriaxone": 0.25, "Cefepime": 0.20, "Meropenem": 0.05,
        "Imipenem": 0.05, "Trimethoprim-Sulfamethoxazole": 0.55,
        "Piperacillin-Tazobactam": 0.15, "Tetracycline": 0.50,
        "Doxycycline": 0.35, "Colistin": 0.02, "Vancomycin": 0.95
    },
    "Klebsiella pneumoniae": {
        "Amoxicillin": 0.85, "Ampicillin": 0.90, "Ciprofloxacin": 0.40,
        "Levofloxacin": 0.35, "Gentamicin": 0.35, "Amikacin": 0.15,
        "Ceftriaxone": 0.40, "Cefepime": 0.35, "Meropenem": 0.10,
        "Imipenem": 0.10, "Trimethoprim-Sulfamethoxazole": 0.50,
        "Piperacillin-Tazobactam": 0.25, "Tetracycline": 0.45,
        "Doxycycline": 0.40, "Colistin": 0.05, "Vancomycin": 0.95
    },
    "Staphylococcus aureus": {
        "Amoxicillin": 0.60, "Ampicillin": 0.60, "Ciprofloxacin": 0.30,
        "Levofloxacin": 0.25, "Gentamicin": 0.20, "Amikacin": 0.15,
        "Ceftriaxone": 0.40, "Cefepime": 0.35, "Meropenem": 0.10,
        "Imipenem": 0.10, "Trimethoprim-Sulfamethoxazole": 0.25,
        "Piperacillin-Tazobactam": 0.20, "Tetracycline": 0.35,
        "Doxycycline": 0.20, "Colistin": 0.95, "Vancomycin": 0.02
    },
    "Pseudomonas aeruginosa": {
        "Amoxicillin": 0.95, "Ampicillin": 0.95, "Ciprofloxacin": 0.35,
        "Levofloxacin": 0.40, "Gentamicin": 0.25, "Amikacin": 0.15,
        "Ceftriaxone": 0.85, "Cefepime": 0.30, "Meropenem": 0.20,
        "Imipenem": 0.25, "Trimethoprim-Sulfamethoxazole": 0.90,
        "Piperacillin-Tazobactam": 0.20, "Tetracycline": 0.90,
        "Doxycycline": 0.85, "Colistin": 0.05, "Vancomycin": 0.95
    },
    "Acinetobacter baumannii": {
        "Amoxicillin": 0.95, "Ampicillin": 0.95, "Ciprofloxacin": 0.60,
        "Levofloxacin": 0.55, "Gentamicin": 0.50, "Amikacin": 0.35,
        "Ceftriaxone": 0.80, "Cefepime": 0.70, "Meropenem": 0.45,
        "Imipenem": 0.50, "Trimethoprim-Sulfamethoxazole": 0.70,
        "Piperacillin-Tazobactam": 0.60, "Tetracycline": 0.55,
        "Doxycycline": 0.40, "Colistin": 0.08, "Vancomycin": 0.95
    },
    "Enterococcus faecalis": {
        "Amoxicillin": 0.10, "Ampicillin": 0.10, "Ciprofloxacin": 0.35,
        "Levofloxacin": 0.30, "Gentamicin": 0.40, "Amikacin": 0.45,
        "Ceftriaxone": 0.90, "Cefepime": 0.90, "Meropenem": 0.85,
        "Imipenem": 0.15, "Trimethoprim-Sulfamethoxazole": 0.80,
        "Piperacillin-Tazobactam": 0.15, "Tetracycline": 0.55,
        "Doxycycline": 0.40, "Colistin": 0.95, "Vancomycin": 0.05
    },
    "Enterococcus faecium": {
        "Amoxicillin": 0.80, "Ampicillin": 0.85, "Ciprofloxacin": 0.65,
        "Levofloxacin": 0.60, "Gentamicin": 0.55, "Amikacin": 0.50,
        "Ceftriaxone": 0.95, "Cefepime": 0.95, "Meropenem": 0.90,
        "Imipenem": 0.85, "Trimethoprim-Sulfamethoxazole": 0.80,
        "Piperacillin-Tazobactam": 0.80, "Tetracycline": 0.60,
        "Doxycycline": 0.50, "Colistin": 0.95, "Vancomycin": 0.25
    },
    "Streptococcus pneumoniae": {
        "Amoxicillin": 0.15, "Ampicillin": 0.15, "Ciprofloxacin": 0.10,
        "Levofloxacin": 0.08, "Gentamicin": 0.70, "Amikacin": 0.65,
        "Ceftriaxone": 0.10, "Cefepime": 0.12, "Meropenem": 0.05,
        "Imipenem": 0.05, "Trimethoprim-Sulfamethoxazole": 0.40,
        "Piperacillin-Tazobactam": 0.10, "Tetracycline": 0.30,
        "Doxycycline": 0.20, "Colistin": 0.95, "Vancomycin": 0.01
    },
    "Proteus mirabilis": {
        "Amoxicillin": 0.50, "Ampicillin": 0.55, "Ciprofloxacin": 0.30,
        "Levofloxacin": 0.25, "Gentamicin": 0.25, "Amikacin": 0.10,
        "Ceftriaxone": 0.20, "Cefepime": 0.15, "Meropenem": 0.03,
        "Imipenem": 0.03, "Trimethoprim-Sulfamethoxazole": 0.45,
        "Piperacillin-Tazobactam": 0.10, "Tetracycline": 0.70,
        "Doxycycline": 0.60, "Colistin": 0.95, "Vancomycin": 0.95
    },
    "Serratia marcescens": {
        "Amoxicillin": 0.90, "Ampicillin": 0.90, "Ciprofloxacin": 0.20,
        "Levofloxacin": 0.18, "Gentamicin": 0.20, "Amikacin": 0.08,
        "Ceftriaxone": 0.25, "Cefepime": 0.15, "Meropenem": 0.05,
        "Imipenem": 0.05, "Trimethoprim-Sulfamethoxazole": 0.35,
        "Piperacillin-Tazobactam": 0.10, "Tetracycline": 0.60,
        "Doxycycline": 0.50, "Colistin": 0.90, "Vancomycin": 0.95
    },
}

# Known resistance genes mapped to antibiotics (CARD-inspired)
CARD_GENE_MAP = {
    "Amoxicillin": ["blaTEM-1", "blaSHV-1", "blaOXA-1"],
    "Ampicillin": ["blaTEM-1", "blaSHV-1", "blaOXA-1"],
    "Ciprofloxacin": ["qnrA", "qnrB", "aac(6')-Ib-cr", "gyrA_mutation"],
    "Levofloxacin": ["qnrA", "qnrB", "gyrA_mutation", "parC_mutation"],
    "Gentamicin": ["aac(3)-IIa", "ant(2'')-Ia", "aph(3')-IIIa"],
    "Amikacin": ["aac(6')-Ib", "armA", "rmtB"],
    "Ceftriaxone": ["blaCTX-M-15", "blaCMY-2", "blaTEM-1"],
    "Cefepime": ["blaCTX-M-15", "blaKPC-2", "blaNDM-1"],
    "Meropenem": ["blaKPC-2", "blaNDM-1", "blaOXA-48", "blaVIM-1"],
    "Imipenem": ["blaKPC-2", "blaNDM-1", "blaOXA-48", "blaIMP-1"],
    "Trimethoprim-Sulfamethoxazole": ["dfrA1", "sul1", "sul2"],
    "Piperacillin-Tazobactam": ["blaTEM-1", "blaOXA-1", "blaCTX-M-15"],
    "Tetracycline": ["tet(A)", "tet(B)", "tet(M)"],
    "Doxycycline": ["tet(A)", "tet(B)", "tet(M)"],
    "Colistin": ["mcr-1", "mcr-2", "pmrA_mutation"],
    "Vancomycin": ["vanA", "vanB", "vanC"],
}


def generate_mendeley_dataset(n_samples=2000):
    """Generate primary Mendeley-style AMR dataset."""
    records = []
    
    for i in range(n_samples):
        organism = np.random.choice(ORGANISMS, p=[0.25, 0.20, 0.15, 0.10, 0.08,
                                                    0.06, 0.05, 0.04, 0.04, 0.03])
        age = int(np.clip(np.random.normal(55, 20), 1, 100))
        gender = np.random.choice(GENDERS)
        specimen = np.random.choice(SPECIMEN_TYPES, p=[0.30, 0.25, 0.15, 0.15, 0.05, 0.10])
        department = np.random.choice(DEPARTMENTS, p=[0.25, 0.30, 0.15, 0.15, 0.15])
        
        row = {
            "Isolate_ID": f"MND-{i+1:05d}",
            "Organism": organism,
            "Age": age,
            "Gender": gender,
            "Specimen_Type": specimen,
            "Department": department,
        }
        
        # Generate resistance results based on organism profile
        profile = RESISTANCE_PROFILES[organism]
        for abx in ANTIBIOTICS:
            base_prob = profile[abx]
            # Add noise for realism
            prob = np.clip(base_prob + np.random.normal(0, 0.08), 0, 1)
            
            # Introduce ~5% missing values (realistic in clinical data)
            if np.random.random() < 0.05:
                row[abx] = np.nan
            else:
                r = np.random.random()
                if r < prob:
                    row[abx] = "Resistant"
                elif r < prob + 0.10:  # ~10% Intermediate
                    row[abx] = "Intermediate"
                else:
                    row[abx] = "Susceptible"
        
        records.append(row)
    
    return pd.DataFrame(records)


def generate_kaggle_dataset(n_samples=1500):
    """Generate secondary Kaggle-style multi-resistance dataset."""
    records = []
    
    for i in range(n_samples):
        organism = np.random.choice(ORGANISMS[:7])  # Subset of organisms
        
        row = {
            "Sample_ID": f"KAG-{i+1:05d}",
            "Bacteria": organism,
            "Patient_Age": int(np.clip(np.random.normal(50, 22), 0, 95)),
            "Patient_Gender": np.random.choice(["M", "F"]),
            "Sample_Source": np.random.choice(["blood", "urine", "respiratory", "wound"]),
        }
        
        profile = RESISTANCE_PROFILES[organism]
        # Kaggle dataset uses a subset of antibiotics with 0/1 encoding
        kaggle_antibiotics = ANTIBIOTICS[:12]  # first 12
        for abx in kaggle_antibiotics:
            base_prob = profile[abx]
            prob = np.clip(base_prob + np.random.normal(0, 0.10), 0, 1)
            
            if np.random.random() < 0.03:
                row[f"{abx}_R"] = np.nan
            else:
                row[f"{abx}_R"] = 1 if np.random.random() < prob else 0
        
        # Number of drugs resistant to (MDR indicator)
        resistant_count = sum(1 for abx in kaggle_antibiotics 
                            if row.get(f"{abx}_R") == 1)
        row["MDR"] = 1 if resistant_count >= 3 else 0
        
        records.append(row)
    
    return pd.DataFrame(records)


def generate_card_database():
    """Generate CARD-style gene annotation reference."""
    records = []
    for abx, genes in CARD_GENE_MAP.items():
        for gene in genes:
            records.append({
                "Gene_Name": gene,
                "Antibiotic_Class": abx,
                "Resistance_Mechanism": np.random.choice([
                    "Antibiotic inactivation", "Target alteration",
                    "Efflux pump", "Target protection",
                    "Reduced permeability"
                ]),
                "Gene_Family": gene.split("-")[0].split("(")[0].split("_")[0],
                "CARD_ID": f"ARO:{np.random.randint(3000000, 3999999):07d}",
            })
    return pd.DataFrame(records)


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    print("Generating Mendeley AMR dataset...")
    mendeley_df = generate_mendeley_dataset(2000)
    mendeley_df.to_csv("data/raw/mendeley_amr.csv", index=False)
    print(f"  Shape: {mendeley_df.shape}")
    print(f"  Columns: {list(mendeley_df.columns)}")
    print(f"  Organisms: {mendeley_df['Organism'].value_counts().to_dict()}")
    
    print("\nGenerating Kaggle multi-resistance dataset...")
    kaggle_df = generate_kaggle_dataset(1500)
    kaggle_df.to_csv("data/raw/kaggle_resistance.csv", index=False)
    print(f"  Shape: {kaggle_df.shape}")
    print(f"  Columns: {list(kaggle_df.columns)}")
    
    print("\nGenerating CARD gene annotation database...")
    card_df = generate_card_database()
    card_df.to_csv("data/raw/card_genes.csv", index=False)
    print(f"  Shape: {card_df.shape}")
    
    print("\n✅ All datasets generated successfully in data/raw/")
