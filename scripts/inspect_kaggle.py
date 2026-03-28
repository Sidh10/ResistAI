import pandas as pd
import sys

k = pd.read_csv("data/raw/Bacteria_dataset_Multiresictance.csv")
print(f"KAGGLE Shape: {k.shape}")

abx_cols = ['AMX/AMP','AMC','CZ','FOX','CTX/CRO','IPM','GEN','AN',
            'Acide nalidixique','ofx','CIP','C','Co-trimoxazole','Furanes','colistine']

print("\nAntibiotic Class Distribution:")
for c in abx_cols:
    vc = k[c].value_counts()
    print(f"\n  {c}:")
    for val, cnt in vc.items():
        print(f"    {val:15s} = {cnt:5d} ({cnt/len(k)*100:.1f}%)")

# Extract organism from Souches
k['Organism'] = k['Souches'].str.replace(r'^S\d+\s*', '', regex=True)
print("\nOrganism distribution (top 15):")
orgs = k['Organism'].value_counts().head(15)
for val, cnt in orgs.items():
    print(f"  {val:35s} = {cnt:5d} ({cnt/len(k)*100:.1f}%)")

# age/gender split
print("\nage/gender format sample:", k['age/gender'].head(10).tolist())
print("Comorbidity columns:")
for c in ['Diabetes', 'Hypertension', 'Hospital_before', 'Infection_Freq']:
    print(f"  {c}: {k[c].value_counts().to_dict()}")

sys.stdout.flush()
