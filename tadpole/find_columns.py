import pandas as pd
import numpy as np

df = pd.read_csv("TADPOLE_D1_D2.csv", dtype=str)

columns = df.columns
print(len(columns))

possible_cog_tests = ["cdr", "adas", "ravlt", "mmse"]
modality_markers = ["ucsff", "baipet", "av45", "av1451", "dtiroi", "upennbio"]
modality_names = ["mri", "fdg_pet", "av45_pet", "av1451_pet", "dti", "csf"]

modality_columns = {}


# Split columns into 8 Modalities 
modality_columns["cognitive_tests"] = ["RID", "PTID", "VISCODE"] + [c for c in columns if any(s in c.lower() for s in possible_cog_tests)]
for i in range(len(modality_markers)):
    modality_columns[modality_names[i]] = ["RID", "PTID", "VISCODE"] + [c for c in columns if modality_markers[i] in c.lower()]
    if modality_names[i] == "fdg_pet":
        modality_columns[modality_names[i]] += ["FDG", "FDG_bl"]
    elif modality_names[i] == "mri":
        modality_columns[modality_names[i]] += ["Ventricles", "Hippocampus","WholeBrain","Entorhinal","Fusiform","MidTemp","Ventricles_bl","Hippocampus_bl","WholeBrain_bl","Entorhinal_bl","Fusiform_bl","MidTemp_bl"]

all_mod = np.concatenate(list(modality_columns.values()) + [["FLDSTRENG", "FSVERSION", 'FLDSTRENG_bl', 'FSVERSION_bl', "PIB", "PIB_bl", "M"]])
modality_columns["clinical"] = ["RID", "PTID", "VISCODE"] + [c for c in columns if c not in all_mod and "ecog" not in c.lower()]

total = 0
for name, cols in modality_columns.items():
    total += len(cols)
    df[cols].to_csv("./tadpole_dataset_8modality/" + name + ".csv", index=False)

print(total)
print(modality_columns["clinical"])






