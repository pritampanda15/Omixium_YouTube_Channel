from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
import pandas as pd
import os

def calc_properties(sdf_file):
    suppl = Chem.SDMolSupplier(sdf_file)
    data = []
    for mol in suppl:
        if mol is None:
            continue
        mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else os.path.basename(sdf_file)
        molwt = Descriptors.MolWt(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        logp = Crippen.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        # pKa, ClogD require external tool or prediction model
        data.append({
            "Name": mol_name,
            "MolWt": molwt,
            "TPSA": tpsa,
            "ALogP": logp,
            "HBD": hbd,
            "HBA": hba
        })
    return pd.DataFrame(data)

def process_tranche_list(list_file, sdf_dir, output_dir):
    with open(list_file) as f:
        sdf_files = [line.strip() for line in f if line.strip()]
    for sdf_file in sdf_files:
        sdf_path = os.path.join(sdf_dir, sdf_file)
        if not os.path.exists(sdf_path):
            print(f"Missing: {sdf_path}")
            continue
        df = calc_properties(sdf_path)
        out_csv = os.path.join(output_dir, os.path.splitext(sdf_file)[0] + ".csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved {out_csv}")

# Example usage:
process_tranche_list("file.txt", "tranches", "processed_molecules")
