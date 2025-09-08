#!/usr/bin/env python3
"""
Batch ADMET Analyzer for SMILES Lists and SDF Files
Handles large datasets efficiently with progress tracking

Requirements: pip install rdkit-pypi pandas numpy matplotlib seaborn tqdm
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumHBD, CalcNumHBA, CalcNumRotatableBonds
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

class BatchADMETAnalyzer:
    """
    Batch ADMET analyzer for processing SMILES lists and SDF files
    """

    def __init__(self):
        self.results = []
        self.failed_compounds = []

    def read_smiles_file(self, file_path, smiles_column='SMILES', name_column=None, delimiter=','):
        """
        Read SMILES from various file formats

        Parameters:
        - file_path: Path to file (CSV, TXT, TSV)
        - smiles_column: Column name containing SMILES
        - name_column: Column name containing compound names (optional)
        - delimiter: File delimiter (default: comma)

        Returns:
        - List of tuples: [(smiles, name), ...]
        """
        try:
            # Determine file format
            file_ext = os.path.splitext(file_path)[1].lower()

            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext == '.tsv':
                df = pd.read_csv(file_path, sep='\t')
            elif file_ext == '.txt':
                # Try to detect delimiter
                with open(file_path, 'r') as f:
                    first_line = f.readline()
                if '\t' in first_line:
                    df = pd.read_csv(file_path, sep='\t')
                elif ',' in first_line:
                    df = pd.read_csv(file_path, sep=',')
                else:
                    # Assume single column of SMILES
                    df = pd.read_csv(file_path, header=None, names=['SMILES'])
                    smiles_column = 'SMILES'
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")

            # Extract SMILES and names
            smiles_list = df[smiles_column].dropna().tolist()

            if name_column and name_column in df.columns:
                names_list = df[name_column].fillna('Unknown').tolist()
            else:
                names_list = [f"Compound_{i+1}" for i in range(len(smiles_list))]

            compounds = list(zip(smiles_list, names_list))

            print(f"📁 Successfully read {len(compounds)} compounds from {file_path}")
            return compounds

        except Exception as e:
            print(f"❌ Error reading SMILES file: {e}")
            return []

    def read_sdf_file(self, file_path, name_property=None):
        """
        Read compounds from SDF file

        Parameters:
        - file_path: Path to SDF file
        - name_property: Property name for compound names (optional)

        Returns:
        - List of tuples: [(smiles, name), ...]
        """
        try:
            compounds = []
            suppl = Chem.SDMolSupplier(file_path)

            print(f"📁 Reading SDF file: {file_path}")

            for i, mol in enumerate(tqdm(suppl, desc="Reading SDF")):
                if mol is not None:
                    # Convert to SMILES
                    smiles = Chem.MolToSmiles(mol)

                    # Get compound name
                    if name_property and mol.HasProp(name_property):
                        name = mol.GetProp(name_property)
                    elif mol.HasProp('_Name'):
                        name = mol.GetProp('_Name')
                    else:
                        name = f"Compound_{i+1}"

                    compounds.append((smiles, name))
                else:
                    print(f"⚠️  Skipped invalid molecule at index {i}")

            print(f"✅ Successfully read {len(compounds)} compounds from SDF")
            return compounds

        except Exception as e:
            print(f"❌ Error reading SDF file: {e}")
            return []

    def calculate_admet_properties(self, mol):
        """Calculate comprehensive ADMET properties"""
        if mol is None:
            return {}

        try:
            # Basic molecular properties
            mw = Descriptors.MolWt(mol)
            alogp98 = Crippen.MolLogP(mol)
            psa_2d = CalcTPSA(mol)
            hbd = CalcNumHBD(mol)
            hba = CalcNumHBA(mol)
            rotbonds = CalcNumRotatableBonds(mol)
            aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)

            # Human Intestinal Absorption
            absorption_t2_2d = self._calculate_absorption_t2_2d(mol, alogp98, psa_2d, mw)

            if psa_2d >= 150.0 or alogp98 <= -2.0 or alogp98 >= 7.0:
                absorption_level = 3
                absorption_desc = "Very Poor"
            elif absorption_t2_2d < 6.1261:
                absorption_level = 0
                absorption_desc = "Good"
            elif 6.1261 <= absorption_t2_2d < 9.6026:
                absorption_level = 1
                absorption_desc = "Moderate"
            else:
                absorption_level = 2
                absorption_desc = "Poor"

            # Aqueous Solubility
            log_sw = self._calculate_log_sw_ds(mol, alogp98, mw, aromatic_rings, rotbonds)

            if log_sw < -8.0:
                sol_level, sol_desc = 0, "Extremely low"
            elif -8.0 <= log_sw < -6.0:
                sol_level, sol_desc = 1, "Very low, but possible"
            elif -6.0 <= log_sw < -4.1:
                sol_level, sol_desc = 2, "Low"
            elif -4.1 <= log_sw < -2.0:
                sol_level, sol_desc = 3, "Good"
            elif -2.0 <= log_sw < 0.0:
                sol_level, sol_desc = 4, "Optimal"
            else:
                sol_level, sol_desc = 5, "Too soluble"

            # BBB Penetration
            log_bb = self._calculate_log_bb_ds(mol, alogp98, psa_2d, mw)

            if log_bb >= 0.7:
                bbb_level = "Very High"
            elif 0 <= log_bb < 0.7:
                bbb_level = "High"
            elif -0.52 < log_bb < 0:
                bbb_level = "Medium"
            else:
                bbb_level = "Low"

            # CYP2D6 Inhibition
            cyp2d6_score = self._calculate_cyp2d6_score(mol, alogp98, mw, psa_2d)
            cyp2d6_inhibitor = "Yes" if cyp2d6_score > 0.161 else "No"

            # Hepatotoxicity
            hepatotox_score = self._calculate_hepatotoxicity_score(mol)
            hepatotoxic = "Yes" if hepatotox_score > -4.154 else "No"

            # Plasma Protein Binding
            ppb_score = self._calculate_ppb_score(mol)
            high_ppb = "Yes" if ppb_score > -2.209 else "No"

            # Drug-likeness (Lipinski)
            lipinski_violations = sum([
                mw > 500,
                alogp98 > 5,
                hbd > 5,
                hba > 10
            ])
            lipinski_compliant = lipinski_violations <= 1

            return {
                'Molecular_Weight': round(mw, 2),
                'AlogP98': round(alogp98, 2),
                'PSA_2D': round(psa_2d, 2),
                'HBD': hbd,
                'HBA': hba,
                'Rotatable_Bonds': rotbonds,
                'Aromatic_Rings': aromatic_rings,

                # ADMET Properties
                'Absorption_Level': absorption_level,
                'Absorption_Description': absorption_desc,
                'Absorption_T2_2D': round(absorption_t2_2d, 4),

                'Solubility_Level': sol_level,
                'Solubility_Description': sol_desc,
                'LogSw': round(log_sw, 3),

                'BBB_Level': bbb_level,
                'LogBB': round(log_bb, 3),

                'CYP2D6_Score': round(cyp2d6_score, 4),
                'CYP2D6_Inhibitor': cyp2d6_inhibitor,

                'Hepatotoxic_Score': round(hepatotox_score, 4),
                'Hepatotoxic': hepatotoxic,

                'PPB_Score': round(ppb_score, 4),
                'High_PPB': high_ppb,

                'Lipinski_Violations': lipinski_violations,
                'Lipinski_Compliant': lipinski_compliant
            }

        except Exception as e:
            print(f"Error calculating properties: {e}")
            return {}

    def _calculate_absorption_t2_2d(self, mol, alogp98, psa_2d, mw):
        """Calculate absorption T2_2D score"""
        hbd = CalcNumHBD(mol)
        hba = CalcNumHBA(mol)
        rotbonds = CalcNumRotatableBonds(mol)

        t2_2d = (0.1 * psa_2d + 0.3 * abs(alogp98) + 0.2 * mw/100 +
                0.4 * hbd + 0.2 * rotbonds + 0.1 * hba)
        return max(0, t2_2d)

    def _calculate_log_sw_ds(self, mol, alogp98, mw, aromatic_rings, rotbonds):
        """Calculate DS-style log(Sw)"""
        hbd = CalcNumHBD(mol)
        hba = CalcNumHBA(mol)

        log_sw = (0.16 - 0.63 * alogp98 - 0.0062 * mw +
                 0.066 * aromatic_rings - 0.74 +
                 0.1 * hbd - 0.05 * rotbonds + 0.02 * hba)
        return log_sw

    def _calculate_log_bb_ds(self, mol, alogp98, psa_2d, mw):
        """Calculate DS-style logBB"""
        log_bb = (-0.0148 * psa_2d + 0.152 * alogp98 + 0.139 - 0.0001 * mw)
        return log_bb

    def _calculate_cyp2d6_score(self, mol, alogp98, mw, psa_2d):
        """Calculate CYP2D6 score"""
        hbd = CalcNumHBD(mol)
        hba = CalcNumHBA(mol)
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)

        score = (0.1 * alogp98 + 0.001 * mw + 0.002 * psa_2d +
                0.05 * aromatic_rings + 0.02 * hba - 0.01 * hbd - 0.3)
        return max(-1, min(1, score))

    def _calculate_hepatotoxicity_score(self, mol):
        """Calculate hepatotoxicity score"""
        alogp98 = Crippen.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        smiles = Chem.MolToSmiles(mol)

        score = -5.0  # Base score

        if 'N(=O)=O' in smiles or '[N+](=O)[O-]' in smiles:
            score += 2.0
        if 'C(=O)Cl' in smiles:
            score += 1.5
        if alogp98 > 5:
            score += 0.5
        if mw > 600:
            score += 0.3

        return score

    def _calculate_ppb_score(self, mol):
        """Calculate PPB score"""
        alogp98 = Crippen.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        psa_2d = CalcTPSA(mol)

        score = (0.3 * alogp98 + 0.001 * mw - 0.01 * psa_2d - 3.0)
        return score

    def process_compounds_batch(self, compounds, batch_size=100):
        """
        Process compounds in batches with progress tracking

        Parameters:
        - compounds: List of (smiles, name) tuples
        - batch_size: Number of compounds to process at once

        Returns:
        - pandas DataFrame with results
        """
        results = []
        failed_compounds = []

        print(f"🚀 Processing {len(compounds)} compounds in batches of {batch_size}...")

        # Process in batches
        for i in tqdm(range(0, len(compounds), batch_size), desc="Processing batches"):
            batch = compounds[i:i+batch_size]

            for smiles, name in batch:
                try:
                    mol = Chem.MolFromSmiles(smiles)

                    if mol is not None:
                        properties = self.calculate_admet_properties(mol)

                        if properties:
                            result = {
                                'Compound_Name': name,
                                'SMILES': smiles,
                                **properties
                            }
                            results.append(result)
                        else:
                            failed_compounds.append((name, smiles, "Property calculation failed"))
                    else:
                        failed_compounds.append((name, smiles, "Invalid SMILES"))

                except Exception as e:
                    failed_compounds.append((name, smiles, str(e)))

        # Store failed compounds
        self.failed_compounds = failed_compounds

        if failed_compounds:
            print(f"⚠️  {len(failed_compounds)} compounds failed processing")

            # Save failed compounds
            failed_df = pd.DataFrame(failed_compounds, columns=['Name', 'SMILES', 'Error'])
            failed_df.to_csv('failed_compounds.csv', index=False)
            print("❌ Failed compounds saved to: failed_compounds.csv")

        print(f"✅ Successfully processed {len(results)} compounds")

        return pd.DataFrame(results)

    def analyze_from_smiles_file(self, file_path, smiles_column='SMILES', name_column=None,
                                delimiter=',', batch_size=100, output_file=None):
        """
        Analyze compounds from SMILES file

        Parameters:
        - file_path: Path to SMILES file
        - smiles_column: Column name containing SMILES
        - name_column: Column name containing names
        - delimiter: File delimiter
        - batch_size: Batch processing size
        - output_file: Output CSV file name
        """
        print("📋 ANALYZING COMPOUNDS FROM SMILES FILE")
        print("=" * 50)

        # Read SMILES file
        compounds = self.read_smiles_file(file_path, smiles_column, name_column, delimiter)

        if not compounds:
            print("❌ No compounds to analyze")
            return None

        # Process compounds
        results_df = self.process_compounds_batch(compounds, batch_size)

        if results_df.empty:
            print("❌ No results generated")
            return None

        # Save results
        if output_file is None:
            output_file = f"admet_results_{os.path.splitext(os.path.basename(file_path))[0]}.csv"

        results_df.to_csv(output_file, index=False)
        print(f"💾 Results saved to: {output_file}")

        # Generate summary
        self.generate_summary_report(results_df)

        return results_df

    def analyze_from_sdf_file(self, file_path, name_property=None, batch_size=100, output_file=None):
        """
        Analyze compounds from SDF file

        Parameters:
        - file_path: Path to SDF file
        - name_property: Property name for compound names
        - batch_size: Batch processing size
        - output_file: Output CSV file name
        """
        print("📋 ANALYZING COMPOUNDS FROM SDF FILE")
        print("=" * 50)

        # Read SDF file
        compounds = self.read_sdf_file(file_path, name_property)

        if not compounds:
            print("❌ No compounds to analyze")
            return None

        # Process compounds
        results_df = self.process_compounds_batch(compounds, batch_size)

        if results_df.empty:
            print("❌ No results generated")
            return None

        # Save results
        if output_file is None:
            output_file = f"admet_results_{os.path.splitext(os.path.basename(file_path))[0]}.csv"

        results_df.to_csv(output_file, index=False)
        print(f"💾 Results saved to: {output_file}")

        # Generate summary
        self.generate_summary_report(results_df)

        return results_df

    def analyze_from_smiles_list(self, smiles_list, names_list=None, batch_size=100, output_file=None):
        """
        Analyze compounds from Python list of SMILES

        Parameters:
        - smiles_list: List of SMILES strings
        - names_list: List of compound names (optional)
        - batch_size: Batch processing size
        - output_file: Output CSV file name
        """
        print("📋 ANALYZING COMPOUNDS FROM SMILES LIST")
        print("=" * 50)

        # Prepare compounds
        if names_list is None:
            names_list = [f"Compound_{i+1}" for i in range(len(smiles_list))]

        compounds = list(zip(smiles_list, names_list))

        # Process compounds
        results_df = self.process_compounds_batch(compounds, batch_size)

        if results_df.empty:
            print("❌ No results generated")
            return None

        # Save results
        if output_file is None:
            output_file = "admet_results_from_list.csv"

        results_df.to_csv(output_file, index=False)
        print(f"💾 Results saved to: {output_file}")

        # Generate summary
        self.generate_summary_report(results_df)

        return results_df

    def generate_summary_report(self, df):
        """Generate summary statistics"""
        print("\n📊 ADMET ANALYSIS SUMMARY")
        print("=" * 40)
        print(f"Total compounds analyzed: {len(df)}")

        # Absorption summary
        if 'Absorption_Description' in df.columns:
            print(f"\n🧬 Human Intestinal Absorption:")
            absorption_counts = df['Absorption_Description'].value_counts()
            for desc, count in absorption_counts.items():
                print(f"  {desc}: {count} ({count/len(df)*100:.1f}%)")

        # Solubility summary
        if 'Solubility_Description' in df.columns:
            print(f"\n💧 Aqueous Solubility:")
            sol_counts = df['Solubility_Description'].value_counts()
            for desc, count in sol_counts.items():
                print(f"  {desc}: {count} ({count/len(df)*100:.1f}%)")

        # BBB summary
        if 'BBB_Level' in df.columns:
            print(f"\n🧠 BBB Penetration:")
            bbb_counts = df['BBB_Level'].value_counts()
            for level, count in bbb_counts.items():
                print(f"  {level}: {count} ({count/len(df)*100:.1f}%)")

        # Drug-likeness summary
        if 'Lipinski_Compliant' in df.columns:
            compliant = df['Lipinski_Compliant'].sum()
            print(f"\n💊 Drug-likeness:")
            print(f"  Lipinski compliant: {compliant} ({compliant/len(df)*100:.1f}%)")

        # Toxicity summary
        if 'CYP2D6_Inhibitor' in df.columns:
            cyp_inhibitors = (df['CYP2D6_Inhibitor'] == 'Yes').sum()
            print(f"\n⚠️  Toxicity Alerts:")
            print(f"  CYP2D6 inhibitors: {cyp_inhibitors} ({cyp_inhibitors/len(df)*100:.1f}%)")

        if 'Hepatotoxic' in df.columns:
            hepatotoxic = (df['Hepatotoxic'] == 'Yes').sum()
            print(f"  Hepatotoxic: {hepatotoxic} ({hepatotoxic/len(df)*100:.1f}%)")

    def create_summary_plots(self, df, output_file='admet_summary_plots.png'):
        """Create summary plots for batch analysis"""
        if df.empty:
            return

        print("🎨 Creating summary plots...")

        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'ADMET Analysis Summary ({len(df)} compounds)', fontsize=14, fontweight='bold')

            # Absorption distribution
            if 'Absorption_Description' in df.columns:
                absorption_counts = df['Absorption_Description'].value_counts()
                axes[0, 0].pie(absorption_counts.values, labels=absorption_counts.index, autopct='%1.1f%%')
                axes[0, 0].set_title('Intestinal Absorption')

            # Solubility distribution
            if 'Solubility_Description' in df.columns:
                sol_counts = df['Solubility_Description'].value_counts()
                axes[0, 1].bar(range(len(sol_counts)), sol_counts.values)
                axes[0, 1].set_xticks(range(len(sol_counts)))
                axes[0, 1].set_xticklabels(sol_counts.index, rotation=45, ha='right')
                axes[0, 1].set_title('Solubility Distribution')
                axes[0, 1].set_ylabel('Count')

            # BBB distribution
            if 'BBB_Level' in df.columns:
                bbb_counts = df['BBB_Level'].value_counts()
                colors = ['blue', 'green', 'cyan', 'orange']
                color_map = {'Very High': 'blue', 'High': 'green', 'Medium': 'cyan', 'Low': 'orange'}
                plot_colors = [color_map.get(level, 'gray') for level in bbb_counts.index]
                axes[0, 2].bar(bbb_counts.index, bbb_counts.values, color=plot_colors)
                axes[0, 2].set_title('BBB Penetration')
                axes[0, 2].set_ylabel('Count')

            # Molecular weight distribution
            if 'Molecular_Weight' in df.columns:
                axes[1, 0].hist(df['Molecular_Weight'], bins=20, alpha=0.7)
                axes[1, 0].axvline(500, color='red', linestyle='--', label='Lipinski limit')
                axes[1, 0].set_xlabel('Molecular Weight')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Molecular Weight')
                axes[1, 0].legend()

            # LogP distribution
            if 'AlogP98' in df.columns:
                axes[1, 1].hist(df['AlogP98'], bins=20, alpha=0.7)
                axes[1, 1].axvline(5, color='red', linestyle='--', label='Lipinski limit')
                axes[1, 1].set_xlabel('AlogP98')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('Lipophilicity')
                axes[1, 1].legend()

            # Drug-likeness summary
            if 'Lipinski_Compliant' in df.columns:
                compliant_counts = df['Lipinski_Compliant'].value_counts()
                colors = ['red', 'green']
                axes[1, 2].bar(['Non-compliant', 'Compliant'],
                              [compliant_counts.get(False, 0), compliant_counts.get(True, 0)],
                              color=colors)
                axes[1, 2].set_title('Lipinski Compliance')
                axes[1, 2].set_ylabel('Count')

            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"📊 Summary plots saved as: {output_file}")
            plt.close()

        except Exception as e:
            print(f"Error creating plots: {e}")


def main():
    """Main function with examples for different input types"""

    print("🧬 BATCH ADMET ANALYZER")
    print("=" * 50)
    print("Supports:")
    print("1. SMILES files (CSV, TSV, TXT)")
    print("2. SDF files")
    print("3. Python lists of SMILES")
    print()

    # Initialize analyzer
    analyzer = BatchADMETAnalyzer()

    # Example 1: Analyze from SMILES list
    print("📋 Example 1: Analyzing from SMILES list")
    example_smiles = [
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
        'CC(=O)OC1=CC=CC=C1C(=O)O',        # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',    # Caffeine
        'CC(=O)NC1=CC=C(C=C1)O',           # Acetaminophen
        'CCO',                              # Ethanol
    ]

    example_names = ['Ibuprofen', 'Aspirin', 'Caffeine', 'Acetaminophen', 'Ethanol']

    results = analyzer.analyze_from_smiles_list(
        smiles_list=example_smiles,
        names_list=example_names,
        output_file='example_results.csv'
    )

    if results is not None:
        # Create summary plots
        analyzer.create_summary_plots(results, 'example_summary_plots.png')

    print("\n" + "="*80)
    print("📖 HOW TO USE WITH YOUR DATA:")
    print("="*80)

    usage_examples = """
# 1. ANALYZE FROM SMILES FILE (CSV/TSV/TXT)
analyzer = BatchADMETAnalyzer()

# CSV file with SMILES column
results = analyzer.analyze_from_smiles_file(
    file_path='your_compounds.csv',
    smiles_column='SMILES',           # Column name containing SMILES
    name_column='Name',               # Column name containing names (optional)
    batch_size=100,                   # Process in batches
    output_file='my_results.csv'      # Output file name
)

# 2. ANALYZE FROM SDF FILE
results = analyzer.analyze_from_sdf_file(
    file_path='your_compounds.sdf',
    name_property='_Name',            # Property containing compound names
    batch_size=100,
    output_file='my_sdf_results.csv'
)

# 3. ANALYZE FROM PYTHON LIST
my_smiles = ['SMILES1', 'SMILES2', 'SMILES3', ...]
my_names = ['Name1', 'Name2', 'Name3', ...]      # Optional

results = analyzer.analyze_from_smiles_list(
    smiles_list=my_smiles,
    names_list=my_names,              # Optional
    batch_size=100,
    output_file='my_list_results.csv'
)

# 4. CREATE SUMMARY PLOTS
analyzer.create_summary_plots(results, 'my_plots.png')
    """

    print(usage_examples)

    return results

if __name__ == "__main__":
    results = main()
