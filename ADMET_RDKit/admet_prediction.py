#!/usr/bin/env python3
"""
Discovery Studio Style ADMET Properties Calculator
Implements DS-style ADMET predictions with proper classification levels

Requirements: pip install rdkit pandas numpy matplotlib seaborn scikit-learn tqdm
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumHBD, CalcNumHBA, CalcNumRotatableBonds
import warnings
warnings.filterwarnings('ignore')

class DSStyleADMETCalculator:
    """
    Discovery Studio Style ADMET properties calculator
    Implements sophisticated ADMET predictions similar to DS
    """

    def __init__(self):
        self.results = []

    def calculate_molecular_properties(self, mol):
        """Calculate basic molecular properties"""
        if mol is None:
            return {}

        try:
            properties = {
                'Molecular_Weight': Descriptors.MolWt(mol),
                'AlogP98': Crippen.MolLogP(mol),  # Approximation of AlogP98
                'PSA_2D': CalcTPSA(mol),  # Polar Surface Area
                'HBD': CalcNumHBD(mol),
                'HBA': CalcNumHBA(mol),
                'Rotatable_Bonds': CalcNumRotatableBonds(mol),
                'Aromatic_Rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'Heavy_Atoms': mol.GetNumHeavyAtoms(),
                'Formal_Charge': Chem.rdmolops.GetFormalCharge(mol),
                'Molar_Refractivity': Crippen.MolMR(mol)
            }
            return properties
        except Exception as e:
            print(f"Error calculating molecular properties: {e}")
            return {}

    def predict_human_intestinal_absorption(self, mol):
        """
        Predict Human Intestinal Absorption using DS-style classification

        Levels:
        0 (Good): ADMET_Absorption_T2_2D < 6.1261
        1 (Moderate): 6.1261 ≤ ADMET_Absorption_T2_2D < 9.6026
        2 (Poor): 9.6026 < ADMET_Absorption_T2_2D
        3 (Very Poor): PSA_2D ≥ 150.0 or AlogP98 ≤ -2.0 or AlogP98 ≥ 7.0
        """
        if mol is None:
            return {}

        try:
            alogp98 = Crippen.MolLogP(mol)
            psa_2d = CalcTPSA(mol)
            mw = Descriptors.MolWt(mol)

            # Calculate approximate Absorption_T2_2D score
            # This is an approximation based on molecular properties
            absorption_t2_2d = self._calculate_absorption_t2_2d(mol, alogp98, psa_2d, mw)

            # Apply DS classification rules
            if psa_2d >= 150.0 or alogp98 <= -2.0 or alogp98 >= 7.0:
                absorption_level = 3
                absorption_desc = "Very Poor"
            elif absorption_t2_2d < 6.1261:
                absorption_level = 0
                absorption_desc = "Good"
            elif 6.1261 <= absorption_t2_2d < 9.6026:
                absorption_level = 1
                absorption_desc = "Moderate"
            else:  # absorption_t2_2d >= 9.6026
                absorption_level = 2
                absorption_desc = "Poor"

            return {
                'ADMET_Absorption_Level': absorption_level,
                'ADMET_Absorption_Description': absorption_desc,
                'ADMET_Absorption_T2_2D': round(absorption_t2_2d, 4),
                'ADMET_AlogP98': round(alogp98, 2),
                'ADMET_PSA_2D': round(psa_2d, 2)
            }
        except Exception as e:
            print(f"Error predicting intestinal absorption: {e}")
            return {}

    def _calculate_absorption_t2_2d(self, mol, alogp98, psa_2d, mw):
        """
        Calculate approximate Absorption T2_2D score
        This is a simplified approximation of the DS algorithm
        """
        # Simplified model based on molecular properties
        # This approximates the DS Absorption_T2_2D calculation

        hbd = CalcNumHBD(mol)
        hba = CalcNumHBA(mol)
        rotbonds = CalcNumRotatableBonds(mol)

        # Empirical formula (approximation)
        t2_2d = (0.1 * psa_2d +
                0.3 * abs(alogp98) +
                0.2 * mw/100 +
                0.4 * hbd +
                0.2 * rotbonds +
                0.1 * hba)

        return max(0, t2_2d)

    def predict_aqueous_solubility_ds(self, mol):
        """
        Predict Aqueous Solubility using DS-style classification

        Levels:
        0: log(Sw) < -8.0 (Extremely low)
        1: -8.0 < log(Sw) < -6.0 (Very low, but possible)
        2: -6.0 < log(Sw) < -4.1 (Low)
        3: -4.1 < log(Sw) < -2.0 (Good)
        4: -2.0 < log(Sw) < 0.0 (Optimal)
        5: 0.0 < log(Sw) (Too soluble)
        """
        if mol is None:
            return {}

        try:
            alogp98 = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
            rotbonds = CalcNumRotatableBonds(mol)

            # Enhanced solubility prediction (approximation of DS algorithm)
            log_sw = self._calculate_log_sw_ds(mol, alogp98, mw, aromatic_rings, rotbonds)

            # DS-style classification
            if log_sw < -8.0:
                sol_level = 0
                sol_desc = "Extremely low"
                drug_like = "No"
            elif -8.0 <= log_sw < -6.0:
                sol_level = 1
                sol_desc = "Very low, but possible"
                drug_like = "No, very low, but possible"
            elif -6.0 <= log_sw < -4.1:
                sol_level = 2
                sol_desc = "Low"
                drug_like = "Yes, low"
            elif -4.1 <= log_sw < -2.0:
                sol_level = 3
                sol_desc = "Good"
                drug_like = "Yes, good"
            elif -2.0 <= log_sw < 0.0:
                sol_level = 4
                sol_desc = "Optimal"
                drug_like = "Yes, optimal"
            else:  # log_sw >= 0.0
                sol_level = 5
                sol_desc = "Too soluble"
                drug_like = "No, too soluble"

            return {
                'ADMET_Solubility_Level': sol_level,
                'ADMET_Solubility_Description': sol_desc,
                'ADMET_Drug_Likeness': drug_like,
                'ADMET_LogSw': round(log_sw, 3)
            }
        except Exception as e:
            print(f"Error predicting DS-style solubility: {e}")
            return {}

    def _calculate_log_sw_ds(self, mol, alogp98, mw, aromatic_rings, rotbonds):
        """
        Calculate DS-style log(Sw) approximation
        """
        hbd = CalcNumHBD(mol)
        hba = CalcNumHBA(mol)

        # Enhanced solubility model (DS approximation)
        log_sw = (0.16 - 0.63 * alogp98 - 0.0062 * mw +
                 0.066 * aromatic_rings - 0.74 +
                 0.1 * hbd - 0.05 * rotbonds +
                 0.02 * hba)

        return log_sw

    def predict_bbb_penetration_ds(self, mol):
        """
        Predict BBB penetration using DS-style classification

        Levels based on logBB:
        - Very high penetrants: logBB ≥ 0.7
        - High penetrants: 0 ≤ logBB < 0.7
        - Medium penetrants: -0.52 < logBB < 0
        - Low penetrants: logBB ≤ -0.52
        """
        if mol is None:
            return {}

        try:
            alogp98 = Crippen.MolLogP(mol)
            psa_2d = CalcTPSA(mol)
            mw = Descriptors.MolWt(mol)

            # Calculate approximate logBB
            log_bb = self._calculate_log_bb_ds(mol, alogp98, psa_2d, mw)

            # DS-style classification
            if log_bb >= 0.7:
                bbb_level = "Very High"
                bbb_color = "blue"
            elif 0 <= log_bb < 0.7:
                bbb_level = "High"
                bbb_color = "green"
            elif -0.52 < log_bb < 0:
                bbb_level = "Medium"
                bbb_color = "cyan"
            else:  # log_bb <= -0.52
                bbb_level = "Low"
                bbb_color = "orange"

            return {
                'ADMET_BBB_Level': bbb_level,
                'ADMET_BBB_Color': bbb_color,
                'ADMET_LogBB': round(log_bb, 3)
            }
        except Exception as e:
            print(f"Error predicting DS-style BBB: {e}")
            return {}

    def _calculate_log_bb_ds(self, mol, alogp98, psa_2d, mw):
        """
        Calculate DS-style logBB approximation
        """
        # Enhanced BBB model based on literature
        log_bb = (-0.0148 * psa_2d + 0.152 * alogp98 + 0.139 -
                 0.0001 * mw)

        return log_bb

    def predict_cyp2d6_inhibition(self, mol):
        """
        Predict CYP2D6 inhibition using simplified Bayesian-like scoring
        DS uses Bayesian score with cutoff of 0.161
        """
        if mol is None:
            return {}

        try:
            alogp98 = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            psa_2d = CalcTPSA(mol)

            # Simplified Bayesian-like score calculation
            bayesian_score = self._calculate_cyp2d6_score(mol, alogp98, mw, psa_2d)

            # Apply DS cutoff
            is_inhibitor = "Yes" if bayesian_score > 0.161 else "No"

            # Calculate approximate Mahalanobis distance (simplified)
            md = self._calculate_mahalanobis_distance(mol)
            md_pvalue = max(0.001, min(1.0, np.exp(-md/2)))  # Approximation

            # Applicability assessment
            if md_pvalue > 0.05:
                applicability = "Reliable"
            elif md_pvalue > 0.01:
                applicability = "Moderate"
            else:
                applicability = "Low"

            return {
                'ADMET_CYP2D6_Score': round(bayesian_score, 4),
                'ADMET_CYP2D6_Inhibitor': is_inhibitor,
                'ADMET_CYP2D6_Applicability': applicability,
                'ADMET_CYP2D6_MD': round(md, 3),
                'ADMET_CYP2D6_MDpvalue': round(md_pvalue, 4)
            }
        except Exception as e:
            print(f"Error predicting CYP2D6: {e}")
            return {}

    def _calculate_cyp2d6_score(self, mol, alogp98, mw, psa_2d):
        """
        Calculate simplified CYP2D6 Bayesian-like score
        """
        # Simplified model based on molecular features
        hbd = CalcNumHBD(mol)
        hba = CalcNumHBA(mol)
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)

        # Empirical scoring function (approximation)
        score = (0.1 * alogp98 +
                0.001 * mw +
                0.002 * psa_2d +
                0.05 * aromatic_rings +
                0.02 * hba -
                0.01 * hbd - 0.3)

        return max(-1, min(1, score))  # Bound between -1 and 1

    def predict_hepatotoxicity(self, mol):
        """
        Predict Hepatotoxicity using simplified Bayesian-like scoring
        DS uses cutoff of -4.154
        """
        if mol is None:
            return {}

        try:
            # Calculate simplified hepatotoxicity score
            bayesian_score = self._calculate_hepatotoxicity_score(mol)

            # Apply DS cutoff
            is_hepatotoxic = "Yes" if bayesian_score > -4.154 else "No"

            # Calculate approximate metrics
            md = self._calculate_mahalanobis_distance(mol)
            md_pvalue = max(0.001, min(1.0, np.exp(-md/2)))

            if md_pvalue > 0.05:
                applicability = "Reliable"
            elif md_pvalue > 0.01:
                applicability = "Moderate"
            else:
                applicability = "Low"

            return {
                'ADMET_Hepatotoxic_Score': round(bayesian_score, 4),
                'ADMET_Hepatotoxic': is_hepatotoxic,
                'ADMET_Hepatotoxic_Applicability': applicability,
                'ADMET_Hepatotoxic_MD': round(md, 3),
                'ADMET_Hepatotoxic_MDpvalue': round(md_pvalue, 4)
            }
        except Exception as e:
            print(f"Error predicting hepatotoxicity: {e}")
            return {}

    def _calculate_hepatotoxicity_score(self, mol):
        """
        Calculate simplified hepatotoxicity Bayesian-like score
        """
        alogp98 = Crippen.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        psa_2d = CalcTPSA(mol)

        # Simplified scoring based on toxicophores and properties
        smiles = Chem.MolToSmiles(mol)

        score = -5.0  # Base score (non-toxic)

        # Add toxicity alerts
        if 'N(=O)=O' in smiles or '[N+](=O)[O-]' in smiles:
            score += 2.0  # Nitro groups
        if 'C(=O)Cl' in smiles:
            score += 1.5  # Acyl chlorides
        if alogp98 > 5:
            score += 0.5  # High lipophilicity
        if mw > 600:
            score += 0.3  # High molecular weight

        return score

    def predict_plasma_protein_binding(self, mol):
        """
        Predict Plasma Protein Binding (PPB)
        DS uses cutoff of -2.209 for ≥90% bound
        """
        if mol is None:
            return {}

        try:
            # Calculate simplified PPB score
            bayesian_score = self._calculate_ppb_score(mol)

            # Apply DS cutoff
            high_binding = "Yes" if bayesian_score > -2.209 else "No"

            # Calculate approximate metrics
            md = self._calculate_mahalanobis_distance(mol)
            md_pvalue = max(0.001, min(1.0, np.exp(-md/2)))

            if md_pvalue > 0.05:
                applicability = "Reliable"
            elif md_pvalue > 0.01:
                applicability = "Moderate"
            else:
                applicability = "Low"

            return {
                'ADMET_PPB_Score': round(bayesian_score, 4),
                'ADMET_PPB_HighBinding': high_binding,
                'ADMET_PPB_Applicability': applicability,
                'ADMET_PPB_MD': round(md, 3),
                'ADMET_PPB_MDpvalue': round(md_pvalue, 4)
            }
        except Exception as e:
            print(f"Error predicting PPB: {e}")
            return {}

    def _calculate_ppb_score(self, mol):
        """
        Calculate simplified PPB Bayesian-like score
        """
        alogp98 = Crippen.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        psa_2d = CalcTPSA(mol)

        # PPB typically increases with lipophilicity and size
        score = (0.3 * alogp98 +
                0.001 * mw -
                0.01 * psa_2d - 3.0)

        return score

    def _calculate_mahalanobis_distance(self, mol):
        """
        Calculate simplified Mahalanobis distance approximation
        """
        # Simplified calculation using normalized properties
        alogp98 = Crippen.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        psa_2d = CalcTPSA(mol)

        # Normalize properties (rough approximation)
        norm_alogp = (alogp98 - 2.5) / 2.0
        norm_mw = (mw - 400) / 200
        norm_psa = (psa_2d - 70) / 50

        # Simplified MD calculation
        md = np.sqrt(norm_alogp**2 + norm_mw**2 + norm_psa**2)

        return md

    def analyze_compound(self, smiles, compound_name=None):
        """Comprehensive DS-style ADMET analysis for a single compound"""
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            print(f"Invalid SMILES: {smiles}")
            return None

        # Calculate all properties
        result = {'SMILES': smiles}
        if compound_name:
            result['Compound_Name'] = compound_name

        result.update(self.calculate_molecular_properties(mol))
        result.update(self.predict_human_intestinal_absorption(mol))
        result.update(self.predict_aqueous_solubility_ds(mol))
        result.update(self.predict_bbb_penetration_ds(mol))
        result.update(self.predict_cyp2d6_inhibition(mol))
        result.update(self.predict_hepatotoxicity(mol))
        result.update(self.predict_plasma_protein_binding(mol))

        return result

    def analyze_compounds(self, compounds_data):
        """Analyze multiple compounds"""
        results = []

        if isinstance(compounds_data, list):
            if isinstance(compounds_data[0], str):
                # List of SMILES
                for i, smiles in enumerate(compounds_data):
                    result = self.analyze_compound(smiles, f"Compound_{i+1}")
                    if result:
                        results.append(result)
            elif isinstance(compounds_data[0], dict):
                # List of dictionaries
                for compound in compounds_data:
                    smiles = compound.get('smiles')
                    name = compound.get('name', 'Unknown')
                    result = self.analyze_compound(smiles, name)
                    if result:
                        results.append(result)
        elif isinstance(compounds_data, dict):
            # Dictionary with name: smiles pairs
            for name, smiles in compounds_data.items():
                result = self.analyze_compound(smiles, name)
                if result:
                    results.append(result)

        self.results = results
        return pd.DataFrame(results)

    def generate_ds_report(self, df, output_file=None):
        """Generate a DS-style ADMET report"""
        if df.empty:
            print("No data to generate report")
            return

        print("=" * 80)
        print("DISCOVERY STUDIO STYLE ADMET ANALYSIS REPORT")
        print("=" * 80)

        print(f"\nNumber of compounds analyzed: {len(df)}")

        # Human Intestinal Absorption summary
        print("\n" + "="*60)
        print("HUMAN INTESTINAL ABSORPTION SUMMARY")
        print("="*60)

        if 'ADMET_Absorption_Level' in df.columns:
            absorption_counts = df['ADMET_Absorption_Description'].value_counts()
            for desc, count in absorption_counts.items():
                print(f"{desc}: {count}/{len(df)} ({count/len(df)*100:.1f}%)")

        # Aqueous Solubility summary
        print("\n" + "="*60)
        print("AQUEOUS SOLUBILITY SUMMARY")
        print("="*60)

        if 'ADMET_Solubility_Level' in df.columns:
            sol_counts = df['ADMET_Solubility_Description'].value_counts()
            for desc, count in sol_counts.items():
                print(f"{desc}: {count}/{len(df)} ({count/len(df)*100:.1f}%)")

        # BBB Penetration summary
        print("\n" + "="*60)
        print("BBB PENETRATION SUMMARY")
        print("="*60)

        if 'ADMET_BBB_Level' in df.columns:
            bbb_counts = df['ADMET_BBB_Level'].value_counts()
            for level, count in bbb_counts.items():
                print(f"{level}: {count}/{len(df)} ({count/len(df)*100:.1f}%)")

        # CYP2D6 Inhibition summary
        print("\n" + "="*60)
        print("CYP2D6 INHIBITION SUMMARY")
        print("="*60)

        if 'ADMET_CYP2D6_Inhibitor' in df.columns:
            cyp_counts = df['ADMET_CYP2D6_Inhibitor'].value_counts()
            inhibitors = cyp_counts.get('Yes', 0)
            print(f"CYP2D6 Inhibitors: {inhibitors}/{len(df)} ({inhibitors/len(df)*100:.1f}%)")

        # Hepatotoxicity summary
        print("\n" + "="*60)
        print("HEPATOTOXICITY SUMMARY")
        print("="*60)

        if 'ADMET_Hepatotoxic' in df.columns:
            hepato_counts = df['ADMET_Hepatotoxic'].value_counts()
            hepatotoxic = hepato_counts.get('Yes', 0)
            print(f"Hepatotoxic compounds: {hepatotoxic}/{len(df)} ({hepatotoxic/len(df)*100:.1f}%)")

        # Plasma Protein Binding summary
        print("\n" + "="*60)
        print("PLASMA PROTEIN BINDING SUMMARY")
        print("="*60)

        if 'ADMET_PPB_HighBinding' in df.columns:
            ppb_counts = df['ADMET_PPB_HighBinding'].value_counts()
            high_binding = ppb_counts.get('Yes', 0)
            print(f"High PPB (≥90%): {high_binding}/{len(df)} ({high_binding/len(df)*100:.1f}%)")

        # Save detailed results
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"\nDetailed results saved to: {output_file}")

        print("\n" + "="*80)

    def plot_ds_properties(self, df, figsize=(16, 12), save_plots=True):
        """Generate DS-style visualization plots"""
        if df.empty:
            print("❌ No data to plot")
            return

        print("🎨 Creating DS-style ADMET plots...")

        try:
            plt.close('all')

            fig = plt.figure(figsize=figsize)
            fig.suptitle('Discovery Studio Style ADMET Analysis', fontsize=16, fontweight='bold')

            gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

            # Plot 1: Absorption vs Solubility
            ax1 = fig.add_subplot(gs[0, 0])
            if 'ADMET_Absorption_Level' in df.columns and 'ADMET_Solubility_Level' in df.columns:
                scatter = ax1.scatter(df['ADMET_Absorption_Level'], df['ADMET_Solubility_Level'],
                                    alpha=0.7, s=60, c=df.index, cmap='viridis')
                ax1.set_xlabel('Absorption Level (0=Good, 3=Very Poor)')
                ax1.set_ylabel('Solubility Level (0=Extremely Low, 5=Too Soluble)')
                ax1.set_title('Absorption vs Solubility')
                ax1.grid(True, alpha=0.3)

            # Plot 2: BBB Penetration Distribution
            ax2 = fig.add_subplot(gs[0, 1])
            if 'ADMET_BBB_Level' in df.columns:
                bbb_counts = df['ADMET_BBB_Level'].value_counts()
                colors = ['blue', 'green', 'cyan', 'orange']
                color_map = {'Very High': 'blue', 'High': 'green', 'Medium': 'cyan', 'Low': 'orange'}
                plot_colors = [color_map.get(level, 'gray') for level in bbb_counts.index]

                bars = ax2.bar(bbb_counts.index, bbb_counts.values, color=plot_colors, alpha=0.7)
                ax2.set_ylabel('Number of Compounds')
                ax2.set_title('BBB Penetration Levels')
                ax2.tick_params(axis='x', rotation=45, labelsize=8)

                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(height)}', ha='center', va='bottom', fontsize=8)

            # Plot 3: CYP2D6 vs Hepatotoxicity
            ax3 = fig.add_subplot(gs[0, 2])
            if 'ADMET_CYP2D6_Inhibitor' in df.columns and 'ADMET_Hepatotoxic' in df.columns:
                # Create crosstab
                crosstab = pd.crosstab(df['ADMET_CYP2D6_Inhibitor'], df['ADMET_Hepatotoxic'])
                crosstab.plot(kind='bar', ax=ax3, color=['green', 'red'], alpha=0.7)
                ax3.set_xlabel('CYP2D6 Inhibitor')
                ax3.set_ylabel('Number of Compounds')
                ax3.set_title('CYP2D6 vs Hepatotoxicity')
                ax3.legend(title='Hepatotoxic', labels=['No', 'Yes'])
                ax3.tick_params(axis='x', rotation=0)

            # Plot 4: Molecular Properties Distribution
            ax4 = fig.add_subplot(gs[1, 0])
            if 'ADMET_AlogP98' in df.columns:
                ax4.hist(df['ADMET_AlogP98'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
                ax4.axvline(-2, color='red', linestyle='--', alpha=0.7, label='Lower limit')
                ax4.axvline(7, color='red', linestyle='--', alpha=0.7, label='Upper limit')
                ax4.set_xlabel('AlogP98')
                ax4.set_ylabel('Frequency')
                ax4.set_title('AlogP98 Distribution')
                ax4.legend(fontsize=8)

            # Plot 5: PSA Distribution
            ax5 = fig.add_subplot(gs[1, 1])
            if 'ADMET_PSA_2D' in df.columns:
                ax5.hist(df['ADMET_PSA_2D'], bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
                ax5.axvline(131.62, color='blue', linestyle='--', alpha=0.7, label='95% limit')
                ax5.axvline(148.12, color='magenta', linestyle='--', alpha=0.7, label='99% limit')
                ax5.axvline(150, color='red', linestyle='--', alpha=0.7, label='Absorption limit')
                ax5.set_xlabel('PSA_2D (Ų)')
                ax5.set_ylabel('Frequency')
                ax5.set_title('Polar Surface Area Distribution')
                ax5.legend(fontsize=8)

            # Plot 6: Bayesian Scores Comparison
            ax6 = fig.add_subplot(gs[1, 2])
            scores_data = []
            score_labels = []

            for col, label in [('ADMET_CYP2D6_Score', 'CYP2D6'),
                              ('ADMET_Hepatotoxic_Score', 'Hepatotoxic'),
                              ('ADMET_PPB_Score', 'PPB')]:
                if col in df.columns:
                    scores_data.append(df[col])
                    score_labels.append(label)

            if scores_data:
                ax6.boxplot(scores_data, labels=score_labels)
                ax6.set_ylabel('Bayesian Score')
                ax6.set_title('Bayesian Scores Distribution')
                ax6.tick_params(axis='x', rotation=45, labelsize=8)

            # Plot 7: Absorption T2_2D vs PSA
            ax7 = fig.add_subplot(gs[2, 0])
            if 'ADMET_Absorption_T2_2D' in df.columns and 'ADMET_PSA_2D' in df.columns:
                colors = df['ADMET_Absorption_Level']
                scatter = ax7.scatter(df['ADMET_PSA_2D'], df['ADMET_Absorption_T2_2D'],
                                    c=colors, cmap='RdYlGn_r', alpha=0.7, s=60)
                ax7.axhline(6.1261, color='blue', linestyle='--', alpha=0.7, label='Good/Moderate')
                ax7.axhline(9.6026, color='red', linestyle='--', alpha=0.7, label='Moderate/Poor')
                ax7.set_xlabel('PSA_2D (Ų)')
                ax7.set_ylabel('Absorption T2_2D')
                ax7.set_title('Absorption Space')
                ax7.legend(fontsize=8)
                ax7.grid(True, alpha=0.3)

                cbar = plt.colorbar(scatter, ax=ax7, shrink=0.8)
                cbar.set_label('Absorption Level', fontsize=8)

            # Plot 8: Solubility vs LogBB
            ax8 = fig.add_subplot(gs[2, 1])
            if 'ADMET_LogSw' in df.columns and 'ADMET_LogBB' in df.columns:
                scatter = ax8.scatter(df['ADMET_LogSw'], df['ADMET_LogBB'],
                                    alpha=0.7, s=60, c=df.index, cmap='plasma')
                ax8.axhline(0.7, color='blue', linestyle='--', alpha=0.7, label='Very High BBB')
                ax8.axhline(0, color='green', linestyle='--', alpha=0.7, label='High BBB')
                ax8.axhline(-0.52, color='orange', linestyle='--', alpha=0.7, label='Low BBB')
                ax8.set_xlabel('LogSw')
                ax8.set_ylabel('LogBB')
                ax8.set_title('Solubility vs BBB')
                ax8.legend(fontsize=8)
                ax8.grid(True, alpha=0.3)

            # Plot 9: Applicability Assessment
            ax9 = fig.add_subplot(gs[2, 2])
            applicability_cols = ['ADMET_CYP2D6_Applicability', 'ADMET_Hepatotoxic_Applicability', 'ADMET_PPB_Applicability']
            reliable_counts = []
            labels = ['CYP2D6', 'Hepatotoxic', 'PPB']

            for col in applicability_cols:
                if col in df.columns:
                    reliable_count = (df[col] == 'Reliable').sum()
                    reliable_counts.append(reliable_count)

            if reliable_counts:
                bars = ax9.bar(labels, reliable_counts, color=['blue', 'red', 'green'], alpha=0.7)
                ax9.set_ylabel('Reliable Predictions')
                ax9.set_title('Model Applicability')

                for bar in bars:
                    height = bar.get_height()
                    ax9.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(height)}', ha='center', va='bottom', fontsize=8)

            plt.tight_layout()

            if save_plots:
                plot_filename = 'ds_style_admet_plots.png'
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"📊 DS-style plots saved as: {plot_filename}")

            print("📈 DS-style plots created successfully!")
            plt.close()

        except Exception as e:
            print(f"❌ Error creating DS-style plots: {e}")


def main():
    """Main function with DS-style ADMET analysis"""

    # Initialize DS-style calculator
    calc = DSStyleADMETCalculator()

    # Example compounds for testing
    example_compounds = {
        'Aspirin': 'CC(=O)OC1=CC=CC=C1C(=O)O',
        'Ibuprofen': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
        'Acetaminophen': 'CC(=O)NC1=CC=C(C=C1)O',
        'Caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'Diazepam': 'CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3',
        'Atorvastatin': 'CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4'
    }

    print("🧬 Starting Discovery Studio Style ADMET Analysis...")
    print("📝 Analyzing example compounds...")

    # Analyze compounds
    results_df = calc.analyze_compounds(example_compounds)

    # Display results
    print(f"\n✅ Analysis complete! Analyzed {len(results_df)} compounds successfully")

    # Generate DS-style report
    calc.generate_ds_report(results_df, 'ds_style_admet_results.csv')

    # Generate DS-style plots
    print("\n🎨 Generating DS-style plots...")
    calc.plot_ds_properties(results_df, save_plots=True)

    # Display key DS-style results
    print("\n📊 DS-STYLE SUMMARY TABLE:")
    summary_cols = ['Compound_Name', 'ADMET_Absorption_Description', 'ADMET_Solubility_Description',
                   'ADMET_BBB_Level', 'ADMET_CYP2D6_Inhibitor', 'ADMET_Hepatotoxic']

    if all(col in results_df.columns for col in summary_cols):
        print(results_df[summary_cols].to_string(index=False))

    return results_df

if __name__ == "__main__":
    # Run the DS-style analysis
    results = main()

    print("\n" + "="*80)
    print("🔬 TO ANALYZE YOUR OWN COMPOUNDS:")
    print("="*80)
    print("""
# Use your own SMILES:
my_compounds = {
    'Compound_1': 'YOUR_SMILES_HERE',
    'Compound_2': 'YOUR_SMILES_HERE',
    # ... add more compounds
}

# Run DS-style analysis:
calc = DSStyleADMETCalculator()
results = calc.analyze_compounds(my_compounds)
calc.generate_ds_report(results, 'my_ds_results.csv')
calc.plot_ds_properties(results, save_plots=True)
    """)
