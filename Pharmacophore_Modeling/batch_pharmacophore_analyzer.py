"""
BATCH PHARMACOPHORE ANALYZER
Process multiple molecules from SDF files
Perfect for analyzing compound libraries
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor, rdMolDescriptors
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from rdkit import DataStructs

# For visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image, ImageDraw
import io
import warnings
warnings.filterwarnings('ignore')

class BatchPharmacophoreAnalyzer:
    """
    Batch processor for pharmacophore analysis of multiple molecules
    """
    
    def __init__(self):
        """Initialize the batch analyzer"""
        print("="*60)
        print("BATCH PHARMACOPHORE ANALYZER")
        print("="*60)
        
        # Load feature factory
        fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        self.feat_factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
        
        # Feature colors
        self.feature_colors = {
            'Donor': '#2ECC40',
            'Acceptor': '#FF4136',
            'Aromatic': '#FFDC00',
            'Hydrophobe': '#01C9E1',
            'PosIonizable': '#0074D9',
            'NegIonizable': '#B10DC9',
            'LumpedHydrophobe': '#FF851B'
        }
        
        # Statistics storage
        self.batch_statistics = []
        self.similarity_matrix = None
        
        print("[✓] Batch analyzer initialized")
        print()
    
    def process_sdf_file(self, sdf_path, output_dir="batch_output", 
                         max_molecules=None, generate_images=True, generate_3d=False):
        """
        Process all molecules in an SDF file
        
        Parameters:
        -----------
        sdf_path : str
            Path to SDF file
        output_dir : str
            Output directory for results
        max_molecules : int or None
            Maximum number of molecules to process (None = all)
        generate_images : bool
            Whether to generate visualization images (can be slow for large batches)
        """
        start_time = time.time()
        
        print(f"Processing SDF file: {sdf_path}")
        print("-"*60)
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create subdirectories
        subdirs = ['molecules', 'visualizations', 'csv_data', 'fingerprints']
        for subdir in subdirs:
            path = os.path.join(output_dir, subdir)
            if not os.path.exists(path):
                os.makedirs(path)
        
        # Load molecules from SDF
        supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
        
        # Process each molecule
        molecules_data = []
        all_features = []
        all_fingerprints = []
        failed_molecules = []
        
        mol_count = 0
        for idx, mol in enumerate(supplier):
            if max_molecules and mol_count >= max_molecules:
                break
            
            if mol is None:
                failed_molecules.append(idx)
                continue
            
            mol_count += 1
            
            # Get molecule name or use index
            if mol.HasProp("_Name"):
                mol_name = mol.GetProp("_Name")
            else:
                mol_name = f"Molecule_{idx}"
            
            print(f"\nProcessing {mol_count}: {mol_name}")
            print("-"*40)
            
            try:
                # Process molecule
                mol_data = self.analyze_single_molecule(
                    mol, mol_name, idx, 
                    output_dir, generate_images, generate_3d
                )
                molecules_data.append(mol_data)
                
            except Exception as e:
                print(f"  [!] Error processing {mol_name}: {str(e)}")
                failed_molecules.append(idx)
                continue
        
        print("\n" + "="*60)
        print("BATCH PROCESSING COMPLETE")
        print("="*60)
        
        # Generate summary reports
        self.generate_batch_summary(molecules_data, output_dir)
        
        # Generate similarity matrix
        if len(molecules_data) > 1:
            self.generate_similarity_matrix(molecules_data, output_dir)
        
        # Generate batch comparison visualization
        if generate_images and len(molecules_data) > 1:
            self.generate_batch_comparison(molecules_data, output_dir)
        
        # Print summary
        elapsed_time = time.time() - start_time
        print(f"\n[✓] Processed {mol_count} molecules in {elapsed_time:.2f} seconds")
        print(f"[✓] Failed molecules: {len(failed_molecules)}")
        print(f"[✓] Results saved to: {output_dir}")
        
        return molecules_data
    
    def analyze_single_molecule(self, mol, mol_name, idx, output_dir, generate_images, generate_3d):
        """
        Analyze a single molecule with complete data export
        """
        # Ensure 3D coordinates
        if generate_3d:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
        if mol.GetNumConformers() == 0:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
        
        # Extract features
        raw_features = self.feat_factory.GetFeaturesForMol(mol)
        features = [(feat.GetFamily(), feat.GetAtomIds()) for feat in raw_features]
        
        # Count features
        feature_counts = {}
        for feat_type, _ in features:
            if feat_type not in feature_counts:
                feature_counts[feat_type] = 0
            feature_counts[feat_type] += 1
        
        print(f"  Features found: {sum(feature_counts.values())}")
        for feat_type, count in feature_counts.items():
            print(f"    {feat_type}: {count}")
        
        # Generate fingerprint
        sig_factory = Gobbi_Pharm2D.factory
        fp = Generate.Gen2DFingerprint(mol, sig_factory)
        
        # Save fingerprint to fingerprints folder
        fp_bits = list(fp.GetOnBits())
        fp_data = {
            'Molecule': mol_name,
            'Fingerprint_Size': fp.GetNumBits(),
            'Active_Bits': len(fp_bits),
            'Bit_Density': round(len(fp_bits)/fp.GetNumBits(), 4),
            'First_100_Bits': str(fp_bits[:100])
        }
        fp_df = pd.DataFrame([fp_data])
        fp_path = os.path.join(output_dir, 'fingerprints', f"{mol_name}_fingerprint.csv")
        fp_df.to_csv(fp_path, index=False)
        
        # Calculate molecular properties
        mol_props = {
            'Index': idx,
            'Name': mol_name,
            'Formula': rdMolDescriptors.CalcMolFormula(mol),
            'MW': round(Descriptors.MolWt(mol), 2),
            'HeavyAtoms': Descriptors.HeavyAtomCount(mol),
            'RotatableBonds': Descriptors.NumRotatableBonds(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'TPSA': round(Descriptors.TPSA(mol), 2),
            'LogP': round(Descriptors.MolLogP(mol), 2),
            'TotalFeatures': sum(feature_counts.values())
        }
        
        # Add feature counts to properties
        for feat_type in self.feature_colors.keys():
            mol_props[f'n_{feat_type}'] = feature_counts.get(feat_type, 0)
        
        # Save molecular properties to csv_data folder
        mol_props_df = pd.DataFrame([mol_props])
        props_path = os.path.join(output_dir, 'csv_data', f"{mol_name}_properties.csv")
        mol_props_df.to_csv(props_path, index=False)
        
        # Save features to CSV
        features_data = []
        for i, (feat_type, atom_ids) in enumerate(features):
            features_data.append({
                'Feature_ID': f'{feat_type}_{i+1}',
                'Feature_Type': feat_type,
                'Atom_Indices': str(atom_ids),
                'Num_Atoms': len(atom_ids)
            })
        
        df_features = pd.DataFrame(features_data)
        
        # Create molecule directory
        mol_dir = os.path.join(output_dir, 'molecules', mol_name)
        if not os.path.exists(mol_dir):
            os.makedirs(mol_dir)
        
        # Save to molecule folder
        mol_features_path = os.path.join(mol_dir, f"{mol_name}_features.csv")
        df_features.to_csv(mol_features_path, index=False)
        
        # Also save to csv_data folder
        csv_features_path = os.path.join(output_dir, 'csv_data', f"{mol_name}_features.csv")
        df_features.to_csv(csv_features_path, index=False)
        
        # Generate visualizations if requested
        if generate_images:
            self.generate_molecule_visualizations(mol, mol_name, features, output_dir)

        if generate_3d:
                mol_dir = os.path.join(output_dir, 'molecules', mol_name)
                if not os.path.exists(mol_dir):
                    os.makedirs(mol_dir)
                html_path = os.path.join(mol_dir, f"{mol_name}_3d.html")
                self.visualize_3d_pharmacophore(mol, features, save_path=html_path)
        
        # Store data for batch analysis
        mol_data = {
            'mol': mol,
            'name': mol_name,
            'features': features,
            'fingerprint': fp,
            'properties': mol_props,
            'feature_counts': feature_counts
        }
        
        return mol_data
    
    def generate_molecule_visualizations(self, mol, mol_name, features, output_dir):
        """
        Generate visualizations for a single molecule
        """
        # 2D structure with pharmacophore overlay
        rdDepictor.Compute2DCoords(mol)
        
        # Prepare highlights
        highlight_atoms = []
        highlight_colors = {}
        
        for feat_type, atom_ids in features:
            color = self.feature_colors.get(feat_type, '#888888')
            rgb = tuple(int(color[i:i+2], 16)/255 for i in (1, 3, 5))
            
            for atom_id in atom_ids:
                highlight_atoms.append(atom_id)
                highlight_colors[atom_id] = rgb
        
        # Create drawing
        drawer = rdMolDraw2D.MolDraw2DCairo(600, 400)
        
        # Try to add atom indices
        try:
            drawer.drawOptions().addAtomIndices = True
        except AttributeError:
            pass  # Skip if not available in this RDKit version
        
        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=highlight_colors
        )
        drawer.FinishDrawing()
        
        # Get the drawing data
        img_data = drawer.GetDrawingText()
        
        # Save to molecule folder
        mol_dir = os.path.join(output_dir, 'molecules', mol_name)
        if not os.path.exists(mol_dir):
            os.makedirs(mol_dir)
        mol_img_path = os.path.join(mol_dir, f"{mol_name}_2d.png")
        with open(mol_img_path, 'wb') as f:
            f.write(img_data)
        
        # Save to visualizations folder
        viz_img_path = os.path.join(output_dir, 'visualizations', f"{mol_name}_2d.png")
        with open(viz_img_path, 'wb') as f:
            f.write(img_data)
    
    def generate_batch_summary(self, molecules_data, output_dir):
        """
        Generate comprehensive batch summary
        """
        print("\nGenerating batch summary...")
        
        # Create summary DataFrame
        summary_data = []
        for mol_data in molecules_data:
            summary_data.append(mol_data['properties'])
        
        df_summary = pd.DataFrame(summary_data)
        
        # Save main summary
        summary_path = os.path.join(output_dir, 'batch_summary.csv')
        df_summary.to_csv(summary_path, index=False)
        print(f"  [✓] Saved: batch_summary.csv")
        
        # Generate statistics
        stats_data = {
            'Metric': [],
            'Mean': [],
            'Std': [],
            'Min': [],
            'Max': []
        }
        
        for col in ['MW', 'HeavyAtoms', 'RotatableBonds', 'HBD', 'HBA', 'TPSA', 'LogP', 'TotalFeatures']:
            if col in df_summary.columns:
                stats_data['Metric'].append(col)
                stats_data['Mean'].append(round(df_summary[col].mean(), 2))
                stats_data['Std'].append(round(df_summary[col].std(), 2))
                stats_data['Min'].append(round(df_summary[col].min(), 2))
                stats_data['Max'].append(round(df_summary[col].max(), 2))
        
        df_stats = pd.DataFrame(stats_data)
        stats_path = os.path.join(output_dir, 'batch_statistics.csv')
        df_stats.to_csv(stats_path, index=False)
        print(f"  [✓] Saved: batch_statistics.csv")
        
        # Print statistics
        print("\nBatch Statistics:")
        print("-"*40)
        print(df_stats.to_string(index=False))
    
    def generate_similarity_matrix(self, molecules_data, output_dir):
        """
        Generate Tanimoto similarity matrix for all molecules
        """
        print("\nCalculating similarity matrix...")
        
        n_mols = len(molecules_data)
        similarity_matrix = np.zeros((n_mols, n_mols))
        mol_names = [mol['name'] for mol in molecules_data]
        
        # Calculate pairwise similarities
        for i in range(n_mols):
            for j in range(i, n_mols):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity = DataStructs.TanimotoSimilarity(
                        molecules_data[i]['fingerprint'],
                        molecules_data[j]['fingerprint']
                    )
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
        
        # Save similarity matrix
        df_similarity = pd.DataFrame(
            similarity_matrix,
            index=mol_names,
            columns=mol_names
        )
        
        similarity_path = os.path.join(output_dir, 'similarity_matrix.csv')
        df_similarity.to_csv(similarity_path)
        print(f"  [✓] Saved: similarity_matrix.csv")
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            df_similarity,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            square=True,
            cbar_kws={'label': 'Tanimoto Similarity'},
            ax=ax
        )
        ax.set_title('Pharmacophore Similarity Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        heatmap_path = os.path.join(output_dir, 'similarity_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [✓] Saved: similarity_heatmap.png")
        
        # Find most similar pairs
        print("\nMost Similar Molecule Pairs:")
        print("-"*40)
        
        # Get upper triangle indices
        upper_tri = np.triu_indices(n_mols, k=1)
        similarities = similarity_matrix[upper_tri]
        
        # Sort and get top pairs
        sorted_idx = np.argsort(similarities)[::-1]
        
        num_pairs = min(5, len(sorted_idx))
        for i in range(num_pairs):
            idx = sorted_idx[i]
            row = upper_tri[0][idx]
            col = upper_tri[1][idx]
            sim = similarities[idx]
            print(f"  {mol_names[row]} <-> {mol_names[col]}: {sim:.3f}")
        
        return df_similarity

    def visualize_3d_pharmacophore(self, mol, features, save_path=None):
        """
        Create interactive 3D pharmacophore visualization
        """
        print("Creating 3D interactive pharmacophore...")
        
        conf = mol.GetConformer()
        fig = go.Figure()
        
        # Add bonds
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            
            begin_pos = conf.GetAtomPosition(begin_idx)
            end_pos = conf.GetAtomPosition(end_idx)
            
            fig.add_trace(go.Scatter3d(
                x=[begin_pos.x, end_pos.x],
                y=[begin_pos.y, end_pos.y],
                z=[begin_pos.z, end_pos.z],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add atoms
        atom_x, atom_y, atom_z = [], [], []
        atom_colors = []
        atom_labels = [] 
        
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            atom_x.append(pos.x)
            atom_y.append(pos.y)
            atom_z.append(pos.z)
            
            atom = mol.GetAtomWithIdx(i)
            element = atom.GetSymbol()
            atom_labels.append(f"{element}{i}")
            
            if element == 'C': atom_colors.append('black')
            elif element == 'N': atom_colors.append('blue')
            elif element == 'O': atom_colors.append('red')
            elif element == 'S': atom_colors.append('yellow')
            else: atom_colors.append('gray')
        
        fig.add_trace(go.Scatter3d(
            x=atom_x, y=atom_y, z=atom_z,
            mode='markers+text',  # Add text mode
            marker=dict(
                size=10,  # Slightly bigger
                color=atom_colors,
                opacity=0.9,
                line=dict(color='darkgray', width=0.5)
            ),
            text=atom_labels,
            textposition='top center',
            textfont=dict(size=9, color='black'),
            hovertemplate='<b>Atom %{text}</b><br>' +
                          'Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
                          '<extra></extra>',
            showlegend=False,
            name='Atoms'
        ))
        
        # Add pharmacophore features
        feature_traces = {}
        for feat_type, atom_ids in features:
            if feat_type not in feature_traces:
                feature_traces[feat_type] = {'x': [], 'y': [], 'z': []}
            
            # Calculate centroid
            coords = []
            for atom_id in atom_ids:
                pos = conf.GetAtomPosition(atom_id)
                coords.append([pos.x, pos.y, pos.z])
            
            centroid = np.mean(coords, axis=0)
            feature_traces[feat_type]['x'].append(centroid[0])
            feature_traces[feat_type]['y'].append(centroid[1])
            feature_traces[feat_type]['z'].append(centroid[2])
        
        # Add feature spheres
        for feat_type, positions in feature_traces.items():
            color = self.feature_colors.get(feat_type, '#888888')
            feature_labels = [f"{feat_type[:3]}{i+1}" for i in range(len(positions['x']))]
            fig.add_trace(go.Scatter3d(
                x=positions['x'],
                y=positions['y'],
                z=positions['z'],
                mode='markers+text',
                marker=dict(
                    size=25,
                    color=color,
                    opacity=0.4,
                    line=dict(color=color, width=2)
                ),
                text=feature_labels,
                        textposition='top center',
                        textfont=dict(size=11, color=color, family='Arial Black'),
                        name=feat_type,
                        hovertemplate=f'<b>{feat_type}</b><br>' +
                                      'Feature: %{text}<br>' +
                                      'Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
                                      '<extra></extra>'
                ))
        
        # Update layout
        mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "Molecule"
        fig.update_layout(
        title=f"3D Pharmacophore Model - {mol_name}",
        scene=dict(
            xaxis_title="X (Å)",
            yaxis_title="Y (Å)",
            zaxis_title="Z (Å)",
            aspectmode='data',
            # ADD THIS for better camera angle:
            camera=dict(
                eye=dict(x=1.25, y=1.25, z=1.25),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
        ),
        width=1280,  
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        autosize=True,
        margin=dict(l=0, r=0, t=50, b=0),  # Minimize margins
        hovermode='closest'
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"  ✓ Saved: {save_path}")
        
        return fig
    
    def generate_batch_comparison(self, molecules_data, output_dir):
        """
        Generate comparison visualizations for the batch
        """
        print("\nGenerating batch comparison plots...")
        
        # Feature distribution comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Feature counts bar plot
        ax = axes[0, 0]
        feature_types = list(self.feature_colors.keys())
        mol_names = [mol['name'][:15] for mol in molecules_data[:10]]  # Limit to 10 for clarity
        
        feature_matrix = []
        for mol in molecules_data[:10]:
            counts = [mol['feature_counts'].get(ft, 0) for ft in feature_types]
            feature_matrix.append(counts)
        
        feature_matrix = np.array(feature_matrix).T
        
        x = np.arange(len(mol_names))
        width = 0.8 / len(feature_types)
        
        for i, feat_type in enumerate(feature_types):
            offset = (i - len(feature_types)/2) * width + width/2
            ax.bar(x + offset, feature_matrix[i], width, 
                  label=feat_type, color=self.feature_colors[feat_type])
        
        ax.set_xlabel('Molecule')
        ax.set_ylabel('Count')
        ax.set_title('Feature Distribution Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(mol_names, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        
        # 2. Property distributions
        ax = axes[0, 1]
        properties = ['MW', 'LogP', 'TPSA', 'HBD']
        data_for_box = []
        labels_for_box = []
        
        for prop in properties:
            values = [mol['properties'][prop] for mol in molecules_data]
            data_for_box.append(values)
            labels_for_box.append(prop)
        
        bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']):
            patch.set_facecolor(color)
        
        ax.set_title('Molecular Property Distributions')
        ax.set_ylabel('Value')
        ax.grid(axis='y', alpha=0.3)
        
        # 3. Total features histogram
        ax = axes[1, 0]
        total_features = [mol['properties']['TotalFeatures'] for mol in molecules_data]
        ax.hist(total_features, bins=20, color='#667EEA', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Total Pharmacophore Features')
        ax.set_ylabel('Number of Molecules')
        ax.set_title('Distribution of Total Features')
        ax.grid(axis='y', alpha=0.3)
        
        # Add statistics
        mean_features = np.mean(total_features)
        ax.axvline(mean_features, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_features:.1f}')
        ax.legend()
        
        # 4. Lipinski's Rule of Five compliance
        ax = axes[1, 1]
        ro5_pass = 0
        ro5_fail = 0
        
        for mol in molecules_data:
            props = mol['properties']
            if (props['MW'] <= 500 and props['LogP'] <= 5 and 
                props['HBD'] <= 5 and props['HBA'] <= 10):
                ro5_pass += 1
            else:
                ro5_fail += 1
        
        ax.pie([ro5_pass, ro5_fail], 
               labels=['Pass Ro5', 'Fail Ro5'],
               colors=['#2ECC40', '#FF4136'],
               autopct='%1.1f%%',
               startangle=90)
        ax.set_title("Lipinski's Rule of Five Compliance")
        
        plt.suptitle('Batch Pharmacophore Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        comparison_path = os.path.join(output_dir, 'batch_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [✓] Saved: batch_comparison.png")
    
    def filter_molecules(self, molecules_data, criteria):
        """
        Filter molecules based on criteria
        
        Parameters:
        -----------
        criteria : dict
            Dictionary with property names and (min, max) tuples
            Example: {'MW': (200, 500), 'LogP': (0, 5)}
        """
        filtered = []
        
        for mol_data in molecules_data:
            props = mol_data['properties']
            pass_filter = True
            
            for prop, (min_val, max_val) in criteria.items():
                if prop in props:
                    if props[prop] < min_val or props[prop] > max_val:
                        pass_filter = False
                        break
            
            if pass_filter:
                filtered.append(mol_data)
        
        return filtered
    
    def export_filtered_sdf(self, molecules_data, output_path):
        """
        Export filtered molecules to new SDF file
        """
        writer = Chem.SDWriter(output_path)
        
        for mol_data in molecules_data:
            mol = mol_data['mol']
            mol.SetProp("_Name", mol_data['name'])
            mol.SetProp("TotalFeatures", str(mol_data['properties']['TotalFeatures']))
            writer.write(mol)
        
        writer.close()
        print(f"[✓] Exported {len(molecules_data)} molecules to {output_path}")

def main():
    """
    Main execution for batch processing
    """
    print("\n" + "="*60)
    print("     BATCH PHARMACOPHORE ANALYSIS TUTORIAL")
    print("="*60 + "\n")
    
    # Initialize batch analyzer
    analyzer = BatchPharmacophoreAnalyzer()
    
    # Example 1: Process SDF file
    print("EXAMPLE 1: Processing Multiple Molecules from SDF")
    print("-"*60)
    
    # Create example SDF file with multiple molecules
    print("Creating example SDF file with FDA-approved drugs...")
    
    drugs = [
        ("Imatinib", "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"),
        ("Dasatinib", "CC1=NC(=CC(=N1)N2CCN(CC2)CCO)NC3=C(C=CC=C3Cl)C(=O)NS(=O)(=O)C4=CC=CC=C4C"),
        ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
        ("Ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    ]
    
    # Create SDF file
    writer = Chem.SDWriter("example_drugs.sdf")
    for name, smiles in drugs:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            mol.SetProp("_Name", name)
            writer.write(mol)
    writer.close()
    print("[✓] Created example_drugs.sdf\n")
    
    # Process the SDF file
    molecules_data = analyzer.process_sdf_file(
        "example_drugs.sdf",
        output_dir="batch_output",
        max_molecules=None,
        generate_images=True
    )
    
    # Example 2: Filter molecules
    print("\n" + "-"*60)
    print("EXAMPLE 2: Filtering Molecules by Properties")
    print("-"*60)
    
    # Filter for drug-like molecules
    criteria = {
        'MW': (150, 500),
        'LogP': (-0.4, 5.6),
        'HBD': (0, 5),
        'HBA': (0, 10)
    }
    
    filtered = analyzer.filter_molecules(molecules_data, criteria)
    print(f"\nFiltered molecules: {len(filtered)} out of {len(molecules_data)}")
    for mol in filtered:
        print(f"  - {mol['name']}: MW={mol['properties']['MW']}, LogP={mol['properties']['LogP']}")
    
    # Export filtered molecules
    if filtered:
        analyzer.export_filtered_sdf(filtered, "filtered_drugs.sdf")
    
    print("\n" + "="*60)
    print("        BATCH ANALYSIS COMPLETE!")
    print("="*60 + "\n")
    
    print("OUTPUTS GENERATED:")
    print("-"*40)
    print("📁 batch_output/")
    print("  ├── batch_summary.csv - Overview of all molecules")
    print("  ├── batch_statistics.csv - Statistical analysis")
    print("  ├── similarity_matrix.csv - Pairwise similarities")
    print("  ├── similarity_heatmap.png - Visual similarity matrix")
    print("  ├── batch_comparison.png - Comparative analysis plots")
    print("  ├── molecules/ - Individual molecule data")
    print("  ├── visualizations/ - 2D structure images")
    print("  ├── csv_data/ - All CSV files in one place")
    print("  └── fingerprints/ - Fingerprint data for each molecule")
    print()
    print("Perfect for:")
    print("  • High-throughput screening")
    print("  • SAR analysis")
    print("  • Lead optimization")
    print("  • Library design")

if __name__ == "__main__":
    main()