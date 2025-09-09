"""
COMPLETE PHARMACOPHORE MODELING TUTORIAL
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor, rdMolDescriptors
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from rdkit import DataStructs

# For 3D visualization
import plotly.graph_objects as go
import plotly.express as px

# For image processing
from PIL import Image, ImageDraw, ImageFont
import io
import warnings
warnings.filterwarnings('ignore')

class CompletePharmacophoreAnalyzer:
    """
    Complete Pharmacophore Analysis and Visualization System
    """
    
    def __init__(self):
        """Initialize the complete system"""
        print("="*60)
        print("INITIALIZING PHARMACOPHORE ANALYSIS SYSTEM")
        print("="*60)
        
        # Load feature factory
        fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        self.feat_factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
        
        # Define color scheme for pharmacophore features
        self.feature_colors = {
            'Donor': '#2ECC40',        # Bright Green
            'Acceptor': '#FF4136',     # Bright Red
            'Aromatic': '#FFDC00',     # Gold
            'Hydrophobe': '#01C9E1',   # Cyan
            'PosIonizable': '#0074D9', # Blue
            'NegIonizable': '#B10DC9', # Purple
            'LumpedHydrophobe': '#FF851B' # Orange
        }
        
        # Feature symbols for labeling
        self.feature_symbols = {
            'Donor': 'D',
            'Acceptor': 'A',
            'Aromatic': 'R',
            'Hydrophobe': 'H',
            'PosIonizable': '+',
            'NegIonizable': '-',
            'LumpedHydrophobe': 'L'
        }
        
        print("✓ System initialized successfully!")
        print(f"✓ Available features: {', '.join(self.feature_colors.keys())}")
        print()
    
    # ========================================
    # PART 1: MOLECULE LOADING AND PROCESSING
    # ========================================
    
    def load_molecule(self, input_data, input_type='smiles', name=None):
        """
        Load molecule from SMILES or SDF file
        """
        print(f"Loading molecule ({input_type})...")
        
        if input_type.lower() == 'smiles':
            mol = Chem.MolFromSmiles(input_data)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {input_data}")
            # Add 3D coordinates
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
        
        elif input_type.lower() == 'sdf':
            supplier = Chem.SDMolSupplier(input_data, removeHs=False)
            mol = next(supplier)
            if mol is None:
                raise ValueError(f"Could not read SDF file: {input_data}")
        
        else:
            raise ValueError("input_type must be 'smiles' or 'sdf'")
        
        # Store molecule name
        if name:
            mol.SetProp("_Name", name)
        
        # Print molecule info
        print(f"✓ Molecule loaded: {name if name else 'Unknown'}")
        print(f"  Formula: {rdMolDescriptors.CalcMolFormula(mol)}")
        print(f"  MW: {Descriptors.MolWt(mol):.2f}")
        print(f"  Atoms: {mol.GetNumAtoms()}")
        print(f"  Bonds: {mol.GetNumBonds()}")
        print()
        
        return mol
    
    # ========================================
    # PART 2: PHARMACOPHORE ANALYSIS
    # ========================================
    
    def extract_pharmacophore_features(self, mol):
        """
        Extract all pharmacophore features from molecule
        """
        print("Extracting pharmacophore features...")
        print("-"*40)
        
        # Get features
        raw_features = self.feat_factory.GetFeaturesForMol(mol)
        
        # Organize features
        features = []
        feature_summary = {}
        
        for feat in raw_features:
            feat_type = feat.GetFamily()
            atom_ids = feat.GetAtomIds()
            features.append((feat_type, atom_ids))
            
            # Count features
            if feat_type not in feature_summary:
                feature_summary[feat_type] = 0
            feature_summary[feat_type] += 1
        
        # Print summary
        for feat_type, count in feature_summary.items():
            color_code = self.feature_colors.get(feat_type, '#888888')
            print(f"  {feat_type}: {count} features")
        
        print(f"\nTotal features found: {len(features)}")
        print()
        
        return features
    
    def calculate_pharmacophore_distances(self, mol, features):
        """
        Calculate distances between pharmacophore features
        """
        print("Calculating inter-feature distances...")
        
        conf = mol.GetConformer()
        feature_coords = []
        feature_labels = []
        
        # Get centroids for each feature
        for i, (feat_type, atom_ids) in enumerate(features):
            coords = []
            for atom_id in atom_ids:
                pos = conf.GetAtomPosition(atom_id)
                coords.append([pos.x, pos.y, pos.z])
            
            centroid = np.mean(coords, axis=0)
            feature_coords.append(centroid)
            feature_labels.append(f"{self.feature_symbols.get(feat_type, 'X')}{i+1}")
        
        # Calculate distance matrix
        n_features = len(feature_coords)
        distances = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(i+1, n_features):
                dist = np.linalg.norm(
                    np.array(feature_coords[i]) - np.array(feature_coords[j])
                )
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Create DataFrame
        df_distances = pd.DataFrame(
            distances,
            index=feature_labels,
            columns=feature_labels
        )
        
        # Print key distances
        print("-"*40)
        print("Key pharmacophore distances (Å):")
        count = 0
        for i in range(n_features):
            for j in range(i+1, n_features):
                if count < 5:  # Show first 5 distances
                    print(f"  {feature_labels[i]} <-> {feature_labels[j]}: {distances[i, j]:.2f} Å")
                    count += 1
        
        if count < (n_features * (n_features-1) / 2):
            print(f"  ... and {int(n_features * (n_features-1) / 2 - count)} more distances")
        
        print()
        return df_distances, feature_coords, feature_labels
    
    def generate_pharmacophore_fingerprint(self, mol):
        """
        Generate pharmacophore fingerprint for the molecule
        """
        print("Generating pharmacophore fingerprint...")
        
        # Generate fingerprint
        sig_factory = Gobbi_Pharm2D.factory
        fp = Generate.Gen2DFingerprint(mol, sig_factory)
        
        # Get statistics
        on_bits = fp.GetOnBits()
        
        print(f"  Fingerprint size: {fp.GetNumBits()} bits")
        print(f"  Active bits: {len(on_bits)}")
        print(f"  Bit density: {len(on_bits)/fp.GetNumBits():.3f}")
        print()
        
        return fp
    
    def compare_molecules(self, mol1, mol2, name1="Molecule 1", name2="Molecule 2"):
        """
        Compare pharmacophores of two molecules
        """
        print(f"Comparing {name1} vs {name2}...")
        
        # Generate fingerprints
        fp1 = self.generate_pharmacophore_fingerprint(mol1)
        fp2 = self.generate_pharmacophore_fingerprint(mol2)
        
        # Calculate similarity
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        
        print(f"  Tanimoto similarity: {similarity:.3f}")
        
        if similarity > 0.8:
            print("  → Very similar pharmacophores")
        elif similarity > 0.6:
            print("  → Moderately similar pharmacophores")
        elif similarity > 0.4:
            print("  → Some similarity")
        else:
            print("  → Different pharmacophores")
        
        print()
        return similarity
    
    # ========================================
    # PART 3: VISUALIZATION FUNCTIONS
    # ========================================
    
    def visualize_2d_pharmacophore(self, mol, features, save_path=None):
        """
        Create 2D structure with pharmacophore overlay and atom indices
        """
        print("Creating 2D pharmacophore visualization...")
        
        # Generate 2D coordinates
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
        
        # Create drawing with atom indices
        drawer = rdMolDraw2D.MolDraw2DCairo(1000, 800)  # Bigger size for legend
        
        # Set drawing options to show atom indices (corrected version)
        drawer.drawOptions().addAtomIndices = True
        drawer.drawOptions().addStereoAnnotation = True
        
        # Try different attribute names for font size (for compatibility)
        try:
            drawer.drawOptions().atomLabelFontSize = 16
        except AttributeError:
            try:
                drawer.drawOptions().annotationFontScale = 0.7
            except AttributeError:
                pass  # Skip if neither attribute exists
        
        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=highlight_colors
        )
        drawer.FinishDrawing()
        
        # Get the image and add legend
        img_str = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(img_str))
        
        # Add comprehensive legend
        img = self._add_enhanced_legend(img, features)
        
        # Save or return
        if save_path:
            img.save(save_path, dpi=(300, 300))
            print(f"  ✓ Saved: {save_path}")
        
        return img
    
    def _add_enhanced_legend(self, img, features):
        """
        Add enhanced legend with feature counts and atom indices
        """
        from PIL import ImageFont
        
        # Count features
        feature_counts = {}
        feature_atoms = {}
        for feat_type, atom_ids in features:
            if feat_type not in feature_counts:
                feature_counts[feat_type] = 0
                feature_atoms[feat_type] = []
            feature_counts[feat_type] += 1
            feature_atoms[feat_type].append(atom_ids)
        
        # Create new image with space for legend
        width, height = img.size
        legend_height = 150 + len(feature_counts) * 25
        new_height = height + legend_height
        new_img = Image.new('RGB', (width, new_height), 'white')
        new_img.paste(img, (0, 0))
        
        # Draw legend
        draw = ImageDraw.Draw(new_img)
        
        # Try to use a better font, fallback to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
            font_bold = font
        
        legend_y = height + 20
        legend_x = 30
        
        # Title
        draw.text((legend_x, legend_y), "PHARMACOPHORE FEATURES", fill='black', font=font_bold)
        legend_y += 30
        
        # Draw feature legend with counts and sample atom indices
        for feat_type, color in self.feature_colors.items():
            if feat_type in feature_counts:
                # Color box
                draw.rectangle([legend_x, legend_y, legend_x+20, legend_y+20], 
                              fill=color, outline='black', width=2)
                
                # Feature name and count
                text = f"{feat_type}: {feature_counts[feat_type]} features"
                draw.text((legend_x+30, legend_y+3), text, fill='black', font=font)
                
                # Add first few atom indices as example
                if feature_atoms[feat_type]:
                    first_atoms = feature_atoms[feat_type][0]
                    atoms_str = f" (atoms: {', '.join(map(str, first_atoms[:3]))}"
                    if len(first_atoms) > 3:
                        atoms_str += "..."
                    atoms_str += ")"
                    draw.text((legend_x+200, legend_y+3), atoms_str, fill='gray', font=font)
                
                legend_y += 25
        
        # Add note about atom indices
        legend_y += 10
        draw.text((legend_x, legend_y), 
                  "Note: Numbers on structure show atom indices", 
                  fill='gray', font=font)
        
        return new_img
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
    
    def visualize_distance_heatmap(self, df_distances, save_path=None):
        """
        Create distance heatmap
        """
        print("Creating distance heatmap...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            df_distances,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Distance (Å)'},
            square=True,
            linewidths=0.5,
            ax=ax
        )
        
        ax.set_title('Pharmacophore Feature Distance Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {save_path}")
            plt.close()
        
        return fig
    
    def visualize_feature_distribution(self, features, mol_name="", save_path=None):
        """
        Create feature distribution charts
        """
        print("Creating feature distribution plots...")
        
        # Count features
        feature_counts = {}
        for feat_type, _ in features:
            if feat_type not in feature_counts:
                feature_counts[feat_type] = 0
            feature_counts[feat_type] += 1
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart
        features_list = list(feature_counts.keys())
        counts = list(feature_counts.values())
        colors = [self.feature_colors.get(f, '#888888') for f in features_list]
        
        bars = ax1.bar(features_list, counts, color=colors, edgecolor='black', linewidth=2)
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_xlabel('Pharmacophore Feature Type', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax1.set_title('Feature Distribution', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Pie chart
        if feature_counts:
            wedges, texts, autotexts = ax2.pie(
                counts,
                labels=features_list,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                wedgeprops=dict(edgecolor='black', linewidth=2)
            )
            
            for autotext in autotexts:
                autotext.set_fontweight('bold')
            
            ax2.set_title('Feature Proportions', fontsize=14, fontweight='bold')
        
        plt.suptitle(f'Pharmacophore Analysis - {mol_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {save_path}")
            plt.close()
        
        return fig
    
    # ========================================
    # PART 4: COMPLETE ANALYSIS PIPELINE
    # ========================================
    
    def complete_analysis(self, input_data, input_type='smiles', mol_name="Molecule", output_dir="output"):
        """
        Run complete pharmacophore analysis pipeline
        """
        print("\n" + "="*60)
        print(f"COMPLETE PHARMACOPHORE ANALYSIS: {mol_name}")
        print("="*60 + "\n")
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}\n")
        
        # Step 1: Load molecule
        mol = self.load_molecule(input_data, input_type, mol_name)
        
        # Step 2: Extract features
        features = self.extract_pharmacophore_features(mol)
        
        # Step 3: Calculate distances
        df_distances, feature_coords, feature_labels = self.calculate_pharmacophore_distances(mol, features)
        
        # Step 4: Generate fingerprint
        fingerprint = self.generate_pharmacophore_fingerprint(mol)
        
        # Step 5: Create visualizations
        print("="*40)
        print("GENERATING VISUALIZATIONS")
        print("="*40 + "\n")
        
        # 2D overlay
        self.visualize_2d_pharmacophore(
            mol, features,
            save_path=os.path.join(output_dir, f"{mol_name}_2d_pharmacophore.png")
        )
        
        # 3D interactive
        self.visualize_3d_pharmacophore(
            mol, features,
            save_path=os.path.join(output_dir, f"{mol_name}_3d_pharmacophore.html")
        )
        
        # Distance heatmap
        self.visualize_distance_heatmap(
            df_distances,
            save_path=os.path.join(output_dir, f"{mol_name}_distances.png")
        )
        
        # Feature distribution
        self.visualize_feature_distribution(
            features, mol_name,
            save_path=os.path.join(output_dir, f"{mol_name}_distribution.png")
        )
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"\nAll results saved in: {output_dir}/")
        print("\nGenerated files:")
        print(f"  1. {mol_name}_2d_pharmacophore.png - 2D structure with features")
        print(f"  2. {mol_name}_3d_pharmacophore.html - Interactive 3D model")
        print(f"  3. {mol_name}_distances.png - Feature distance heatmap")
        print(f"  4. {mol_name}_distribution.png - Feature statistics")
        
        

        # Step 4.5: Save data to CSV files
        print("Saving data to CSV files...")

        # 1. Save pharmacophore features to CSV
        features_data = []
        for i, (feat_type, atom_ids) in enumerate(features):
            features_data.append({
                'Feature_ID': f'{feat_type}_{i+1}',
                'Feature_Type': feat_type,
                'Atom_Indices': str(atom_ids),
                'Num_Atoms': len(atom_ids)
            })

        df_features = pd.DataFrame(features_data)
        df_features.to_csv(
            os.path.join(output_dir, f"{mol_name}_features.csv"),
            index=False
        )
        print(f"  ✓ Saved: {mol_name}_features.csv")

        # 2. Save distance matrix to CSV
        df_distances.to_csv(
            os.path.join(output_dir, f"{mol_name}_distances.csv")
        )
        print(f"  ✓ Saved: {mol_name}_distances.csv")

        # 3. Save fingerprint bits to CSV
        fp_bits = list(fingerprint.GetOnBits())
        fp_data = {
            'Fingerprint_Size': [fingerprint.GetNumBits()],
            'Active_Bits': [len(fp_bits)],
            'Bit_Density': [len(fp_bits)/fingerprint.GetNumBits()],
            'Active_Bit_Positions': [str(fp_bits[:100])]  # First 100 bits for readability
        }
        df_fingerprint = pd.DataFrame(fp_data)
        df_fingerprint.to_csv(
            os.path.join(output_dir, f"{mol_name}_fingerprint_summary.csv"),
            index=False
        )
        print(f"  ✓ Saved: {mol_name}_fingerprint_summary.csv")

        # 4. Save molecular properties to CSV
        mol_props = {
            'Property': ['Molecular_Formula', 'Molecular_Weight', 'Num_Atoms', 
                         'Num_Bonds', 'Num_Heavy_Atoms', 'Num_Rotatable_Bonds',
                         'Num_HBD', 'Num_HBA', 'TPSA', 'LogP'],
            'Value': [
                rdMolDescriptors.CalcMolFormula(mol),
                Descriptors.MolWt(mol),
                mol.GetNumAtoms(),
                mol.GetNumBonds(),
                Descriptors.HeavyAtomCount(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol),
                Descriptors.MolLogP(mol)
            ]
        }
        df_props = pd.DataFrame(mol_props)
        df_props.to_csv(
            os.path.join(output_dir, f"{mol_name}_properties.csv"),
            index=False
        )
        print(f"  ✓ Saved: {mol_name}_properties.csv")

        return mol, features, fingerprint
# ========================================
# MAIN TUTORIAL EXECUTION
# ========================================

def main():
    """
    Main tutorial script for YouTube video
    """
    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + " "*15 + "PHARMACOPHORE MODELING TUTORIAL" + " "*12 + "║")
    print("║" + " "*18 + "Complete Python Pipeline" + " "*16 + "║")
    print("╚" + "═"*58 + "╝\n")
    
    # Initialize analyzer
    analyzer = CompletePharmacophoreAnalyzer()
    
    # ========================================
    # EXAMPLE 1: Single Drug Analysis
    # ========================================
    print("\n" + "━"*60)
    print("EXAMPLE 1: ANALYZING IMATINIB (GLEEVEC)")
    print("━"*60)
    
    # Imatinib SMILES
    imatinib_smiles = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"
    
    # Run complete analysis
    mol1, features1, fp1 = analyzer.complete_analysis(
        imatinib_smiles,
        input_type='smiles',
        mol_name='Imatinib',
        output_dir='output_imatinib'
    )
    
    # ========================================
    # EXAMPLE 2: Drug Comparison
    # ========================================
    print("\n" + "━"*60)
    print("EXAMPLE 2: COMPARING WITH DASATINIB")
    print("━"*60)
    
    # Dasatinib SMILES
    dasatinib_smiles = "CC1=NC(=CC(=N1)N2CCN(CC2)CCO)NC3=C(C=CC=C3Cl)C(=O)NS(=O)(=O)C4=CC=CC=C4C"
    
    # Analyze second molecule
    mol2, features2, fp2 = analyzer.complete_analysis(
        dasatinib_smiles,
        input_type='smiles',
        mol_name='Dasatinib',
        output_dir='output_dasatinib'
    )
    
    # Compare the two drugs
    print("\n" + "━"*60)
    print("PHARMACOPHORE COMPARISON")
    print("━"*60 + "\n")
    
    similarity = analyzer.compare_molecules(mol1, mol2, "Imatinib", "Dasatinib")

    # Save comparison results to CSV
    comparison_data = {
        'Molecule_1': ['Imatinib'],
        'Molecule_2': ['Dasatinib'],
        'Tanimoto_Similarity': [similarity],
        'Interpretation': ['Very similar' if similarity > 0.8 else 
                          'Moderately similar' if similarity > 0.6 else
                          'Some similarity' if similarity > 0.4 else 
                          'Different']
    }
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison.to_csv('pharmacophore_comparison.csv', index=False)
    print(f"✓ Saved: pharmacophore_comparison.csv")
    
    # ========================================
    # EXAMPLE 3: Loading from SDF (Template)
    # ========================================
    print("\n" + "━"*60)
    print("EXAMPLE 3: LOADING FROM SDF FILE")
    print("━"*60)
    print("\nTo load from an SDF file, use:")
    print("─"*40)
    print("mol, features, fp = analyzer.complete_analysis(")
    print("    'path/to/your/molecule.sdf',")
    print("    input_type='sdf',")
    print("    mol_name='YourMolecule',")
    print("    output_dir='output_your_molecule'")
    print(")")
    
    # ========================================
    # Final Summary
    # ========================================
    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + " "*20 + "PHARMACOPHORE ANALYSIS COMPLETED!" + " "*20 + "║")
    print("╚" + "═"*58 + "╝\n")
    
    print("KEY TAKEAWAYS")
    print("─"*40)
    print("✓ Pharmacophores represent 3D arrangements of features")
    print("✓ Features include H-bond donors/acceptors, aromatic, hydrophobic")
    print("✓ Distance matrices define the 3D pharmacophore pattern")
    print("✓ Fingerprints enable rapid database searching")
    print("✓ Similar drugs often share pharmacophore patterns")
    print()
    print("APPLICATIONS:")
    print("─"*40)
    print("• Virtual screening of compound libraries")
    print("• Lead optimization in drug design")
    print("• Understanding structure-activity relationships")
    print("• Scaffold hopping to find new drug classes")
    print()
    print("Thank you for watching! Don't forget to like and subscribe!")

if __name__ == "__main__":
    main()
