#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced HOMO-LUMO Pipeline with Nature-style Publication Quality Figures
and Automated PyMOL Script Generation

Author: Enhanced for publication-quality output
"""

import os, pathlib, time
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import polars as pl
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
from pyscf import gto, scf, dft, tools

# -----------------------------
# USER INPUTS
# -----------------------------
INPUT_CSV = "molecules.csv"   # CSV with id,smiles,charge,multiplicity
OUTDIR = "results"
METHOD = "DFT-PBE0"
BASIS = "6-31g*"
ISOSURFACE_GRID = [80, 80, 80]  # Higher resolution for publication
ISOSURFACE_LEVEL = 0.03  # For PyMOL scripts

# Publication settings
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'  # or 'pdf', 'svg'
PYMOL_RENDER_DPI = 300

# -----------------------------
# NATURE-STYLE MATPLOTLIB SETUP
# -----------------------------
# Nature journal style parameters
def setup_nature_style():
    """Configure matplotlib for Nature-style publication quality"""
    plt.style.use('default')  # Start fresh
    
    # Font settings (Nature prefers Arial/Helvetica)
    mpl.rcParams.update({
        'font.family': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,           # Nature standard
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 12,
        
        # Line and marker settings
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'patch.linewidth': 0.5,
        
        # Axes settings
        'axes.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': 'black',
        'axes.labelweight': 'normal',
        'axes.axisbelow': True,
        
        # Tick settings
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.3,
        'ytick.minor.width': 0.3,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        
        # Grid settings
        'axes.grid': True,
        'grid.color': '#E5E5E5',
        'grid.linewidth': 0.3,
        'grid.alpha': 0.8,
        
        # Figure settings
        'figure.dpi': FIGURE_DPI,
        'savefig.dpi': FIGURE_DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'savefig.transparent': False,
        'savefig.facecolor': 'white',
        
        # Color settings
        'axes.prop_cycle': mpl.cycler('color', [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ])
    })

# Nature color palette
NATURE_COLORS = {
    'blue': '#1f77b4',
    'orange': '#ff7f0e', 
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
    'olive': '#bcbd22',
    'cyan': '#17becf',
    'homo': '#d62728',      # Red for HOMO
    'lumo': '#1f77b4',      # Blue for LUMO
    'gap': '#9467bd',       # Purple for gap
    'occupied': '#ff7f0e',  # Orange for occupied
    'virtual': '#2ca02c'    # Green for virtual
}

# -----------------------------
# ENHANCED DATACLASS
# -----------------------------
@dataclass
class CalcResult:
    id: str
    smiles: str
    method: str
    basis: str
    n_elec: int
    homo_idx: int
    lumo_idx: int
    homo_h: float
    lumo_h: float
    gap_h: float
    homo_ev: float
    lumo_ev: float
    gap_ev: float
    mo_energies_h: list
    geometry_xyz: str
    e_tot: float
    dipole_moment: float = 0.0
    molecular_weight: float = 0.0
    logp: float = 0.0

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def ensure_outdir(path: str):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def rdkit_build(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    AllChem.EmbedMolecule(mol, params)
    AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
    return mol

def rdkit_to_xyz(mol: Chem.Mol) -> str:
    conf = mol.GetConformer()
    lines = []
    for i in range(mol.GetNumAtoms()):
        a = mol.GetAtomWithIdx(i)
        x, y, z = conf.GetAtomPosition(i)
        lines.append(f"{a.GetSymbol()} {x:.6f} {y:.6f} {z:.6f}")
    return "\n".join(lines)

def calculate_molecular_properties(mol: Chem.Mol):
    """Calculate additional molecular properties"""
    mol_no_h = Chem.RemoveHs(mol)
    return {
        'molecular_weight': Descriptors.MolWt(mol_no_h),
        'logp': Descriptors.MolLogP(mol_no_h),
        'tpsa': Descriptors.TPSA(mol_no_h),
        'rotatable_bonds': Descriptors.NumRotatableBonds(mol_no_h)
    }

def save_structures(mol: Chem.Mol, outdir: str, mol_id: str):
    subdir = os.path.join(outdir, mol_id)
    ensure_outdir(subdir)
    
    # SDF
    sdf_path = os.path.join(subdir, "molecule.sdf")
    w = Chem.SDWriter(sdf_path)
    w.write(mol)
    w.close()
    
    # XYZ
    xyz_path = os.path.join(subdir, "molecule.xyz")
    with open(xyz_path,"w") as f:
        conf = mol.GetConformer()
        f.write(f"{mol.GetNumAtoms()}\n{mol_id} - RDKit optimized\n")
        for i in range(mol.GetNumAtoms()):
            a = mol.GetAtomWithIdx(i)
            x, y, z = conf.GetAtomPosition(i)
            f.write(f"{a.GetSymbol():2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")
    
    # PDB for PyMOL compatibility
    pdb_path = os.path.join(subdir, "molecule.pdb")
    Chem.MolToPDBFile(mol, pdb_path)
    
    # Enhanced 2D structure with better quality
    png_path = os.path.join(subdir, "molecule_2d.png")
    mol_2d = Chem.RemoveHs(mol)
    img = Draw.MolToImage(mol_2d, size=(600, 450), dpi=FIGURE_DPI)
    img.save(png_path, dpi=(FIGURE_DPI, FIGURE_DPI))
    
    return xyz_path, subdir, pdb_path

def pyscf_run(xyz, charge, multiplicity, method, basis):
    spin = multiplicity - 1
    mol_p = gto.Mole()
    mol_p.build(atom=xyz, charge=charge, spin=spin, basis=basis, unit="Angstrom")
    
    if method.upper() == "RHF" and spin == 0:
        mf = scf.RHF(mol_p)
    elif method.upper().startswith("DFT"):
        xc = method.split("-",1)[1] if "-" in method else "PBE0"
        mf = dft.RKS(mol_p)
        mf.xc = xc
    else:
        mf = scf.ROHF(mol_p)
    
    mf.conv_tol = 1e-8
    mf.verbose = 0
    energy = mf.run()
    e_tot = energy.e_tot
    
    # Calculate dipole moment
    dipole = mf.dip_moment()
    dipole_magnitude = np.linalg.norm(dipole)
    
    mo_e = mf.mo_energy.tolist()
    n_elec = mol_p.nelectron
    homo = n_elec//2 - 1
    lumo = homo + 1
    
    hartree_to_ev = 27.211386245988
    
    return CalcResult(
        id="", smiles="", method=method, basis=basis, n_elec=n_elec,
        homo_idx=homo, lumo_idx=lumo,
        homo_h=mo_e[homo], lumo_h=mo_e[lumo], gap_h=mo_e[lumo]-mo_e[homo],
        homo_ev=mo_e[homo]*hartree_to_ev, lumo_ev=mo_e[lumo]*hartree_to_ev,
        gap_ev=(mo_e[lumo]-mo_e[homo])*hartree_to_ev,
        mo_energies_h=mo_e, geometry_xyz=xyz, e_tot=e_tot,
        dipole_moment=dipole_magnitude
    ), mol_p, mf

def write_cubes(mol, mf, homo_idx, lumo_idx, outdir):
    """Write cube files with higher resolution"""
    homo_cube = os.path.join(outdir, "homo.cube")
    lumo_cube = os.path.join(outdir, "lumo.cube")
    
    tools.cubegen.orbital(mol, homo_cube, mf.mo_coeff[:,homo_idx],
                          nx=ISOSURFACE_GRID[0], ny=ISOSURFACE_GRID[1], nz=ISOSURFACE_GRID[2])
    tools.cubegen.orbital(mol, lumo_cube, mf.mo_coeff[:,lumo_idx],
                          nx=ISOSURFACE_GRID[0], ny=ISOSURFACE_GRID[1], nz=ISOSURFACE_GRID[2])
    
    return homo_cube, lumo_cube

# -----------------------------
# PYMOL SCRIPT GENERATION
# -----------------------------
def generate_pymol_script(mol_id: str, subdir: str, pdb_path: str, 
                         homo_cube: str, lumo_cube: str, result: CalcResult):
    """Generate publication-quality PyMOL script for orbital visualization"""
    
    script_content = f'''# =============================================================================
# PyMOL Script for {mol_id} - HOMO/LUMO Visualization
# Generated by Enhanced HOMO-LUMO Pipeline
# =============================================================================

# Setup for publication quality
set ray_trace_mode, 1
set antialias, 2
set ray_opaque_background, off
set depth_cue, 0
set spec_reflect, 0
set shininess, 0
set ambient, 0.4
set direct, 0.6
set reflect, 0.0
set specular, 0.0

# Background and lighting
bg_color white
set light_count, 2
set_color nature_red, [214, 39, 40]
set_color nature_blue, [31, 119, 180] 
set_color nature_gray, [127, 127, 127]

# Load molecular structure
load {os.path.basename(pdb_path)}, {mol_id}
hide everything, {mol_id}
show sticks, {mol_id}
set stick_radius, 0.15, {mol_id}
set stick_transparency, 0.0, {mol_id}

# Beautiful atom colors
util.cbaw {mol_id}
color gray70, (elem C and {mol_id})
color white, (elem H and {mol_id})
color red, (elem O and {mol_id})
color blue, (elem N and {mol_id})
color yellow, (elem S and {mol_id})

# Load orbital cube files
load {os.path.basename(homo_cube)}, homo_map
load {os.path.basename(lumo_cube)}, lumo_map

# HOMO orbital surfaces (positive and negative phases)
isosurface homo_pos, homo_map, {ISOSURFACE_LEVEL}
isosurface homo_neg, homo_map, -{ISOSURFACE_LEVEL}
color nature_red, homo_pos
color salmon, homo_neg
set transparency, 0.3, homo_pos
set transparency, 0.3, homo_neg

# LUMO orbital surfaces (positive and negative phases)  
isosurface lumo_pos, lumo_map, {ISOSURFACE_LEVEL}
isosurface lumo_neg, lumo_map, -{ISOSURFACE_LEVEL}
color nature_blue, lumo_pos
color lightblue, lumo_neg
set transparency, 0.3, lumo_pos
set transparency, 0.3, lumo_neg

# Initial view setup
orient {mol_id}
zoom {mol_id}, 3.0
set orthoscopic, 1

# =============================================================================
# Scene 1: HOMO orbital
# =============================================================================
disable lumo_pos
disable lumo_neg
enable homo_pos
enable homo_neg
enable {mol_id}

# Add energy label
pseudoatom homo_label, pos=[0,0,0]
label homo_label, "HOMO: {result.homo_ev:.2f} eV"
set label_size, 20
set label_font_id, 7
set label_color, nature_red
set label_position, [0, 5, 0]
hide everything, homo_label

scene homo_view, store

# High-quality render
set ray_trace_frames, 1
ray {PYMOL_RENDER_DPI*2}, {int(PYMOL_RENDER_DPI*1.5)}
png {mol_id}_HOMO_publication.png, dpi={PYMOL_RENDER_DPI}

# =============================================================================  
# Scene 2: LUMO orbital
# =============================================================================
disable homo_pos
disable homo_neg
disable homo_label
enable lumo_pos
enable lumo_neg

# Add energy label
pseudoatom lumo_label, pos=[0,0,0]
label lumo_label, "LUMO: {result.lumo_ev:.2f} eV"
set label_color, nature_blue
hide everything, lumo_label

scene lumo_view, store

# High-quality render
ray {PYMOL_RENDER_DPI*2}, {int(PYMOL_RENDER_DPI*1.5)}
png {mol_id}_LUMO_publication.png, dpi={PYMOL_RENDER_DPI}

# =============================================================================
# Scene 3: Combined view (both orbitals, side by side)
# =============================================================================
disable lumo_label
enable homo_pos
enable homo_neg
enable lumo_pos  
enable lumo_neg

# Adjust transparency for combined view
set transparency, 0.4, homo_pos
set transparency, 0.4, homo_neg
set transparency, 0.4, lumo_pos
set transparency, 0.4, lumo_neg

scene combined_view, store

# Side-by-side grid view
set grid_mode, 1

# HOMO panel
set grid_slot, 1
scene homo_view, recall
enable homo_label
ray {PYMOL_RENDER_DPI}, {PYMOL_RENDER_DPI}

# LUMO panel  
set grid_slot, 2
scene lumo_view, recall
enable lumo_label
ray {PYMOL_RENDER_DPI}, {PYMOL_RENDER_DPI}

# Save combined figure
png {mol_id}_HOMO_LUMO_combined.png, dpi={PYMOL_RENDER_DPI}

# Reset to single view
set grid_mode, 0
scene combined_view, recall

# =============================================================================
# Animation: 360-degree rotation
# =============================================================================
# Create smooth rotation movie (6 seconds at 30 fps = 180 frames)
mset 1 x180

# Rotation around Y-axis
util.mroll 1, 1, 180

# Save movie frames
mpng {mol_id}_rotation_

# =============================================================================
# Final summary information
# =============================================================================
print "============================================="
print "Molecular Orbital Analysis Summary"
print "============================================="
print "Molecule ID: {mol_id}"
print "HOMO Energy: {result.homo_ev:.3f} eV"
print "LUMO Energy: {result.lumo_ev:.3f} eV" 
print "HOMO-LUMO Gap: {result.gap_ev:.3f} eV"
print "Total Energy: {result.e_tot:.6f} Hartree"
print "Dipole Moment: {result.dipole_moment:.3f} Debye"
print "============================================="

# Useful commands for manual exploration:
# scene homo_view, recall    # Show HOMO
# scene lumo_view, recall    # Show LUMO  
# scene combined_view, recall # Show both
# set transparency, X, object_name  # Adjust transparency (0-1)
# set isosurface_level, X, object_name  # Adjust isosurface level
'''

    script_path = os.path.join(subdir, f"{mol_id}_pymol_orbitals.pml")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"[PyMOL] Script saved: {script_path}")
    return script_path

# -----------------------------
# ENHANCED PLOTTING FUNCTIONS
# -----------------------------
def plot_mo_ladder_nature(res, ax):
    """Nature-style MO ladder plot"""
    H2eV = 27.211386245988
    energies = np.array(res.mo_energies_h) * H2eV
    
    # Plot all MO levels
    for j, e in enumerate(energies):
        if j == res.homo_idx:
            ax.hlines(e, 0.85, 1.15, colors=NATURE_COLORS['homo'], 
                     linewidth=2.5, label='HOMO', zorder=3)
        elif j == res.lumo_idx:
            ax.hlines(e, 0.85, 1.15, colors=NATURE_COLORS['lumo'], 
                     linewidth=2.5, label='LUMO', zorder=3)
        elif j < res.homo_idx:  # Occupied
            ax.hlines(e, 0.9, 1.1, colors=NATURE_COLORS['occupied'], 
                     linewidth=1.5, alpha=0.7, zorder=2)
        else:  # Virtual
            ax.hlines(e, 0.9, 1.1, colors=NATURE_COLORS['virtual'], 
                     linewidth=1.5, alpha=0.7, zorder=2)
    
    # Highlight HOMO-LUMO gap
    gap_height = energies[res.lumo_idx] - energies[res.homo_idx]
    rect = Rectangle((0.87, energies[res.homo_idx]), 0.26, gap_height,
                    facecolor=NATURE_COLORS['gap'], alpha=0.15, zorder=1)
    ax.add_patch(rect)
    
    # Formatting
    ax.set_xlim(0.8, 1.2)
    ax.set_xticks([])
    ax.set_ylabel('Energy (eV)', fontweight='bold')
    ax.set_title(f'{res.id}', fontweight='bold', pad=10)
    
    # Add gap annotation
    gap_center = (energies[res.homo_idx] + energies[res.lumo_idx]) / 2
    ax.annotate(f'Gap: {res.gap_ev:.2f} eV', 
                xy=(1.25, gap_center), xytext=(1.4, gap_center),
                fontsize=7, ha='left', va='center',
                arrowprops=dict(arrowstyle='->', lw=0.5, color='black'))
    
    # Legend
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), frameon=False, fontsize=7)

def plot_dos_nature(res, ax):
    """Nature-style density of states plot"""
    H2eV = 27.211386245988
    energies = np.array(res.mo_energies_h) * H2eV
    
    # Create histogram
    counts, bins, patches = ax.hist(energies, bins=30, color=NATURE_COLORS['gray'], 
                                   alpha=0.6, edgecolor='white', linewidth=0.5)
    
    # Color occupied vs virtual states
    for i, (count, bin_left, bin_right) in enumerate(zip(counts, bins[:-1], bins[1:])):
        bin_center = (bin_left + bin_right) / 2
        if bin_center <= energies[res.homo_idx]:
            patches[i].set_facecolor(NATURE_COLORS['occupied'])
            patches[i].set_alpha(0.7)
        else:
            patches[i].set_facecolor(NATURE_COLORS['green'])
            patches[i].set_alpha(0.7)
    
    # Mark HOMO and LUMO
    ax.axvline(energies[res.homo_idx], color=NATURE_COLORS['homo'], 
              linestyle='--', linewidth=2, label='HOMO', alpha=0.8)
    ax.axvline(energies[res.lumo_idx], color=NATURE_COLORS['lumo'], 
              linestyle='--', linewidth=2, label='LUMO', alpha=0.8)
    
    # Formatting
    ax.set_xlabel('Energy (eV)', fontweight='bold')
    ax.set_ylabel('Density of States', fontweight='bold')
    ax.set_title(f'{res.id} - Electronic DOS', fontweight='bold', pad=10)
    ax.legend(frameon=False, fontsize=7)
    
    # Minor ticks
    ax.minorticks_on()

def create_summary_figure(results, output_path):
    """Create comprehensive Nature-style summary figure"""
    n_mols = len(results)
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 4, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.3)
    
    # Panel A: HOMO-LUMO gap comparison
    ax_gap = fig.add_subplot(gs[0, :2])
    mol_names = [r.id for r in results]
    gaps = [r.gap_ev for r in results]
    homos = [r.homo_ev for r in results]
    lumos = [r.lumo_ev for r in results]
    
    x = np.arange(len(mol_names))
    width = 0.35
    
    bars1 = ax_gap.bar(x - width/2, [-h for h in homos], width, 
                       label='HOMO', color=NATURE_COLORS['homo'], alpha=0.8)
    bars2 = ax_gap.bar(x + width/2, lumos, width, 
                       label='LUMO', color=NATURE_COLORS['lumo'], alpha=0.8)
    
    ax_gap.set_xlabel('Molecule', fontweight='bold')
    ax_gap.set_ylabel('Energy (eV)', fontweight='bold') 
    ax_gap.set_title('HOMO-LUMO Energy Levels', fontweight='bold', fontsize=12)
    ax_gap.set_xticks(x)
    ax_gap.set_xticklabels(mol_names, rotation=45, ha='right')
    ax_gap.legend(frameon=False)
    ax_gap.grid(True, alpha=0.3)
    ax_gap.axhline(0, color='black', linewidth=0.5)
    
    # Panel B: Gap vs Total Energy scatter
    ax_scatter = fig.add_subplot(gs[0, 2:])
    total_energies = [r.e_tot for r in results]
    sc = ax_scatter.scatter(gaps, total_energies, c=gaps, cmap='viridis', 
                           s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    for i, mol_id in enumerate(mol_names):
        ax_scatter.annotate(mol_id, (gaps[i], total_energies[i]), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=7, alpha=0.8)
    
    ax_scatter.set_xlabel('HOMO-LUMO Gap (eV)', fontweight='bold')
    ax_scatter.set_ylabel('Total Energy (Hartree)', fontweight='bold')
    ax_scatter.set_title('Energy Correlation', fontweight='bold', fontsize=12)
    cbar = plt.colorbar(sc, ax=ax_scatter, shrink=0.8)
    cbar.set_label('Gap (eV)', fontweight='bold')
    
    # Panel C: Individual MO ladders
    for i, res in enumerate(results[:2]):  # Show first 2 molecules
        ax_ladder = fig.add_subplot(gs[1, i*2:(i+1)*2])
        plot_mo_ladder_nature(res, ax_ladder)
    
    # Panel D: Summary statistics table
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('tight')
    ax_table.axis('off')
    
    table_data = []
    for res in results:
        table_data.append([
            res.id,
            f"{res.gap_ev:.2f}",
            f"{res.homo_ev:.2f}", 
            f"{res.lumo_ev:.2f}",
            f"{res.dipole_moment:.2f}"
        ])
    
    table = ax_table.table(cellText=table_data,
                          colLabels=['Molecule', 'Gap (eV)', 'HOMO (eV)', 'LUMO (eV)', 'Dipole (D)'],
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(mol_names) + 1):
        for j in range(5):
            if i == 0:  # Header
                table[(i, j)].set_facecolor('#E5E5E5')
                table[(i, j)].set_text_props(weight='bold')
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.suptitle('HOMO-LUMO Analysis Summary', fontsize=14, fontweight='bold', y=0.98)
    
    # Save high-quality figure
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"[PLOT] Nature-style summary figure saved: {output_path}")

# -----------------------------
# MAIN FUNCTION
# -----------------------------
def main():
    # Setup publication style
    setup_nature_style()
    
    ensure_outdir(OUTDIR)
    t0 = time.time()
    
    print("="*60)
    print("Enhanced HOMO-LUMO Pipeline - Nature Publication Quality")
    print("="*60)
    
    df_input = pl.read_csv(INPUT_CSV)
    results = []
    pymol_scripts = []
    
    for row in df_input.iter_rows(named=True):
        if "id" not in row:
            raise ValueError("Input CSV is missing the 'id' column.")
        
        mol_id = row["id"]
        smiles = row["smiles"]
        charge = int(row.get("charge") or 0)
        mult = int(row.get("multiplicity") or 1)
        
        print(f"\n[INFO] Processing {mol_id}...")
        print(f"       SMILES: {smiles}")
        
        # Build molecule
        mol = rdkit_build(smiles)
        mol_props = calculate_molecular_properties(mol)
        
        # Save structures
        xyz, subdir, pdb_path = save_structures(mol, OUTDIR, mol_id)
        
        # Quantum calculation
        res, mol_p, mf = pyscf_run(xyz, charge, mult, METHOD, BASIS)
        res.id = mol_id
        res.smiles = smiles
        res.molecular_weight = mol_props['molecular_weight']
        res.logp = mol_props['logp']
        
        # Generate cube files
        homo_cube, lumo_cube = write_cubes(mol_p, mf, res.homo_idx, res.lumo_idx, subdir)
        
        # Generate PyMOL script
        pymol_script = generate_pymol_script(mol_id, subdir, pdb_path, 
                                           homo_cube, lumo_cube, res)
        pymol_scripts.append(pymol_script)
        
        results.append(res)
        
        # Individual publication-quality plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        plot_mo_ladder_nature(res, ax1)
        plot_dos_nature(res, ax2)
        
        plt.tight_layout()
        individual_plot = os.path.join(subdir, f"{mol_id}_analysis.{FIGURE_FORMAT}")
        plt.savefig(individual_plot, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        print(f"       ✓ HOMO: {res.homo_ev:.2f} eV")
        print(f"       ✓ LUMO: {res.lumo_ev:.2f} eV") 
        print(f"       ✓ Gap:  {res.gap_ev:.2f} eV")
    
    # Enhanced summary CSV with additional properties
    summary_data = []
    for r in results:
        summary_data.append({
            "id": r.id,
            "smiles": r.smiles,
            "method": r.method,
            "basis": r.basis,
            "molecular_weight": r.molecular_weight,
            "logp": r.logp,
            "n_electrons": r.n_elec,
            "HOMO_eV": r.homo_ev,
            "LUMO_eV": r.lumo_ev,
            "gap_eV": r.gap_ev,
            "HOMO_idx": r.homo_idx,
            "LUMO_idx": r.lumo_idx,
            "E_total_Ha": r.e_tot,
            "dipole_moment_D": r.dipole_moment
        })
    
    summary = pl.DataFrame(summary_data)
    summary_path = os.path.join(OUTDIR, "enhanced_summary.csv")
    summary.write_csv(summary_path)
    
    # Create Nature-style summary figure
    summary_fig_path = os.path.join(OUTDIR, f"publication_summary.{FIGURE_FORMAT}")
    create_summary_figure(results, summary_fig_path)
    
    # Generate master PyMOL script
    master_script_path = os.path.join(OUTDIR, "master_pymol_script.pml")
    with open(master_script_path, 'w') as f:
        f.write("# Master PyMOL Script - Load All Molecules\n")
        f.write("# Run individual scripts with: @script_name.pml\n\n")
        for i, script in enumerate(pymol_scripts):
            rel_path = os.path.relpath(script, OUTDIR)
            f.write(f"# Molecule {i+1}\n")
            f.write(f"# @{rel_path}\n\n")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Processed {len(results)} molecules")
    print(f"Total runtime: {time.time()-t0:.2f} seconds")
    print(f"\nOutputs saved in: {OUTDIR}/")
    print(f"• Enhanced summary: {summary_path}")
    print(f"• Publication figure: {summary_fig_path}")
    print(f"• PyMOL scripts: {len(pymol_scripts)} individual + 1 master")
    print("="*60)

if __name__ == "__main__":
    main()
