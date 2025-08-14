#!/usr/bin/env python3
import os
import re
import sys

def extract_poses(dlg_file):
    """
    Extract all poses from an AutoDock DLG file and save them as separate PDBQT files.
    The output files are saved in a folder named after the ligand.
    
    Args:
        dlg_file (str): Path to the input DLG file
    """
    # Get the base name of the DLG file (without extension)
    base_name = os.path.splitext(os.path.basename(dlg_file))[0]
    
    # Create a folder named after the ligand
    output_folder = os.path.join(os.path.dirname(dlg_file), base_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    with open(dlg_file, 'r') as f:
        content = f.read()
    
    # Split content into models
    models = re.split(r'DOCKED: MODEL\s+\d+', content)[1:]  # Skip content before first model
    
    for i, model in enumerate(models, 1):
        # Extract the pose coordinates and metadata
        pose_lines = []
        energy = None
        
        # Extract binding energy
        energy_match = re.search(r'Estimated Free Energy of Binding\s+=\s+([-\d.]+)', model)
        if energy_match:
            energy = float(energy_match.group(1))
        
        # Collect all DOCKED: lines that are part of the PDBQT structure
        for line in model.split('\n'):
            if line.startswith('DOCKED: REMARK') or \
               line.startswith('DOCKED: ROOT') or \
               line.startswith('DOCKED: ATOM') or \
               line.startswith('DOCKED: ENDROOT') or \
               line.startswith('DOCKED: BRANCH') or \
               line.startswith('DOCKED: ENDBRANCH') or \
               line.startswith('DOCKED: TORSDOF'):
                # Remove the 'DOCKED: ' prefix
                cleaned_line = line.replace('DOCKED: ', '')
                pose_lines.append(cleaned_line)
        
        # Write the pose to a PDBQT file
        output_file = os.path.join(output_folder, f'pose_{i}.pdbqt')
        with open(output_file, 'w') as f:
            # Write energy as a remark
            if energy is not None:
                f.write(f'REMARK VINA RESULT:    {energy:>8.2f}      0.000      0.000\n')
            # Write the pose coordinates
            f.write('\n'.join(pose_lines))
            f.write('\n')  # Add final newline
        
        print(f'Wrote pose {i} to {output_file} (Energy: {energy:.2f} kcal/mol)')

def main():
    if len(sys.argv) != 2:
        print("Usage: dlg2pdbqt input.dlg")
        sys.exit(1)
    
    dlg_file = sys.argv[1]
    if not os.path.exists(dlg_file):
        print(f"Error: File '{dlg_file}' not found")
        sys.exit(1)
        
    try:
        extract_poses(dlg_file)
        print(f"\nDone! All poses have been extracted to the '{os.path.splitext(os.path.basename(dlg_file))[0]}' folder.")
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()