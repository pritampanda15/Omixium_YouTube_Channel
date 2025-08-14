#!/bin/bash

# Directories
# Directories
RECEPTOR_ROOT_DIR="/mnt/0b25801b-bcaa-4119-bf74-7f5d160330b8/pritam/Docking_Protocols/Autodock_GPU/docking/rigid"
LIGAND_DIR="/mnt/0b25801b-bcaa-4119-bf74-7f5d160330b8/pritam/Docking_Protocols/Autodock_GPU/docking/ligands"
OUTPUT_ROOT="rigid_docked_ligands"

# AutoDock GPU parameters
NEV="25000000"
HEURMAX="12000000"
NRUN="100"
P="300"
G="50000"
LSMET='ad'
LSIT="500"
AUTOSTOP="1"
ASFREQ="5"
STOPSTD="0.1"
CLUSTERING="1"
#OUTPUT_CLUSTER_POSES="10"
GBEST="1"
XML="1"
DLG="1"
GFPOP="1"
CONTACT="1"

# Create output directory
mkdir -p "$OUTPUT_ROOT"

# Loop through all receptor directories
for receptor_dir in "$RECEPTOR_ROOT_DIR"/*/; do
    receptor_name=$(basename "$receptor_dir")
    maps_fld_file="$receptor_dir/rigidReceptor.maps.fld"  # Path to .maps.fld file

    # Check if .maps.fld file exists
    if [[ ! -f "$maps_fld_file" ]]; then
        echo "Maps file not found for receptor $receptor_name. Skipping..."
        continue
    fi

    # Create output directory for this receptor
    output_dir="$OUTPUT_ROOT/results/$receptor_name"
    mkdir -p "$output_dir"  # Ensure the directory exists

    # Loop through all ligand files
    for ligand in "$LIGAND_DIR"/*.pdbqt; do
        ligand_name=$(basename "$ligand" .pdbqt)
        log_file="$output_dir/${ligand_name}_log.txt"

        # Check if ligand file exists
        if [[ ! -f "$ligand" ]]; then
            echo "Ligand file $ligand not found. Skipping..."
            continue
        fi

        # Copy ligand to output directory
        cp "$ligand" "$output_dir/"

        # Run AutoDock-GPU in the output directory
        echo "Docking $ligand_name to $receptor_name..."
        (
            cd "$output_dir" && \
            autodock_gpu_128wi -lfile "$(basename "$ligand")" -ffile "$maps_fld_file" \
                -nrun "$NRUN" --heuristics 1 --heurmax "$HEURMAX" --nev "$NEV" --lsmet "$LSMET" --lsit "$LSIT" --autostop "$AUTOSTOP" \
                --asfreq "$ASFREQ" --stopstd "$STOPSTD" --clustering "$CLUSTERING" \
                --gbest "$GBEST" --xmloutput "$XML" --dlgoutput "$DLG" \
                -p "$P" -g "$G" 
        )

        # Cleanup copied ligand
        rm -f "$output_dir/$(basename "$ligand")"

        # Check for errors
        if [[ $? -ne 0 ]]; then
            echo "Error docking $ligand_name to $receptor_name. Check $log_file"
        else
            echo "Completed docking $ligand_name to $receptor_name."
        fi
    done
done

echo "All docking jobs completed!"
