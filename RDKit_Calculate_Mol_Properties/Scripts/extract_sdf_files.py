#!/usr/bin/env python3
"""
SDF File Extractor
Extracts all compressed SDF files to a clean directory structure for processing
"""

import os
import gzip
import shutil
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_single_file(args):
    """Extract a single compressed SDF file"""
    gz_file, output_dir = args
    
    try:
        # Create output filename (remove .gz extension)
        gz_path = Path(gz_file)
        output_file = Path(output_dir) / gz_path.stem
        
        # Skip if already exists
        if output_file.exists():
            return f"Skipped (exists): {output_file.name}"
        
        # Extract file
        with gzip.open(gz_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        return f"Extracted: {output_file.name}"
        
    except Exception as e:
        return f"Error extracting {gz_file}: {str(e)}"

def extract_all_sdf_files(input_dir, output_dir, n_processes=None):
    """Extract all compressed SDF files to output directory"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(exist_ok=True)
    
    # Find all .gz files
    gz_files = list(input_path.rglob("*.sdf.gz"))
    logger.info(f"Found {len(gz_files)} compressed SDF files")
    
    if not gz_files:
        logger.warning("No compressed SDF files found")
        return
    
    # Prepare arguments for multiprocessing
    extract_args = [(str(gz_file), str(output_path)) for gz_file in gz_files]
    
    # Extract files in parallel
    n_processes = n_processes or mp.cpu_count()
    logger.info(f"Extracting files using {n_processes} processes...")
    
    with mp.Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(extract_single_file, extract_args),
            total=len(extract_args),
            desc="Extracting SDF files"
        ))
    
    # Log results
    extracted_count = sum(1 for r in results if r.startswith("Extracted:"))
    skipped_count = sum(1 for r in results if r.startswith("Skipped"))
    error_count = sum(1 for r in results if r.startswith("Error"))
    
    logger.info(f"Extraction complete:")
    logger.info(f"  Extracted: {extracted_count}")
    logger.info(f"  Skipped: {skipped_count}")
    logger.info(f"  Errors: {error_count}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Extract compressed SDF files')
    parser.add_argument('input_dir', help='Directory containing compressed SDF files')
    parser.add_argument('-o', '--output_dir', default='all_extracted_sdf', 
                       help='Output directory for extracted files (default: all_extracted_sdf)')
    parser.add_argument('-p', '--processes', type=int, default=None,
                       help='Number of processes (default: CPU count)')
    
    args = parser.parse_args()
    
    # Extract files
    output_path = extract_all_sdf_files(
        args.input_dir,
        args.output_dir,
        args.processes
    )
    
    if output_path:
        logger.info(f"All files extracted to: {output_path}")
        logger.info(f"You can now run: python wager_molecular_filter.py {output_path}")

if __name__ == "__main__":
    main()