#!/usr/bin/env python3
"""
Gene Expression Data Input Formats for ML Pipeline
=================================================

This script demonstrates the expected input data formats for the gene expression
ML pipeline and provides utilities to convert between common formats.
"""

import pandas as pd
import numpy as np

def demonstrate_data_formats():
    """
    Show different input data formats for gene expression analysis
    """
    
    print("=" * 80)
    print("GENE EXPRESSION DATA INPUT FORMATS")
    print("=" * 80)
    
    # =========================================================================
    # FORMAT 1: Standard Format (Samples as Rows, Genes as Columns)
    # =========================================================================
    print("\n1. STANDARD FORMAT (Recommended)")
    print("-" * 40)
    print("Structure: Rows = Samples/Patients, Columns = Genes + Target")
    print("This is the format expected by the ML pipeline.")
    
    # Create example data
    sample_data_standard = {
        'BRCA1': [12.5, 8.3, 15.2, 9.7, 11.8],
        'TP53': [25.1, 18.9, 22.3, 28.5, 20.1],
        'EGFR': [8.7, 12.4, 6.9, 14.2, 10.5],
        'MYC': [15.3, 22.1, 18.7, 12.9, 19.4],
        'PTEN': [7.2, 9.8, 6.5, 11.3, 8.9],
        'disease_status': [1, 0, 1, 0, 1]  # Target variable
    }
    
    df_standard = pd.DataFrame(sample_data_standard, 
                              index=['Patient_001', 'Patient_002', 'Patient_003', 
                                   'Patient_004', 'Patient_005'])
    
    print("\nExample:")
    print(df_standard)
    
    print(f"\nDataset shape: {df_standard.shape}")
    print(f"Samples (rows): {df_standard.shape[0]}")
    print(f"Features (columns excluding target): {df_standard.shape[1] - 1}")
    
    # =========================================================================
    # FORMAT 2: Transposed Format (Common in Bioinformatics)
    # =========================================================================
    print("\n\n2. TRANSPOSED FORMAT (Common in bioinformatics files)")
    print("-" * 50)
    print("Structure: Rows = Genes, Columns = Samples + Metadata")
    print("Often found in GEO, TCGA, and other genomics databases.")
    
    # Create transposed version
    genes = ['BRCA1', 'TP53', 'EGFR', 'MYC', 'PTEN']
    patients = ['Patient_001', 'Patient_002', 'Patient_003', 'Patient_004', 'Patient_005']
    
    expression_values = np.array([
        [12.5, 8.3, 15.2, 9.7, 11.8],   # BRCA1
        [25.1, 18.9, 22.3, 28.5, 20.1], # TP53
        [8.7, 12.4, 6.9, 14.2, 10.5],   # EGFR
        [15.3, 22.1, 18.7, 12.9, 19.4], # MYC
        [7.2, 9.8, 6.5, 11.3, 8.9]      # PTEN
    ])
    
    df_transposed = pd.DataFrame(expression_values, 
                                index=genes, 
                                columns=patients)
    
    print("\nExample (Expression Matrix):")
    print(df_transposed)
    
    # Metadata file (separate)
    metadata = pd.DataFrame({
        'Sample_ID': patients,
        'disease_status': [1, 0, 1, 0, 1],
        'age': [45, 52, 38, 61, 47],
        'gender': ['F', 'M', 'F', 'F', 'M']
    })
    
    print("\nMetadata file (separate):")
    print(metadata)
    
    # =========================================================================
    # FORMAT 3: Multiple Files Format
    # =========================================================================
    print("\n\n3. MULTIPLE FILES FORMAT")
    print("-" * 30)
    print("Common structure:")
    print("- expression_matrix.txt (genes × samples)")
    print("- clinical_data.txt (sample metadata)")
    print("- gene_annotations.txt (gene information)")
    
    # =========================================================================
    # FORMAT 4: Real-world Examples
    # =========================================================================
    print("\n\n4. REAL-WORLD DATA EXAMPLES")
    print("-" * 35)
    
    print("\nA. RNA-seq Count Data:")
    rnaseq_example = pd.DataFrame({
        'ENSG00000139618': [1205, 856, 1456, 923, 1187],  # BRCA2
        'ENSG00000141510': [2341, 1876, 2567, 2103, 2234], # TP53
        'ENSG00000146648': [876, 1234, 698, 1456, 1087],   # EGFR
        'treatment_response': ['Responder', 'Non-responder', 'Responder', 'Non-responder', 'Responder']
    }, index=['Sample_001', 'Sample_002', 'Sample_003', 'Sample_004', 'Sample_005'])
    
    print(rnaseq_example.head())
    print("Note: Count data (integers), needs normalization")
    
    print("\nB. Microarray Data:")
    microarray_example = pd.DataFrame({
        '1007_s_at': [8.45, 7.23, 9.12, 7.89, 8.67],      # Probe ID
        '1053_at': [12.34, 11.56, 13.21, 10.87, 12.45],   # Probe ID
        '117_at': [6.78, 7.45, 6.23, 8.12, 7.01],         # Probe ID
        'cancer_type': ['Adenocarcinoma', 'Normal', 'Squamous', 'Normal', 'Adenocarcinoma']
    }, index=['GSM001', 'GSM002', 'GSM003', 'GSM004', 'GSM005'])
    
    print(microarray_example.head())
    print("Note: Log-transformed intensities (floats)")
    
    print("\nC. Drug Response Data:")
    drug_response_example = pd.DataFrame({
        'BRCA1': [12.5, 8.3, 15.2, 9.7, 11.8],
        'TP53': [25.1, 18.9, 22.3, 28.5, 20.1],
        'EGFR': [8.7, 12.4, 6.9, 14.2, 10.5],
        'IC50_Doxorubicin': [2.3, 15.7, 1.8, 12.4, 3.1],  # Target: drug sensitivity
    }, index=['CellLine_001', 'CellLine_002', 'CellLine_003', 'CellLine_004', 'CellLine_005'])
    
    print(drug_response_example.head())
    print("Note: IC50 values as target (lower = more sensitive)")


def show_file_format_examples():
    """
    Show different file format examples
    """
    print("\n\n" + "=" * 80)
    print("COMMON FILE FORMATS")
    print("=" * 80)
    
    print("\n1. TAB-SEPARATED VALUES (.tsv/.txt)")
    print("-" * 40)
    print("Most common format in bioinformatics")
    print("Example file content:")
    print("""
Sample_ID	BRCA1	TP53	EGFR	MYC	disease_status
Patient_001	12.5	25.1	8.7	15.3	1
Patient_002	8.3	18.9	12.4	22.1	0
Patient_003	15.2	22.3	6.9	18.7	1
    """.strip())
    
    print("\n2. COMMA-SEPARATED VALUES (.csv)")
    print("-" * 35)
    print("Common format, easy to open in Excel")
    print("Example file content:")
    print("""
Sample_ID,BRCA1,TP53,EGFR,MYC,disease_status
Patient_001,12.5,25.1,8.7,15.3,1
Patient_002,8.3,18.9,12.4,22.1,0
Patient_003,15.2,22.3,6.9,18.7,1
    """.strip())
    
    print("\n3. EXCEL FILES (.xlsx/.xls)")
    print("-" * 30)
    print("- Expression data in one sheet")
    print("- Metadata in another sheet")
    print("- Gene annotations in third sheet")


def data_preprocessing_utilities():
    """
    Utilities to convert between different formats
    """
    print("\n\n" + "=" * 80)
    print("DATA CONVERSION UTILITIES")
    print("=" * 80)
    
    print("\n1. Converting Transposed Data to Standard Format")
    print("-" * 50)
    
    code_example_1 = '''
def transpose_expression_data(expression_file, metadata_file, target_column):
    """
    Convert transposed gene expression data to standard format
    
    Parameters:
    -----------
    expression_file : str
        Path to expression matrix (genes as rows, samples as columns)
    metadata_file : str
        Path to metadata file with sample information
    target_column : str
        Name of target column in metadata
    
    Returns:
    --------
    pd.DataFrame : Standard format data
    """
    # Load expression matrix (genes × samples)
    expr_df = pd.read_csv(expression_file, sep='\\t', index_col=0)
    
    # Transpose to get samples × genes
    expr_df = expr_df.T
    
    # Load metadata
    metadata_df = pd.read_csv(metadata_file, sep='\\t')
    metadata_df.set_index('Sample_ID', inplace=True)
    
    # Merge expression and metadata
    combined_df = expr_df.join(metadata_df[target_column])
    
    return combined_df

# Usage example:
# data = transpose_expression_data('expression_matrix.txt', 'clinical_data.txt', 'disease_status')
    '''
    print(code_example_1)
    
    print("\n2. Handling Multiple Data Types")
    print("-" * 35)
    
    code_example_2 = '''
def load_multi_omics_data(file_paths, target_column):
    """
    Load and combine multiple omics datasets
    
    Parameters:
    -----------
    file_paths : dict
        Dictionary with data type names as keys and file paths as values
        Example: {'expression': 'rna_seq.txt', 'mutation': 'mutations.txt'}
    target_column : str
        Name of target column
    
    Returns:
    --------
    pd.DataFrame : Combined dataset
    """
    datasets = []
    
    for data_type, file_path in file_paths.items():
        df = pd.read_csv(file_path, sep='\\t', index_col=0)
        
        # Add prefix to column names to avoid conflicts
        if data_type != 'clinical':
            df.columns = [f"{data_type}_{col}" for col in df.columns if col != target_column]
        
        datasets.append(df)
    
    # Combine all datasets
    combined_df = pd.concat(datasets, axis=1)
    
    return combined_df

# Usage example:
# files = {
#     'expression': 'gene_expression.txt',
#     'methylation': 'dna_methylation.txt', 
#     'clinical': 'clinical_data.txt'
# }
# data = load_multi_omics_data(files, 'survival_status')
    '''
    print(code_example_2)
    
    print("\n3. Data Quality Checks")
    print("-" * 25)
    
    code_example_3 = '''
def check_data_quality(df, target_column):
    """
    Perform quality checks on gene expression data
    """
    print("DATA QUALITY REPORT")
    print("=" * 40)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Samples: {df.shape[0]}")
    print(f"Features: {df.shape[1] - 1}")
    
    # Check for missing values
    missing_samples = df.isnull().any(axis=1).sum()
    missing_features = df.isnull().any(axis=0).sum()
    print(f"Samples with missing values: {missing_samples}")
    print(f"Features with missing values: {missing_features}")
    
    # Check target distribution
    print(f"\\nTarget variable distribution:")
    print(df[target_column].value_counts())
    
    # Check for constant features
    constant_features = (df.drop(columns=[target_column]).nunique() == 1).sum()
    print(f"\\nConstant features: {constant_features}")
    
    # Check data types
    print(f"\\nData types:")
    print(df.dtypes.value_counts())
    
    return df

# Usage:
# data = check_data_quality(df, 'disease_status')
    '''
    print(code_example_3)


def create_sample_datasets():
    """
    Create sample datasets in different formats for testing
    """
    print("\n\n" + "=" * 80)
    print("CREATING SAMPLE DATASETS")
    print("=" * 80)
    
    np.random.seed(42)
    
    # 1. Standard format dataset
    print("\nCreating standard format dataset...")
    
    n_samples = 100
    n_genes = 50
    
    # Generate gene names
    gene_names = [f'GENE_{i:04d}' for i in range(n_genes)]
    sample_names = [f'Sample_{i:03d}' for i in range(n_samples)]
    
    # Generate expression data
    expression_data = np.random.lognormal(mean=2, sigma=1, size=(n_samples, n_genes))
    
    # Create target variable (binary classification)
    # Make it somewhat correlated with first few genes
    target = (expression_data[:, :5].mean(axis=1) > np.median(expression_data[:, :5].mean(axis=1))).astype(int)
    
    # Create DataFrame
    df_standard = pd.DataFrame(expression_data, columns=gene_names, index=sample_names)
    df_standard['disease_status'] = target
    
    # Save as TSV (recommended format)
    df_standard.to_csv('sample_gene_expression_standard.tsv', sep='\t')
    print("✓ Standard format saved as 'sample_gene_expression_standard.tsv'")
    
    # 2. Transposed format with separate metadata
    print("\nCreating transposed format dataset...")
    
    # Expression matrix (genes × samples)
    df_transposed = pd.DataFrame(expression_data.T, columns=sample_names, index=gene_names)
    df_transposed.to_csv('sample_expression_matrix.tsv', sep='\t')
    
    # Metadata file
    metadata = pd.DataFrame({
        'Sample_ID': sample_names,
        'disease_status': target,
        'age': np.random.randint(25, 75, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'batch': np.random.choice(['Batch_A', 'Batch_B', 'Batch_C'], n_samples)
    })
    metadata.to_csv('sample_metadata.tsv', sep='\t', index=False)
    
    print("✓ Transposed format saved as 'sample_expression_matrix.tsv' and 'sample_metadata.tsv'")
    
    # 3. Multi-class classification example
    print("\nCreating multi-class classification dataset...")
    
    # Create 3-class target
    target_multiclass = np.random.choice(['Normal', 'Disease_A', 'Disease_B'], n_samples)
    
    df_multiclass = df_standard.copy()
    df_multiclass['disease_type'] = target_multiclass
    df_multiclass.to_csv('sample_multiclass_data.tsv', sep='\t')
    
    print("✓ Multi-class dataset saved as 'sample_multiclass_data.tsv'")
    
    # 4. Drug response dataset
    print("\nCreating drug response dataset...")
    
    # Generate IC50 values (continuous target)
    ic50_values = np.random.lognormal(mean=1, sigma=0.5, size=n_samples)
    
    df_drug_response = df_standard.drop(columns=['disease_status']).copy()
    df_drug_response['IC50_Drug_X'] = ic50_values
    
    # Binary sensitive/resistant classification
    sensitive_threshold = np.median(ic50_values)
    df_drug_response['drug_sensitive'] = (ic50_values < sensitive_threshold).astype(int)
    
    df_drug_response.to_csv('sample_drug_response_data.tsv', sep='\t')
    print("✓ Drug response dataset saved as 'sample_drug_response_data.tsv'")
    
    print(f"\n📁 All sample datasets created! You can use them with the ML pipeline:")
    print(f"   pipeline.load_and_inspect_data('sample_gene_expression_standard.tsv', 'disease_status')")


if __name__ == "__main__":
    demonstrate_data_formats()
    show_file_format_examples()
    data_preprocessing_utilities()
    create_sample_datasets()
