# Comprehensive Single-Cell RNA-seq Analysis with SCANPY: A Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Environment Setup](#environment-setup)
3. [Hardware Configuration](#hardware-configuration)
4. [Data Acquisition](#data-acquisition)
5. [Data Loading and Preprocessing](#data-loading-and-preprocessing)
6. [Quality Control](#quality-control)
7. [Normalization and Feature Selection](#normalization-and-feature-selection)
8. [Dimensionality Reduction](#dimensionality-reduction)
9. [Clustering Analysis](#clustering-analysis)
10. [Differential Expression Analysis](#differential-expression-analysis)
11. [Cell Type Annotation](#cell-type-annotation)
12. [Trajectory Analysis](#trajectory-analysis)
13. [Gene Set Enrichment and Pathway Analysis](#gene-set-enrichment-and-pathway-analysis)
14. [Advanced Visualization](#advanced-visualization)
15. [Integration with Other Tools](#integration-with-other-tools)
16. [Best Practices and Tips](#best-practices-and-tips)

## Introduction

Single-cell RNA sequencing (scRNA-seq) has revolutionized our understanding of cellular heterogeneity. This guide provides a complete workflow for analyzing scRNA-seq data using SCANPY (Single-Cell Analysis in Python) along with complementary tools for comprehensive biological insights.

### What You'll Learn
- Complete scRNA-seq analysis pipeline from raw data to biological insights
- Parameter optimization strategies for each analysis step
- Integration with pathway analysis and functional annotation tools
- Performance optimization using GPU acceleration
- Troubleshooting common issues

## Environment Setup

### 1. Creating a Conda Environment

```bash
# Create a new conda environment
conda create -n scanpy-env python=3.10
conda activate scanpy-env

# Install core packages
pip install scanpy[all]==1.9.6
pip install pandas numpy matplotlib seaborn
pip install leidenalg python-igraph
pip install bbknn scvelo cellrank
pip install decoupler-py gseapy
```

### 2. GPU Support Installation (Optional but Recommended)

```bash
# For NVIDIA GPUs - Install RAPIDS for GPU acceleration
conda install -c rapidsai -c nvidia -c conda-forge rapids=23.10 python=3.10 cudatoolkit=11.8

# Install GPU-enabled packages
pip install rapids-singlecell
pip install cupy-cuda11x
```

### 3. Additional Analysis Tools

```bash
# Pathway analysis and annotation
pip install gseapy pyscenic
pip install celltypist scarches

# Visualization enhancements
pip install plotly nbformat>=4.2.0
pip install scvelo[all]

# Data integration tools
pip install harmony-pytorch scanorama
```

## Hardware Configuration

### Optimal Resource Management

```python
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure SCANPY settings
sc.settings.verbosity = 3  # Verbosity level
sc.settings.set_figure_params(dpi=80, facecolor='white')
sc.settings.n_jobs = -1  # Use all available cores

# Memory optimization
sc.settings.max_memory = 120  # GB, adjust based on your system

# GPU configuration (if available)
try:
    import rapids_singlecell as rsc
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU acceleration enabled!")
except ImportError:
    GPU_AVAILABLE = False
    print("Running on CPU")

# Set up plotting parameters
import matplotlib
matplotlib.rcParams['figure.figsize'] = (4, 4)
matplotlib.rcParams['pdf.fonttype'] = 42  # For publication-quality figures
```

### Memory Management Tips

```python
def optimize_adata(adata):
    """
    Optimize AnnData object memory usage
    """
    # Convert to sparse matrix if not already
    if not scipy.sparse.issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)
    
    # Optimize data types
    for col in adata.obs.columns:
        if adata.obs[col].dtype == 'object':
            try:
                adata.obs[col] = adata.obs[col].astype('category')
            except:
                pass
    
    # Remove unnecessary data
    if 'raw' in adata.uns:
        del adata.uns['raw']
    
    return adata
```

## Data Acquisition

### 1. Downloading Public Datasets

```python
# Method 1: Using scanpy built-in datasets
adata = sc.datasets.pbmc3k()  # 3k PBMCs from 10X Genomics
print(f"Dataset shape: {adata.shape}")

# Method 2: Download from GEO
import GEOparse
import urllib.request

def download_geo_data(geo_id, output_dir='./data'):
    """
    Download data from GEO database
    
    Parameters:
    -----------
    geo_id : str
        GEO accession number (e.g., 'GSE123456')
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    gse = GEOparse.get_GEO(geo=geo_id, destdir=output_dir)
    
    # Get supplementary files
    for gsm_name, gsm in gse.gsms.items():
        for supp_file in gsm.supplementary_file_url:
            urllib.request.urlretrieve(supp_file, 
                                     f"{output_dir}/{gsm_name}_{supp_file.split('/')[-1]}")
    
    return gse

# Example usage
# gse = download_geo_data('GSE123456')
```

### 2. Loading 10X Genomics Data

```python
def load_10x_data(data_path, min_genes=200, min_cells=3):
    """
    Load 10X Genomics data with initial filtering
    
    Parameters:
    -----------
    data_path : str
        Path to the directory containing matrix.mtx, genes.tsv, barcodes.tsv
    min_genes : int
        Minimum number of genes expressed per cell
    min_cells : int
        Minimum number of cells expressing a gene
    """
    # Read 10X data
    adata = sc.read_10x_mtx(
        data_path,
        var_names='gene_symbols',  # use gene symbols for gene names
        cache=True  # write a cache file for faster subsequent reading
    )
    
    # Make variable names unique
    adata.var_names_make_unique()
    
    # Initial filtering
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    print(f"Loaded dataset: {adata.shape[0]} cells × {adata.shape[1]} genes")
    
    return adata

# Example usage
# adata = load_10x_data('./data/filtered_gene_bc_matrices/hg19/')
```

### 3. Loading Multiple Samples

```python
def load_multiple_samples(sample_dict, batch_key='batch'):
    """
    Load and concatenate multiple samples
    
    Parameters:
    -----------
    sample_dict : dict
        Dictionary with sample names as keys and paths as values
    batch_key : str
        Key to store batch information
    """
    adatas = {}
    
    for sample_name, path in sample_dict.items():
        print(f"Loading {sample_name}...")
        adata = sc.read_10x_mtx(path, var_names='gene_symbols', cache=True)
        adata.obs[batch_key] = sample_name
        adatas[sample_name] = adata
    
    # Concatenate all samples
    adata_concat = sc.concat(adatas, label=batch_key, keys=list(adatas.keys()))
    
    print(f"Combined dataset: {adata_concat.shape[0]} cells × {adata_concat.shape[1]} genes")
    
    return adata_concat

# Example usage
# samples = {
#     'Control': './data/control/',
#     'Treatment': './data/treatment/'
# }
# adata = load_multiple_samples(samples)
```

## Quality Control

### Comprehensive QC Metrics

```python
def calculate_qc_metrics(adata, 
                         mt_prefix='MT-',
                         rb_prefix='RPS|RPL',
                         hb_genes=['HBA1', 'HBA2', 'HBB'],
                         min_genes=200,
                         max_genes=2500,
                         max_mt_percent=5):
    """
    Calculate and visualize comprehensive QC metrics
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix
    mt_prefix : str
        Prefix for mitochondrial genes
    rb_prefix : str
        Prefix pattern for ribosomal genes
    hb_genes : list
        List of hemoglobin genes
    min_genes : int
        Minimum genes per cell threshold
    max_genes : int
        Maximum genes per cell threshold
    max_mt_percent : float
        Maximum mitochondrial percentage threshold
    """
    
    # Calculate mitochondrial genes
    adata.var['mt'] = adata.var_names.str.startswith(mt_prefix)
    
    # Calculate ribosomal genes
    import re
    adata.var['ribo'] = adata.var_names.str.contains(f'^({rb_prefix})')
    
    # Calculate hemoglobin genes
    adata.var['hb'] = adata.var_names.isin(hb_genes)
    
    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt', 'ribo', 'hb'], 
                               percent_top=None, log1p=False, inplace=True)
    
    # Add additional metrics
    adata.obs['n_genes'] = (adata.X > 0).sum(axis=1).A1 if scipy.sparse.issparse(adata.X) else (adata.X > 0).sum(axis=1)
    adata.obs['log10_total_counts'] = np.log10(adata.obs['total_counts'] + 1)
    
    # Create QC plots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Violin plots for main metrics
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'pct_counts_ribo'],
                 jitter=0.4, multi_panel=True, ax=axes[0])
    
    # Scatter plots for relationships
    axes[1, 0].scatter(adata.obs['total_counts'], adata.obs['n_genes_by_counts'], 
                      c=adata.obs['pct_counts_mt'], s=1, cmap='viridis', alpha=0.5)
    axes[1, 0].set_xlabel('Total counts')
    axes[1, 0].set_ylabel('Number of genes')
    
    # Distribution plots
    axes[1, 1].hist(adata.obs['n_genes_by_counts'], bins=50, alpha=0.7)
    axes[1, 1].axvline(min_genes, color='r', linestyle='--', label=f'Min: {min_genes}')
    axes[1, 1].axvline(max_genes, color='r', linestyle='--', label=f'Max: {max_genes}')
    axes[1, 1].set_xlabel('Number of genes')
    axes[1, 1].legend()
    
    axes[1, 2].hist(adata.obs['pct_counts_mt'], bins=50, alpha=0.7)
    axes[1, 2].axvline(max_mt_percent, color='r', linestyle='--', label=f'Max: {max_mt_percent}%')
    axes[1, 2].set_xlabel('Mitochondrial gene percentage')
    axes[1, 2].legend()
    
    # Complexity plot
    axes[1, 3].scatter(adata.obs['log10_total_counts'], 
                      adata.obs['n_genes_by_counts'] / adata.obs['total_counts'],
                      s=1, alpha=0.5)
    axes[1, 3].set_xlabel('Log10 total counts')
    axes[1, 3].set_ylabel('Genes per UMI')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\nQC Statistics:")
    print(f"Cells before filtering: {adata.n_obs}")
    print(f"Median genes per cell: {adata.obs['n_genes_by_counts'].median():.0f}")
    print(f"Median counts per cell: {adata.obs['total_counts'].median():.0f}")
    print(f"Median MT percentage: {adata.obs['pct_counts_mt'].median():.2f}%")
    
    return adata

# Apply QC metrics
adata = calculate_qc_metrics(adata)
```

### Adaptive Filtering

```python
def adaptive_filtering(adata, 
                       mad_threshold=5,
                       min_genes=200,
                       min_counts=500):
    """
    Apply adaptive filtering based on MAD (Median Absolute Deviation)
    
    Parameters:
    -----------
    mad_threshold : float
        Number of MADs from median for outlier detection
    """
    from scipy import stats
    
    # Calculate MAD-based thresholds
    def mad_based_outlier(data, threshold=3):
        median = np.median(data)
        mad = stats.median_abs_deviation(data)
        lower = median - threshold * mad
        upper = median + threshold * mad
        return lower, upper
    
    # Calculate thresholds
    n_genes_lower, n_genes_upper = mad_based_outlier(adata.obs['n_genes_by_counts'], mad_threshold)
    counts_lower, counts_upper = mad_based_outlier(adata.obs['total_counts'], mad_threshold)
    mt_lower, mt_upper = mad_based_outlier(adata.obs['pct_counts_mt'], mad_threshold)
    
    # Apply minimum thresholds
    n_genes_lower = max(n_genes_lower, min_genes)
    counts_lower = max(counts_lower, min_counts)
    
    print(f"Adaptive thresholds:")
    print(f"  Genes: {n_genes_lower:.0f} - {n_genes_upper:.0f}")
    print(f"  Counts: {counts_lower:.0f} - {counts_upper:.0f}")
    print(f"  MT%: 0 - {mt_upper:.2f}")
    
    # Filter cells
    adata = adata[
        (adata.obs['n_genes_by_counts'] >= n_genes_lower) &
        (adata.obs['n_genes_by_counts'] <= n_genes_upper) &
        (adata.obs['total_counts'] >= counts_lower) &
        (adata.obs['total_counts'] <= counts_upper) &
        (adata.obs['pct_counts_mt'] < mt_upper)
    ].copy()
    
    print(f"Cells after filtering: {adata.n_obs}")
    
    return adata

# Apply adaptive filtering
adata = adaptive_filtering(adata)
```

## Normalization and Feature Selection

### 1. Normalization Methods

```python
def normalize_data(adata, 
                  method='standard',
                  target_sum=1e4,
                  exclude_highly_expressed=True):
    """
    Normalize expression data with multiple method options
    
    Parameters:
    -----------
    method : str
        'standard', 'scran', or 'sctransform'
    target_sum : float
        Target sum for normalization
    exclude_highly_expressed : bool
        Whether to exclude highly expressed genes from normalization
    """
    
    # Save raw counts
    adata.raw = adata
    
    if method == 'standard':
        # Standard log-normalization
        sc.pp.normalize_total(adata, target_sum=target_sum, 
                             exclude_highly_expressed=exclude_highly_expressed)
        sc.pp.log1p(adata)
        
    elif method == 'scran':
        # scran pooling-based size factor normalization
        import scanpy.external as sce
        
        # Preliminary clustering for size factor calculation
        adata_pp = adata.copy()
        sc.pp.normalize_total(adata_pp, target_sum=target_sum)
        sc.pp.log1p(adata_pp)
        sc.pp.pca(adata_pp, n_comps=15)
        sc.pp.neighbors(adata_pp)
        sc.tl.leiden(adata_pp, resolution=0.5)
        
        # Calculate size factors
        adata.obs['size_factors'] = sce.pp.scran_normalize(
            adata, 
            clusters=adata_pp.obs['leiden'],
            return_size_factors=True
        )
        
        # Normalize with size factors
        adata.X /= adata.obs['size_factors'].values[:, None]
        sc.pp.log1p(adata)
        
    elif method == 'sctransform':
        # SCTransform normalization
        try:
            import scanpy.external as sce
            sce.pp.sctransform(adata)
        except ImportError:
            print("SCTransform not available, using standard normalization")
            sc.pp.normalize_total(adata, target_sum=target_sum)
            sc.pp.log1p(adata)
    
    print(f"Normalization complete using {method} method")
    return adata

# Apply normalization
adata = normalize_data(adata, method='standard')
```

### 2. Highly Variable Gene Selection

```python
def select_highly_variable_genes(adata, 
                                 method='seurat',
                                 n_top_genes=2000,
                                 min_mean=0.0125,
                                 max_mean=3,
                                 min_disp=0.5,
                                 batch_key=None):
    """
    Select highly variable genes with different methods
    
    Parameters:
    -----------
    method : str
        'seurat', 'cell_ranger', or 'seurat_v3'
    n_top_genes : int
        Number of highly variable genes to select
    batch_key : str
        Key for batch correction in HVG selection
    """
    
    if batch_key:
        # Batch-aware HVG selection
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor=method,
            batch_key=batch_key,
            subset=False
        )
    else:
        # Standard HVG selection
        sc.pp.highly_variable_genes(
            adata,
            min_mean=min_mean,
            max_mean=max_mean,
            min_disp=min_disp,
            n_top_genes=n_top_genes,
            flavor=method,
            subset=False
        )
    
    # Plot HVG selection
    sc.pl.highly_variable_genes(adata)
    
    # Print statistics
    print(f"Number of highly variable genes: {adata.var['highly_variable'].sum()}")
    
    # Store full data and filter
    adata.raw = adata
    adata = adata[:, adata.var['highly_variable']]
    
    return adata

# Select HVGs
adata = select_highly_variable_genes(adata, n_top_genes=2000)
```

### 3. Data Scaling and Regression

```python
def scale_data(adata, 
              vars_to_regress=None,
              max_value=10,
              use_gpu=False):
    """
    Scale data and optionally regress out unwanted sources of variation
    
    Parameters:
    -----------
    vars_to_regress : list
        Variables to regress out (e.g., ['total_counts', 'pct_counts_mt'])
    max_value : float
        Clip values exceeding this threshold
    use_gpu : bool
        Use GPU acceleration if available
    """
    
    if use_gpu and GPU_AVAILABLE:
        import rapids_singlecell as rsc
        rsc.pp.scale(adata, max_value=max_value)
        
        if vars_to_regress:
            rsc.pp.regress_out(adata, vars_to_regress)
    else:
        # Scale to unit variance
        sc.pp.scale(adata, max_value=max_value)
        
        # Regress out unwanted sources of variation
        if vars_to_regress:
            sc.pp.regress_out(adata, vars_to_regress)
    
    print(f"Data scaled. Max value: {max_value}")
    if vars_to_regress:
        print(f"Regressed out: {vars_to_regress}")
    
    return adata

# Scale data
adata = scale_data(adata, vars_to_regress=['total_counts', 'pct_counts_mt'])
```

## Dimensionality Reduction

### 1. Principal Component Analysis (PCA)

```python
def run_pca(adata, 
           n_comps=50,
           use_hvg=True,
           svd_solver='auto',
           random_state=0):
    """
    Run PCA with optimal component selection
    
    Parameters:
    -----------
    n_comps : int
        Number of principal components
    use_hvg : bool
        Use only highly variable genes
    svd_solver : str
        SVD solver to use ('auto', 'full', 'arpack', 'randomized')
    """
    
    # Run PCA
    sc.tl.pca(adata, n_comps=n_comps, svd_solver=svd_solver, random_state=random_state)
    
    # Determine optimal number of PCs
    # Method 1: Elbow plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Variance explained
    axes[0].plot(range(1, n_comps+1), adata.uns['pca']['variance_ratio'], 'o-')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Variance Explained Ratio')
    axes[0].set_title('PCA Variance Explained')
    
    # Cumulative variance
    cumsum_var = np.cumsum(adata.uns['pca']['variance_ratio'])
    axes[1].plot(range(1, n_comps+1), cumsum_var, 'o-')
    axes[1].axhline(y=0.9, color='r', linestyle='--', label='90% variance')
    axes[1].set_xlabel('Principal Component')
    axes[1].set_ylabel('Cumulative Variance Explained')
    axes[1].set_title('Cumulative Variance')
    axes[1].legend()
    
    # Find elbow point (simplified)
    from kneed import KneeLocator
    try:
        kn = KneeLocator(range(1, n_comps+1), 
                        adata.uns['pca']['variance_ratio'], 
                        curve='convex', 
                        direction='decreasing')
        elbow = kn.elbow
        axes[0].axvline(x=elbow, color='r', linestyle='--', label=f'Elbow at PC{elbow}')
        axes[0].legend()
        print(f"Suggested number of PCs (elbow method): {elbow}")
    except:
        print("Could not determine elbow point automatically")
    
    # Method 2: Plot top genes for first PCs
    sc.pl.pca_loadings(adata, components=[1,2,3,4], ax=axes[2])
    
    plt.tight_layout()
    plt.show()
    
    # Additional visualization
    sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True)
    
    return adata

# Run PCA
adata = run_pca(adata, n_comps=50)
```

### 2. UMAP and t-SNE

```python
def run_umap_tsne(adata, 
                 n_neighbors=30,
                 n_pcs=40,
                 min_dist=0.3,
                 spread=1.0,
                 perplexity=30,
                 learning_rate=200,
                 use_gpu=False):
    """
    Run UMAP and t-SNE with parameter optimization
    
    Parameters:
    -----------
    n_neighbors : int
        Number of neighbors for UMAP
    n_pcs : int
        Number of PCs to use
    min_dist : float
        Minimum distance for UMAP
    spread : float
        Spread parameter for UMAP
    perplexity : int
        Perplexity for t-SNE
    """
    
    # Compute neighbor graph
    print(f"Computing neighbor graph with {n_neighbors} neighbors...")
    if use_gpu and GPU_AVAILABLE:
        import rapids_singlecell as rsc
        rsc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    else:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    
    # Run UMAP
    print("Running UMAP...")
    if use_gpu and GPU_AVAILABLE:
        import rapids_singlecell as rsc
        rsc.tl.umap(adata, min_dist=min_dist, spread=spread)
    else:
        sc.tl.umap(adata, min_dist=min_dist, spread=spread)
    
    # Run t-SNE
    print("Running t-SNE...")
    if use_gpu and GPU_AVAILABLE:
        import rapids_singlecell as rsc
        rsc.tl.tsne(adata, perplexity=perplexity, learning_rate=learning_rate)
    else:
        sc.tl.tsne(adata, perplexity=perplexity, learning_rate=learning_rate)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc.pl.umap(adata, ax=axes[0], show=False)
    sc.pl.tsne(adata, ax=axes[1], show=False)
    plt.tight_layout()
    plt.show()
    
    return adata

# Run dimensionality reduction
adata = run_umap_tsne(adata, n_neighbors=30, n_pcs=40)
```

### 3. Diffusion Maps (Alternative)

```python
def run_diffusion_map(adata, n_comps=15):
    """
    Run diffusion map for trajectory analysis
    """
    # Compute diffusion map
    sc.tl.diffmap(adata, n_comps=n_comps)
    
    # Plot
    sc.pl.diffmap(adata, color='n_genes', components=['1,2', '1,3'])
    
    return adata
```

## Clustering Analysis

### 1. Leiden Clustering with Resolution Optimization

```python
def optimize_clustering_resolution(adata, 
                                  resolution_range=(0.1, 2.0, 0.1),
                                  n_neighbors=30,
                                  metric='euclidean',
                                  random_state=0):
    """
    Optimize clustering resolution using multiple metrics
    
    Parameters:
    -----------
    resolution_range : tuple
        (start, stop, step) for resolution search
    """
    import pandas as pd
    from sklearn import metrics
    
    resolutions = np.arange(*resolution_range)
    
    # Store metrics
    results = []
    
    for res in resolutions:
        # Run clustering
        sc.tl.leiden(adata, resolution=res, random_state=random_state, key_added=f'leiden_{res}')
        
        # Calculate metrics
        labels = adata.obs[f'leiden_{res}'].astype(int)
        n_clusters = len(np.unique(labels))
        
        # Silhouette score (subsampled for speed)
        if adata.n_obs > 5000:
            subsample_idx = np.random.choice(adata.n_obs, 5000, replace=False)
            sil_score = metrics.silhouette_score(
                adata.obsm['X_pca'][subsample_idx], 
                labels[subsample_idx]
            )
        else:
            sil_score = metrics.silhouette_score(adata.obsm['X_pca'], labels)
        
        # Calinski-Harabasz score
        ch_score = metrics.calinski_harabasz_score(adata.obsm['X_pca'], labels)
        
        # Davies-Bouldin score (lower is better)
        db_score = metrics.davies_bouldin_score(adata.obsm['X_pca'], labels)
        
        results.append({
            'resolution': res,
            'n_clusters': n_clusters,
            'silhouette': sil_score,
            'calinski_harabasz': ch_score,
            'davies_bouldin': db_score
        })
        
    results_df = pd.DataFrame(results)
    
    # Plot metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(results_df['resolution'], results_df['n_clusters'], 'o-')
    axes[0, 0].set_xlabel('Resolution')
    axes[0, 0].set_ylabel('Number of Clusters')
    axes[0, 0].set_title('Clusters vs Resolution')
    
    axes[0, 1].plot(results_df['resolution'], results_df['silhouette'], 'o-')
    axes[0, 1].set_xlabel('Resolution')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].set_title('Silhouette Score (higher is better)')
    
    axes[1, 0].plot(results_df['resolution'], results_df['calinski_harabasz'], 'o-')
    axes[1, 0].set_xlabel('Resolution')
    axes[1, 0].set_ylabel('Calinski-Harabasz Score')
    axes[1, 0].set_title('Calinski-Harabasz Score (higher is better)')
    
    axes[1, 1].plot(results_df['resolution'], results_df['davies_bouldin'], 'o-')
    axes[1, 1].set_xlabel('Resolution')
    axes[1, 1].set_ylabel('Davies-Bouldin Score')
    axes[1, 1].set_title('Davies-Bouldin Score (lower is better)')
    
    plt.tight_layout()
    plt.show()
    
    # Find optimal resolution (based on silhouette score)
    optimal_res = results_df.loc[results_df['silhouette'].idxmax(), 'resolution']
    print(f"\nOptimal resolution (by silhouette score): {optimal_res}")
    print(f"Number of clusters: {results_df.loc[results_df['resolution']==optimal_res, 'n_clusters'].values[0]}")
    
    # Set final clustering
    adata.obs['leiden'] = adata.obs[f'leiden_{optimal_res}']
    
    # Clean up temporary clusterings
    for res in resolutions:
        del adata.obs[f'leiden_{res}']
    
    return adata, results_df

# Optimize clustering
adata, clustering_metrics = optimize_clustering_resolution(adata)
```

### 2. Alternative Clustering Methods

```python
def compare_clustering_algorithms(adata, resolution=1.0):
    """
    Compare different clustering algorithms
    """
    
    # Leiden
    sc.tl.leiden(adata, resolution=resolution, key_added='leiden')
    
    # Louvain
    sc.tl.louvain(adata, resolution=resolution, key_added='louvain')
    
    # Spectral clustering
    from sklearn.cluster import SpectralClustering
    spec_clustering = SpectralClustering(
        n_clusters=len(adata.obs['leiden'].unique()),
        affinity='precomputed',
        random_state=0
    )
    connectivity = adata.obsp['connectivities'].toarray()
    adata.obs['spectral'] = spec_clustering.fit_predict(connectivity).astype(str)
    
    # Visualize all methods
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sc.pl.umap(adata, color='leiden', ax=axes[0], show=False, title='Leiden')
    sc.pl.umap(adata, color='louvain', ax=axes[1], show=False, title='Louvain')
    sc.pl.umap(adata, color='spectral', ax=axes[2], show=False, title='Spectral')
    plt.tight_layout()
    plt.show()
    
    return adata

# Compare clustering methods
adata = compare_clustering_algorithms(adata)
```

## Differential Expression Analysis

### 1. Finding Marker Genes

```python
def find_all_markers(adata, 
                    groupby='leiden',
                    method='wilcoxon',
                    min_fold_change=0.25,
                    min_pct=0.1,
                    n_genes=25):
    """
    Find marker genes for all clusters
    
    Parameters:
    -----------
    method : str
        'wilcoxon', 't-test', 'logreg', or 'wilcoxon_pval'
    min_fold_change : float
        Minimum log fold change
    min_pct : float
        Minimum percentage of cells expressing the gene
    """
    
    # Calculate marker genes
    print(f"Finding markers using {method} test...")
    sc.tl.rank_genes_groups(
        adata, 
        groupby=groupby, 
        method=method,
        min_fold_change=min_fold_change,
        n_genes=n_genes,
        use_raw=True
    )
    
    # Filter by expression percentage
    sc.tl.filter_rank_genes_groups(
        adata,
        min_in_group_fraction=min_pct,
        min_fold_change=min_fold_change
    )
    
    # Visualize top markers
    sc.pl.rank_genes_groups(adata, n_genes=n_genes, sharey=False)
    
    # Create marker gene DataFrame
    markers_df = pd.DataFrame()
    for cluster in adata.obs[groupby].unique():
        cluster_genes = pd.DataFrame({
            'gene': adata.uns['rank_genes_groups']['names'][cluster][:n_genes],
            'score': adata.uns['rank_genes_groups']['scores'][cluster][:n_genes],
            'logfoldchange': adata.uns['rank_genes_groups']['logfoldchanges'][cluster][:n_genes],
            'pval': adata.uns['rank_genes_groups']['pvals'][cluster][:n_genes],
            'pval_adj': adata.uns['rank_genes_groups']['pvals_adj'][cluster][:n_genes],
            'cluster': cluster
        })
        markers_df = pd.concat([markers_df, cluster_genes])
    
    # Save markers
    markers_df.to_csv('marker_genes.csv', index=False)
    print(f"Saved marker genes to marker_genes.csv")
    
    return adata, markers_df

# Find markers
adata, markers = find_all_markers(adata)
```

### 2. Visualization of Marker Genes

```python
def visualize_markers(adata, markers_df, n_top=5):
    """
    Create comprehensive marker gene visualizations
    """
    
    # Get top markers per cluster
    top_markers = markers_df.groupby('cluster').head(n_top)['gene'].unique()
    
    # Dot plot
    sc.pl.dotplot(adata, top_markers, groupby='leiden', 
                  dendrogram=True, standard_scale='var')
    
    # Heatmap
    sc.pl.heatmap(adata, top_markers, groupby='leiden', 
                  swap_axes=True, standard_scale='var', cmap='RdBu_r')
    
    # Stacked violin plot
    sc.pl.stacked_violin(adata, top_markers, groupby='leiden', 
                         rotation=90, dendrogram=True)
    
    # Matrix plot
    sc.pl.matrixplot(adata, top_markers, groupby='leiden', 
                     dendrogram=True, standard_scale='var', cmap='RdBu_r')
    
    return adata

# Visualize markers
adata = visualize_markers(adata, markers)
```

## Cell Type Annotation

### 1. Manual Annotation with Known Markers

```python
def manual_annotation(adata, marker_dict, groupby='leiden'):
    """
    Manually annotate cell types based on known markers
    
    Parameters:
    -----------
    marker_dict : dict
        Dictionary with cell types as keys and marker lists as values
    """
    
    # Example marker dictionary for PBMCs
    if marker_dict is None:
        marker_dict = {
            'CD4 T cells': ['CD3D', 'CD4', 'IL7R'],
            'CD8 T cells': ['CD3D', 'CD8A', 'CD8B'],
            'NK cells': ['GNLY', 'NKG7', 'KLRD1'],
            'B cells': ['MS4A1', 'CD79A', 'CD79B'],
            'Monocytes': ['CD14', 'LYZ', 'S100A8', 'S100A9'],
            'DCs': ['FCER1A', 'CST3', 'CLEC10A'],
            'Platelets': ['PPBP', 'PF4']
        }
    
    # Score each cell type
    for cell_type, markers in marker_dict.items():
        # Filter markers that exist in the dataset
        valid_markers = [m for m in markers if m in adata.var_names]
        if valid_markers:
            sc.tl.score_genes(adata, valid_markers, score_name=f'{cell_type}_score')
    
    # Create score matrix
    score_cols = [f'{ct}_score' for ct in marker_dict.keys()]
    score_matrix = adata.obs[score_cols].values
    
    # Assign cell types based on highest score per cluster
    cluster_annotations = {}
    for cluster in adata.obs[groupby].unique():
        cluster_mask = adata.obs[groupby] == cluster
        cluster_scores = score_matrix[cluster_mask].mean(axis=0)
        best_type = list(marker_dict.keys())[np.argmax(cluster_scores)]
        cluster_annotations[cluster] = best_type
    
    # Add annotations
    adata.obs['cell_type'] = adata.obs[groupby].map(cluster_annotations)
    
    # Visualize
    sc.pl.umap(adata, color=['leiden', 'cell_type'], legend_loc='on data')
    
    return adata

# Annotate cell types
adata = manual_annotation(adata, marker_dict=None)
```

### 2. Automated Annotation with CellTypist

```python
def automated_annotation_celltypist(adata, model='Immune_All_Low.pkl'):
    """
    Automated cell type annotation using CellTypist
    """
    import celltypist
    from celltypist import models
    
    # Download model if needed
    models.download_models(force_update=False)
    
    # Load model
    model = models.Model.load(model=model)
    
    # Predict cell types
    predictions = celltypist.annotate(
        adata, 
        model=model,
        majority_voting=True
    )
    
    # Add predictions to adata
    adata = predictions.to_adata()
    
    # Visualize
    sc.pl.umap(adata, color=['majority_voting', 'conf_score'], 
               legend_loc='on data', frameon=False)
    
    return adata

# Example usage (requires celltypist installation)
# adata = automated_annotation_celltypist(adata)
```

## Trajectory Analysis

### 1. Pseudotime Analysis with Diffusion Pseudotime

```python
def calculate_pseudotime(adata, root_cluster='0', groupby='leiden'):
    """
    Calculate pseudotime using diffusion maps
    
    Parameters:
    -----------
    root_cluster : str
        Cluster to use as root for pseudotime
    """
    
    # Calculate diffusion map if not already done
    if 'X_diffmap' not in adata.obsm:
        sc.tl.diffmap(adata)
    
    # Set root cell (cell with highest expression of root markers)
    root_mask = adata.obs[groupby] == root_cluster
    root_indices = np.where(root_mask)[0]
    
    # Use the cell closest to the median of the root cluster
    root_cell = root_indices[0]
    adata.uns['iroot'] = root_cell
    
    # Calculate diffusion pseudotime
    sc.tl.dpt(adata)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sc.pl.umap(adata, color=groupby, ax=axes[0], show=False, title='Clusters')
    sc.pl.umap(adata, color='dpt_pseudotime', ax=axes[1], show=False, title='Pseudotime')
    sc.pl.diffmap(adata, color='dpt_pseudotime', ax=axes[2], show=False)
    plt.tight_layout()
    plt.show()
    
    return adata

# Calculate pseudotime
# adata = calculate_pseudotime(adata, root_cluster='0')
```

### 2. RNA Velocity Analysis

```python
def run_velocity_analysis(adata, loom_file=None):
    """
    Run RNA velocity analysis using scVelo
    
    Parameters:
    -----------
    loom_file : str
        Path to loom file with spliced/unspliced counts
    """
    import scvelo as scv
    
    if loom_file:
        # Load velocity data
        ldata = scv.read(loom_file, cache=True)
        
        # Merge with adata
        adata = scv.utils.merge(adata, ldata)
    
    # Preprocess
    scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=2000)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    
    # Run velocity
    scv.tl.velocity(adata)
    scv.tl.velocity_graph(adata)
    
    # Project velocity
    scv.pl.velocity_embedding_stream(adata, basis='umap', color='leiden')
    
    # Calculate velocity confidence
    scv.tl.velocity_confidence(adata)
    
    # Visualize
    scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, dpi=120)
    
    return adata

# Example usage (requires loom file)
# adata = run_velocity_analysis(adata, loom_file='velocyto_output.loom')
```

## Gene Set Enrichment and Pathway Analysis

### 1. Gene Set Scoring

```python
def score_gene_sets(adata, gene_sets=None, organism='human'):
    """
    Score predefined gene sets
    
    Parameters:
    -----------
    gene_sets : dict
        Dictionary of gene sets
    organism : str
        'human' or 'mouse'
    """
    
    # Default gene sets if none provided
    if gene_sets is None:
        gene_sets = {
            'Cell_cycle': ['MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 
                          'GINS2', 'MCM6', 'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'CENPU', 'HELLS'],
            'Apoptosis': ['CASP3', 'CASP7', 'BAX', 'BAK1', 'BID', 'BBC3', 'PMAIP1'],
            'Hypoxia': ['VEGFA', 'PGK1', 'ENO1', 'LDHA', 'GAPDH', 'SLC2A1', 'HIF1A'],
            'Inflammation': ['TNF', 'IL1B', 'IL6', 'CXCL8', 'PTGS2', 'NFKB1', 'STAT3']
        }
    
    # Score each gene set
    for name, genes in gene_sets.items():
        # Filter genes present in dataset
        valid_genes = [g for g in genes if g in adata.var_names]
        if valid_genes:
            sc.tl.score_genes(adata, valid_genes, score_name=name)
            print(f"Scored {name}: {len(valid_genes)}/{len(genes)} genes found")
    
    # Visualize scores
    score_names = list(gene_sets.keys())
    sc.pl.umap(adata, color=score_names, ncols=2, cmap='RdBu_r', use_raw=False)
    
    return adata

# Score gene sets
adata = score_gene_sets(adata)
```

### 2. Pathway Enrichment Analysis

```python
def pathway_enrichment_analysis(adata, 
                               cluster_key='leiden',
                               organism='human',
                               pval_cutoff=0.05):
    """
    Perform pathway enrichment analysis using GSEAPY
    
    Parameters:
    -----------
    cluster_key : str
        Key for cluster annotations
    organism : str
        'human' or 'mouse'
    pval_cutoff : float
        P-value cutoff for significance
    """
    import gseapy as gp
    
    # Get marker genes per cluster
    if 'rank_genes_groups' not in adata.uns:
        sc.tl.rank_genes_groups(adata, groupby=cluster_key, method='wilcoxon')
    
    # Prepare gene sets
    gene_sets = ['KEGG_2021_Human', 'GO_Biological_Process_2021', 'Reactome_2022']
    
    enrichment_results = {}
    
    for cluster in adata.obs[cluster_key].unique():
        print(f"Analyzing cluster {cluster}...")
        
        # Get top genes for cluster
        genes = adata.uns['rank_genes_groups']['names'][str(cluster)][:200]
        
        # Run enrichment
        enr = gp.enrichr(
            gene_list=list(genes),
            gene_sets=gene_sets,
            organism=organism.capitalize(),
            outdir=None,
            cutoff=pval_cutoff
        )
        
        enrichment_results[cluster] = enr.results
    
    # Visualize top pathways
    for cluster, results in enrichment_results.items():
        if not results.empty:
            top_pathways = results.nsmallest(10, 'P-value')
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(top_pathways)), -np.log10(top_pathways['P-value']))
            plt.yticks(range(len(top_pathways)), top_pathways['Term'])
            plt.xlabel('-log10(P-value)')
            plt.title(f'Top Enriched Pathways - Cluster {cluster}')
            plt.tight_layout()
            plt.show()
    
    return enrichment_results

# Run pathway analysis
# enrichment_results = pathway_enrichment_analysis(adata)
```

### 3. Transcription Factor Analysis with Decoupler

```python
def transcription_factor_analysis(adata):
    """
    Infer transcription factor activities using decoupler
    """
    import decoupler as dc
    
    # Load TF-target networks
    net = dc.get_collectri(organism='human', split_complexes=False)
    
    # Run enrichment analysis
    dc.run_mlm(
        mat=adata,
        net=net,
        source='source',
        target='target',
        weight='weight',
        verbose=True,
        use_raw=False
    )
    
    # Extract activities
    acts = dc.get_acts(adata, obsm_key='mlm_estimate')
    
    # Find top TFs per cluster
    mean_acts = dc.summarize_acts(acts, groupby='leiden', min_std=1)
    
    # Visualize top TFs
    n_top = 3
    top_tfs = mean_acts.var.iloc[:n_top].index.tolist()
    
    sc.pl.umap(adata, color=top_tfs, cmap='RdBu_r', vcenter=0)
    
    # Plot TF activities heatmap
    sc.pl.matrixplot(
        acts, 
        top_tfs, 
        groupby='leiden',
        dendrogram=True,
        standard_scale='var',
        cmap='RdBu_r'
    )
    
    return adata

# Run TF analysis
# adata = transcription_factor_analysis(adata)
```

## Advanced Visualization

### 1. Publication-Ready Plots

```python
def create_publication_figure(adata, save_path='figures/'):
    """
    Create publication-ready multi-panel figures
    """
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import os
    
    os.makedirs(save_path, exist_ok=True)
    
    # Set publication parameters
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.size'] = 8
    plt.rcParams['font.family'] = 'Arial'
    
    # Create complex figure layout
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel A: UMAP with clusters
    ax1 = fig.add_subplot(gs[0, 0:2])
    sc.pl.umap(adata, color='leiden', ax=ax1, show=False, frameon=False, 
               title='', legend_loc='none')
    ax1.set_title('A. Cell Clusters', fontweight='bold', loc='left')
    
    # Panel B: UMAP with cell types
    ax2 = fig.add_subplot(gs[0, 2:4])
    sc.pl.umap(adata, color='cell_type', ax=ax2, show=False, frameon=False,
               title='', legend_loc='right margin')
    ax2.set_title('B. Cell Types', fontweight='bold', loc='left')
    
    # Panel C: Dot plot of markers
    ax3 = fig.add_subplot(gs[1, :])
    marker_genes = ['CD3D', 'CD4', 'CD8A', 'MS4A1', 'CD14', 'FCGR3A', 'NKG7']
    sc.pl.dotplot(adata, marker_genes, groupby='leiden', ax=ax3, show=False,
                  dendrogram=False)
    ax3.set_title('C. Marker Expression', fontweight='bold', loc='left')
    
    # Panel D: Violin plots
    ax4 = fig.add_subplot(gs[2, :2])
    sc.pl.violin(adata, ['n_genes_by_counts', 'pct_counts_mt'], 
                 groupby='leiden', ax=ax4, show=False, rotation=45)
    ax4.set_title('D. QC Metrics', fontweight='bold', loc='left')
    
    # Panel E: Pathway scores
    ax5 = fig.add_subplot(gs[2, 2])
    if 'Cell_cycle' in adata.obs.columns:
        sc.pl.umap(adata, color='Cell_cycle', ax=ax5, show=False, frameon=False,
                   cmap='RdBu_r', title='')
        ax5.set_title('E. Cell Cycle Score', fontweight='bold', loc='left')
    
    # Save figure
    fig.savefig(f'{save_path}/main_figure.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f'{save_path}/main_figure.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print(f"Figures saved to {save_path}")

# Create publication figure
create_publication_figure(adata)
```

### 2. Interactive Visualizations

```python
def create_interactive_plots(adata):
    """
    Create interactive plots using plotly
    """
    import plotly.express as px
    import plotly.graph_objects as go
    
    # Prepare data
    df = pd.DataFrame({
        'UMAP1': adata.obsm['X_umap'][:, 0],
        'UMAP2': adata.obsm['X_umap'][:, 1],
        'cluster': adata.obs['leiden'].astype(str),
        'cell_type': adata.obs.get('cell_type', adata.obs['leiden']),
        'n_genes': adata.obs['n_genes_by_counts'],
        'mt_percent': adata.obs['pct_counts_mt']
    })
    
    # Interactive scatter plot
    fig = px.scatter(
        df, 
        x='UMAP1', 
        y='UMAP2',
        color='cluster',
        hover_data=['cell_type', 'n_genes', 'mt_percent'],
        title='Interactive UMAP',
        width=800,
        height=600
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    fig.show()
    
    # Save as HTML
    fig.write_html('interactive_umap.html')
    
    return fig

# Create interactive plots
# interactive_fig = create_interactive_plots(adata)
```

## Integration with Other Tools

### 1. Batch Correction with Harmony

```python
def batch_correction_harmony(adata, batch_key='batch', n_pcs=50):
    """
    Perform batch correction using Harmony
    """
    import scanpy.external as sce
    
    # Run Harmony
    sce.pp.harmony_integrate(
        adata, 
        batch_key,
        basis='X_pca',
        adjusted_basis='X_pca_harmony'
    )
    
    # Recompute neighbors and UMAP with corrected PCs
    sc.pp.neighbors(adata, use_rep='X_pca_harmony')
    sc.tl.umap(adata)
    sc.tl.leiden(adata)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc.pl.umap(adata, color=batch_key, ax=axes[0], show=False, 
               title='Batch Distribution')
    sc.pl.umap(adata, color='leiden', ax=axes[1], show=False,
               title='Clusters after Harmony')
    plt.tight_layout()
    plt.show()
    
    return adata

# Example usage with batch data
# adata = batch_correction_harmony(adata, batch_key='batch')
```

### 2. Integration with Seurat (via SeuratDisk)

```python
def export_to_seurat(adata, filename='adata_for_seurat.h5ad'):
    """
    Export AnnData object for Seurat
    """
    # Save as h5ad
    adata.write_h5ad(filename)
    
    print(f"Saved to {filename}")
    print("To load in R with Seurat:")
    print("""
    library(Seurat)
    library(SeuratDisk)
    
    # Convert h5ad to h5seurat
    Convert("adata_for_seurat.h5ad", dest = "h5seurat")
    
    # Load into Seurat
    seurat_obj <- LoadH5Seurat("adata_for_seurat.h5seurat")
    """)
    
    return filename

# Export for Seurat
# export_to_seurat(adata)
```

## Best Practices and Tips

### 1. Reproducibility Checklist

```python
def ensure_reproducibility(adata):
    """
    Ensure analysis reproducibility
    """
    import datetime
    
    # Record analysis parameters
    adata.uns['analysis_params'] = {
        'date': datetime.datetime.now().isoformat(),
        'scanpy_version': sc.__version__,
        'python_version': sys.version,
        'n_hvg': sum(adata.var['highly_variable']) if 'highly_variable' in adata.var else 0,
        'n_pcs': adata.obsm['X_pca'].shape[1] if 'X_pca' in adata.obsm else 0,
        'clustering_resolution': adata.uns.get('leiden', {}).get('params', {}).get('resolution', 'NA'),
        'random_seed': 0
    }
    
    # Set random seeds
    np.random.seed(0)
    import random
    random.seed(0)
    
    print("Reproducibility parameters recorded:")
    for key, value in adata.uns['analysis_params'].items():
        print(f"  {key}: {value}")
    
    return adata
```

### 2. Performance Optimization Tips

```python
def optimize_performance():
    """
    Tips for optimizing scRNA-seq analysis performance
    """
    
    tips = """
    Performance Optimization Guidelines:
    
    1. **Memory Management**:
       - Use sparse matrices: adata.X = scipy.sparse.csr_matrix(adata.X)
       - Clear unnecessary data: del adata.uns['large_data']
       - Use chunked operations for large datasets
       - Consider downsampling for initial exploration
    
    2. **CPU Optimization**:
       - Set n_jobs=-1 in scanpy settings
       - Use multiprocessing for batch processing
       - Profile code with line_profiler for bottlenecks
    
    3. **GPU Acceleration**:
       - Use rapids-singlecell for GPU support
       - Prefer GPU for: PCA, neighbors, UMAP, clustering
       - Monitor GPU memory with nvidia-smi
    
    4. **Disk I/O**:
       - Use .h5ad format with compression
       - Cache intermediate results
       - Use backed mode for very large datasets:
         adata = sc.read_h5ad('file.h5ad', backed='r')
    
    5. **Algorithm Selection**:
       - Use approximate methods for large data (randomized PCA)
       - Consider mini-batch approaches for clustering
       - Use subset of cells for parameter optimization
    
    6. **Parallel Processing Example**:
       from multiprocessing import Pool
       
       def process_sample(sample_path):
           adata = sc.read_10x_mtx(sample_path)
           # Process...
           return adata
       
       with Pool(processes=4) as pool:
           results = pool.map(process_sample, sample_paths)
    """
    
    print(tips)
    return tips

# Display optimization tips
optimize_performance()
```

### 3. Common Troubleshooting

```python
def troubleshooting_guide():
    """
    Common issues and solutions in scRNA-seq analysis
    """
    
    solutions = {
        "Memory Error": [
            "Use sparse matrices",
            "Reduce n_top_genes in HVG selection",
            "Process in chunks",
            "Use backed mode for h5ad files",
            "Increase system swap space"
        ],
        
        "Batch Effects": [
            "Use Harmony or Scanorama for integration",
            "Include batch in HVG selection",
            "Regress out batch effects in scaling",
            "Use batch-aware clustering (bbknn)",
            "Verify with mixing metrics"
        ],
        
        "Poor Clustering": [
            "Optimize resolution parameter",
            "Try different numbers of PCs",
            "Adjust n_neighbors parameter",
            "Remove doublets before clustering",
            "Check for technical artifacts"
        ],
        
        "Low Quality Data": [
            "Adjust QC thresholds adaptively",
            "Check for ambient RNA contamination",
            "Consider using SoupX for correction",
            "Verify sequencing depth adequacy",
            "Check for batch-specific issues"
        ],
        
        "Annotation Issues": [
            "Use multiple marker genes per type",
            "Combine automated and manual methods",
            "Verify with known biology",
            "Use reference datasets for mapping",
            "Consider sub-clustering"
        ]
    }
    
    for issue, solutions_list in solutions.items():
        print(f"\n{issue}:")
        for solution in solutions_list:
            print(f"  • {solution}")
    
    return solutions

# Display troubleshooting guide
troubleshooting_guide()
```

## Conclusion

This comprehensive guide covers the complete workflow for single-cell RNA-seq analysis using SCANPY and complementary tools. Key takeaways:

1. **Start with Quality**: Rigorous QC is essential for reliable results
2. **Optimize Parameters**: Use data-driven approaches for parameter selection
3. **Validate Results**: Cross-reference findings with known biology
4. **Document Everything**: Ensure reproducibility with detailed records
5. **Leverage GPU**: Use acceleration when available for large datasets
6. **Integrate Tools**: Combine multiple tools for comprehensive analysis

### Next Steps

- Explore advanced methods like spatial transcriptomics integration
- Implement custom analysis pipelines for specific biological questions
- Contribute to open-source tools and share your workflows
- Stay updated with the rapidly evolving scRNA-seq field

### Resources

- **SCANPY Documentation**: https://scanpy.readthedocs.io/
- **scRNA-seq Course**: https://scrnaseq-course.org/
- **Single Cell Portal**: https://singlecell.broadinstitute.org/
- **Cell Ranger**: https://support.10xgenomics.com/
- **Seurat**: https://satijalab.org/seurat/

### Citation

If you use this guide, please cite:
- Wolf et al., SCANPY: large-scale single-cell gene expression data analysis. Genome Biology (2018)
- Relevant tool papers for additional packages used

---

*Last updated: 2025*
*Version: 1.0*
*License: MIT*