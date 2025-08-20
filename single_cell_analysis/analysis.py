import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import scanpy as sc
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Configure plotting
    plt.style.use('default')
    sns.set_palette("husl")

    # Configure scanpy
    sc.settings.verbosity = 3
    sc.settings.set_figure_params(dpi=100, facecolor='white', figsize=(6, 4))

    print(" Modern scRNA-seq analysis setup complete")
    print(" Using: Scanpy + Polars + Matplotlib + Marimo")
    return np, pl, plt, sc, sns


@app.cell
def _(pl, sc):
    # Load 10x h5 data
    adata_raw = sc.read_10x_h5('dataset.h5')

    # Make gene names unique
    adata_raw.var_names_make_unique()

    # Store gene information
    adata_raw.var['ensembl_ids'] = adata_raw.var['gene_ids']
    adata_raw.var['gene_symbols'] = adata_raw.var.index

    print(f"Dataset shape: {adata_raw.shape}")
    print(f"Genes: {adata_raw.n_vars}, Cells: {adata_raw.n_obs}")
    print(f"Gene names are unique: {adata_raw.var_names.is_unique}")

    # Quick overview with polars
    var_df = pl.from_pandas(adata_raw.var)
    print(f"\nFeature types: {var_df['feature_types'].unique().to_list()}")
    print(f"First 10 genes: {adata_raw.var_names[:10].tolist()}")
    return (adata_raw,)


@app.cell
def _(adata_raw, pl, sc):
    # Calculate QC metrics
    adata_qc = adata_raw.copy()

    # Identify gene types
    adata_qc.var['mt'] = adata_qc.var_names.str.startswith('MT-')
    adata_qc.var['ribo'] = adata_qc.var_names.str.startswith(('RPS', 'RPL'))
    adata_qc.var['hemoglobin'] = adata_qc.var_names.str.contains('^HB[^(P)]')

    # Calculate per-cell QC metrics
    sc.pp.calculate_qc_metrics(adata_qc, percent_top=None, log1p=False, inplace=True)

    # Add mitochondrial and ribosomal gene percentages
    adata_qc.obs['pct_counts_mt'] = adata_qc[:, adata_qc.var['mt']].X.sum(axis=1).A1 / adata_qc.obs['total_counts'] * 100
    adata_qc.obs['pct_counts_ribo'] = adata_qc[:, adata_qc.var['ribo']].X.sum(axis=1).A1 / adata_qc.obs['total_counts'] * 100

    print(f"Found {adata_qc.var['mt'].sum()} mitochondrial genes")
    print(f"Found {adata_qc.var['ribo'].sum()} ribosomal genes")

    # Create QC dataframe for analysis
    qc_stats = pl.from_pandas(adata_qc.obs[['total_counts', 'n_genes_by_counts', 'pct_counts_mt', 'pct_counts_ribo']])

    print("\nQC Statistics:")
    print(qc_stats.describe())
    return adata_qc, qc_stats


@app.cell
def _(plt, qc_stats):
    # Create QC plots
    def plot_qc_metrics():
        plot_data = qc_stats.to_pandas()
    
        print(f" Creating QC plots for {len(plot_data):,} cells")
    
        # Plot 1: Total UMI counts
        plt.figure(figsize=(10, 6))
        plt.hist(plot_data['total_counts'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        plt.xlabel('Total UMI Counts')
        plt.ylabel('Number of Cells')
        plt.title('Total UMI Counts Distribution')
        plt.grid(True, alpha=0.3)
        plt.axvline(plot_data['total_counts'].median(), color='red', linestyle='--', label=f'Median: {plot_data["total_counts"].median():.0f}')
        plt.legend()
        plt.show()
    
        # Plot 2: Genes per cell
        plt.figure(figsize=(10, 6))
        plt.hist(plot_data['n_genes_by_counts'], bins=50, color='orange', alpha=0.7, edgecolor='black')
        plt.xlabel('Genes per Cell')
        plt.ylabel('Number of Cells')
        plt.title('Genes per Cell Distribution')
        plt.grid(True, alpha=0.3)
        plt.axvline(plot_data['n_genes_by_counts'].median(), color='red', linestyle='--', label=f'Median: {plot_data["n_genes_by_counts"].median():.0f}')
        plt.legend()
        plt.show()
    
        # Plot 3: Mitochondrial percentage
        plt.figure(figsize=(10, 6))
        plt.hist(plot_data['pct_counts_mt'], bins=50, color='red', alpha=0.7, edgecolor='black')
        plt.xlabel('% Mitochondrial Genes')
        plt.ylabel('Number of Cells')
        plt.title('Mitochondrial Gene % Distribution')
        plt.grid(True, alpha=0.3)
        plt.axvline(plot_data['pct_counts_mt'].median(), color='darkred', linestyle='--', label=f'Median: {plot_data["pct_counts_mt"].median():.1f}%')
        plt.legend()
        plt.show()

    plot_qc_metrics()
    return


@app.cell
def _(adata_qc, adata_raw, pl, qc_stats, sc):
    # Calculate filtering thresholds
    filtering_thresholds = qc_stats.select([
        pl.col('total_counts').quantile(0.02).alias('min_counts_2pct'),
        pl.col('total_counts').quantile(0.98).alias('max_counts_98pct'),
        pl.col('n_genes_by_counts').quantile(0.02).alias('min_genes_2pct'), 
        pl.col('n_genes_by_counts').quantile(0.98).alias('max_genes_98pct'),
        pl.col('pct_counts_mt').quantile(0.95).alias('max_mt_95pct'),
        pl.col('pct_counts_mt').quantile(0.98).alias('max_mt_98pct')
    ])

    print("Suggested filtering thresholds:")
    print(filtering_thresholds)

    # Apply filtering
    adata_filtered = adata_qc.copy()

    print(f"Cells before filtering: {adata_filtered.n_obs}")
    print(f"Genes before filtering: {adata_filtered.n_vars}")

    # Get thresholds
    thresh = filtering_thresholds.to_dicts()[0]
    min_genes = max(200, int(thresh['min_genes_2pct']))
    max_genes = min(6000, int(thresh['max_genes_98pct']))
    max_mt = min(25, thresh['max_mt_95pct'])

    print(f"Using thresholds: min_genes={min_genes}, max_genes={max_genes}, max_mt={max_mt:.1f}%")

    # Apply cell filters
    sc.pp.filter_cells(adata_filtered, min_genes=min_genes)
    adata_filtered = adata_filtered[adata_filtered.obs.n_genes_by_counts < max_genes, :]
    adata_filtered = adata_filtered[adata_filtered.obs.pct_counts_mt < max_mt, :]

    # Apply gene filters
    min_cells = max(3, int(adata_filtered.n_obs * 0.0001))  # Genes in at least 0.01% of cells
    sc.pp.filter_genes(adata_filtered, min_cells=min_cells)

    print(f"Cells after filtering: {adata_filtered.n_obs}")
    print(f"Genes after filtering: {adata_filtered.n_vars}")
    print(f"Removed {adata_raw.n_obs - adata_filtered.n_obs:,} cells and {adata_raw.n_vars - adata_filtered.n_vars:,} genes")
    return (adata_filtered,)


@app.cell
def _(adata_filtered, plt, sc):
    # Preprocessing pipeline
    adata_processed = adata_filtered.copy()

    # Save raw data
    adata_processed.raw = adata_processed

    # Normalization and log transformation
    sc.pp.normalize_total(adata_processed, target_sum=1e4)
    sc.pp.log1p(adata_processed)

    # Find highly variable genes
    sc.pp.highly_variable_genes(adata_processed, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=3000)

    print(f"Found {adata_processed.var.highly_variable.sum()} highly variable genes")

    # Visualize highly variable genes
    hvg_data = adata_processed.var.copy()

    plt.figure(figsize=(10, 6))
    plt.scatter(hvg_data.loc[~hvg_data.highly_variable, 'means'], 
               hvg_data.loc[~hvg_data.highly_variable, 'dispersions_norm'], 
               c='lightgray', s=1, alpha=0.5, label='Not HVG')
    plt.scatter(hvg_data.loc[hvg_data.highly_variable, 'means'], 
               hvg_data.loc[hvg_data.highly_variable, 'dispersions_norm'], 
               c='red', s=1, alpha=0.7, label='Highly Variable')
    plt.xlabel('Mean Expression (log)')
    plt.ylabel('Normalized Dispersion')
    plt.title('Highly Variable Genes Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Keep only highly variable genes for downstream analysis
    adata_hvg = adata_processed[:, adata_processed.var.highly_variable].copy()
    print(f"Dataset for downstream analysis: {adata_hvg.shape}")
    return (adata_hvg,)


@app.cell
def _(adata_hvg, plt, sc):
    # Dimensionality reduction pipeline
    adata_dimred = adata_hvg.copy()

    # Scale data
    sc.pp.scale(adata_dimred, max_value=10)

    # Principal component analysis
    sc.tl.pca(adata_dimred, svd_solver='arpack')

    # Visualize PCA
    pca_coords = adata_dimred.obsm['X_pca']
    total_counts = adata_dimred.obs['total_counts']

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_coords[:, 0], pca_coords[:, 1], 
                         c=total_counts, s=0.5, alpha=0.6, cmap='viridis')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'PCA Plot (n={len(pca_coords):,} cells)')
    plt.colorbar(scatter, label='Total Counts')
    plt.grid(True, alpha=0.3)
    plt.show()

    # Compute neighborhood graph
    sc.pp.neighbors(adata_dimred, n_neighbors=10, n_pcs=40)

    # UMAP embedding
    sc.tl.umap(adata_dimred)

    print(" Dimensionality reduction complete")
    print(f"PCA: {adata_dimred.obsm['X_pca'].shape}")
    print(f"UMAP: {adata_dimred.obsm['X_umap'].shape}")
    return (adata_dimred,)


@app.cell
def _(adata_dimred, pl, sc):
    # Perform Leiden clustering
    adata_clustered = adata_dimred.copy()

    # Try different resolutions
    resolutions = [0.3, 0.5, 0.8]
    for res in resolutions:
        sc.tl.leiden(adata_clustered, resolution=res, key_added=f'leiden_res_{res}')

    # Use resolution 0.5 as default
    sc.tl.leiden(adata_clustered, resolution=0.5)

    # Calculate cluster statistics
    cluster_stats = []
    for cluster in adata_clustered.obs['leiden'].cat.categories:
        cluster_mask = adata_clustered.obs['leiden'] == cluster
        cluster_cells = adata_clustered[cluster_mask]
    
        stats = {
            'cluster': cluster,
            'n_cells': cluster_mask.sum(),
            'mean_counts': cluster_cells.obs['total_counts'].mean(),
            'mean_genes': cluster_cells.obs['n_genes_by_counts'].mean(),
            'mean_mt_pct': cluster_cells.obs['pct_counts_mt'].mean()
        }
        cluster_stats.append(stats)

    cluster_df = pl.DataFrame(cluster_stats)
    print("Cluster Statistics:")
    print(cluster_df)

    print(f" Identified {len(adata_clustered.obs['leiden'].cat.categories)} clusters")
    return (adata_clustered,)


@app.cell
def _(adata_clustered, np, plt):
    # Create comprehensive UMAP visualizations
    umap_coordinates = adata_clustered.obsm['X_umap']
    cluster_labels = adata_clustered.obs['leiden']
    cell_total_counts = adata_clustered.obs['total_counts']
    cell_n_genes = adata_clustered.obs['n_genes_by_counts']
    cell_pct_mt = adata_clustered.obs['pct_counts_mt']

    # Cluster plot
    plt.figure(figsize=(12, 10))
    unique_cluster_ids = cluster_labels.cat.categories
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_cluster_ids)))

    for i, cluster_id in enumerate(unique_cluster_ids):
        cluster_mask_umap = cluster_labels == cluster_id
        plt.scatter(umap_coordinates[cluster_mask_umap, 0], umap_coordinates[cluster_mask_umap, 1], 
                   c=[cluster_colors[i]], s=0.5, alpha=0.7, label=f'Cluster {cluster_id}')

    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title(f'UMAP Clustering (n={len(umap_coordinates):,} cells)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # QC metrics overlays
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Total counts
    scatter_counts = axes[0].scatter(umap_coordinates[:, 0], umap_coordinates[:, 1], 
                                    c=cell_total_counts, s=0.5, alpha=0.6, cmap='viridis')
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')
    axes[0].set_title('Total UMI Counts')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter_counts, ax=axes[0])

    # Number of genes
    scatter_genes = axes[1].scatter(umap_coordinates[:, 0], umap_coordinates[:, 1], 
                                   c=cell_n_genes, s=0.5, alpha=0.6, cmap='plasma')
    axes[1].set_xlabel('UMAP 1')
    axes[1].set_ylabel('UMAP 2')
    axes[1].set_title('Genes per Cell')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter_genes, ax=axes[1])

    # Mitochondrial percentage
    scatter_mito = axes[2].scatter(umap_coordinates[:, 0], umap_coordinates[:, 1], 
                                  c=cell_pct_mt, s=0.5, alpha=0.6, cmap='Reds')
    axes[2].set_xlabel('UMAP 1')
    axes[2].set_ylabel('UMAP 2')
    axes[2].set_title('% Mitochondrial Genes')
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(scatter_mito, ax=axes[2])

    plt.tight_layout()
    plt.show()
    return (umap_coordinates,)


@app.cell
def _(adata_clustered, pl, plt, sc, sns, umap_coordinates):
    # Find marker genes for each cluster
    sc.tl.rank_genes_groups(adata_clustered, 'leiden', method='wilcoxon')

    # CD8+ T cell markers to check
    cd8_marker_genes = ['CD8A', 'CD8B', 'CD3D', 'CD3E', 'GZMB', 'PRF1', 'IFNG', 'TNF', 'CCR7', 'IL7R']
    available_marker_genes = [gene for gene in cd8_marker_genes if gene in adata_clustered.var_names]

    print(f"Available CD8+ T cell markers: {available_marker_genes}")

    if len(available_marker_genes) > 0:
        # Create expression heatmap
        marker_expression_data = []
    
        for gene_name in available_marker_genes:
            gene_expression = adata_clustered[:, gene_name].X.toarray().flatten()
            for cell_idx, cluster_marker in enumerate(adata_clustered.obs['leiden']):
                marker_expression_data.append({
                    'gene': gene_name,
                    'cluster': cluster_marker,
                    'expression': gene_expression[cell_idx]
                })
    
        marker_expr_df = pl.DataFrame(marker_expression_data)
    
        # Calculate mean expression per cluster
        cluster_mean_expr = (marker_expr_df
                            .group_by(['gene', 'cluster'])
                            .agg(pl.col('expression').mean().alias('mean_expression'))
                            .sort(['gene', 'cluster']))
    
        # Create heatmap
        heatmap_pivot_data = cluster_mean_expr.to_pandas().pivot(index='gene', columns='cluster', values='mean_expression')
    
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_pivot_data, annot=True, cmap='Reds', fmt='.2f', 
                    cbar_kws={'label': 'Mean Expression'})
        plt.title('CD8+ T Cell Marker Expression by Cluster')
        plt.xlabel('Leiden Cluster')
        plt.ylabel('Gene')
        plt.tight_layout()
        plt.show()
    
        # Expression on UMAP for key markers
        if 'CD8A' in available_marker_genes:
            cd8a_expression = adata_clustered[:, 'CD8A'].X.toarray().flatten()
        
            plt.figure(figsize=(10, 8))
            cd8a_scatter = plt.scatter(umap_coordinates[:, 0], umap_coordinates[:, 1], 
                                      c=cd8a_expression, s=0.5, alpha=0.7, cmap='Reds')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.title('CD8A Expression on UMAP')
            plt.colorbar(cd8a_scatter, label='CD8A Expression')
            plt.grid(True, alpha=0.3)
            plt.show()
    
        if 'GZMB' in available_marker_genes:
            gzmb_expression = adata_clustered[:, 'GZMB'].X.toarray().flatten()
        
            plt.figure(figsize=(10, 8))
            gzmb_scatter = plt.scatter(umap_coordinates[:, 0], umap_coordinates[:, 1], 
                                      c=gzmb_expression, s=0.5, alpha=0.7, cmap='Purples')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.title('GZMB Expression on UMAP')
            plt.colorbar(gzmb_scatter, label='GZMB Expression')
            plt.grid(True, alpha=0.3)
            plt.show()

    else:
        print("No CD8+ markers found. Showing top marker genes per cluster:")
        # Show top marker genes using scanpy
        sc.pl.rank_genes_groups(adata_clustered, n_genes=5, sharey=False)
    return (available_marker_genes,)


@app.cell
def _(
    adata_clustered,
    adata_filtered,
    adata_hvg,
    adata_raw,
    available_marker_genes,
):
    # Final summary
    print(" ANALYSIS COMPLETE!")
    print("=" * 50)
    print(f" Original dataset: {adata_raw.shape}")
    print(f" After QC filtering: {adata_filtered.shape}")
    print(f"  Highly variable genes: {adata_hvg.shape}")
    print(f" Final clusters: {len(adata_clustered.obs['leiden'].cat.categories)}")
    print(f" Cells analyzed: {adata_clustered.n_obs:,}")

    print("\n Key Results:")
    print(f"• Identified {len(adata_clustered.obs['leiden'].cat.categories)} distinct cell populations")
    print(f"• Average UMI counts per cell: {adata_clustered.obs['total_counts'].mean():.0f}")
    print(f"• Average genes per cell: {adata_clustered.obs['n_genes_by_counts'].mean():.0f}")
    print(f"• Mitochondrial gene %: {adata_clustered.obs['pct_counts_mt'].mean():.1f}%")

    if len(available_marker_genes) > 0:
        print(f"• CD8+ T cell markers detected: {len(available_marker_genes)}")
        print(f"• Available markers: {', '.join(available_marker_genes)}")

    print("\n Modern Tools Used:")
    print("• Polars: Ultra-fast data manipulation")
    print("• Matplotlib: Publication-quality plots")
    print("• Marimo: Reactive notebook environment")
    print("• Scanpy: Single-cell analysis pipeline")

    print("\n For YouTube Tutorial:")
    print("• Beautiful visualizations for all 55K+ cells")
    print("• No sampling required - plot everything!")
    print("• Fast processing with modern Python stack")
    print("• Publication-ready figures")

    # Save final results
    final_results = {
        'n_cells_original': adata_raw.n_obs,
        'n_genes_original': adata_raw.n_vars,
        'n_cells_final': adata_clustered.n_obs,
        'n_genes_final': adata_clustered.n_vars,
        'n_clusters': len(adata_clustered.obs['leiden'].cat.categories),
        'mean_umi_counts': float(adata_clustered.obs['total_counts'].mean()),
        'mean_genes_per_cell': float(adata_clustered.obs['n_genes_by_counts'].mean()),
        'mean_mt_percent': float(adata_clustered.obs['pct_counts_mt'].mean())
    }

    print(f"\n Analysis summary: {final_results}")
    return


if __name__ == "__main__":
    app.run()
