import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def quick_molecular_analysis(csv_file, output_prefix="mol_analysis"):
    """
    Quick and essential molecular property analysis
    Perfect for immediate insights into your compound library
    """
    # Load data
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} compounds from {csv_file}")
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("Set2")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Lipinski's Rule of Five Radar Plot
    ax1 = plt.subplot(3, 3, 1)
    lipinski_violations = []
    lipinski_violations.append(sum(df['MolWt'] > 500))
    lipinski_violations.append(sum(df['ALogP'] > 5))
    lipinski_violations.append(sum(df['HBD'] > 5))
    lipinski_violations.append(sum(df['HBA'] > 10))
    
    labels = ['MW>500', 'LogP>5', 'HBD>5', 'HBA>10']
    x = range(len(labels))
    bars = ax1.bar(x, lipinski_violations, color=['red', 'orange', 'yellow', 'pink'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Violations')
    ax1.set_title('Lipinski Rule Violations')
    
    # Add percentages
    for bar, count in zip(bars, lipinski_violations):
        height = bar.get_height()
        percentage = (count / len(df)) * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{percentage:.1f}%', ha='center', va='bottom')
    
    # 2. MW vs LogP (Drug-like space)
    ax2 = plt.subplot(3, 3, 2)
    scatter = ax2.scatter(df['ALogP'], df['MolWt'], c=df['TPSA'], 
                         cmap='viridis', alpha=0.6, s=50)
    ax2.axhline(500, color='red', linestyle='--', alpha=0.7, label='MW limit')
    ax2.axvline(5, color='red', linestyle='--', alpha=0.7, label='LogP limit')
    ax2.set_xlabel('ALogP')
    ax2.set_ylabel('Molecular Weight')
    ax2.set_title('Chemical Space (MW vs LogP)')
    ax2.legend()
    plt.colorbar(scatter, ax=ax2, label='TPSA')
    
    # 3. TPSA Distribution
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(df['TPSA'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(140, color='red', linestyle='--', label='BBB limit (140)')
    ax3.axvline(90, color='orange', linestyle='--', label='Oral limit (90)')
    ax3.set_xlabel('TPSA (Ų)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Topological Polar Surface Area')
    ax3.legend()
    
    # 4. Molecular Weight Distribution
    ax4 = plt.subplot(3, 3, 4)
    ax4.hist(df['MolWt'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax4.axvline(500, color='red', linestyle='--', label='Lipinski limit')
    ax4.axvline(df['MolWt'].mean(), color='blue', linestyle=':', label=f'Mean: {df["MolWt"].mean():.1f}')
    ax4.set_xlabel('Molecular Weight (Da)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Molecular Weight Distribution')
    ax4.legend()
    
    # 5. LogP Distribution
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(df['ALogP'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax5.axvline(5, color='red', linestyle='--', label='Lipinski limit')
    ax5.axvline(df['ALogP'].mean(), color='blue', linestyle=':', label=f'Mean: {df["ALogP"].mean():.2f}')
    ax5.set_xlabel('ALogP')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Lipophilicity Distribution')
    ax5.legend()
    
    # 6. H-Bond Donors vs Acceptors
    ax6 = plt.subplot(3, 3, 6)
    scatter2 = ax6.scatter(df['HBD'], df['HBA'], c=df['MolWt'], 
                          cmap='plasma', alpha=0.6, s=50)
    ax6.axhline(10, color='red', linestyle='--', alpha=0.7, label='HBA limit')
    ax6.axvline(5, color='red', linestyle='--', alpha=0.7, label='HBD limit')
    ax6.set_xlabel('Hydrogen Bond Donors')
    ax6.set_ylabel('Hydrogen Bond Acceptors')
    ax6.set_title('H-Bond Profile')
    ax6.legend()
    plt.colorbar(scatter2, ax=ax6, label='MW')
    
    # 7. Property Correlation Matrix (simplified)
    ax7 = plt.subplot(3, 3, 7)
    props = ['MolWt', 'TPSA', 'ALogP', 'HBD', 'HBA']
    corr_matrix = df[props].corr()
    im = ax7.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax7.set_xticks(range(len(props)))
    ax7.set_yticks(range(len(props)))
    ax7.set_xticklabels(props, rotation=45)
    ax7.set_yticklabels(props)
    ax7.set_title('Property Correlations')
    
    # Add correlation values
    for i in range(len(props)):
        for j in range(len(props)):
            text = ax7.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax7)
    
    # 8. Drug-likeness Summary
    ax8 = plt.subplot(3, 3, 8)
    
    # Calculate drug-like compounds
    lipinski_pass = ((df['MolWt'] <= 500) & (df['ALogP'] <= 5) & 
                     (df['HBD'] <= 5) & (df['HBA'] <= 10))
    lead_like = ((df['MolWt'] <= 350) & (df['ALogP'] <= 3) & 
                 (df['HBD'] <= 3) & (df['HBA'] <= 6))
    fragment_like = ((df['MolWt'] <= 300) & (df['ALogP'] <= 3) & 
                     (df['HBD'] <= 3) & (df['HBA'] <= 3))
    
    categories = ['All', 'Lipinski', 'Lead-like', 'Fragment']
    counts = [len(df), lipinski_pass.sum(), lead_like.sum(), fragment_like.sum()]
    
    bars = ax8.bar(categories, counts, color=['gray', 'green', 'blue', 'orange'])
    ax8.set_ylabel('Number of Compounds')
    ax8.set_title('Drug-likeness Categories')
    
    # Add percentages
    for bar, count in zip(bars, counts):
        if count > 0:
            percentage = (count / len(df)) * 100
            ax8.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{percentage:.1f}%', ha='center', va='bottom')
    
    # 9. Summary Statistics Table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('tight')
    ax9.axis('off')
    
    # Create summary statistics
    summary_stats = df[props].describe().round(2)
    table_data = []
    for prop in props:
        table_data.append([
            prop,
            f"{summary_stats.loc['mean', prop]:.2f}",
            f"{summary_stats.loc['std', prop]:.2f}",
            f"{summary_stats.loc['min', prop]:.2f}",
            f"{summary_stats.loc['max', prop]:.2f}"
        ])
    
    table = ax9.table(cellText=table_data,
                     colLabels=['Property', 'Mean', 'Std', 'Min', 'Max'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax9.set_title('Summary Statistics')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = f"{output_prefix}_comprehensive.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comprehensive analysis saved as: {output_file}")
    plt.show()
    
    # Print summary
    print("\n" + "="*50)
    print("MOLECULAR PROPERTY SUMMARY")
    print("="*50)
    print(f"Total compounds: {len(df)}")
    print(f"Lipinski compliant: {lipinski_pass.sum()} ({lipinski_pass.sum()/len(df)*100:.1f}%)")
    print(f"Lead-like: {lead_like.sum()} ({lead_like.sum()/len(df)*100:.1f}%)")
    print(f"Fragment-like: {fragment_like.sum()} ({fragment_like.sum()/len(df)*100:.1f}%)")
    print("\nProperty ranges:")
    for prop in props:
        print(f"  {prop}: {df[prop].min():.2f} - {df[prop].max():.2f} (mean: {df[prop].mean():.2f})")
    
    return df

def compare_datasets_quick(csv_files, labels=None, output_prefix="comparison"):
    """Quick comparison of multiple datasets"""
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(csv_files))]
    
    datasets = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        datasets.append(df)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    properties = ['MolWt', 'ALogP', 'TPSA', 'HBD', 'HBA']
    colors = plt.cm.Set3(np.linspace(0, 1, len(datasets)))
    
    for i, prop in enumerate(properties):
        ax = axes[i]
        for j, (dataset, label, color) in enumerate(zip(datasets, labels, colors)):
            if prop in dataset.columns:
                ax.hist(dataset[prop], alpha=0.6, label=label, bins=20, color=color)
        
        ax.set_xlabel(prop)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{prop} Distribution Comparison')
        ax.legend()
    
    # Summary comparison table
    ax = axes[5]
    ax.axis('tight')
    ax.axis('off')
    
    # Create comparison table
    comparison_data = []
    for i, (dataset, label) in enumerate(zip(datasets, labels)):
        lipinski_pass = ((dataset['MolWt'] <= 500) & (dataset['ALogP'] <= 5) & 
                        (dataset['HBD'] <= 5) & (dataset['HBA'] <= 10))
        comparison_data.append([
            label,
            len(dataset),
            f"{lipinski_pass.sum()}",
            f"{lipinski_pass.sum()/len(dataset)*100:.1f}%",
            f"{dataset['MolWt'].mean():.1f}",
            f"{dataset['ALogP'].mean():.2f}"
        ])
    
    table = ax.table(cellText=comparison_data,
                    colLabels=['Dataset', 'Count', 'Lipinski Pass', 'Pass %', 'Avg MW', 'Avg LogP'],
                    cellLoc='center',
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('Dataset Comparison Summary')
    
    plt.tight_layout()
    output_file = f"{output_prefix}_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Dataset comparison saved as: {output_file}")
    plt.show()

# Example usage
if __name__ == "__main__":
    # Single dataset analysis
    csv_file = "AFAAMM.xaa.csv"  # Replace with your actual file path
    df = quick_molecular_analysis(csv_file)
    
    # Compare multiple datasets
    # csv_files = ["output_A.csv", "output_B.csv", "output_C.csv"]
    # labels = ["Tranche A", "Tranche B", "Tranche C"]
    # compare_datasets_quick(csv_files, labels)
