import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import numpy as np

def create_interactive_plots(csv_file):
    """
    Create interactive molecular property plots using Plotly
    Perfect for exploring data dynamically in Jupyter notebooks or web browsers
    """
    df = pd.read_csv(csv_file)
    print(f"Creating interactive plots for {len(df)} compounds...")
    
    # Add drug-likeness categories
    df['Lipinski_Compliant'] = ((df['MolWt'] <= 500) & (df['ALogP'] <= 5) & 
                                (df['HBD'] <= 5) & (df['HBA'] <= 10))
    df['Lead_like'] = ((df['MolWt'] <= 350) & (df['ALogP'] <= 3) & 
                       (df['HBD'] <= 3) & (df['HBA'] <= 6))
    df['Fragment_like'] = ((df['MolWt'] <= 300) & (df['ALogP'] <= 3) & 
                          (df['HBD'] <= 3) & (df['HBA'] <= 3))
    
    # Create drug-likeness category
    def categorize_compound(row):
        if row['Fragment_like']:
            return 'Fragment-like'
        elif row['Lead_like']:
            return 'Lead-like'
        elif row['Lipinski_Compliant']:
            return 'Drug-like'
        else:
            return 'Non-drug-like'
    
    df['Drug_Category'] = df.apply(categorize_compound, axis=1)
    
    # 1. 3D Chemical Space Plot
    print("Creating 3D chemical space plot...")
    fig_3d = px.scatter_3d(df, x='ALogP', y='MolWt', z='TPSA',
                          color='Drug_Category',
                          size='HBA',
                          hover_data=['Name', 'HBD'],
                          title='3D Chemical Space Explorer',
                          labels={'ALogP': 'Lipophilicity (ALogP)',
                                 'MolWt': 'Molecular Weight (Da)',
                                 'TPSA': 'Topological Polar Surface Area (Ų)',
                                 'Drug_Category': 'Drug-likeness Category'},
                          color_discrete_map={
                              'Fragment-like': '#1f77b4',
                              'Lead-like': '#ff7f0e', 
                              'Drug-like': '#2ca02c',
                              'Non-drug-like': '#d62728'
                          })
    
    fig_3d.update_layout(
        scene=dict(
            xaxis_title='ALogP',
            yaxis_title='Molecular Weight (Da)',
            zaxis_title='TPSA (Ų)'
        ),
        width=900,
        height=700
    )
    fig_3d.show()
    
    # 2. Interactive Correlation Matrix
    print("Creating interactive correlation matrix...")
    props = ['MolWt', 'TPSA', 'ALogP', 'HBD', 'HBA']
    corr_matrix = df[props].corr()
    
    fig_corr = px.imshow(corr_matrix, 
                        text_auto=True, 
                        aspect="auto",
                        title="Interactive Molecular Property Correlation Matrix",
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1)
    fig_corr.show()
    
    # 3. Property Distribution Dashboard
    print("Creating property distribution dashboard...")
    fig_dist = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Molecular Weight', 'Lipophilicity (ALogP)', 'TPSA', 
                       'H-Bond Donors', 'H-Bond Acceptors', 'Drug Categories'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"type": "pie"}]]
    )
    
    # Molecular Weight histogram
    fig_dist.add_trace(go.Histogram(x=df['MolWt'], name='MW', nbinsx=30), row=1, col=1)
    fig_dist.add_vline(x=500, line_dash="dash", line_color="red", row=1, col=1)
    
    # ALogP histogram
    fig_dist.add_trace(go.Histogram(x=df['ALogP'], name='ALogP', nbinsx=30), row=1, col=2)
    fig_dist.add_vline(x=5, line_dash="dash", line_color="red", row=1, col=2)
    
    # TPSA histogram
    fig_dist.add_trace(go.Histogram(x=df['TPSA'], name='TPSA', nbinsx=30), row=1, col=3)
    fig_dist.add_vline(x=140, line_dash="dash", line_color="red", row=1, col=3)
    
    # HBD histogram
    fig_dist.add_trace(go.Histogram(x=df['HBD'], name='HBD'), row=2, col=1)
    
    # HBA histogram
    fig_dist.add_trace(go.Histogram(x=df['HBA'], name='HBA'), row=2, col=2)
    
    # Drug category pie chart
    category_counts = df['Drug_Category'].value_counts()
    fig_dist.add_trace(go.Pie(labels=category_counts.index, values=category_counts.values,
                             name="Drug Categories"), row=2, col=3)
    
    fig_dist.update_layout(height=800, title_text="Molecular Property Distribution Dashboard")
    fig_dist.show()
    
    # 4. Chemical Space with Filters
    print("Creating filterable chemical space plot...")
    fig_filter = px.scatter(df, x='ALogP', y='MolWt', 
                           color='Drug_Category',
                           size='TPSA',
                           hover_data=['Name', 'HBD', 'HBA'],
                           title='Interactive Chemical Space (MW vs LogP)',
                           labels={'ALogP': 'Lipophilicity (ALogP)',
                                  'MolWt': 'Molecular Weight (Da)'},
                           color_discrete_map={
                               'Fragment-like': '#1f77b4',
                               'Lead-like': '#ff7f0e', 
                               'Drug-like': '#2ca02c',
                               'Non-drug-like': '#d62728'
                           })
    
    # Add Lipinski boundaries
    fig_filter.add_hline(y=500, line_dash="dash", line_color="red", 
                        annotation_text="MW = 500 (Lipinski)")
    fig_filter.add_vline(x=5, line_dash="dash", line_color="red", 
                        annotation_text="LogP = 5 (Lipinski)")
    
    fig_filter.update_layout(width=900, height=600)
    fig_filter.show()
    
    # 5. Parallel Coordinates Plot
    print("Creating parallel coordinates plot...")
    # Normalize data for better visualization
    df_norm = df.copy()
    for col in ['MolWt', 'TPSA', 'ALogP', 'HBD', 'HBA']:
        df_norm[f'{col}_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    
    # Sample data for readability (parallel coordinates can get crowded)
    sample_size = min(200, len(df))
    df_sample = df_norm.sample(n=sample_size, random_state=42)
    
    fig_parallel = px.parallel_coordinates(
        df_sample,
        dimensions=['MolWt', 'TPSA', 'ALogP', 'HBD', 'HBA'],
        color='Drug_Category',
        title=f"Parallel Coordinates Plot (Sample of {sample_size} compounds)",
        color_discrete_map={
            'Fragment-like': '#1f77b4',
            'Lead-like': '#ff7f0e', 
            'Drug-like': '#2ca02c',
            'Non-drug-like': '#d62728'
        }
    )
    fig_parallel.show()
    
    # 6. Property vs Property Scatter Matrix
    print("Creating scatter plot matrix...")
    fig_matrix = px.scatter_matrix(df.sample(n=min(500, len(df)), random_state=42),
                                  dimensions=['MolWt', 'TPSA', 'ALogP', 'HBD', 'HBA'],
                                  color='Drug_Category',
                                  title="Molecular Property Scatter Matrix",
                                  color_discrete_map={
                                      'Fragment-like': '#1f77b4',
                                      'Lead-like': '#ff7f0e', 
                                      'Drug-like': '#2ca02c',
                                      'Non-drug-like': '#d62728'
                                  })
    fig_matrix.update_layout(height=800)
    fig_matrix.show()
    
    # 7. Animated Bubble Chart (if you have multiple datasets)
    print("Creating summary statistics table...")
    
    # Create summary statistics visualization
    summary_stats = df.groupby('Drug_Category').agg({
        'MolWt': ['count', 'mean', 'std'],
        'ALogP': ['mean', 'std'],
        'TPSA': ['mean', 'std'],
        'HBD': ['mean', 'std'],
        'HBA': ['mean', 'std']
    }).round(2)
    
    print("\nSUMMARY STATISTICS BY DRUG CATEGORY:")
    print("="*60)
    for category in df['Drug_Category'].unique():
        subset = df[df['Drug_Category'] == category]
        print(f"\n{category.upper()}:")
        print(f"  Count: {len(subset)}")
        print(f"  MW: {subset['MolWt'].mean():.1f} ± {subset['MolWt'].std():.1f}")
        print(f"  LogP: {subset['ALogP'].mean():.2f} ± {subset['ALogP'].std():.2f}")
        print(f"  TPSA: {subset['TPSA'].mean():.1f} ± {subset['TPSA'].std():.1f}")
    
    return df

def create_comparison_dashboard(csv_files, labels=None):
    """Create interactive comparison dashboard for multiple datasets"""
    if labels is None:
        labels = [f"Dataset_{i+1}" for i in range(len(csv_files))]
    
    # Load all datasets
    all_data = []
    for csv_file, label in zip(csv_files, labels):
        df = pd.read_csv(csv_file)
        df['Dataset'] = label
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Create comparison plots
    print("Creating multi-dataset comparison dashboard...")
    
    # 1. Property distribution comparison
    fig_comp = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Molecular Weight', 'ALogP', 'TPSA', 
                       'H-Bond Donors', 'H-Bond Acceptors', 'Dataset Sizes')
    )
    
    properties = ['MolWt', 'ALogP', 'TPSA', 'HBD', 'HBA']
    
    for i, prop in enumerate(properties):
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        for dataset in labels:
            subset = combined_df[combined_df['Dataset'] == dataset]
            fig_comp.add_trace(
                go.Histogram(x=subset[prop], name=f"{dataset}_{prop}", 
                           opacity=0.7, nbinsx=20),
                row=row, col=col
            )
    
    # Dataset size comparison
    dataset_sizes = combined_df['Dataset'].value_counts()
    fig_comp.add_trace(
        go.Bar(x=dataset_sizes.index, y=dataset_sizes.values, name="Dataset Sizes"),
        row=2, col=3
    )
    
    fig_comp.update_layout(height=800, title_text="Multi-Dataset Comparison Dashboard")
    fig_comp.show()
    
    # 2. 3D comparison plot
    fig_3d_comp = px.scatter_3d(combined_df, x='ALogP', y='MolWt', z='TPSA',
                               color='Dataset',
                               hover_data=['Name'],
                               title='3D Chemical Space Comparison')
    fig_3d_comp.show()
    
    return combined_df

# Example usage and utility functions
def save_interactive_plots_as_html(csv_file, output_dir="interactive_plots"):
    """Save all interactive plots as HTML files for sharing"""
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df = create_interactive_plots(csv_file)
    print(f"Interactive plots would be saved to {output_dir}/ directory")
    print("Use fig.write_html('filename.html') to save individual plots")
    
    return df

# Example usage
if __name__ == "__main__":
    # Single dataset interactive analysis
    csv_file = "AFAAMM.xaa.csv"  # Replace with your actual file
    df = create_interactive_plots(csv_file)
    
    # Multi-dataset comparison
    # csv_files = ["output_A.csv", "output_B.csv", "output_C.csv"]
    # labels = ["Tranche A", "Tranche B", "Tranche C"]
    # combined_df = create_comparison_dashboard(csv_files, labels)
