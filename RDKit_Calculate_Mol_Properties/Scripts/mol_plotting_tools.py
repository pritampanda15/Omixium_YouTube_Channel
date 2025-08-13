import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats

class MolecularPropertyPlotter:
    def __init__(self, csv_file):
        """Initialize with molecular property CSV file"""
        self.df = pd.read_csv(csv_file)
        self.setup_style()
    
    def setup_style(self):
        """Set up plotting styles"""
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
    def lipinski_analysis(self, save_path=None):
        """Lipinski's Rule of Five analysis with violations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Lipinski's Rule of Five Analysis", fontsize=16, fontweight='bold')
        
        # Define Lipinski violations
        violations = {
            'MW_violation': self.df['MolWt'] > 500,
            'LogP_violation': self.df['ALogP'] > 5,
            'HBD_violation': self.df['HBD'] > 5,
            'HBA_violation': self.df['HBA'] > 10
        }
        
        # Plot 1: Molecular Weight
        axes[0,0].hist(self.df['MolWt'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].axvline(500, color='red', linestyle='--', linewidth=2, label='Lipinski limit (500)')
        axes[0,0].set_xlabel('Molecular Weight (Da)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Molecular Weight Distribution')
        axes[0,0].legend()
        
        # Plot 2: LogP
        axes[0,1].hist(self.df['ALogP'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0,1].axvline(5, color='red', linestyle='--', linewidth=2, label='Lipinski limit (5)')
        axes[0,1].set_xlabel('ALogP')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Lipophilicity Distribution')
        axes[0,1].legend()
        
        # Plot 3: HBD vs HBA
        scatter = axes[1,0].scatter(self.df['HBD'], self.df['HBA'], 
                                  c=self.df['MolWt'], cmap='viridis', alpha=0.6)
        axes[1,0].axhline(10, color='red', linestyle='--', alpha=0.7, label='HBA limit (10)')
        axes[1,0].axvline(5, color='red', linestyle='--', alpha=0.7, label='HBD limit (5)')
        axes[1,0].set_xlabel('Hydrogen Bond Donors')
        axes[1,0].set_ylabel('Hydrogen Bond Acceptors')
        axes[1,0].set_title('H-Bond Donors vs Acceptors')
        axes[1,0].legend()
        plt.colorbar(scatter, ax=axes[1,0], label='Molecular Weight')
        
        # Plot 4: Violation summary
        violation_counts = [sum(violations[v]) for v in violations.keys()]
        violation_labels = ['MW > 500', 'LogP > 5', 'HBD > 5', 'HBA > 10']
        bars = axes[1,1].bar(violation_labels, violation_counts, color=['red', 'orange', 'yellow', 'pink'])
        axes[1,1].set_ylabel('Number of Violations')
        axes[1,1].set_title('Lipinski Rule Violations')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # Add violation percentages on bars
        for bar, count in zip(bars, violation_counts):
            height = bar.get_height()
            percentage = (count / len(self.df)) * 100
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                          f'{count}\n({percentage:.1f}%)', 
                          ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return violations
    
    def chemical_space_plot(self, save_path=None):
        """2D chemical space visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # MW vs LogP (classic chemical space)
        scatter1 = axes[0].scatter(self.df['ALogP'], self.df['MolWt'], 
                                  c=self.df['TPSA'], cmap='plasma', 
                                  alpha=0.7, s=60)
        axes[0].set_xlabel('ALogP')
        axes[0].set_ylabel('Molecular Weight (Da)')
        axes[0].set_title('Chemical Space: MW vs LogP')
        plt.colorbar(scatter1, ax=axes[0], label='TPSA')
        
        # Add drug-like space boundaries
        axes[0].axhline(500, color='red', linestyle='--', alpha=0.5)
        axes[0].axvline(5, color='red', linestyle='--', alpha=0.5)
        axes[0].text(0.1, 480, 'Drug-like space', fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # TPSA vs LogP
        scatter2 = axes[1].scatter(self.df['ALogP'], self.df['TPSA'], 
                                  c=self.df['MolWt'], cmap='viridis', 
                                  alpha=0.7, s=60)
        axes[1].set_xlabel('ALogP')
        axes[1].set_ylabel('TPSA (Ų)')
        axes[1].set_title('Permeability Space: TPSA vs LogP')
        plt.colorbar(scatter2, ax=axes[1], label='Molecular Weight')
        
        # Add permeability guidelines
        axes[1].axhline(140, color='orange', linestyle='--', alpha=0.5, label='BBB permeability limit')
        axes[1].axhline(90, color='green', linestyle='--', alpha=0.5, label='Oral bioavailability limit')
        axes[1].legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def correlation_heatmap(self, save_path=None):
        """Correlation matrix heatmap"""
        numerical_cols = ['MolWt', 'TPSA', 'ALogP', 'HBD', 'HBA']
        corr_matrix = self.df[numerical_cols].corr()
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        heatmap = sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', 
                             center=0, square=True, linewidths=0.5, 
                             cbar_kws={"shrink": .8}, fmt='.3f')
        
        plt.title('Molecular Property Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def property_distributions(self, save_path=None):
        """Distribution plots for all properties"""
        numerical_cols = ['MolWt', 'TPSA', 'ALogP', 'HBD', 'HBA']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(numerical_cols):
            # Histogram with KDE
            sns.histplot(data=self.df, x=col, kde=True, ax=axes[i], alpha=0.7)
            
            # Add statistics
            mean_val = self.df[col].mean()
            median_val = self.df[col].median()
            axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            axes[i].axvline(median_val, color='orange', linestyle='--', label=f'Median: {median_val:.2f}')
            
            axes[i].set_title(f'{col} Distribution')
            axes[i].legend()
        
        # Remove empty subplot
        if len(numerical_cols) < len(axes):
            axes[-1].remove()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def interactive_plots(self):
        """Create interactive plots using Plotly"""
        # 3D scatter plot
        fig = px.scatter_3d(self.df, x='ALogP', y='MolWt', z='TPSA',
                           color='HBA', size='HBD',
                           hover_data=['Name'],
                           title='3D Chemical Space Visualization',
                           labels={'color': 'HBA Count'})
        fig.show()
        
        # Interactive correlation matrix
        numerical_cols = ['MolWt', 'TPSA', 'ALogP', 'HBD', 'HBA']
        corr_matrix = self.df[numerical_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                       text_auto=True, 
                       aspect="auto",
                       title="Interactive Correlation Matrix",
                       color_continuous_scale='RdBu_r')
        fig.show()
        
        # Parallel coordinates plot
        fig = px.parallel_coordinates(
            self.df.head(50),  # Limit for readability
            dimensions=['MolWt', 'TPSA', 'ALogP', 'HBD', 'HBA'],
            title="Parallel Coordinates Plot (First 50 compounds)"
        )
        fig.show()
    
    def compare_datasets(self, *other_csv_files, save_path=None):
        """Compare multiple datasets"""
        datasets = [self.df]
        labels = ['Dataset 1']
        
        for i, csv_file in enumerate(other_csv_files):
            df_other = pd.read_csv(csv_file)
            datasets.append(df_other)
            labels.append(f'Dataset {i+2}')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        properties = ['MolWt', 'ALogP', 'TPSA', 'HBD']
        
        for i, prop in enumerate(properties):
            ax = axes[i//2, i%2]
            
            for j, (dataset, label) in enumerate(zip(datasets, labels)):
                if prop in dataset.columns:
                    ax.hist(dataset[prop], alpha=0.6, label=label, bins=20)
            
            ax.set_xlabel(prop)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{prop} Distribution Comparison')
            ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def drug_like_filter_analysis(self, save_path=None):
        """Analyze drug-likeness with multiple filters"""
        # Define filters
        filters = {
            'Lipinski': (
                (self.df['MolWt'] <= 500) & 
                (self.df['ALogP'] <= 5) & 
                (self.df['HBD'] <= 5) & 
                (self.df['HBA'] <= 10)
            ),
            'Veber': (
                (self.df['TPSA'] <= 140) & 
                (self.df['HBD'] + self.df['HBA'] <= 12)
            ),
            'Lead-like': (
                (self.df['MolWt'] <= 350) & 
                (self.df['ALogP'] <= 3) & 
                (self.df['HBD'] <= 3) & 
                (self.df['HBA'] <= 6)
            ),
            'Fragment-like': (
                (self.df['MolWt'] <= 300) & 
                (self.df['ALogP'] <= 3) & 
                (self.df['HBD'] <= 3) & 
                (self.df['HBA'] <= 3)
            )
        }
        
        # Calculate pass rates
        filter_results = {}
        for name, condition in filters.items():
            pass_count = condition.sum()
            pass_rate = (pass_count / len(self.df)) * 100
            filter_results[name] = {'count': pass_count, 'percentage': pass_rate}
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot of pass counts
        names = list(filter_results.keys())
        counts = [filter_results[name]['count'] for name in names]
        percentages = [filter_results[name]['percentage'] for name in names]
        
        bars1 = ax1.bar(names, counts, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        ax1.set_ylabel('Number of Compounds')
        ax1.set_title('Compounds Passing Drug-likeness Filters')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add percentage labels on bars
        for bar, percentage in zip(bars1, percentages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{percentage:.1f}%', ha='center', va='bottom')
        
        # Pie chart
        ax2.pie(counts, labels=names, autopct='%1.1f%%', startangle=90,
               colors=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        ax2.set_title('Distribution of Filter Compliance')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return filter_results

# Usage example and utility functions
def analyze_molecular_properties(csv_file):
    """Complete analysis workflow"""
    plotter = MolecularPropertyPlotter(csv_file)
    
    print("Starting comprehensive molecular property analysis...")
    
    # Run all analyses
    print("\n1. Lipinski's Rule of Five Analysis")
    violations = plotter.lipinski_analysis()
    
    print("\n2. Chemical Space Visualization")
    plotter.chemical_space_plot()
    
    print("\n3. Property Correlations")
    plotter.correlation_heatmap()
    
    print("\n4. Property Distributions")
    plotter.property_distributions()
    
    print("\n5. Drug-likeness Filter Analysis")
    filter_results = plotter.drug_like_filter_analysis()
    
    print("\n6. Interactive Plots (if using Jupyter)")
    # plotter.interactive_plots()  # Uncomment for interactive plots
    
    return plotter, violations, filter_results

# Example usage:
if __name__ == "__main__":
    # For single dataset analysis
    csv_file = "AFAAMM.xaa.csv"  # Replace with your CSV file path
    plotter = MolecularPropertyPlotter(csv_file)
    
    # Generate all plots
    plotter.lipinski_analysis(save_path="lipinski_analysis.png")
    plotter.chemical_space_plot(save_path="chemical_space.png")
    plotter.correlation_heatmap(save_path="correlation_matrix.png")
    plotter.property_distributions(save_path="property_distributions.png")
    plotter.drug_like_filter_analysis(save_path="drug_likeness.png")
    
    # For comparing datasets A, B, C
    # plotter.compare_datasets("output_B/file.csv", "output_C/file.csv", 
    #                         save_path="dataset_comparison.png")
