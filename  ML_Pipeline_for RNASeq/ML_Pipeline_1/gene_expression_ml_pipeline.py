#!/usr/bin/env python3
"""
Comprehensive Machine Learning Pipeline for Gene Expression Analysis
==================================================================

This pipeline handles gene expression datasets for predicting disease/phenotype/drug response/survival.
Includes preprocessing, multiple ML algorithms, evaluation metrics, and biological interpretation.

Author: Bioinformatics ML Pipeline
Date: 2025
"""
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, RFE
)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, 
    confusion_matrix, roc_curve, precision_recall_curve
)
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class GeneExpressionMLPipeline:
    """
    Comprehensive ML pipeline for gene expression analysis
    """
    
    def __init__(self, random_state=RANDOM_SEED):
        self.random_state = random_state
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        self.models = {}
        self.results = {}
        self.feature_names = None
        self.selected_features = None
        
    def load_and_inspect_data(self, file_path, target_column, sep='\t'):
        """
        Load gene expression dataset and perform initial inspection
        
        Parameters:
        -----------
        file_path : str
            Path to the gene expression file
        target_column : str
            Name of the target column (phenotype/outcome)
        sep : str
            File separator (default: '\t' for TSV files)
        """
        print("=== Loading and Inspecting Dataset ===")
        
        # Load data
        self.data = pd.read_csv(file_path, sep=sep, index_col=0)
        
        # Separate features and target
        if target_column in self.data.columns:
            self.y = self.data[target_column]
            self.X = self.data.drop(columns=[target_column])
        else:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        self.feature_names = self.X.columns.tolist()
        
        # Basic dataset info
        print(f"Dataset shape: {self.X.shape}")
        print(f"Number of samples: {self.X.shape[0]}")
        print(f"Number of genes/features: {self.X.shape[1]}")
        print(f"Target distribution:\n{self.y.value_counts()}")
        print(f"Missing values in features: {self.X.isnull().sum().sum()}")
        print(f"Missing values in target: {self.y.isnull().sum()}")
        
        return self
    
    def preprocess_data(self, 
                       log_transform=True, 
                       variance_threshold=0.01, 
                       k_best_features=5000,
                       apply_pca=False, 
                       pca_components=0.95):
        """
        Comprehensive preprocessing for gene expression data
        
        Parameters:
        -----------
        log_transform : bool
            Apply log2(x + 1) transformation
        variance_threshold : float
            Remove features with variance below threshold
        k_best_features : int
            Number of top features to select using univariate tests
        apply_pca : bool
            Apply PCA for dimensionality reduction
        pca_components : int or float
            Number of PCA components (if int) or variance explained (if float)
        """
        print("\n=== Data Preprocessing ===")
        
        # Handle missing values
        if self.X.isnull().sum().sum() > 0:
            print("Handling missing values...")
            self.X = self.X.fillna(self.X.median())
        
        # Remove samples with missing target
        if self.y.isnull().sum() > 0:
            mask = ~self.y.isnull()
            self.X = self.X[mask]
            self.y = self.y[mask]
            print(f"Removed {(~mask).sum()} samples with missing target")
        
        # Log transformation (common for gene expression data)
        if log_transform:
            print("Applying log2(x + 1) transformation...")
            self.X = np.log2(self.X + 1)
        
        # Remove low variance features
        print(f"Removing features with variance < {variance_threshold}...")
        var_selector = VarianceThreshold(threshold=variance_threshold)
        self.X = pd.DataFrame(
            var_selector.fit_transform(self.X),
            index=self.X.index,
            columns=self.X.columns[var_selector.get_support()]
        )
        print(f"Features after variance filtering: {self.X.shape[1]}")
        
        # Feature selection using univariate statistical tests
        if k_best_features and k_best_features < self.X.shape[1]:
            print(f"Selecting top {k_best_features} features using univariate tests...")
            selector = SelectKBest(score_func=f_classif, k=k_best_features)
            self.X = pd.DataFrame(
                selector.fit_transform(self.X, self.y),
                index=self.X.index,
                columns=self.X.columns[selector.get_support()]
            )
            self.selected_features = self.X.columns.tolist()
            print(f"Features after selection: {self.X.shape[1]}")
        
        # Normalization/Standardization
        print("Applying robust scaling...")
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        X_scaled = self.scaler.fit_transform(self.X)
        self.X = pd.DataFrame(X_scaled, index=self.X.index, columns=self.X.columns)
        
        # PCA (optional)
        if apply_pca:
            print(f"Applying PCA (components: {pca_components})...")
            self.pca = PCA(n_components=pca_components, random_state=self.random_state)
            X_pca = self.pca.fit_transform(self.X)
            
            if isinstance(pca_components, float):
                n_components = X_pca.shape[1]
            else:
                n_components = pca_components
                
            pca_columns = [f'PC{i+1}' for i in range(n_components)]
            self.X = pd.DataFrame(X_pca, index=self.X.index, columns=pca_columns)
            print(f"PCA components: {self.X.shape[1]}")
            if hasattr(self.pca, 'explained_variance_ratio_'):
                print(f"Explained variance: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        return self
    
    def split_data(self, test_size=0.2, stratify=True):
        """
        Split data into training and testing sets
        """
        print(f"\n=== Data Splitting ===")
        
        stratify_y = self.y if stratify else None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=stratify_y
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Training target distribution:\n{self.y_train.value_counts()}")
        print(f"Test target distribution:\n{self.y_test.value_counts()}")
        
        return self
    
    def apply_pca_for_visualization(self, n_components=2):
        """
        Apply PCA properly - fit on training data only, transform both train and test
        This avoids data leakage and provides clean data for visualization
        """
        print(f"\n=== Applying PCA for Visualization (n_components={n_components}) ===")
        
        if not hasattr(self, 'X_train') or self.X_train is None:
            raise ValueError("Data must be split first. Call split_data() before apply_pca_for_visualization()")
        
        # Fit PCA only on training data
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        
        # Convert DataFrames to numpy arrays if needed
        X_train_array = self.X_train.values if hasattr(self.X_train, 'values') else self.X_train
        X_test_array = self.X_test.values if hasattr(self.X_test, 'values') else self.X_test
        
        # Fit on training data and transform both sets
        self.X_train_pca = self.pca.fit_transform(X_train_array)
        self.X_test_pca = self.pca.transform(X_test_array)
        
        print(f"PCA applied successfully!")
        print(f"Training data PCA shape: {self.X_train_pca.shape}")
        print(f"Test data PCA shape: {self.X_test_pca.shape}")
        
        if hasattr(self.pca, 'explained_variance_ratio_'):
            total_variance = self.pca.explained_variance_ratio_.sum()
            print(f"Total explained variance: {total_variance:.3f}")
            for i, var_ratio in enumerate(self.pca.explained_variance_ratio_):
                print(f"PC{i+1} explains {var_ratio:.3f} of variance")
        
        return self
    
    def define_models(self):
        """
        Define machine learning models with initial parameters
        """
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'SVM': SVC(
                probability=True,
                random_state=self.random_state
            ),
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'XGBoost': xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss'
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=self.random_state,
                max_iter=500
            )
        }
        
        return self
    
    def train_and_evaluate_models(self, cv_folds=5):
        """
        Train models and evaluate performance using cross-validation
        """
        print(f"\n=== Model Training and Evaluation ===")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train, 
                cv=cv, scoring='roc_auc', n_jobs=-1
            )
            
            # Fit model on full training data
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            self.results[name] = {
                'model': model,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': accuracy,
                'test_roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"CV ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
            print(f"Test Accuracy: {accuracy:.3f}")
            print(f"Test ROC-AUC: {roc_auc:.3f}")
        
        return self
    
    def hyperparameter_optimization(self, models_to_optimize=None):
        """
        Optimize hyperparameters for selected models
        """
        print(f"\n=== Hyperparameter Optimization ===")
        
        if models_to_optimize is None:
            models_to_optimize = ['Random Forest', 'XGBoost']
        
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        for name in models_to_optimize:
            if name in self.models and name in param_grids:
                print(f"\nOptimizing {name}...")
                
                grid_search = GridSearchCV(
                    self.models[name],
                    param_grids[name],
                    cv=cv,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )
                
                grid_search.fit(self.X_train, self.y_train)
                
                # Update model with best parameters
                self.models[name] = grid_search.best_estimator_
                
                # Re-evaluate with optimized model
                y_pred = grid_search.predict(self.X_test)
                y_pred_proba = grid_search.predict_proba(self.X_test)[:, 1]
                
                self.results[name].update({
                    'model': grid_search.best_estimator_,
                    'best_params': grid_search.best_params_,
                    'best_cv_score': grid_search.best_score_,
                    'test_accuracy': accuracy_score(self.y_test, y_pred),
                    'test_roc_auc': roc_auc_score(self.y_test, y_pred_proba),
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                })
                
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best CV ROC-AUC: {grid_search.best_score_:.3f}")
                print(f"Optimized Test ROC-AUC: {roc_auc_score(self.y_test, y_pred_proba):.3f}")
        
        return self
    
    def visualize_results(self):
        """
        Create comprehensive visualizations
        """
        print(f"\n=== Creating Visualizations ===")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Model Comparison
        plt.subplot(2, 4, 1)
        model_names = list(self.results.keys())
        cv_means = [self.results[name]['cv_mean'] for name in model_names]
        cv_stds = [self.results[name]['cv_std'] for name in model_names]
        
        bars = plt.bar(model_names, cv_means, yerr=cv_stds, capsize=5)
        plt.title('Cross-Validation ROC-AUC Scores')
        plt.ylabel('ROC-AUC')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, cv_means):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{mean_val:.3f}', ha='center', va='bottom')
        
        # 2. ROC Curves
        plt.subplot(2, 4, 2)
        for name in model_names:
            y_pred_proba = self.results[name]['y_pred_proba']
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc_score = self.results[name]['test_roc_auc']
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Confusion Matrix (Best Model)
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['test_roc_auc'])
        plt.subplot(2, 4, 3)
        cm = confusion_matrix(self.y_test, self.results[best_model_name]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 4. Feature Importance (for tree-based models)
        plt.subplot(2, 4, 4)
        if 'Random Forest' in self.results:
            rf_model = self.results['Random Forest']['model']
            if hasattr(rf_model, 'feature_importances_'):
                importances = rf_model.feature_importances_
                indices = np.argsort(importances)[::-1][:20]  # Top 20 features
                
                plt.barh(range(len(indices)), importances[indices])
                plt.yticks(range(len(indices)), [self.X.columns[i] for i in indices])
                plt.title('Top 20 Feature Importances (Random Forest)')
                plt.xlabel('Importance')
        
        # 5. PCA Plot (if PCA was applied)
        plt.subplot(2, 4, 5)
        if self.pca is not None:
            # Transform test data for visualization
            X_test_pca = self.pca.transform(self.scaler.transform(
                pd.DataFrame(self.X_test, columns=self.X.columns)
            ))
            
            for label in self.y_test.unique():
                mask = self.y_test == label
                plt.scatter(X_test_pca[mask, 0], X_test_pca[mask, 1], 
                          label=f'Class {label}', alpha=0.6)
            
            plt.xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)')
            plt.ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.title('PCA Visualization')
            plt.legend()
        else:
            # If no PCA, show first two features
            for label in self.y_test.unique():
                mask = self.y_test == label
                plt.scatter(self.X_test.iloc[mask, 0], self.X_test.iloc[mask, 1], 
                          label=f'Class {label}', alpha=0.6)
            
            plt.xlabel(self.X.columns[0])
            plt.ylabel(self.X.columns[1])
            plt.title('Feature Space Visualization')
            plt.legend()
        
        # 6. Precision-Recall Curves
        plt.subplot(2, 4, 6)
        for name in model_names:
            y_pred_proba = self.results[name]['y_pred_proba']
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
            plt.plot(recall, precision, label=name)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        
        # 7. Learning Curves (for best model)
        plt.subplot(2, 4, 7)
        best_model = self.results[best_model_name]['model']
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        from sklearn.model_selection import learning_curve
        train_sizes, train_scores, val_scores = learning_curve(
            best_model, self.X_train, self.y_train, 
            train_sizes=train_sizes, cv=3, scoring='roc_auc'
        )
        
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training')
        plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation')
        plt.fill_between(train_sizes, 
                        np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                        np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                        alpha=0.1)
        plt.fill_between(train_sizes, 
                        np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                        np.mean(val_scores, axis=1) + np.std(val_scores, axis=1),
                        alpha=0.1)
        
        plt.xlabel('Training Set Size')
        plt.ylabel('ROC-AUC')
        plt.title(f'Learning Curves - {best_model_name}')
        plt.legend()
        
        # 8. Model Performance Summary
        plt.subplot(2, 4, 8)
        metrics_data = []
        for name in model_names:
            metrics_data.append([
                self.results[name]['test_accuracy'],
                self.results[name]['test_roc_auc']
            ])
        
        metrics_df = pd.DataFrame(metrics_data, 
                                index=model_names, 
                                columns=['Accuracy', 'ROC-AUC'])
        sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5)
        plt.title('Model Performance Heatmap')
        
        plt.tight_layout()
        # Create results directory
        if not os.path.exists('results'):
            os.makedirs('results')
            print(" Created 'results' folder")

        # Generate timestamp  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save PNG
        plot_filename = f'results/gene_expression_analysis_{timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f" Plot saved as: {plot_filename}")

        # Save PDF  
        pdf_filename = f'results/gene_expression_analysis_{timestamp}.pdf'
        plt.savefig(pdf_filename, bbox_inches='tight', facecolor='white')
        print(f"📄 PDF saved as: {pdf_filename}")

        # Now show
        plt.show()
        
        
        return self
    
    def generate_biological_interpretation(self, top_n_genes=20):
        """
        Generate biological interpretation of top predictive genes
        """
        print(f"\n=== Biological Interpretation ===")
        
        # Get feature importance from Random Forest (if available)
        if 'Random Forest' in self.results:
            rf_model = self.results['Random Forest']['model']
            if hasattr(rf_model, 'feature_importances_'):
                importances = rf_model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'Gene': self.X.columns,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                print(f"Top {top_n_genes} most important genes:")
                print("=" * 50)
                
                top_genes = feature_importance_df.head(top_n_genes)
                for idx, (_, row) in enumerate(top_genes.iterrows(), 1):
                    print(f"{idx:2d}. {row['Gene']:15s} (Importance: {row['Importance']:.4f})")
                
                # Save to file for further analysis
                feature_importance_df.to_csv('top_predictive_genes.csv', index=False)
                print(f"\nAll feature importances saved to 'top_predictive_genes.csv'")
                
                # Store the results but return self for method chaining
                self.top_genes = top_genes
                return self  # ✅ FIXED: Return self instead of top_genes
        
        return self  # ✅ FIXED: Always return self
    
    def print_detailed_results(self):
        """
        Print comprehensive results summary
        """
        print(f"\n" + "="*60)
        print("COMPREHENSIVE RESULTS SUMMARY")
        print("="*60)
        
        # Dataset info
        print(f"\nDataset Information:")
        print(f"- Samples: {self.X.shape[0]}")
        print(f"- Features after preprocessing: {self.X.shape[1]}")
        print(f"- Classes: {list(self.y.unique())}")
        
        # Model performance
        print(f"\nModel Performance Summary:")
        print(f"{'Model':<20} {'CV ROC-AUC':<12} {'Test ROC-AUC':<13} {'Test Accuracy':<13}")
        print("-" * 60)
        
        for name, results in self.results.items():
            cv_score = f"{results['cv_mean']:.3f}±{results['cv_std']:.3f}"
            test_roc = f"{results['test_roc_auc']:.3f}"
            test_acc = f"{results['test_accuracy']:.3f}"
            print(f"{name:<20} {cv_score:<12} {test_roc:<13} {test_acc:<13}")
        
        # Best model
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['test_roc_auc'])
        print(f"\nBest performing model: {best_model_name}")
        print(f"Best ROC-AUC: {self.results[best_model_name]['test_roc_auc']:.3f}")
        
        # Detailed classification report for best model
        print(f"\nDetailed Classification Report for {best_model_name}:")
        print("-" * 50)
        y_pred = self.results[best_model_name]['y_pred']
        y_test_array = self.y_test.values if hasattr(self.y_test, 'values') else self.y_test
        print(classification_report(y_test_array, y_pred))


def main():
    """
    Main function to run the complete pipeline
    
    Usage example with your own data:
    """
    
    # Initialize pipeline
    pipeline = GeneExpressionMLPipeline(random_state=42)
    
    # Example usage (uncomment and modify for your data):
    """
    pipeline = (pipeline
                .load_and_inspect_data('your_gene_expression_data.tsv', 'disease_status')
                .preprocess_data(
                    log_transform=True,
                    variance_threshold=0.01,
                    k_best_features=5000,
                    apply_pca=False  # Don't use PCA in preprocessing for better ML performance
                )
                .split_data(test_size=0.2, stratify=True)
                .apply_pca_for_visualization(n_components=2)  # Apply PCA properly after splitting
                .define_models()
                .train_and_evaluate_models(cv_folds=5)
                .hyperparameter_optimization(['Random Forest', 'XGBoost'])
                .visualize_results()  # Now works without indexing errors!
                .generate_biological_interpretation(top_n_genes=20)
                .print_detailed_results())
    """
    
    # Create sample data for demonstration
    print("Creating sample dataset for demonstration...")
    
    # Generate synthetic gene expression data
    n_samples = 200
    n_genes = 1000
    
    np.random.seed(42)
    
    # Create correlated gene groups (simulating biological pathways)
    gene_data = []
    gene_names = []
    
    for pathway in range(10):  # 10 pathways
        pathway_genes = np.random.multivariate_normal(
            mean=np.zeros(20),  # 20 genes per pathway
            cov=np.eye(20) * 0.5 + np.ones((20, 20)) * 0.3,  # Correlated genes
            size=n_samples
        )
        gene_data.append(pathway_genes)
        gene_names.extend([f'Gene_P{pathway}_{i}' for i in range(20)])
    
    # Add random genes
    random_genes = np.random.normal(0, 1, (n_samples, n_genes - 200))
    gene_data.append(random_genes)
    gene_names.extend([f'Gene_R{i}' for i in range(n_genes - 200)])
    
    X_synthetic = np.concatenate(gene_data, axis=1)
    
    # Make expression values positive (typical for gene expression)
    X_synthetic = np.exp(X_synthetic)
    
    # Create phenotype based on first few pathways (simulating disease-related genes)
    disease_signal = (X_synthetic[:, :60].mean(axis=1) + 
                     np.random.normal(0, 0.5, n_samples))
    y_synthetic = (disease_signal > np.median(disease_signal)).astype(int)
    
    # Create DataFrame
    df_synthetic = pd.DataFrame(X_synthetic, columns=gene_names)
    df_synthetic['disease_status'] = y_synthetic
    
    # Save synthetic data
    df_synthetic.to_csv('synthetic_gene_expression_data.csv', index=False)
    print("Synthetic dataset saved as 'synthetic_gene_expression_data.csv'")
    
    # Run pipeline on synthetic data
    pipeline = (pipeline
                .load_and_inspect_data('synthetic_gene_expression_data.csv', 'disease_status', sep=',')
                .preprocess_data(
                    log_transform=True,
                    variance_threshold=0.01,
                    k_best_features=500,
                    apply_pca=False
                )
                .split_data(test_size=0.2, stratify=True)
                .define_models()
                .train_and_evaluate_models(cv_folds=5)
                .hyperparameter_optimization(['Random Forest', 'XGBoost'])
                .visualize_results()
                .generate_biological_interpretation(top_n_genes=20)
                .print_detailed_results())
    
    return pipeline


if __name__ == "__main__":
    main()