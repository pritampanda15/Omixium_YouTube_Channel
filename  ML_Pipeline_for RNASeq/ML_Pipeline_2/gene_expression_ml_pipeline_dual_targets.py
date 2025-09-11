#!/usr/bin/env python3
"""
Enhanced Dual-Target Machine Learning Pipeline for Gene Expression
================================================================

Enhanced version with comprehensive visualizations and auto-saving capabilities.
Supports both classification (drug_sensitive) and regression (IC50) analysis.

Author: Enhanced Bioinformatics ML Pipeline
Date: 2025
"""
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, StratifiedKFold,
    learning_curve
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, 
    confusion_matrix, roc_curve, precision_recall_curve,
    mean_squared_error, r2_score, mean_absolute_error
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class EnhancedGeneExpressionMLPipeline:
    """
    Enhanced ML pipeline supporting both classification and regression targets
    with comprehensive visualization and auto-saving capabilities
    """

    def __init__(self, random_state=RANDOM_SEED):
        self.random_state = random_state
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        self.class_models = {}
        self.reg_models = {}
        self.results = {}
        self.feature_names = None
        self.selected_features = None
        self.classification_target = None
        self.regression_target = None
        
        # Create output directory with timestamp
        self.output_dir = f"ml_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Results will be saved to: {self.output_dir}")

    def load_and_inspect_data(self, file_path, classification_target=None, regression_target=None, sep='\t'):
        """
        Load gene expression data and separate classification and regression targets
        """
        print("=== Loading and Inspecting Dataset ===")
        self.data = pd.read_csv(file_path, sep=sep, index_col=None)
        
        self.classification_target = classification_target
        self.regression_target = regression_target

        if classification_target and classification_target in self.data.columns:
            self.y_class = self.data[classification_target]
        elif classification_target:
            raise ValueError(f"Classification target '{classification_target}' not found.")

        if regression_target and regression_target in self.data.columns:
            self.y_reg = self.data[regression_target]
        elif regression_target:
            raise ValueError(f"Regression target '{regression_target}' not found.")

        # Features
        drop_cols = []
        if classification_target:
            drop_cols.append(classification_target)
        if regression_target:
            drop_cols.append(regression_target)

        self.X = self.data.drop(columns=drop_cols)
        self.feature_names = self.X.columns.tolist()

        print(f"Dataset shape: {self.X.shape}")
        print(f"Number of samples: {self.X.shape[0]}")
        print(f"Number of features: {self.X.shape[1]}")

        if classification_target:
            print(f"Classification target distribution:\n{self.y_class.value_counts()}")
        if regression_target:
            print(f"Regression target summary:\n{self.y_reg.describe()}")

        print(f"Missing values in features: {self.X.isnull().sum().sum()}")
        if classification_target:
            print(f"Missing values in classification target: {self.y_class.isnull().sum()}")
        if regression_target:
            print(f"Missing values in regression target: {self.y_reg.isnull().sum()}")

        return self

    def preprocess_data(self, log_transform=True, variance_threshold=0.01, k_best_features=None, apply_pca=False, pca_components=0.95):
        """
        Preprocess features with enhanced feature selection
        """
        print("\n=== Data Preprocessing ===")

        # Fill missing values
        if self.X.isnull().sum().sum() > 0:
            print("Handling missing values...")
            self.X = self.X.fillna(self.X.median())

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

        # Feature selection - use appropriate score function based on available targets
        if k_best_features and k_best_features < self.X.shape[1]:
            print(f"Selecting top {k_best_features} features using univariate tests...")
            
            # Use classification target for feature selection if available, otherwise regression
            target_for_selection = self.y_class if hasattr(self, 'y_class') else self.y_reg
            score_func = f_classif if hasattr(self, 'y_class') else f_regression
            
            selector = SelectKBest(score_func=score_func, k=k_best_features)
            self.X = pd.DataFrame(
                selector.fit_transform(self.X, target_for_selection),
                index=self.X.index,
                columns=self.X.columns[selector.get_support()]
            )
            self.selected_features = self.X.columns.tolist()
            print(f"Features after selection: {self.X.shape[1]}")

        # Scaling
        print("Applying robust scaling...")
        self.scaler = RobustScaler()
        self.X = pd.DataFrame(self.scaler.fit_transform(self.X), index=self.X.index, columns=self.X.columns)

        # PCA (optional)
        if apply_pca:
            print(f"Applying PCA (components={pca_components})...")
            self.pca = PCA(n_components=pca_components, random_state=self.random_state)
            X_pca = self.pca.fit_transform(self.X)
            n_components = X_pca.shape[1] if isinstance(pca_components, float) else pca_components
            self.X = pd.DataFrame(X_pca, index=self.X.index, columns=[f'PC{i+1}' for i in range(n_components)])
            print(f"PCA components: {self.X.shape[1]}")

        return self

    def split_data(self, test_size=0.2):
        """
        Split data into training and testing sets
        """
        print("\n=== Splitting Data ===")
        stratify_y = self.y_class if hasattr(self, 'y_class') else None
        
        # Handle both targets or single target
        targets = []
        if hasattr(self, 'y_class'):
            targets.append(self.y_class)
        if hasattr(self, 'y_reg'):
            targets.append(self.y_reg)
            
        if len(targets) == 2:
            self.X_train, self.X_test, self.y_class_train, self.y_class_test, self.y_reg_train, self.y_reg_test = train_test_split(
                self.X, self.y_class, self.y_reg,
                test_size=test_size,
                random_state=self.random_state,
                stratify=stratify_y
            )
        elif len(targets) == 1:
            if hasattr(self, 'y_class'):
                self.X_train, self.X_test, self.y_class_train, self.y_class_test = train_test_split(
                    self.X, self.y_class,
                    test_size=test_size,
                    random_state=self.random_state,
                    stratify=stratify_y
                )
            else:
                self.X_train, self.X_test, self.y_reg_train, self.y_reg_test = train_test_split(
                    self.X, self.y_reg,
                    test_size=test_size,
                    random_state=self.random_state
                )
        
        print(f"Training samples: {self.X_train.shape[0]}")
        print(f"Testing samples: {self.X_test.shape[0]}")
        return self

    def define_models(self, classification_models=None, regression_models=None):
        """
        Define enhanced models for classification and regression
        """
        self.class_models = {}
        self.reg_models = {}

        if classification_models is None:
            classification_models = ['Random Forest', 'SVM', 'Logistic Regression', 'XGBoost', 'Neural Network']
        if regression_models is None:
            regression_models = ['Random Forest', 'SVM', 'Linear Regression', 'XGBoost', 'Neural Network']

        # Classification models
        if hasattr(self, 'y_class_train'):
            for name in classification_models:
                if name == 'Random Forest':
                    self.class_models[name] = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
                elif name == 'SVM':
                    self.class_models[name] = SVC(probability=True, random_state=self.random_state)
                elif name == 'Logistic Regression':
                    self.class_models[name] = LogisticRegression(max_iter=1000, random_state=self.random_state)
                elif name == 'XGBoost':
                    self.class_models[name] = xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss')
                elif name == 'Neural Network':
                    self.class_models[name] = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=self.random_state)

        # Regression models
        if hasattr(self, 'y_reg_train'):
            for name in regression_models:
                if name == 'Random Forest':
                    self.reg_models[name] = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
                elif name == 'SVM':
                    self.reg_models[name] = SVR()
                elif name == 'Linear Regression':
                    self.reg_models[name] = LinearRegression()
                elif name == 'XGBoost':
                    self.reg_models[name] = xgb.XGBRegressor(random_state=self.random_state, eval_metric='rmse')
                elif name == 'Neural Network':
                    self.reg_models[name] = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=self.random_state)

        return self

    def train_and_evaluate_models(self):
        """
        Train classification and regression models with cross-validation
        """
        print("\n=== Training and Evaluating Models ===")
        
        # Classification
        if self.class_models and hasattr(self, 'y_class_train'):
            print("\n-- Classification Models --")
            for name, model in self.class_models.items():
                print(f"\nTraining {name}...")
                
                # Cross-validation
                cv_scores = cross_val_score(model, self.X_train, self.y_class_train, 
                                          cv=5, scoring='roc_auc')
                
                # Train on full training set
                model.fit(self.X_train, self.y_class_train)
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, "predict_proba") else None
                
                accuracy = accuracy_score(self.y_class_test, y_pred)
                roc_auc = roc_auc_score(self.y_class_test, y_pred_proba) if y_pred_proba is not None else None

                self.results[f'class_{name}'] = {
                    'model': model,
                    'test_accuracy': accuracy,
                    'test_roc_auc': roc_auc,
                    'cv_scores': cv_scores,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                print(f"{name} - Test Accuracy: {accuracy:.3f}, Test ROC-AUC: {roc_auc:.3f}, CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Regression
        if self.reg_models and hasattr(self, 'y_reg_train'):
            print("\n-- Regression Models --")
            for name, model in self.reg_models.items():
                print(f"\nTraining {name}...")
                
                # Cross-validation
                cv_scores = cross_val_score(model, self.X_train, self.y_reg_train, 
                                          cv=5, scoring='r2')
                
                # Train on full training set
                model.fit(self.X_train, self.y_reg_train)
                y_pred = model.predict(self.X_test)
                
                mse = mean_squared_error(self.y_reg_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(self.y_reg_test, y_pred)
                r2 = r2_score(self.y_reg_test, y_pred)

                self.results[f'reg_{name}'] = {
                    'model': model,
                    'test_mse': mse,
                    'test_rmse': rmse,
                    'test_mae': mae,
                    'test_r2': r2,
                    'cv_scores': cv_scores,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_pred': y_pred
                }
                print(f"{name} - Test R2: {r2:.3f}, Test RMSE: {rmse:.3f}, CV R2: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        return self

    def create_comprehensive_visualizations(self):
        """
        Create comprehensive visualizations similar to the example image
        """
        print("\n=== Creating Comprehensive Visualizations ===")
        
        # Set up the figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Cross-Validation Scores (Classification)
        if self.class_models:
            ax1 = plt.subplot(3, 4, 1)
            model_names = []
            cv_means = []
            cv_stds = []
            
            for name, res in self.results.items():
                if name.startswith('class_'):
                    model_names.append(name.replace('class_', ''))
                    cv_means.append(res['cv_mean'])
                    cv_stds.append(res['cv_std'])
            
            bars = ax1.bar(model_names, cv_means, yerr=cv_stds, capsize=5, 
                          color='lightcoral', alpha=0.7)
            ax1.set_title('Cross-Validation ROC-AUC Scores')
            ax1.set_ylabel('ROC-AUC')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, mean_val in zip(bars, cv_means):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{mean_val:.3f}', ha='center', va='bottom')

        # 2. ROC Curves
        if self.class_models:
            ax2 = plt.subplot(3, 4, 2)
            for name, res in self.results.items():
                if name.startswith('class_') and res['y_pred_proba'] is not None:
                    fpr, tpr, _ = roc_curve(self.y_class_test, res['y_pred_proba'])
                    auc_score = res['test_roc_auc']
                    ax2.plot(fpr, tpr, label=f"{name.replace('class_', '')} (AUC = {auc_score:.3f})")
            
            ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('ROC Curves')
            ax2.legend()

        # 3. Confusion Matrix (best classification model)
        if self.class_models:
            ax3 = plt.subplot(3, 4, 3)
            best_class_model = max(self.results.items(), 
                                 key=lambda x: x[1].get('test_roc_auc', 0) if x[0].startswith('class_') else 0)
            
            cm = confusion_matrix(self.y_class_test, best_class_model[1]['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
            ax3.set_title(f'Confusion Matrix - {best_class_model[0].replace("class_", "")}')
            ax3.set_xlabel('Predicted Label')
            ax3.set_ylabel('True Label')

        # 4. Feature Importance (Random Forest)
        ax4 = plt.subplot(3, 4, 4)
        rf_results = next((res for name, res in self.results.items() 
                          if 'Random Forest' in name and hasattr(res['model'], 'feature_importances_')), None)
        
        if rf_results:
            importances = rf_results['model'].feature_importances_
            indices = np.argsort(importances)[::-1][:20]
            top_features = [self.X.columns[i] for i in indices]
            top_importances = importances[indices]
            
            ax4.barh(range(len(top_importances)), top_importances, color='lightcoral', alpha=0.7)
            ax4.set_yticks(range(len(top_importances)))
            ax4.set_yticklabels(top_features)
            ax4.set_title('Top 20 Feature Importances (Random Forest)')
            ax4.set_xlabel('Importance')

        # 5. PCA Visualization
        ax5 = plt.subplot(3, 4, 5)
        pca_viz = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca_viz.fit_transform(self.X)
        
        if hasattr(self, 'y_class'):
            scatter = ax5.scatter(X_pca[:, 0], X_pca[:, 1], c=self.y_class, 
                                cmap='RdYlBu', alpha=0.7)
            ax5.set_title('PCA Visualization')
            plt.colorbar(scatter, ax=ax5, label='Class')
        
        ax5.set_xlabel(f'PC1 ({pca_viz.explained_variance_ratio_[0]:.1%} variance)')
        ax5.set_ylabel(f'PC2 ({pca_viz.explained_variance_ratio_[1]:.1%} variance)')

        # 6. Precision-Recall Curves
        if self.class_models:
            ax6 = plt.subplot(3, 4, 6)
            for name, res in self.results.items():
                if name.startswith('class_') and res['y_pred_proba'] is not None:
                    precision, recall, _ = precision_recall_curve(self.y_class_test, res['y_pred_proba'])
                    ax6.plot(recall, precision, label=name.replace('class_', ''))
            
            ax6.set_xlabel('Recall')
            ax6.set_ylabel('Precision')
            ax6.set_title('Precision-Recall Curves')
            ax6.legend()

        # 7. Learning Curves
        if self.class_models:
            ax7 = plt.subplot(3, 4, 7)
            # Use best performing model for learning curve
            best_model_name = max(self.results.items(), 
                                key=lambda x: x[1].get('test_roc_auc', 0) if x[0].startswith('class_') else 0)[0]
            best_model = self.results[best_model_name]['model']
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            train_sizes, train_scores, val_scores = learning_curve(
                best_model, self.X_train, self.y_class_train, cv=cv, 
                train_sizes=np.linspace(0.1, 1.0, 10), scoring='roc_auc'
            )
            
            ax7.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training', color='blue')
            ax7.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation', color='red')
            ax7.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                           train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1, color='blue')
            ax7.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                           val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1, color='red')
            ax7.set_xlabel('Training Set Size')
            ax7.set_ylabel('ROC-AUC')
            ax7.set_title(f'Learning Curves - {best_model_name.replace("class_", "")}')
            ax7.legend()

        # 8. Model Performance Heatmap
        ax8 = plt.subplot(3, 4, 8)
        
        # Collect all metrics for heatmap
        models = []
        metrics_data = []
        
        for name, res in self.results.items():
            if name.startswith('class_'):
                models.append(name.replace('class_', ''))
                metrics_data.append([res['test_accuracy'], res['test_roc_auc']])
        
        if metrics_data:
            heatmap_data = pd.DataFrame(metrics_data, 
                                      index=models, 
                                      columns=['Accuracy', 'ROC-AUC'])
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='Greens', ax=ax8)
            ax8.set_title('Model Performance Heatmap')

        # 9-12. Regression plots (if available)
        if self.reg_models:
            # Regression scatter plots
            reg_model_names = [name for name in self.results.keys() if name.startswith('reg_')]
            
            for i, name in enumerate(reg_model_names[:4]):  # Show up to 4 regression models
                ax = plt.subplot(3, 4, 9 + i)
                res = self.results[name]
                
                ax.scatter(self.y_reg_test, res['y_pred'], alpha=0.6)
                ax.plot([self.y_reg_test.min(), self.y_reg_test.max()], 
                       [self.y_reg_test.min(), self.y_reg_test.max()], 'r--')
                
                r2 = res['test_r2']
                ax.set_xlabel('True IC50')
                ax.set_ylabel('Predicted IC50')
                ax.set_title(f'{name.replace("reg_", "")} (R² = {r2:.3f})')

        plt.tight_layout()
        
        # Save the comprehensive plot
        plt.savefig(os.path.join(self.output_dir, 'comprehensive_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return self

    def save_detailed_results(self):
        """
        Save detailed results to files
        """
        print("\n=== Saving Detailed Results ===")
        
        # Create results summary
        results_summary = []
        
        for name, res in self.results.items():
            if name.startswith('class_'):
                results_summary.append({
                    'Model': name.replace('class_', ''),
                    'Type': 'Classification',
                    'Test_Accuracy': res.get('test_accuracy', 'N/A'),
                    'Test_ROC_AUC': res.get('test_roc_auc', 'N/A'),
                    'CV_Mean': res.get('cv_mean', 'N/A'),
                    'CV_Std': res.get('cv_std', 'N/A')
                })
            elif name.startswith('reg_'):
                results_summary.append({
                    'Model': name.replace('reg_', ''),
                    'Type': 'Regression',
                    'Test_R2': res.get('test_r2', 'N/A'),
                    'Test_RMSE': res.get('test_rmse', 'N/A'),
                    'Test_MAE': res.get('test_mae', 'N/A'),
                    'CV_Mean': res.get('cv_mean', 'N/A'),
                    'CV_Std': res.get('cv_std', 'N/A')
                })
        
        # Save results to CSV
        results_df = pd.DataFrame(results_summary)
        results_df.to_csv(os.path.join(self.output_dir, 'model_results_summary.csv'), index=False)
        
        # Save feature importance (if available)
        rf_results = next((res for name, res in self.results.items() 
                          if 'Random Forest' in name and hasattr(res['model'], 'feature_importances_')), None)
        
        if rf_results:
            importances = rf_results['model'].feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': self.X.columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            feature_importance_df.to_csv(os.path.join(self.output_dir, 'feature_importance.csv'), index=False)
        
        # Save experiment parameters
        params = {
            'Classification_Target': self.classification_target,
            'Regression_Target': self.regression_target,
            'Number_of_Features': self.X.shape[1],
            'Number_of_Samples': self.X.shape[0],
            'Test_Size': len(self.X_test) if hasattr(self, 'X_test') else 'N/A',
            'Random_Seed': self.random_state
        }
        
        with open(os.path.join(self.output_dir, 'experiment_parameters.txt'), 'w') as f:
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
        
        print(f"All results saved to directory: {self.output_dir}")
        return self

    def generate_biological_interpretation(self, top_n_genes=20):
        """
        Generate biological interpretation with enhanced analysis
        """
        print("\n=== Biological Interpretation ===")
        
        # Feature importance analysis
        for name, res in self.results.items():
            model = res['model']
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:top_n_genes]
                top_features = [self.X.columns[i] for i in indices]
                top_importances = importances[indices]
                
                print(f"\nTop {top_n_genes} features for {name.replace('_', ' ')}:")
                for i, (feature, importance) in enumerate(zip(top_features, top_importances)):
                    print(f"{i+1:2d}. {feature}: {importance:.4f}")
                
                # Create individual feature importance plot
                plt.figure(figsize=(10, 8))
                plt.barh(range(len(top_importances)), top_importances, color='lightcoral', alpha=0.7)
                plt.yticks(range(len(top_importances)), top_features)
                plt.xlabel('Feature Importance')
                plt.title(f'Top {top_n_genes} Feature Importances - {name.replace("_", " ")}')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                
                # Save individual plot
                safe_name = name.replace(' ', '_').replace('_', '_')
                plt.savefig(os.path.join(self.output_dir, f'feature_importance_{safe_name}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.show()
        
        return self

    def run_complete_analysis(self, file_path, classification_target=None, regression_target=None, **kwargs):
        """
        Run complete analysis pipeline with all visualizations and auto-saving
        """
        print("🧬 Starting Complete Gene Expression ML Analysis 🧬")
        print("=" * 60)
        
        # Load and preprocess data
        self.load_and_inspect_data(file_path, classification_target, regression_target)
        self.preprocess_data(**kwargs)
        self.split_data()
        
        # Define and train models
        self.define_models()
        self.train_and_evaluate_models()
        
        # Create all visualizations
        self.create_comprehensive_visualizations()
        self.generate_biological_interpretation()
        
        # Save all results
        self.save_detailed_results()
        
        print("\n" + "=" * 60)
        print("🎉 Analysis completed successfully!")
        print(f"📁 All results saved to: {self.output_dir}")
        print("=" * 60)
        
        return self