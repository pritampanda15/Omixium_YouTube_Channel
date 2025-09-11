# run_analysis_2.py - FIXED DATA TYPE ISSUE
from gene_expression_ml_pipeline_dual_targets import EnhancedGeneExpressionMLPipeline
import pandas as pd
import numpy as np

# Initialize pipeline
pipeline = EnhancedGeneExpressionMLPipeline(random_state=42)

# Load data first
print("Loading data...")
pipeline.load_and_inspect_data(
    'sample_drug_response_data.tsv',
    classification_target='drug_sensitive',
    regression_target='IC50_Drug_X'
)

# FIX THE DATA TYPE ISSUE - this is the actual problem
print("Fixing data types...")
print(f"Original X shape: {pipeline.X.shape}")
print(f"Data types before fix:\n{pipeline.X.dtypes.value_counts()}")

# Convert all feature columns to numeric, replacing non-numeric with NaN
for col in pipeline.X.columns:
    pipeline.X[col] = pd.to_numeric(pipeline.X[col], errors='coerce')

# Fill NaN values with column median (better than 0)
pipeline.X = pipeline.X.fillna(pipeline.X.median())

print(f"Data types after fix:\n{pipeline.X.dtypes.value_counts()}")
print(f"Any remaining NaN values: {pipeline.X.isnull().sum().sum()}")

# Now run the rest of the analysis
print("Running preprocessing...")
pipeline.preprocess_data(
    log_transform=True,  
    k_best_features=20,
    apply_pca=True
)

print("Splitting data...")
pipeline.split_data(test_size=0.2)

print("Defining models...")
pipeline.define_models(
    classification_models=['Logistic Regression', 'Random Forest', 'XGBoost'],
    regression_models=['Random Forest', 'XGBoost']
)

print("Training models...")
pipeline.train_and_evaluate_models()

print("Creating visualizations...")
pipeline.create_comprehensive_visualizations()

print("Generating biological interpretation...")
pipeline.generate_biological_interpretation(top_n_genes=20)

print("Saving results...")
pipeline.save_detailed_results()

print(f"Analysis completed! Results saved to: {pipeline.output_dir}")