from gene_expression_ml_pipeline import GeneExpressionMLPipeline

# Your updated pipeline (no more indexing errors!)
pipeline = GeneExpressionMLPipeline(random_state=42)

pipeline = (pipeline
    .load_and_inspect_data('sample_gene_expression_standard.tsv', 'disease_status')
    .preprocess_data(
        log_transform=True,
        k_best_features=5000,
        apply_pca=False  # Don't use PCA in preprocessing
    )
    .split_data(test_size=0.2, stratify=True)
    .apply_pca_for_visualization(n_components=2)  
    .define_models()
    .train_and_evaluate_models()
    .hyperparameter_optimization(['Random Forest', 'XGBoost'])
    .visualize_results() 
    .generate_biological_interpretation()
    .print_detailed_results())