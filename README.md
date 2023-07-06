# Early detection and diagnosis of cancer using EMethylNet: an interpretable model for DNA methylation pattern detection

-------------------------------------------------

This is the code repo for EMethylNet. We create XGBoost models that can classify 13 cancer types as well as non-cancer tissue samples using DNA methylome data. We utilise the features identified by the multiclass XGBoost model to develop EMethylNET (Explainable Methylome Neural network for Evaluation of Tumours), a multiclass deep neural network. We then run analyses on the features detected by the XGBoost models.


## Folder structure:

1. First start with `data_preprocessing/`. This will download the TCGA data and pre-process it into csv files.

2. Now train some models:
    * Start with `simple_methods/` (logistic regression and SVM)
    * Then run `xgboost/` models
    * Finally, make a neural network in `FoundFeatureNN/`

3. Analyse the models' features:
    * Map the features to genes in `mapping_to_genes/`
    * Run gene ontology analysis in `biology/gProfiler/`
    * Look at literature evidence in `biology/pangaea/`
    * Analyse enriched pathways in `biology/pathways/`
    * Analyse the multiclass lncRNAs in `biology/lncRNAs/`
    * Run survival analysis in `survival_analysis/`


## Code for each figure
- Figure 2:
    - 2a+b: `xgboost/create_classifiers.ipynb`
    - 2c+d: `xgboost/create_classifiers.ipynb` and `xgboost/binary_xgboost_performance.ipynb`
    - 2e: `xgboost/create_classifiers.ipynb`
    - 2f+g: `xgboost/multiclass_xgboost_performance.ipynb`
- Figure 3:
    - 3a+b+d+e: `xgboost/binary_xgboost_external_performance.ipynb`
    - 3c: `xgboost/binary_xgboost_external_COAD_cm.ipynb`
- Figure 4:
    - 4b,c,d: `FoundFeatureNN/multiclass_DNN_external_performance.ipynb`
- Figure 5:
    - 5a: `biology/gProfiler/REVIGO_smaller.r`
    - 5b: `biology/pangaea/analysing_pangea_results.ipynb`
    - 5c: `biology/pathways/KEGGPathwayAnalysis.Rmd`
    - 5d: `biology/pathways/visualising_enriched_kegg_pathways/visualising_enriched_kegg_pathways.ipynb`
- Figure 6:
    - Hand curated networks in pathviso and cytoscape
Figure 7:
    - 7a: `biology/lncRNAs/proportions_of_lncRNAs/gene_stats.Rmd`
    - 7b+c: `biology/lncRNAs/combining_databases_and_pangea.ipynb`
    - 7d: `biology/lncRNAs/validate_with_properties/CLC_comparison_results.Rmd`
    - 7e: `biology/lncRNAs/validate_with_properties/CLC_comparison.Rmd`
Figure 8:
    - 8a: `survival_analysis/SurvivalAnalysisNoTestset.Rmd`
    - 8b+c: `survival_analysis/SurvivalAnalysis.Rmd`

------------------------------------------------


