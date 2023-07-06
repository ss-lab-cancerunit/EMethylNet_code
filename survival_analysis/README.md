# Survival analysis

Here, we run two different survival analyses on the genes found by the binary xgboost models.

First, we run a simple survival analysis in `SurvivalAnalysisNoTestset.Rmd`. This runs standard survival analysis and plots KM curves in `KMs/`.

Then, we run survival analysis with a test set in `SurvivalAnalysis.Rmd`, to see whether the genes can predict survival. We plot time-dependent ROC curves and calculate the AUCs (the AUCs are plotted in `analyse_survival_results.Rmd`, which produces `testsetAUCs_5_year_violin.pdf`).