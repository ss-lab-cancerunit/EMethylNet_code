# XGBoost


## Training XGBoost classifer:
- create_classifiers.ipynb. Can create both binary and multiclass XGBoost models. Also saves the feature importances. Warning: the multiclass model takes about 1 or 2 days to train!

## Saved results and models:
- figs/ - confusion matrices, ROC and precision-recall curves showing performance of the models
- saved_models/ - fitted xgboost models (in pickle format, use joblib.load to load) and a text file that represents the structure of all the decision trees (so you can see which trees are behind the xgboost model)
- feature_importances - files with feature importances and concatenated feature importances for each model. Features start from 0 (so feature 0 = first probe, cg0000029)
- features\_as\_genes - files with gene lists that xgboost found (from Concat\_feature\_importances.ipynb and ProbesToGenesUsingNewMapping). See the README in this folder for the naming convention.
- feature\_stats - Looks at stats for each feature (probe) found important. Csv files show how many genes a probe was mapped to, and where this probe was in relation to the gene and CGI. For summary plots, see the swarm plots.

## Looking at found features and finding genes:
- Firstly, run Concat\_feature_importances.ipynb - takes in multiple feature importance files from feature\__importances and concatenates them to see which features are common on multiple runs.
- Then, run ProbesToGenesUsingNewMapping.Rmd (or .R for multiple cancer types at once) - this maps the concatenated features to genes using the mapping file created and stores them in features\_as\_genes. Also creates feature\_stats files.

## Other files:
- get\_binary\_acc.ipynb: Uses figs/metrics files to caluclate the average accuracy and mcc of xgboost binary classifiers.
- Hyperparam_search.ipynb: Code used for finding hyperparameters. Includes my notes when deciding on them.
- load_and_eval.ipynb: Use this if you want to load a saved model and calculate its performance on the test set.
