# Workflow for preprocessing the data:
--------------------------------------

- run downloading_files.R
- Run DataPreprocessingRda.R (or .Rmd for a readable, single cancer type version)
- Run clean\_data\_preprocessing.ipynb - saves into dataset/pandas folder
- For multiclass data - Run merging\_cancer\_types.R


# Explanation of the other files

- blacklist - contains the list of probes which have been shown to be noisy/unreliable, and notes I made when reading the paper about this



