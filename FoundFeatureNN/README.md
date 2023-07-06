# Found Feature NN

Code (`FoundFeatureNN.ipynb`) and results for the multiclass neural network model (saved in `saved_models/`). We run a hyperparameter search in `FoundFeatureNN_Hyperparam_search.ipynb`.

Note that the cell outputs in `FoundFeatureNN.ipynb` are for the model when no_LUAD=True (ie, without a LUAD class). This is why the cell outputs do not match the multiclass neural network results. However, the results in `figs/` are results (on the test set) with the LUAD class, and should match the multiclass neural network results.

`multiclass_DNN_external_performance.ipynb` evaluates the neural network model on the independent data sets, and saves figures into `multiclass_DNN_external_figs`.