# Mapping to genes

Getting the features found by the XGBoost models and mapping them to genes.

We make the mapping file in `Make_mapping.Rmd`. This maps any probes within this window to the gene:
<---- 1500bp ----->| > Gene > |

We then create gene lists for each xgboot model (all binary models and the multiclass model) in `XGBoost_features_to_genes.Rmd` (saved in `gene_lists/`). This uses the saved XGBoost features and the mapping file. This also creates probe .bed and .csv files (`probe_lists/`) and mapping files for each model (`specific_mappings/`). We also create background gene lists (`mapping_to_genes/`) (as these files are quite big, I have only included the multiclass background genes here).

For the multiclass genes, Shamith went through it and refined the probes mapping to multiple genes, creating the file `multi_probes_mapping_to_multiple_genes (1).xlsx`. We process this in `refining_multi_gene_list.Rmd`. The refined gene list is in `gene_lists/genes__1500.csv` and the refined mapping is in `specific_mappings/mapping_human_refined_.csv`.