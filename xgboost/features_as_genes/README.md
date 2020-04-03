Features as genes
-----------------

Here are the gene lists, resulting from the R notebook ProbesToGenesUsingNewMapping.Rmd, which uses the mapping files I created in '../checking_mapping/makeMyMapping'.

They were either made with the strict mapping file (where CpGs are less than 1500bp upstream or inside the genes they are mapped to), or the relaxed mapping file (where CpGs within a window of 5000bp either side of a gene are accepted). This explains the _strict or _relaxed suffix.

The min count refers to how many times a feature was found by xgboost over multiple runs - only features found min_count times are taken and turned into genes.

only\_islands means only probes within CGIs are taken and turned into genes. near\_islands means probes within a CGI or its shore or shelf are taken (so within 4000bp either side of a CGI).
