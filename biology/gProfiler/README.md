# gProfiler and Revigo workflow:

I ran gProfiler in `gProfiler.Rmd`.
Then I ran the multiclass results (gProfiler_revigo.csv) through revigo using the Small (0.5) setting, with Homo sapiens GO term sizes, and the SimRel similarity measure. I then exported the R script and modified it (see `REVIGO_smaller.r`) to make the figures more how I wanted them.

I chose to use the bonferroni correction method as they report it as being the strictest (see "g:Profiler: a web server for functional enrichment analysis and conversions of gene lists (2019 update)")