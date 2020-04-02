# Script version of DataPreprocessingRda, that can deal with multiple cancer types at once

library(tidyverse)
library(dplyr)
library(TCGAbiolinks)
library(SummarizedExperiment)

# you might have to setwd() (you want it set to data_preprocessing/)
print(getwd())

blacklist.file <- 'blacklist/Naeem2014_blacklist_annotation.txt'


cancer.type.codes <- c('BLCA', 'BRCA', 'COAD', 'ESCA', 'HNSC', 'KIRC', 'KIRP', 'LIHC', 'LUAD', 'LUSC', 'PRAD', 'THCA', 'UCEC')

process <- function(cancer.type.code) {
  print(paste("Doing cancer type", cancer.type.code))
  
  load(paste(cancer.type.code, 'methylation.rda', sep='')) # will create a variable called 'data'
  beta_values <- assay(data)
  
  # removing blacklisted:
  blacklist <- read.csv(file = blacklist.file, header=TRUE, sep='\t') # has a column called 'Flag.discard.keep.' which either is 'discard' or 'keep'
  
  blacklisted_probes <- blacklist %>%
    filter(Flag.discard.keep. == 'discard') %>%
    dplyr::select(probe)
  
  beta_values_no_blacklisted <- beta_values[!rownames(beta_values) %in% blacklisted_probes$probe, ] # get all rows that are not in blacklisted probes
  
  
  # removing * chromosome
  library(GenomicRanges)
  probe_granges <- rowRanges(data)
  sorted_probe_granges <- probe_granges %>%
    as_tibble() %>%
    filter(seqnames != "*") %>% # we also take out unknown (?) chromosomes, get 480457 probes
    filter(Composite.Element.REF %in% rownames(beta_values_no_blacklisted)) # getting the ones that were not blacklisted, 293808 probes
  beta_values_sorted <- beta_values_no_blacklisted[c(sorted_probe_granges$Composite.Element.REF), ] # selecting rows in order of sorted_probe_granges
  
  
  # Remove probes with an na value fraction of more than 0.05
  na_frac <- is.na(beta_values_sorted) %>%
    as_tibble() %>%
    rowMeans()
  beta_values_not_na <- beta_values_sorted[na_frac <= 0.05, ] # only keep probes with less than 0.05 na values
  # getting the names of the probes removed:
  na_probes_removed <- rownames(beta_values_sorted[na_frac > 0.05, ])
  
  
  # Impute (=estimate) the remaining na values:
  library(impute)
  total_frac_na <- mean(is.na(beta_values_not_na))
  
  imputed_betas <- impute.knn(beta_values_not_na, k=10, rowmax=0.25) # we allow any row to have max 25% missing data, no more
  imputed_betas <- imputed_betas[[1]]
  
  
  # get m values
  library(lumi)
  m_values <- beta2m(imputed_betas)
  
  
  # Writing to file (both beta and m values):
  save.path <- 'GDCdata/from_rda/TCGA-'
  print(paste('PROJECT: ', cancer.type.code, ', WRITING M VALUES MATRIX ', sep = ''))
  
  write.csv(m_values, file=paste(save.path, cancer.type.code, '_m_from_rda', sep=''), sep='\t', row.names = TRUE, quote = FALSE)
  
}


for (cancer.type.code in cancer.type.codes) {
  process(cancer.type.code)
}

