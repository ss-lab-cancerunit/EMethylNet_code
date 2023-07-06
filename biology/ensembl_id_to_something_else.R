# read in ensembl id file, use biomart to transform it to hgnc id file

# type is the type of gene identifier you want to convert to, that biomart recognises, eg. "hgnc_id"
ensembl_to_ <- function(ensembl_file, type) {
  library(data.table)
  ensembl_ids <- read.table(ensembl_file)
  
  library(biomaRt)
  ensembl <- useEnsembl(biomart = "ensembl", dataset = "hsapiens_gene_ensembl") #, mirror = "useast") # this mirror change worked when I was having server/connection issues
  # using a different mirror because the default was playing up
  # ensembl=useMart("ensembl")
  # ensembl = useDataset("hsapiens_gene_ensembl",mart=ensembl)
  
  biomart_name_conversion <- getBM(attributes = c("ensembl_gene_id", type),
                                   filters = 'ensembl_gene_id',
                                   values = ensembl_ids,
                                   mart = ensembl
                                   )
  
  # remove duplicated ensembl ids (when they map to multiple 'types')
  biomart_name_conversion <- biomart_name_conversion[!duplicated(biomart_name_conversion$ensembl_gene_id), ]
  
  print(paste("Couldn't find:", sum(biomart_name_conversion[, type] == ''), "out of", nrow(biomart_name_conversion)))
  
  result <- biomart_name_conversion[, type]
  print(biomart_name_conversion)
  result <- result[result != ""]
  return(result)

}


ensembl_to_simpler <- function(ensembl_ids, type, remove_duplicates = TRUE) {
  library(data.table)
  
  library(biomaRt)
  ensembl <- useEnsembl(biomart = "ensembl", dataset = "hsapiens_gene_ensembl") #, mirror = "useast") # this mirror change worked when I was having server/connection issues
  # using a different mirror because the default was playing up
  # ensembl=useMart("ensembl")
  # ensembl = useDataset("hsapiens_gene_ensembl",mart=ensembl)
  
  biomart_name_conversion <- getBM(attributes = c("ensembl_gene_id", type),
                                   filters = 'ensembl_gene_id',
                                   values = ensembl_ids,
                                   mart = ensembl,
                                   useCache = FALSE
  )
  
  # remove duplicated ensembl ids (when they map to multiple 'types')
  if (remove_duplicates == TRUE) {
    biomart_name_conversion <- biomart_name_conversion[!duplicated(biomart_name_conversion$ensembl_gene_id), ]
  }
  
  print(paste("Couldn't find:", sum(biomart_name_conversion[, type] == ''), "out of", nrow(biomart_name_conversion)))
  
  return(biomart_name_conversion)
  
}


to_ensembl_ <- function(ids, id_type) {
  library(data.table)
  
  library(biomaRt)
  ensembl <- useEnsembl(biomart = "ensembl", dataset = "hsapiens_gene_ensembl") #, mirror = "useast") # this mirror change worked when I was having server/connection issues
  # using a different mirror because the default was playing up
  # ensembl=useMart("ensembl")
  # ensembl = useDataset("hsapiens_gene_ensembl",mart=ensembl)
  
  biomart_name_conversion <- getBM(attributes = c(id_type, 'ensembl_gene_id'),
                                   filters = id_type,
                                   values = ids,
                                   mart = ensembl
  )
  
  # remove duplicated ensembl ids (when they map to multiple 'types')
  # biomart_name_conversion <- biomart_name_conversion[!duplicated(biomart_name_conversion$ensembl_gene_id), ]
  
  print(paste("Couldn't find:", sum(biomart_name_conversion[, 'ensembl_gene_id'] == ''), "out of", nrow(biomart_name_conversion)))
  
  return(biomart_name_conversion)
  
}


# all gene lists to names:
all_gene_lists_to_names <- function(cancer_type) {
  ensembl_file = paste0('/Tank/methylation-patterns-code/methylation-patterns-izzy/gene_lists/all_genes_min_count_1__', cancer_type ,'_relaxed.csv')
  output_location = paste0('/Tank/methylation-patterns-code/methylation-patterns-izzy/gene_lists/all_genes_min_count_1__', cancer_type ,'_relaxed_names.csv')
  hgnc_symbols <- ensembl_to_(ensembl_file, "external_gene_name")
  write.table(hgnc_symbols, output_location, quote = F, row.names = F, col.names = F)
}

# cancer_types <- c("BLCA", "BRCA", "COAD", "ESCA", "HNSC", "KIRC", "KIRP", "LIHC", "LUAD", "LUSC", "PRAD", "THCA", "UCEC")
# for (cancer_type in cancer_types) {
#   all_gene_lists_to_names(cancer_type)
# }

