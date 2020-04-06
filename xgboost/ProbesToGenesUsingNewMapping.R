# A less readable script version of ProbesToGenesUsingNewMapping.Rmd. The advantage is it can loop and do multiple cancer types in one run.

cancer_types = c('BLCA', 'BRCA', 'COAD', 'ESCA', 'HNSC', 'KIRC', 'KIRP', 'LIHC', 'LUAD', 'LUSC', 'PRAD', 'THCA', 'UCEC')
# cancer_types = c('') # for multiclass
stricts = c('TRUE', 'FALSE') # does both strict and relaxed

just_stats = TRUE

min_count = 1 # how many times a feature had to be found in consecutive runs of xgboost to be taken and transformed to a gene
only_islands <- FALSE # only take probes within CpG islands
near_islands <- FALSE # takes probes within CpG islands or their shore or shelf (so a 4000bp window around either side of each island)


island_label <- ''
if (only_islands == TRUE) {
  island_label <- 'only_islands'
} else if (near_islands == TRUE) {
  island_label <- 'near_islands'
}


for (cancer_type in cancer_types) {
  for (strict_mapping in stricts) {
    print(paste("Doing cancer type: ", cancer_type, " and strict is ", strict_mapping))  
    
    # change, depending on the cancer type:
    feature_file <- paste('feature_importances/', cancer_type, '_concatenated.csv', sep='') # features found by xgboost
    print("Using features from:")
    print(feature_file)
    
    if (strict_mapping == TRUE) {
      mapping_file <- '../checking_mapping/my_mapping_1500.rda'
      save_file <- paste('features_as_genes/all_genes_min_count_', min_count,'_', island_label, '_', cancer_type, '_strict.csv', sep = '')
      
    } else {
      mapping_file <- '../checking_mapping/my_mapping_5000.rda'  
      save_file <- paste('features_as_genes/all_genes_min_count_', min_count, '_', island_label, '_', cancer_type, '_relaxed.csv', sep = '')
    }
    
    ## ------------------------------------------------------------------------
    
    
    # from the original data, read in the probe list:
    library(data.table)
    if (cancer_type == '') {
      total_probes <- fread('../data_preprocessing/dataset/pandas/m_values/TCGA-all.csv', sep = '\t', select = c(1))  
    } else {
      total_probes <- fread(paste('../data_preprocessing/dataset/pandas/m_values/TCGA-', cancer_type, '.csv', sep = ''), sep = '\t', select = c(1), header = TRUE)
    }
    
    
    ## ------------------------------------------------------------------------
    
    load(mapping_file)
    
    features = read.table(feature_file, sep = ',', header = TRUE)
    
    # filter
    library(dplyr)
    features <- features %>%
      filter(count >= min_count)
    
    # View(features)
    
    
    ## ------------------------------------------------------------------------
    
    probes <- total_probes$probe[features$feature + 1] # feature indexing is from 0, but R indexing is from 1, so we add 1 - I have verified this is correct
    # View(probes)
    
    
    ## ------------------------------------------------------------------------
    
    # given a probe name, gets the gene ids that are mapped to that probe
    get_gene_ids <- function(probe, mapping) {
      rows = mapping[mapping$probe == probe, ]
      ids = rows$ensembl_gene_id
      return(ids)
    }
    
    get_CGI_location <- function(probe, mapping) {
      rows = mapping[mapping$probe == probe, ]
      location = rows$feature_type
      if (identical(location, character(0))) return(".") # if probe didn't map to any row, return not at any CGI location
      return(location[[1]]) # just get first location, as the same across all rows
    }
    
    get_all_info <- function(probe, mapping) {
      rows = mapping[mapping$probe == probe, ]
      return(rows)
    }
    
    ## ------------------------------------------------------------------------
    library(purrr)
    
    # filter probes based on their CGI loction
    if (only_islands == TRUE) {
      location <- map(probes, get_CGI_location, my_mapping)
      probes <- probes[location == "Island"]
    } else if (near_islands == TRUE) {
      location <- map(probes, get_CGI_location, my_mapping)
      probes <- probes[location != "."]
    }
    
    if (just_stats == FALSE) { # get genes and save
      genes <- map(probes, get_gene_ids, my_mapping)
      
      # genes
      
      genes <- unlist(genes)
      
      ## ------------------------------------------------------------------------
      # genes
      write.table(genes, save_file, quote = FALSE, row.names = FALSE, col.names = FALSE)
    }
    
    ## --------------------------------------------------------------------------
    
    # get stats and save
    
    all_info <- map(probes, get_all_info, my_mapping) # NOTE: if a probe maps to more than 1 gene, it will have more than 1 GRanges row
    
    stats <- data.frame(row.names = probes)
    
    stats$before_gene = 0
    stats$within_gene = 0
    stats$after_gene = 0
    stats$N_Shelf = 0
    stats$N_Shore = 0
    stats$Island = 0
    stats$S_Shore = 0
    stats$S_Shelf = 0
    stats$genes_mapped = 0 
    
    for (l in all_info) { # for each found probe and its info
      if (length(l) > 0) {
        stats[l[1, ]$probe, "genes_mapped"] <- length(l)
        for (i in 1:length(l)) {
          # print(i)
          # print(l[i, ])
          
          # Do CGIs
          if (l[i, ]$feature_type != '.') {
            val <- stats[l[i, ]$probe, l[i, ]$feature_type]
            stats[l[i, ]$probe, l[i, ]$feature_type] <- val + 1
          }
          
          # Do genes
          if (countOverlaps(l[i, ]$gene_feature.ranges, ranges(l[i, ])) > 0) { # within gene
            stats[l[i, ]$probe, "within_gene"] <- stats[l[i, ]$probe, "within_gene"] + 1
          }
          else {
            if (start(l[i, ]$gene_feature.ranges) > start(l[i, ])) {
              if (as.vector(l[i, ]$gene_feature.strand)[[1]] == "+") {
                stats[l[i, ]$probe, "before_gene"] <- stats[l[i, ]$probe, "before_gene"] + 1
              }
              else { # different for negative strand
                stats[l[i, ]$probe, "after_gene"] <- stats[l[i, ]$probe, "after_gene"] + 1
              }
            }
            
            if (end(l[i, ]$gene_feature.ranges) < start(l[i, ])) {
              if (as.vector(l[i, ]$gene_feature.strand)[[1]] == "+") {
                stats[l[i, ]$probe, "after_gene"] <- stats[l[i, ]$probe, "after_gene"] + 1
              }
              else { # different for negative strand
                stats[l[i, ]$probe, "before_gene"] <- stats[l[i, ]$probe, "before_gene"] + 1
              }
            }
          }
        }
      }
    }
    
    stats['totals', ] = colSums(stats)
    
    if (strict_mapping == TRUE) {
      savepath <- paste('feature_stats/stats_min_count_', min_count, '_', island_label, '_', cancer_type, '_strict.csv', sep = '')
    } else {
      savepath <- paste('feature_stats/stats_min_count_', min_count, '_', island_label, '_', cancer_type, '_relaxed.csv', sep = '')
    }
    
    write.table(stats, savepath, quote = FALSE, row.names = TRUE, col.names = TRUE)
    
    
  }
}