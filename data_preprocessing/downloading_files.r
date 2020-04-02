####################################################################

# make sure wd is set to 'data_preprocessing'

library(TCGAbiolinks)
library(stringr)

# choosing TCGA projects
# for all projects: cancer.types.codes = c('ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD',
#                       'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM')

# for now, downlaod the ones with the most samples (see 'ExploringTheData/LookingAtAvailableTCGAData.Rmd'):
# cancer.types.codes = c("BRCA", "HNSC", "THCA", "PRAD")

cancer.types.codes = c('BLCA', 'BRCA', 'COAD', 'ESCA', 'HNSC', 'KIRC', 'KIRP', 'LIHC', 'LUAD', 'LUSC', 'PRAD', 'THCA', 'UCEC')


tcga <- 'TCGA'
methylation <- 'methylation.rda'
gene.expression <- 'expression.rda'

for (cancer.type.code in cancer.types.codes) {
  
  print(paste0("Doing type ", cancer.type.code, ".................................................."))
  
  if (file.exists(paste0(cancer.type.code, methylation, sep =''))) { # } & file.exists(paste0(cancer.type.code, gene.expression, sep =''))) {
    print("Done this type already") # if we already have the methylation and expression rda files, we don't need to do anything
  }
  else {
    # search for patients in methylation dataset and gene expression dataset
    
    query.met.check <- GDCquery(project = paste(tcga, cancer.type.code, sep = '-'),
                                legacy = FALSE,
                                data.category = 'DNA Methylation',
                                platform = 'Illumina Human Methylation 450')

    # query.exp.check <- GDCquery(project = paste(tcga, cancer.type.code, sep = '-'),
                                # data.category = 'Transcriptome Profiling',
                                # experimental.strategy = 'RNA-Seq',
                                # data.type = 'Gene Expression Quantification',
                                # workflow.type = 'HTSeq - Counts')
    
    # Downloading all samples, regardless of whether there is a matching expression sample:
    # ----------------------------------------------------------------------------------
    GDCdownload(query.met.check, token.file = file, files.per.chunk = 5)
    met <- (GDCprepare(query.met.check, save = TRUE, save.filename = paste(cancer.type.code, methylation, sep=''), summarizedExperiment = TRUE))
    
    # GDCdownload(query.exp.check, token.file = file, files.per.chunk = 5)
    # exp <- GDCprepare(query.exp.check, save = TRUE, save.filename = paste(cancer.type.code, gene.expression, sep=''))
    
    # Uncomment all below (and comment stuff above) to get data that has matching expression data:
    # -------------------------------------------------------------------------------------------
    
    # intersation of patients (first 12 symbols from TCGA barcode) that are in both datasets
    # common.patients <- intersect(substr(getResults(query.met.check, cols = "cases"), 1, 12),
                                 # substr(getResults(query.exp.check, cols = "cases"), 1, 12))
    
    # selecting and downloading common patients files from methylation and gene expression datasets
    # query.met <- GDCquery(project = paste(tcga, cancer.type.code, sep = '-'),
    #                       legacy = FALSE,
    #                       data.category = 'DNA Methylation',
    #                       platform = 'Illumina Human Methylation 450',
    #                       barcode = common.patients)
    # 
    # GDCdownload(query.met, token.file = file, files.per.chunk = 5)
    # met <- (GDCprepare(query.met, save = TRUE, save.filename = paste(cancer.type.code, methylation, sep=''), summarizedExperiment = TRUE))
    
    # where is the barcode argument here?
    # query.exp <- GDCquery(project = paste(tcga, cancer.type.code, sep = '-'),
    #                       data.category = 'Transcriptome Profiling',
    #                       experimental.strategy = 'RNA-Seq',
    #                       data.type = 'Gene Expression Quantification',
    #                       workflow.type = 'HTSeq - Counts')
    
    # GDCdownload(query.exp, token.file = file, files.per.chunk = 5)
    # exp <- GDCprepare(query.exp, save = TRUE, save.filename = paste(cancer.type.code, gene.expression, sep=''))
  }
  
}

