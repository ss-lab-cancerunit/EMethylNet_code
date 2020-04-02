# after data preprocessing we have a matrix for each cancer type. 
# these need to be merged to create a big data matrix for classification of cancer type. The classification targets are one for each cancer type, and one for normal.

# you might have to setwd() (you want it set to data_preprocessing/)
print(getwd())


library(dplyr)

# cancer.types.codes = c("BRCA", "HNSC", "THCA", "PRAD")
cancer.types.codes = c('BLCA', 'BRCA', 'COAD', 'ESCA', 'HNSC', 'KIRC', 'KIRP', 'LIHC', 'LUAD', 'LUSC', 'PRAD', 'THCA', 'UCEC')
cancer.types.nums = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13) # the target numbers - 1 means BLCA, 2 means BRCA, etc, (0 means normal)
names(cancer.types.nums) = cancer.types.codes

use_beta <- FALSE

all_data <- NULL
all_diagnoses <- NULL


# changes the diagnosis values 0, 1 (normal, tumour) to 0, num of cancer type
# this means each cancer type has a separate diagnosis number and normal is always 0
change_to_type_diagnosis <- function(diagnoses, type) {
  return(ifelse(diagnoses== 1, cancer.types.nums[type], 0))
}


for (cancer.type.code in cancer.types.codes) {
  print(paste("Doing ", cancer.type.code))
  if (use_beta == TRUE) {
    m_values <- read.csv(file=paste('dataset/pandas/beta_values/TCGA-', cancer.type.code, '.csv', sep = ''), header = TRUE, sep ='') # rows = probes, cols = samples
  }
  else {
    m_values <- read.csv(file=paste('dataset/pandas/m_values/TCGA-', cancer.type.code, '.csv', sep = ''), header = TRUE, sep ='') # rows = probes, cols = samples
  }
  
  diagnoses <- read.csv(file=paste('dataset/pandas/diagnoses/TCGA-', cancer.type.code, '.csv', sep = ''), header = TRUE, sep ='')
  diagnoses$X0 <- change_to_type_diagnosis(diagnoses$X0, cancer.type.code)
  
  # need to add on to all data
  if (is.null(all_data)) {
    all_data <- m_values # first one is easy
    all_diagnoses <- diagnoses
    print("All data and all diagnoses is now dim:")
    print(dim(all_data))
    print(dim(all_diagnoses))
  } else {
    # need to add on columns, rows should match up
    all_data <- merge(all_data, m_values, by="probe", sort=FALSE)
    # all_data <- bind_cols(all_data, m_values)
    all_diagnoses <- bind_rows(all_diagnoses, diagnoses) # append on to end of the column
    print("All data and all diagnoses is now dim:")
    print(dim(all_data))
    print(dim(all_diagnoses))
  }
}

print("Writing all data...")
if (use_beta == TRUE) {
  write.table(all_data, '../data_preprocessing/dataset/pandas/beta_values/TCGA-all.csv', sep='\t', row.names = FALSE)
} else {
  write.table(all_data, '../data_preprocessing/dataset/pandas/m_values/TCGA-all.csv', sep='\t', row.names = FALSE )
}
print("Writing diagnoses...")
write.table(all_diagnoses, '../data_preprocessing/dataset/pandas/diagnoses/TCGA-all.csv', sep='\t', quote = FALSE)

# NOTE: this data is NOT shuffled! Will probably need to shuffle before classification algs. Update: using sklearn's train_test_split shuffles it for us when we get the data
