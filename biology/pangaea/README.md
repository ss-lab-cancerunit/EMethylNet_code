# Pangea analysis

Want to make a table of evidence for the multiclass genes, using Pangaea. Want methylation and cancer links. 

Commands I ran:
----------------
## First get multiclass gene names:
awk -F',' 'NR > 1 {gsub(/"/, "", $2);print $2}' ../../mapping_to_genes/gene_lists/genes__1500.csv > genes__1500_names.csv
awk '!visited[$0]++' genes__1500_names.csv  > genes__1500_names_unique.csv

This ignores the header, replaces quotes with the empty string, and writes the second column to genes__1500_names.csv. The second command removes duplicate genes.

## Download the 4 million cancer abstracts, creating the xml file:
papers-parser download --genes genes__1500_names_unique.csv --relations cancer_stems.txt --synonyms 'default' --model rules --number 4227582 --output cancer_output "cancer"


## Then create the json output:
papers-parser local --genes genes__1500_names_unique.csv --relations cancer_stems.txt --synonyms genes_to_synonyms_trimmed.txt --model rules --output cancer_output cancer_output.xml


We then look at these results in `analysing_pangea_results.ipynb` and save a csv with the processed literature evidence: `cancer_abstracts_with_methylation_evidence_cleaned.csv`.