# lncRNA conservation
---------------------

Phast conservation file bigwig downloaded from: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/

multi_lncRNAs and background_lncRNAs bed files made in biology/lncRNA_network/validate_with_properties/CLC_comparison.Rmd.

## Set of commands to get to gat_result:

bigWigToWig hg38.phastCons100way.bw hg38.phastCons100way.wig
wig2bed --max-mem 20G --sort-tmpdir tmp < hg38.phastCons100way.wig > hg38.phastCons100way.bed
awk '($5 > 0.7)' hg38.phastCons100way.bed > hg38.phastCons100way.filtered_7.bed
bedtools merge -i hg38.phastCons100way.filtered_7.bed -c 5 -o mean > hg38.phastCons100way.filtered_7.merged.bed
bedtools sort -i ../background_lncRNAs.bed > background_lncRNAs.sorted.bed
bedtools sort -i ../multi_lncRNAs.bed > multi_lncRNAs.sorted.bed
cut -f1,2,3 hg38.phastCons100way.filtered_7.merged.bed > hg38.phastCons100way.filtered_7.merged.simple.bed
gat-run.py --segment-file=multi_lncRNAs.sorted.bed --workspace-file=workspace.bed --annotation-file=hg38.phastCons100way.filtered_7.merged.simple.bed --stdout=gat_result --ignore-segment-tracks

Also for a cut off of 0.9:
awk '($5 > 0.9)' hg38.phastCons100way.bed > hg38.phastCons100way.filtered_9.bed
bedtools merge -i hg38.phastCons100way.filtered_9.bed -c 5 -o mean > hg38.phastCons100way.filtered_9.merged.bed
cut -f1,2,3 hg38.phastCons100way.filtered_9.merged.bed > hg38.phastCons100way.filtered_9.merged.simple.bed
gat-run.py --segment-file=multi_lncRNAs.sorted.bed --workspace-file=workspace.bed --annotation-file=hg38.phastCons100way.filtered_9.merged.simple.bed --stdout=gat_result_9 --ignore-segment-tracks


How I get intersected_hg38_background_lncRNAs.bed:

bedtools intersect -a hg38.phastCons100way.bed -b background_lncRNAs.sorted.bed -sorted > intersected_hg38_background_lncRNAs.bed
