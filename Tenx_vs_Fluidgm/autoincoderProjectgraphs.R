library('scater')
library('Seurat')
library(biomaRt)
library(ggplot2)
library(reshape2)

# -------------------- Running only on Control treatment ----------------- #

#format 10x data
caf_ctrl_data <- Read10X(data.dir = "C:/Users/natallahADM/Desktop/Cancer Center/Projects/hayward_ratliff/2017_prelimData10xGenomics_CAF/Control-CAF")
caf_ctrl      <- CreateSeuratObject(raw.data = caf_ctrl_data, project = "CAFCTRL")
caf_exprs_mat <- caf_ctrl@raw.data
Tenxmat <- as.matrix(caf_exprs_mat)
dim(Tenxmat)
#33694  3321

#format C1 data
caf_data <- read.table("C:/Users/natallahADM/Desktop/Cancer Center/Projects/Ratliff/CAF_single-cell/GeneCounts_forStats.txt", header = T)
dim(caf_data)

genes_name_ensbl <- caf_data[, 1]
cells_name       <- colnames(caf_data)
caf_data         <- data.matrix(caf_data[,-1])
rownames(caf_data) <- genes_name_ensbl

dim(caf_data)
#57992   109

# ------------------- Creating gene names from ensemble ID ---------------------- #
ensembl = useMart("ensembl", dataset = "hsapiens_gene_ensembl")
filters = listFilters(ensembl)
attributes = listAttributes(ensembl)
filterOptions("ensembl_gene_id", ensembl)
ensembl_genes = as.matrix(rownames(caf_data))
colnames(ensembl_genes) = "gene_id"
caf_data = cbind(ensembl_genes, caf_data)

ensembl_to_gene = getBM(attributes = c('ensembl_gene_id','external_gene_name'), 
                        filters = 'ensembl_gene_id', 
                        values = ensembl_genes, 
                        mart = ensembl)
merged_df = merge(x = caf_data, 
                  y = ensembl_to_gene, 
                  by.x = "gene_id",
                  by.y = 'ensembl_gene_id',
                  all.x = TRUE)

# Because some of the gene names are not unique, making rownames for count matrix 
# is problematic. For this reason, I create a key for the gene, ensembl, 
# and dupilicate gene names

gene_names_for_duplicates = toupper(
  make.names(merged_df$external_gene_name, 
             unique = T))

gene_ensembl_dup_key = data.matrix(cbind(merged_df[,1], 
                                         merged_df$external_gene_name,
                                         gene_names_for_duplicates))
gene_ensembl_dup_key[,1] = rownames(gene_ensembl_dup_key)

ensembl_gene_dup_key = gene_ensembl_dup_key
rownames(ensembl_gene_dup_key) = ensembl_gene_dup_key[,3]

temp2 = merged_df[,2:(dim(merged_df)[2]-1)]
temp2 = sapply(temp2, function(x) as.numeric(as.character(x)))

sc_gene_counts = data.matrix(temp2)
rownames(sc_gene_counts) = gene_names_for_duplicates
sc_gene_counts[1:5,1:5]

cell_names = colnames(sc_gene_counts)
cell_names= unlist(strsplit(cell_names,"_1.fastq_tophat"))[1:ncol(sc_gene_counts)]
colnames(sc_gene_counts) = cell_names

# Gene Names
head(rownames(sc_gene_counts))
# Cell Names
head(colnames(sc_gene_counts))

#find genes in common between the two datasets
inCommon_tenx<-subset(Tenxmat, rownames(Tenxmat)%in%rownames(sc_gene_counts))
dim(inCommon_tenx)
#20428  3321
#3321 cells in the 10X dataset

inCommon_c1<-subset(sc_gene_counts, rownames(sc_gene_counts)%in%rownames(Tenxmat))
dim(inCommon_c1)
#20428   109
#109 cells in the C1 dataset

gene_c1   <- rownames(inCommon_c1)
gene_tenx <- rownames(inCommon_tenx)
ord_gene_c1 <- order(gene_c1)
ord_gene_tenx <- order(gene_tenx)

inCommon_c1   <- inCommon_c1[ord_gene_c1, ]
inCommon_tenx <- inCommon_tenx[ord_gene_tenx, ]

save(inCommon_c1, file = "GenesInCommon_c1.RData")
save(inCommon_tenx, file = "GenesInCommon_tenx.RData")

#count non-zerocells for each gene (row)
nonZero_c1 <- rowSums(inCommon_c1 != 0)
nonZero_tenx <- rowSums(inCommon_tenx != 0)

#divide by number of cells in dataset
C1 <- nonZero_c1/109
Chromium   <- nonZero_tenx/3321
gene_names <- rownames(inCommon_c1)
fractionCells <-data.frame(genes = gene_names, C1 = C1,Chromium = Chromium)

#count non zero genes in each cell
nonZerogenes_c1 <- colSums(inCommon_c1 != 0)
nonZerogenes_tenx <- colSums(inCommon_tenx != 0)

#number of genes seen in Fluidigm and in Chromium


#######Plot fraction of non-zero cells within a gene (each point is a gene)
fractionCells$C1 <- as.factor(fractionCells$C1)
fractionCells$Chromium <- as.factor(fractionCells$Chromium)

ggplot(data = fractionCells, aes(x=Chromium, y=C1)) + geom_point(alpha = 0.4)+ 
  geom_abline(intercept=0, slope=1, color='red', size=1.1) +
  # scale_y_discrete(breaks = seq(0,1,by = 0.1)) + 
  # scale_x_discrete(breaks = seq(0,1,by = 0.1))+
  labs(title="Gene-wise fraction of non-zero cells")

####### Plot fraction of non-zero cells within a gene (each point is a gene) 
#### Plus factors for 10x and C1
fc_melt <- melt(fractionCells, value.name = "cell_frac")
fc_melt$variable <- as.factor(fc_melt$variable)

ggplot(data = fc_melt, aes(x = variable, y = cell_frac)) + 
  geom_boxplot() + ggtitle("Boxplot of fractions of cells")

ggplot(data = fractionCells, aes(x = C1, y = Chromium, color = genes)) + 
  geom_point(alpha = 0.4) + ggtitle("Boxplot of fractions of cells")
