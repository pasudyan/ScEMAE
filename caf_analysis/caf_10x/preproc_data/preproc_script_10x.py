import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import csv
import collections
import scipy.sparse as sp_sparse
import tables
import sys
import scipy as sp

from scipy import stats
from scipy.io import mmread
from random import sample
from math import floor

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from collections import Counter

import preproc_functions_10x

print "Reading in Matrix"
# ====================== Read matrix data in ==================== #
caf_ctrl_count = mmread("../data/Control/GRCh38/matrix.mtx")

# Read genes name in columns
with open("../data/Control/GRCh38/genes.tsv") as genes_all:
	for column in zip(*[line for line in csv.reader(genes_all, dialect="excel-tab")]):
		genes_name = column

genes = np.reshape(genes_name, (len(genes_name), 1))
print "Reading in Matrix completed"

# ============ Preprocessing data for quality control ============= #

# ************* First QC *************** #
# ** Get rid of all 0 cells and genes ** #
print "Filter 0 cells and genes"
# Compute the sum across all cells and genes
gene_sum = caf_ctrl_count.sum(axis = 1)
cell_sum = caf_ctrl_count.sum(axis = 0)

# Getting rid of 0 cells and genes
ind_gene = np.where(gene_sum == 0)[0]
ind_cell = np.where(cell_sum == 0) #no cell with 0 expressions

mask   = np.zeros(caf_ctrl_count.shape[0], dtype = bool)
mask_t = np.ones(len(ind_gene), dtype = bool)
mask[ind_gene] = mask_t

ctrl_count_arr = sp_sparse.coo_matrix.toarray(caf_ctrl_count)
ctrl_count_arr = ctrl_count_arr[~mask, :]
genes = genes[~mask, :]

print "Filter 0 cells and genes completed"

# ********* Compute counts per million ********* #
numcells = ctrl_count_arr.shape[1]
numgenes = ctrl_count_arr.shape[0]

print "Compute CPM of genes"

# test = np.reshape(np.matrix(range(9)), (3,3))
ctrl_norm = preproc_functions_10x.array_normalize(ctrl_count_arr, scale = 1e4)

stats.describe(ctrl_count_arr[:,1])
stats.describe(ctrl_norm[:,1])

# Plot histogram of normalize counts and counts
plt.figure(figsize=(8,8))
plt.hist(ctrl_norm[:,1], bins=10)
plt.xlabel('Norm Counts')
plt.ylabel('Frequency')
plt.title('Normalize counts gene 1')
plt.savefig("Figures/Normalize_counts_gene_1.png")

plt.figure(figsize=(8,8))
plt.hist(ctrl_count_arr[:,1], bins=10)
plt.xlabel('Norm Counts')
plt.ylabel('Frequency')
plt.title('Counts gene 1')
plt.savefig("Figures/Counts_gene_1.png")

# ============== Finding highly variable genes ================== #
# ********** Compute Mean w/w.o. 0's across all genes************ #
print "Compute Mean"

mean_genes_in  = np.mean(ctrl_norm, axis = 1)
mean_genes_out = preproc_functions_10x.compute_mean_zero_out(ctrl_norm)

stats.describe(mean_genes_in)
stats.describe(mean_genes_out)

# Plot histogram of mean zero in and out
plt.figure(figsize=(8,8))
plt.hist(mean_genes_in, bins=10)
plt.xlabel('Mean')
plt.ylabel('Frequency')
plt.title('Histogram of Mean Genes In')
plt.savefig("Figures/hist_mean_gene_in.png")

plt.figure(figsize=(8,8))
plt.hist(mean_genes_out, bins=10)
plt.xlabel('Mean')
plt.ylabel('Frequency')
plt.title('Histogram of Mean Genes out')
plt.savefig("Figures/hist_mean_gene_out.png")

plt.figure(figsize=(8,8))
plt.scatter(mean_genes_in, mean_genes_out)
plt.xlabel('Mean genes in')
plt.ylabel('Mean genes out')
plt.title('Scatter plot of genes and means')
plt.savefig("Figures/scatter_mean_genes_in_out.png")

# ********** Compute SD w/w.o. 0's ************ #
print "Compute sd_genes_zero_in and sd_genes_zero_out"

sd_genes_zero_in  = np.std(ctrl_norm, axis = 1)
sd_genes_zero_out = preproc_functions_10x.compute_sd_zero_out(ctrl_norm)

var_genes_zero_in  = np.power(sd_genes_zero_in, 2)
var_genes_zero_out = np.power(sd_genes_zero_out, 2)

stats.describe(sd_genes_zero_in)
stats.describe(sd_genes_zero_out)

# Plot histogram of sd zero in and out
plt.figure(figsize=(8,8))
plt.hist(sd_genes_zero_in, bins=10)
plt.xlabel('SD')
plt.ylabel('Frequency')
plt.title('Histogram of SD Genes In')
plt.savefig("Figures/hist_sd_gene_in.png")

plt.figure(figsize=(8,8))
plt.hist(sd_genes_zero_out, bins=10)
plt.xlabel('SD')
plt.ylabel('Frequency')
plt.title('Histogram of SD Genes out')
plt.savefig("Figures/hist_sd_gene_out.png")

plt.figure(figsize=(8,8))
plt.scatter(sd_genes_zero_in, sd_genes_zero_out)
plt.xlabel('SD genes in')
plt.ylabel('SD genes out')
plt.title('Scatter plot of genes and sd')
plt.savefig("Figures/scatter_sd_genes_in_out.png")

# ********** Compute LogVMR w/w.o. 0's ************ #

print "Compute log_vmr_zero_in and log_vmr_zero_out"

log_vmr_in  = np.log1p(var_genes_zero_in/mean_genes_in)
log_vmr_out = np.log1p(var_genes_zero_out/mean_genes_out)

stats.describe(log_vmr_in)
stats.describe(log_vmr_out)

# Plot histogram of logVMR in
plt.figure(figsize=(8,8))
plt.hist(log_vmr_in, bins='auto')
plt.xlabel('logVMR per gene')
plt.ylabel('Frequency')
plt.title('Log VMR Distribution')
plt.savefig("Figures/hist_vmr_genes_zero_in.png")

# Plot histogram of logVMR out
plt.figure(figsize=(8,8))
plt.hist(log_vmr_out, bins='auto')
plt.xlabel('logVMR per gene')
plt.ylabel('Frequency')
plt.title('Log VMR Distribution')
plt.savefig("Figures/hist_vmr_genes_zero_out.png")

# Scatter plot of log vmr in and out
plt.figure(figsize=(8,8))
plt.scatter(log_vmr_out, log_vmr_in, alpha = 0.5)
plt.xlabel('Log VMR - out')
plt.ylabel('Log VMR - in')
plt.title('Scatter Plot of Log VMR')
plt.savefig("Figures/scatter_log_vmr_genes.png")

matplotlib.pyplot.close('all')

# ********** Compute Scaled Dispersion w/w.o. 0's ************ #
log_mean_genes_in  = np.log(mean_genes_in)
log_mean_genes_out = np.log(mean_genes_out)

# ******** Finding Variable Genes ******* #
print "Find highly variable genes"

# ********* Plot of common genes with mean_low_cutoff = 0.1, disp_low_cutoff = 0.1 *********** #
mean_low_cutoff = 0.1
disp_low_cutoff = 0.1

ind_hvg_in  = preproc_functions_10x.IndxVariableGenes(log_mean_genes_in, log_vmr_in, numgenes, 
	mean_low_cutoff = mean_low_cutoff, mean_high_cutoff = float("inf"), 
	disp_low_cutoff = disp_low_cutoff, disp_high_cutoff = float("inf"), zero_type = 1)
ind_hvg_out = preproc_functions_10x.IndxVariableGenes(log_mean_genes_out, log_vmr_out, numgenes, 
	mean_low_cutoff = mean_low_cutoff, mean_high_cutoff = float("inf"), 
	disp_low_cutoff = disp_low_cutoff, disp_high_cutoff = float("inf"), zero_type = 2)

print "Plot of common highly variable genes"

plt.figure(figsize=(8,8))
venn2([set(ind_hvg_in['ind_genes']), set(ind_hvg_out['ind_genes'])], ('Zero_In', 'Zero_out'))
plt.title('Venn Diagram of common Highly Variable Genes')
plt.savefig("Figures/venn_common_hvg_genes_mean_0.1_disp_0.1.png")

matplotlib.pyplot.close('all')

# ********* Plot of common genes with mean_low_cutoff = 0.5, disp_low_cutoff = 0.5 *********** #
mean_low_cutoff = 0.5
disp_low_cutoff = 0.5

ind_hvg_in  = preproc_functions_10x.IndxVariableGenes(log_mean_genes_in, log_vmr_in, numgenes, 
	mean_low_cutoff = mean_low_cutoff, mean_high_cutoff = float("inf"), 
	disp_low_cutoff = disp_low_cutoff, disp_high_cutoff = float("inf"), zero_type = 1)
ind_hvg_out = preproc_functions_10x.IndxVariableGenes(log_mean_genes_out, log_vmr_out, numgenes, 
	mean_low_cutoff = mean_low_cutoff, mean_high_cutoff = float("inf"), 
	disp_low_cutoff = disp_low_cutoff, disp_high_cutoff = float("inf"), zero_type = 2)

print "Plot of common highly variable genes"

plt.figure(figsize=(8,8))
venn2([set(ind_hvg_in['ind_genes']), set(ind_hvg_out['ind_genes'])], ('Zero_In', 'Zero_out'))
plt.title('Venn Diagram of common Highly Variable Genes')
plt.savefig("Figures/venn_common_hvg_genes_mean_0.5_disp_0.5.png")

matplotlib.pyplot.close('all')

# ********* Plot of common genes with mean_low_cutoff = 1, disp_low_cutoff = 1 *********** #
mean_low_cutoff = 1
disp_low_cutoff = 1

ind_hvg_in  = preproc_functions_10x.IndxVariableGenes(log_mean_genes_in, log_vmr_in, numgenes, 
	mean_low_cutoff = mean_low_cutoff, mean_high_cutoff = float("inf"), 
	disp_low_cutoff = disp_low_cutoff, disp_high_cutoff = float("inf"), zero_type = 1)
ind_hvg_out = preproc_functions_10x.IndxVariableGenes(log_mean_genes_out, log_vmr_out, numgenes, 
	mean_low_cutoff = mean_low_cutoff, mean_high_cutoff = float("inf"), 
	disp_low_cutoff = disp_low_cutoff, disp_high_cutoff = float("inf"), zero_type = 2)

print "Plot of common highly variable genes"

plt.figure(figsize=(8,8))
venn2([set(ind_hvg_in['ind_genes']), set(ind_hvg_out['ind_genes'])], ('Zero_In', 'Zero_out'))
plt.title('Venn Diagram of common Highly Variable Genes')
plt.savefig("Figures/venn_common_hvg_genes_mean_1_disp_1.png")

matplotlib.pyplot.close('all')

# ********* Plot of common genes with mean_low_cutoff = 2, disp_low_cutoff = 2 *********** #
mean_low_cutoff = 2
disp_low_cutoff = 2

ind_hvg_in  = preproc_functions_10x.IndxVariableGenes(log_mean_genes_in, log_vmr_in, numgenes, 
	mean_low_cutoff = mean_low_cutoff, mean_high_cutoff = float("inf"), 
	disp_low_cutoff = disp_low_cutoff, disp_high_cutoff = float("inf"), zero_type = 1)
ind_hvg_out = preproc_functions_10x.IndxVariableGenes(log_mean_genes_out, log_vmr_out, numgenes, 
	mean_low_cutoff = mean_low_cutoff, mean_high_cutoff = float("inf"), 
	disp_low_cutoff = disp_low_cutoff, disp_high_cutoff = float("inf"), zero_type = 2)

print "Plot of common highly variable genes"

plt.figure(figsize=(8,8))
venn2([set(ind_hvg_in['ind_genes']), set(ind_hvg_out['ind_genes'])], ('Zero_In', 'Zero_out'))
plt.title('Venn Diagram of common Highly Variable Genes')
plt.savefig("Figures/venn_common_hvg_genes_mean_2_disp_2.png")

matplotlib.pyplot.close('all')







