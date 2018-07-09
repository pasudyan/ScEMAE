import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import csv
import collections
import scipy.sparse as sp_sparse
import tables
import sys

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

import preproc_functions

print "Reading in Matrix"
# ====================== Read h5 data in ==================== #
filtered_matrix_h5 = "../../data/1M_neurons_filtered_gene_bc_matrices_h5.h5"
genome = "mm10"
gene_bc_matrix = preproc_functions.get_matrix_from_h5(filtered_matrix_h5, genome)

# Store only expression matrix and gene names 
exp_mat_count = gene_bc_matrix[3]
gene_names    = gene_bc_matrix[1]

print "Reading in Matrix completed"

# ============ Preprocessing data for quality control ============= #

# ************* First QC *************** #
# ** Get rid of all 0 cells and genes ** #

print "Filter 0 cells and genes"
# Compute the sum across all cells and genes
gene_sum = exp_mat_count.sum(axis = 1)
cell_sum = exp_mat_count.sum(axis = 0)

# Getting rid of 0 cells and genes
ind_gene = np.where(gene_sum == 0)
ind_cell = np.where(cell_sum == 0) #no cell with 0 expressions

mask   = np.zeros(exp_mat_count.shape[0], dtype = bool)
mask_t = np.ones(len(ind_gene[0]), dtype = bool)

mask[ind_gene[0]] = mask_t
exp_mat_count = exp_mat_count[~mask, :]
gene_names = gene_names[~mask]

print "Filter 0 cells and genes completed"

# ********* Compute counts per million ********* #
numcells = exp_mat_count.shape[1]
numgenes = exp_mat_count.shape[0]

print "Compute CPM of genes"

exp_mat_cpm = preproc_functions.sparse_cpm_col(exp_mat_count)

print "Convert to CSR matrix"

exp_mat_cpm_csr = exp_mat_cpm.tocsr()

# ============== Finding highly variable genes ================== #
# ********** Compute Mean w/w.o. 0's across all genes************ #
print "Compute Mean"

mean_genes_in  = exp_mat_cpm.mean(axis=1)
mean_genes_out = preproc_functions.sparse_mean_mat(exp_mat_cpm_csr)

# ********** Compute SD w/w.o. 0's ************ #
print "Compute sd_genes_zero_in and log_vmr_in"

sd_genes_zero_in  = preproc_functions.std_sparse_zero_in(exp_mat_cpm)
var_genes_zero_in = np.power(sd_genes_zero_in,2)
log_vmr_in = np.log1p(var_genes_zero_in/mean_genes_in)

print "Compute sd_genes_zero_out and log_vmr_out"
sd_genes_zero_out  = preproc_functions.std_sparse_zero_out_all(exp_mat_cpm_csr)
var_genes_zero_out = np.power(sd_genes_zero_out,2)
log_vmr_out = np.log1p(var_genes_zero_out/mean_genes_out)

# List of genes with 0 variance:
var_0_genes = np.where(var_genes_zero_out == 0)[0]
mask = np.zeros(exp_mat_cpm.shape[0], dtype = bool)
mask_t = np.ones(len(var_0_genes), dtype = bool)
mask[var_0_genes] = mask_t
var_genes_zero_out_sub = var_genes_zero_out[~mask, :]
mean_genes_out_sub = mean_genes_out[~mask, :]

log_vmr_out_sub = np.log1p(var_genes_zero_out_sub/mean_genes_out_sub)

# ********* Save results ******** #
np.savez("save_results/zero_out_summary_data.npz", var_genes_zero_out = var_genes_zero_out, mean_genes_out = mean_genes_out, 
	log_vmr_out = log_vmr_out)

np.savez("save_results/zero_in_summary_data.npz", var_genes_zero_in = var_genes_zero_in, mean_genes_in = mean_genes_in, 
	log_vmr_in = log_vmr_in)

print "Plot standard deviations"

# ********* Load results ******** #
data_out = np.load("save_results/zero_out_summary_data.npz")
data_in  = np.load("save_results/zero_in_summary_data.npz")

var_genes_zero_in = data_in['var_genes_zero_in']
mean_genes_in = data_in['mean_genes_in']
log_vmr_in = data_in['log_vmr_in']

var_genes_zero_out = data_out['var_genes_zero_out']
mean_genes_out = data_out['mean_genes_out']
log_vmr_out = data_out['log_vmr_out']

log_mean_genes_in  = np.log(mean_genes_in)
log_mean_genes_out = np.log(mean_genes_out)

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
plt.scatter(log_vmr_out[:,0], log_vmr_in[:,0], alpha = 0.5)
plt.xlabel('Log VMR - out')
plt.ylabel('Log VMR - in')
plt.title('Scatter Plot of Log VMR')
plt.savefig("Figures/scatter_log_vmr_genes.png")

# Plot scatter of variance in and out
plt.figure(figsize=(8,8))
plt.scatter(np.log1p(var_genes_zero_in[:,0]), np.log1p(var_genes_zero_out[:,0]), alpha = 0.5)
plt.xlabel('Log Variance zero in')
plt.ylabel('Log Variance zero out')
plt.title('Log Variance comparisons')
plt.savefig("Figures/scatter_log_var_compare.png")

# Plot scatter of mean in and out
plt.figure(figsize=(8,8))
plt.scatter(np.log(mean_genes_in[:,0]), np.log(mean_genes_out[:,0]), alpha = 0.5)
plt.xlabel('Log mean zero in')
plt.ylabel('Log mean zero out')
plt.title('Log Mean comparisons')
plt.savefig("Figures/scatter_log_mean_compare.png")

# Plot scatter of mean in and disp
plt.figure(figsize=(8,8))
plt.scatter(np.log(mean_genes_in[:,0]), log_vmr_in, alpha = 0.5)
plt.xlabel('Log mean zero in')
plt.ylabel('Log dispersion non-scaled')
plt.title('Mean dispersion comparisons')
plt.savefig("Figures/scatter_log_mean_disp_ori_in.png")

plt.figure(figsize=(8,8))
plt.scatter(mean_genes_in[:,0], log_vmr_in, alpha = 0.5)
plt.xlabel('Log mean zero in')
plt.ylabel('Log dispersion non-scaled')
plt.title('Mean dispersion comparisons')
plt.savefig("Figures/scatter_mean_disp_ori_in.png")

# Plot scatter of mean out and disp
plt.figure(figsize=(8,8))
plt.scatter(np.log(mean_genes_out[:,0]), log_vmr_out, alpha = 0.5)
plt.xlabel('Log mean zero out')
plt.ylabel('Log dispersion non-scaled')
plt.title('Mean dispersion comparisons')
plt.savefig("Figures/scatter_log_mean_disp_ori_out.png")

plt.figure(figsize=(8,8))
plt.scatter(mean_genes_out[:,0], log_vmr_out, alpha = 0.5)
plt.xlabel('Log mean zero out')
plt.ylabel('Log dispersion non-scaled')
plt.title('Mean dispersion comparisons')
plt.savefig("Figures/scatter_mean_disp_ori_out.png")

# ******** Finding Variable Genes ******* #
print "Find highly variable genes"

# ********* Plot of common genes with mean_low_cutoff = 0.1, disp_low_cutoff = 0.1 *********** #
mean_low_cutoff = 0.1
disp_low_cutoff = 0.1

ind_hvg_in  = preproc_functions.IndxVariableGenes(log_mean_genes_in, log_vmr_in, numgenes, 
	mean_low_cutoff = mean_low_cutoff, mean_high_cutoff = float("inf"), 
	disp_low_cutoff = disp_low_cutoff, disp_high_cutoff = float("inf"), zero_type = 1)
ind_hvg_out = preproc_functions.IndxVariableGenes(log_mean_genes_out, log_vmr_out, numgenes, 
	mean_low_cutoff = mean_low_cutoff, mean_high_cutoff = float("inf"), 
	disp_low_cutoff = disp_low_cutoff, disp_high_cutoff = float("inf"), zero_type = 2)

len_hvg_in  = len(ind_hvg_in['ind_genes'])
len_hvg_out = len(ind_hvg_out['ind_genes'])

genes_dispersion_scaled_in  = ind_hvg_in['genes_dispersion_scaled']
genes_dispersion_scaled_out = ind_hvg_out['genes_dispersion_scaled']

print "Plot of common highly variable genes"

plt.figure(figsize=(8,8))
venn2([set(ind_hvg_in['ind_genes']), set(ind_hvg_out['ind_genes'])], ('Zero_In', 'Zero_out'))
plt.title('Venn Diagram of common Highly Variable Genes')
plt.savefig("Figures/venn_common_hvg_genes_mean_0.1_disp_0.1.png")

matplotlib.pyplot.close('all')


# ********* Plot of common genes with mean_low_cutoff = 0.5, disp_low_cutoff = 0.5 *********** #
mean_low_cutoff = 0.5
disp_low_cutoff = 0.5

ind_hvg_in  = preproc_functions.IndxVariableGenes(log_mean_genes_in, log_vmr_in, numgenes, 
	mean_low_cutoff = mean_low_cutoff, mean_high_cutoff = float("inf"), 
	disp_low_cutoff = disp_low_cutoff, disp_high_cutoff = float("inf"), zero_type = 1)
ind_hvg_out = preproc_functions.IndxVariableGenes(log_mean_genes_out, log_vmr_out, numgenes, 
	mean_low_cutoff = mean_low_cutoff, mean_high_cutoff = float("inf"), 
	disp_low_cutoff = disp_low_cutoff, disp_high_cutoff = float("inf"), zero_type = 2)

len_hvg_in  = len(ind_hvg_in['ind_genes'])
len_hvg_out = len(ind_hvg_out['ind_genes'])

genes_dispersion_scaled_in  = ind_hvg_in['genes_dispersion_scaled']
genes_dispersion_scaled_out = ind_hvg_out['genes_dispersion_scaled']

print "Plot of common highly variable genes"

plt.figure(figsize=(8,8))
venn2([set(ind_hvg_in['ind_genes']), set(ind_hvg_out['ind_genes'])], ('Zero_In', 'Zero_out'))
plt.title('Venn Diagram of common Highly Variable Genes')
plt.savefig("Figures/venn_common_hvg_genes_mean_0.5_disp_0.5.png")

matplotlib.pyplot.close('all')

# ********* Plot of common genes with mean_low_cutoff = 1, disp_low_cutoff = 1 *********** #
mean_low_cutoff = 1
disp_low_cutoff = 1

ind_hvg_in  = preproc_functions.IndxVariableGenes(log_mean_genes_in, log_vmr_in, numgenes, 
	mean_low_cutoff = mean_low_cutoff, mean_high_cutoff = float("inf"), 
	disp_low_cutoff = disp_low_cutoff, disp_high_cutoff = float("inf"), zero_type = 1)
ind_hvg_out = preproc_functions.IndxVariableGenes(log_mean_genes_out, log_vmr_out, numgenes, 
	mean_low_cutoff = mean_low_cutoff, mean_high_cutoff = float("inf"), 
	disp_low_cutoff = disp_low_cutoff, disp_high_cutoff = float("inf"), zero_type = 2)

len_hvg_in  = len(ind_hvg_in['ind_genes'])
len_hvg_out = len(ind_hvg_out['ind_genes'])

genes_dispersion_scaled_in  = ind_hvg_in['genes_dispersion_scaled']
genes_dispersion_scaled_out = ind_hvg_out['genes_dispersion_scaled']

print "Plot of common highly variable genes"

plt.figure(figsize=(8,8))
venn2([set(ind_hvg_in['ind_genes']), set(ind_hvg_out['ind_genes'])], ('Zero_In', 'Zero_out'))
plt.title('Venn Diagram of common Highly Variable Genes')
plt.savefig("Figures/venn_common_hvg_genes_mean_1_disp_1.png")

matplotlib.pyplot.close('all')

# ********* Plot of common genes with mean_low_cutoff = 2, disp_low_cutoff = 2 *********** #
mean_low_cutoff = 2
disp_low_cutoff = 2

ind_hvg_in  = preproc_functions.IndxVariableGenes(log_mean_genes_in, log_vmr_in, numgenes, 
	mean_low_cutoff = mean_low_cutoff, mean_high_cutoff = float("inf"), 
	disp_low_cutoff = disp_low_cutoff, disp_high_cutoff = float("inf"), zero_type = 1)
ind_hvg_out = preproc_functions.IndxVariableGenes(log_mean_genes_out, log_vmr_out, numgenes, 
	mean_low_cutoff = mean_low_cutoff, mean_high_cutoff = float("inf"), 
	disp_low_cutoff = disp_low_cutoff, disp_high_cutof = float("inf"), zero_type = 2)

len_hvg_in  = len(ind_hvg_in['ind_genes'])
len_hvg_out = len(ind_hvg_out['ind_genes'])

genes_dispersion_scaled_in  = ind_hvg_in['genes_dispersion_scaled']
genes_dispersion_scaled_out = ind_hvg_out['genes_dispersion_scaled']

print "Plot of common highly variable genes"

plt.figure(figsize=(8,8))
venn2([set(ind_hvg_in['ind_genes']), set(ind_hvg_out['ind_genes'])], ('Zero_In', 'Zero_out'))
plt.title('Venn Diagram of common Highly Variable Genes')
plt.savefig("Figures/venn_common_hvg_genes_mean_2_disp_2.png")

matplotlib.pyplot.close('all')








