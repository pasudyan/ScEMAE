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
log_vmr_in = np.log(var_genes_zero_in/mean_genes_in)

print "Compute sd_genes_zero_out and log_vmr_out"
sd_genes_zero_out  = preproc_functions.std_sparse_zero_out_all(exp_mat_cpm_csr)
var_genes_zero_out = np.power(sd_genes_zero_out,2)
log_vmr_out = np.log(var_genes_zero_out/mean_genes_out)

# List of genes with 0 variance:
var_0_genes = np.where(var_genes_zero_out == 0)[0]
mask = np.zeros(exp_mat_cpm.shape[0], dtype = bool)
mask_t = np.ones(len(var_0_genes), dtype = bool)
mask[var_0_genes] = mask_t
var_genes_zero_out_sub = var_genes_zero_out[~mask, :]
mean_genes_out_sub = mean_genes_out[~mask, :]

log_vmr_out_sub = np.log(var_genes_zero_out_sub/mean_genes_out_sub)

print "Plot standard deviations"

# Plot histogram of SD genes zero in
plt.figure(figsize=(8,8))
plt.hist(log_vmr_in, bins='auto')
plt.xlabel('logVMR per gene')
plt.ylabel('Frequency')
plt.title('Log VMR Distribution')
plt.savefig("Figures/hist_vmr_genes_zero_in.png")

# Plot histogram of SD genes zero out
plt.figure(figsize=(8,8))
plt.hist(log_vmr_out_sub, bins='auto')
plt.xlabel('logVMR per gene')
plt.ylabel('Frequency')
plt.title('Log VMR Distribution')
plt.savefig("Figures/hist_vmr_genes_zero_out.png")

# Scatter plot of vmr in and out
log_vmr_in_sub = log_vmr_in[~mask, :]
plt.figure(figsize=(8,8))
plt.scatter(log_vmr_out_sub[:,0], log_vmr_in_sub[:,0], alpha = 0.5)
plt.xlabel('Log VMR - out')
plt.ylabel('Log VMR - in')
plt.title('Scatter Plot of Log VMR')
plt.savefig("Figures/scatter_log_vmr_genes.png")

# Plot histogram of VMR in for -Inf VMR out
log_vmr_in_sub_in = log_vmr_in[mask,:]
plt.figure(figsize=(8,8))
plt.hist(log_vmr_in_sub_in, bins='auto')
plt.xlabel('logVMR per gene')
plt.ylabel('Frequency')
plt.title('Log VMR Distribution')
plt.savefig("Figures/hist_vmr_genes_zero_in_sub_in.png")

# ******** Finding Variable Genes ******* #
log_mean_genes_in  = np.log(mean_genes_in)
log_mean_genes_out = np.log(mean_genes_out)

print "Find highly variable genes"

# Whether zero's are 1 = in or 2 = out
zero_type   = 1

ind_hvg_in  = IndxVariableGenes(log_mean_genes_in, log_vmr_in, mean_low_cutoff = 0.1, mean_high_cutoff = 10, 
    disp_low_cutoff = 0.1, disp_high_cutoff = float("inf"), zero_type = 1)
ind_hvg_out = IndxVariableGenes(log_mean_genes_out, log_vmr_out, mean_low_cutoff = 0.1, mean_high_cutoff = 10, 
    disp_low_cutoff = 0.1, disp_high_cutoff = float("inf"), zero_type = 2)

len_hvg_in  = len(ind_hvg_in['ind_genes'])
len_hvg_out = len(ind_hvg_out['ind_genes'])

genes_dispersion_scaled_in  = ind_hvg_in['genes_dispersion_scaled']
genes_dispersion_scaled_out = ind_hvg_out['genes_dispersion_scaled']

print "Plot of common highly variable genes"

plt.figure(figsize=(8,8))
venn2([set(ind_hvg_in), set(ind_hvg_out)], ('Zero_In', 'Zero_out'))
plt.title('Venn Diagram of common Highly Variable Genes')
plt.savefig("Figures/venn_common_hvg_genes.png")

set_in  = set(ind_hvg_in['ind_genes'])
set_out = set(ind_hvg_out['ind_genes'])

int_sets = np.asarray(list(set_in.intersection(set_out)))
common_genes = gene_names[int_sets]

# ******** Find Genes with small VMR out ******* #
ind_vmr_sub = np.where(log_vmr_out <= -25) #there are 5232 genes
exp_mat_count_sub = exp_mat_count[ind_vmr_sub[0],:]
gene_names_sub    = gene_names[ind_vmr_sub[0]]

# ******* Sort Array and take top 100 ****** #
n = 1000

a = np.asarray(np.argsort(log_vmr_in, axis = 0)[::-1][:n]).ravel()
b = np.asarray(np.argsort(log_vmr_out, axis = 0)[::-1][:n]).ravel()

# ********* Plot Venn Diagram ********* #
plt.figure(figsize=(8,8))
venn2([set(a), set(b)], ('Zero_In', 'Zero_out'))
plt.title('Venn Diagram of top 1000 genes')
plt.savefig("Figures/venn_top_1000_genes.png")

n = 3000

a = np.asarray(np.argsort(log_vmr_in, axis = 0)[::-1][:n]).ravel()
b = np.asarray(np.argsort(log_vmr_out, axis = 0)[::-1][:n]).ravel()

plt.figure(figsize=(8,8))
venn2([set(a), set(b)], ('Zero_In', 'Zero_out'))
plt.title('Venn Diagram of top 3000 genes')
plt.savefig("Figures/venn_top_3000_genes.png")

n = 5000

a = np.asarray(np.argsort(log_vmr_in, axis = 0)[::-1][:n]).ravel()
b = np.asarray(np.argsort(log_vmr_out, axis = 0)[::-1][:n]).ravel()

plt.figure(figsize=(8,8))
venn2([set(a), set(b)], ('Zero_In', 'Zero_out'))
plt.title('Venn Diagram of top 5000 genes')
plt.savefig("Figures/venn_top_5000_genes.png")

n = 10000

a = np.asarray(np.argsort(log_vmr_in, axis = 0)[::-1][:n]).ravel()
b = np.asarray(np.argsort(log_vmr_out, axis = 0)[::-1][:n]).ravel()

plt.figure(figsize=(8,8))
venn2([set(a), set(b)], ('Zero_In', 'Zero_out'))
plt.title('Venn Diagram of top 10000 genes')
plt.savefig("Figures/venn_top_10000_genes.png")





