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

import preproc_data_func.py

print "Reading in Matrix"
# ====================== Read h5 data in ==================== #
GeneBCMatrix = collections.namedtuple('GeneBCMatrix', ['gene_ids', 'gene_names',
                                                       'barcodes', 'matrix'])
filtered_matrix_h5 = "../../data/1M_neurons_filtered_gene_bc_matrices_h5.h5"
genome = "mm10"
gene_bc_matrix = get_matrix_from_h5(filtered_matrix_h5, genome)

# Store only expression matrix
exp_mat_count = gene_bc_matrix[3]
gene_names    = gene_bc_matrix[1]

print "Reading in Matrix completed"
# # ================== Looking at the data ========================= #
# exp_min_cells, exp_max_cells = exp_mat.min(axis = 0), exp_mat.max(axis = 0)
# exp_min_genes, exp_max_genes = exp_mat.min(axis = 1), exp_mat.max(axis = 1)

# stats.describe(exp_max_cells.todense(), axis = 1)
# stats.describe(exp_min_cells.todense(), axis = 1)
# stats.describe(exp_max_genes.todense(), axis = 0)
# stats.describe(exp_min_genes.todense(), axis = 0)

# # ================= Test Case ================= # #
indptr = np.array([0, 2, 3, 6, 9])
indices = np.array([0, 2, 2, 0, 1, 2, 0, 1, 2])
data    = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

a = sp_sparse.csc_matrix((data, indices, indptr), shape=(3,4))

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

mask = np.zeros(exp_mat_count.shape[0], dtype = bool)
mask_t = np.ones(len(ind_gene[0]), dtype = bool)

mask[ind_gene[0]] = mask_t
exp_mat_count = exp_mat_count[~mask, :]
gene_names = gene_names[~mask]

print "Filter 0 cells and genes completed"

# ********* Compute counts per million ********* #
numcells = exp_mat_count.shape[1]
numgenes = exp_mat_count.shape[0]



print "Compute CPM of genes"

exp_mat_cpm = sparse_cpm_col(exp_mat_count)

# sp_sparse.save_npz('millcell_filtered_cpm.npz', exp_mat)
# exp_mat = sp_sparse.load_npz('millcell_filtered_cpm.npz')

# print "Compute Log"

exp_mat_cpm_csr = exp_mat_cpm.tocsr()

# log_exp_mat_cpm_zero_in  = exp_mat_cpm.log1p()
# log_exp_mat_cpm_zero_out = exp_mat_cpm.copy()
# log_exp_mat_cpm_zero_out.data = np.log(log_exp_mat_cpm_zero_out.data)

# ============== Finding highly variable genes ================== #
# ********** Compute Mean w/w.o. 0's across all genes************ #
print "Compute Mean"



mean_genes_in  = exp_mat_cpm.mean(axis=1)
mean_genes_out = sparse_mean_mat(exp_mat_cpm_csr)

# ********** Compute SD w/w.o. 0's ************ #


print "Compute sd_genes_zero_in"
sd_genes_zero_in  = std_sparse_zero_in(exp_mat_cpm)
var_genes_zero_in = np.power(sd_genes_zero_in,2)
log_vmr_in = np.log(var_genes_zero_in/mean_genes_in)

print "Compute sd_genes_zero_out"
sd_genes_zero_out  = std_sparse_zero_out_all(exp_mat_cpm_csr)
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

# Whether zero's are 1 = in or 2 = out
zero_type = 1

ind_hvg_in  = IndxVariableGenes(log_mean_genes_in, log_vmr_in, mean_low_cutoff = 0.1, mean_high_cutoff = 10, 
    disp_low_cutoff = 0.1, disp_high_cutoff = float("inf"), zero_type = 1)
ind_hvg_out = IndxVariableGenes(log_mean_genes_out, log_vmr_out, mean_low_cutoff = 0.1, mean_high_cutoff = 10, 
    disp_low_cutoff = 0.1, disp_high_cutoff = float("inf"), zero_type = 2)

len_hvg_in  = len(ind_hvg_in['ind_genes'])
len_hvg_out = len(ind_hvg_out['ind_genes'])

genes_dispersion_scaled_in  = ind_hvg_in['genes_dispersion_scaled']
genes_dispersion_scaled_out = ind_hvg_out['genes_dispersion_scaled']

# Plot of common highly variable genes
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

# # ********* Test ********* #
# indptr  = np.array([0,2,3,6])
# indices = np.array([0,2,2,0,1,2])
# data    = np.array([1,2,3,4,5,6])
# sp_mat  = sp_sparse.csc_matrix((data, indices, indptr), shape=(3,3), dtype=float)

# def std_array_zero_out(x_mat):
#     m = x_mat.shape[0]
#     a = np.zeros((1,m))
#     for i in range(m):
#         b = np.where(x_mat[i,]!=0)
#         a[0,i] = np.std(x_mat[i,b[0]])
#     return a

# ************* Second QC *************** #
# ** Get rid of all genes with sd < 0.1 #

# # Take only highly variable genes
# # Genes with > 0.6 standard deviation
# ind_sd_in  = np.where(sd_genes_zero_in  >= 0.6)
# ind_sd_out = np.where(sd_genes_zero_out >= 0.6)
# final_gn_zero_in_mat  = log_exp_mat_zero_in[ind_sd_in[0], :]
# final_gn_zero_out_mat = log_exp_mat_zero_out[ind_sd_out[0], :]


# ****** Compute Highly Variable Genes - Seurat ******* #
# Compute the dispersion for each method

# ============ Divide data into train/test set ============= #


x_train_sp_in, x_dev_sp_in, x_test_sp_in    = subsample(final_gn_zero_in_mat)
x_train_sp_out, x_dev_sp_out, x_test_sp_out = subsample(final_gn_zero_out_mat)

sp_sparse.save_npz('train_set_zero_in.npz', x_train_sp_in)
sp_sparse.save_npz('dev_set_zero_in.npz', x_dev_sp_in)
sp_sparse.save_npz('test_set_zero_in.npz', x_test_sp_in)

sp_sparse.save_npz('train_set_zero_out.npz', x_train_sp_out)
sp_sparse.save_npz('dev_set_zero_out.npz', x_dev_sp_out)
sp_sparse.save_npz('test_set_zero_out.npz', x_test_sp_out)

# Convert matrices to arrays
x_train = x_train_sp.toarray()
x_dev   = x_dev_sp.toarray()
x_test  = x_test_sp.toarray()

# filename = "gene_name_sub.npz"
# np.savez(filename, gene_names_sub, ind_vmr_sub)
# # ************ save to csv ************* #
# import csv
# file_dir = "geneList_smallVMR.csv" #where you want the file to be downloaded to 

# csv  = open(file_dir, "wb") 

# columnTitleRow = "Gene"
# csv.write(columnTitleRow)

# for j in range(len(gene_names_sub)):
#   gn  = gene_names_sub[j]
#   row = gn + "\n"
#   csv.write(row)
# csv.close()

# # ******** Counter for small VMR ******** #
# store_freq = dict()
# for j in range(len(gene_names_sub)):
#     temp = exp_mat_count_sub[j,:].data
#     store_freq[str(gene_names_sub[j])] = Counter(temp)

# std_freq = np.zeros((len(gene_names_sub),1))
# for j in range(len(gene_names_sub)):
#     temp = exp_mat_count_sub[j,:].data
#     std_freq[j,0] = np.std(temp)

# Counter(std_freq[:,0]) # all std's are 0

# freq_val = []
# for key, value in store_freq.items():
#     freq_val.append(value.keys()[0])

# Counter(freq_val) #1:5224,
# #2:4('Klrc2','Spin2d','Gm11345', 'Defb50')
# #3:2("'Gm30484', 'Gm14718'"), 5:1("Gzmb"), 54:1("Gzma")

# freq_val_np = np.asarray(freq_val)
# which_1 = np.where(freq_val_np == 1)
# which_2 = np.where(freq_val_np == 2)
# which_3 = np.where(freq_val_np == 3)
# which_5 = np.where(freq_val_np == 5)
# which_54 = np.where(freq_val_np == 54)

# exp_mat_cpm_sub = exp_mat_cpm[ind_vmr_sub[0],:]

# store_freq_cpm = dict()
# for j in range(len(gene_names_sub)):
#     temp = exp_mat_cpm_sub[j,:].data
#     store_freq_cpm[str(gene_names_sub[j])] = Counter(temp)




