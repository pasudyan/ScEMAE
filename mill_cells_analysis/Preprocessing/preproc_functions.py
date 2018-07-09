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

GeneBCMatrix = collections.namedtuple('GeneBCMatrix', ['gene_ids', 'gene_names',
                                                       'barcodes', 'matrix'])

def get_matrix_from_h5(filename, genome):
    with tables.open_file(filename, 'r') as f:
        try:
            group = f.get_node(f.root, genome)
        except tables.NoSuchNodeError:
            print "That genome does not exist in this file."
            return None
        gene_ids = getattr(group, 'genes').read()
        gene_names = getattr(group, 'gene_names').read()
        barcodes = getattr(group, 'barcodes').read()
        data = getattr(group, 'data').read()
        indices = getattr(group, 'indices').read()
        indptr = getattr(group, 'indptr').read()
        shape = getattr(group, 'shape').read()
        matrix = sp_sparse.csc_matrix((data, indices, indptr), shape=shape)
        return GeneBCMatrix(gene_ids, gene_names, barcodes, matrix)

def sparse_cpm_row(x_mat_orig):
    x_mat = x_mat_orig.copy()
    if x_mat.format != 'csc':
        msg = 'CPM normalization on csc format only, not {0}'
        msg = msg.format(x_mat.format)
        raise ValueError(msg)
    row_norm = np.bincount(x_mat.indices, weights=x_mat.data)
    x_mat.data = (x_mat.data/np.take(row_norm, x_mat.indices))*1e4
    return(x_mat)

def sparse_cpm_col(x_mat_orig):
    x_mat = x_mat_orig.copy().astype("float64")
    numcells = x_mat.shape[1]
    if x_mat.format != 'csc':
        msg = 'CPM normalization on csc format only, not {0}'
        msg = msg.format(x_mat.format)
        raise ValueError(msg)
    col_sum = x_mat.sum(axis = 0).astype("float64")
    for i in range(numcells):
        print(i)
        x_mat.data[x_mat.indptr[i]:x_mat.indptr[i+1]] = (x_mat.data[x_mat.indptr[i]:x_mat.indptr[i+1]]/col_sum[0,i])*1e4
    return(x_mat)

def sparse_mean(x_mat):
    (x,y,z) = sp_sparse.find(x_mat)
    counts  = np.bincount(x)
    sums    = np.bincount(x, weights = z)
    avg     = sums/counts
    return(avg)

def sparse_mean_mat(x_mat):
    '''
        Input:  a csr matrix
        Output: mean of non-zero elements
        '''
    m = x_mat.shape[0]
    row_avg = np.zeros((m,1))
    for i in range(m):
        row = x_mat.getrow(i)
        row_avg[i,0] = sparse_mean(row)
    return row_avg

def std_sparse_zero_in(x_mat):
    x_data = x_mat.copy()
    x_data.data **= 2
    mean_xsq = x_data.mean(axis = 1)
    mean_x   = x_mat.mean(axis = 1)
    mean_x   = np.power(mean_x, 2)
    var = mean_xsq - (mean_x)
    sd = np.sqrt(var)
    return sd

def std_sparse_zero_out(x_mat):
    x_data = x_mat.copy()
    x_data.data **= 2
    mean_xsq = sparse_mean(x_data)
    mean_x   = sparse_mean(x_mat)
    mean_x **= 2
    var = mean_xsq - (mean_x)
    if (var <= 0):
        var = 1e-11
    sd = np.sqrt(var)
    return sd

def std_sparse_zero_out_all(x_mat):
    m  = x_mat.shape[0]
    sd = np.ones((m, 1))
    for i in range(m):
        row = x_mat.getrow(i).data
        sd[i,0] = np.std(row)
    return sd

def IndxVariableGenes(mean_genes, disp_genes, numgenes, mean_low_cutoff, mean_high_cutoff, disp_low_cutoff, disp_high_cutoff, zero_type):
    # Cutoffs in Log scale
    numbins = 20
    freq, bins = np.histogram(mean_genes, bins = numbins, range = None, 
        normed = False, weights = None)
    pos = np.digitize(mean_genes, bins)
    max_pos = np.where(pos[:,0] == numbins + 1) 
    pos[max_pos,:] = numbins

    mean_y = np.zeros((numbins, 1))
    sd_y   = np.zeros((numbins, 1))
    genes_dispersion_scaled = np.zeros((numgenes, 1))

    for j in range(numbins):
        y = np.where(pos[:,0] == (j+1))[0]
        print(len(y))
        mean_y[j,0] = np.mean(disp_genes[y,:])
        sd_y[j,0] = np.std(disp_genes[y,:])

    # Fix NaN's 
    mean_y[np.isnan(mean_y)[:,0],:] = 0
    sd_y[np.isnan(sd_y)[:,0],:] = 0

    for j in range(numbins+1):
        y = np.where(pos[:,0] == (j+1))[0]
        if (len(y) != 0):
            genes_dispersion_scaled[y,:] = (disp_genes[y,:] - mean_y[j])/sd_y[j]

    genes_dispersion_scaled[np.isnan(genes_dispersion_scaled[:,0])== True,:] = 0

    exp_mean_genes_arr = np.asarray(np.exp(mean_genes))
    mean_genes_arr = np.asarray(mean_genes)

    if (zero_type == 1):
        # Plot Highly variable Genes 
        plt.figure(figsize=(8,8))
        plt.scatter(exp_mean_genes_arr[:,0], genes_dispersion_scaled[:,0], alpha = 0.5)
        plt.xlabel('Average Expression')
        plt.ylabel('Log Dispersion')
        plt.title('Highly Variable Genes')
        plt.savefig("Figures/scatter_mean_disp_in.png")

        # Plot Highly variable Genes 
        plt.figure(figsize=(8,8))
        plt.scatter(mean_genes_arr[:,0], genes_dispersion_scaled[:,0], alpha = 0.5)
        plt.xlabel('Log Average Expression')
        plt.ylabel('Log Dispersion')
        plt.title('Highly Variable Genes')
        plt.savefig("Figures/scatter_log_mean_disp_in.png")

    if (zero_type == 2):
        # Plot Highly variable Genes 
        plt.figure(figsize=(8,8))
        plt.scatter(exp_mean_genes_arr[:,0], genes_dispersion_scaled[:,0], alpha = 0.5)
        plt.xlabel('Average Expression')
        plt.ylabel('Log Dispersion')
        plt.title('Highly Variable Genes')
        plt.savefig("Figures/scatter_mean_disp_out.png")

        # Plot Highly variable Genes 
        plt.figure(figsize=(8,8))
        plt.scatter(mean_genes_arr[:,0], genes_dispersion_scaled[:,0], alpha = 0.5)
        plt.xlabel('Log Average Expression')
        plt.ylabel('Log Dispersion')
        plt.title('Highly Variable Genes')
        plt.savefig("Figures/scatter_log_mean_disp_out.png")

    # Genes that passes cutoffs
    pass_cutoff = np.where((mean_genes < np.log(mean_high_cutoff)) & (mean_genes > np.log(mean_low_cutoff)) & 
        (genes_dispersion_scaled < np.log(disp_high_cutoff)) & (genes_dispersion_scaled > np.log(disp_low_cutoff)))[0]

    if (zero_type == 1):
        filename = "Figures/scatter_hvg_mean_" + str(mean_low_cutoff) + "_disp_" + str(disp_low_cutoff) + "_in.png"

        # Plot Highly variable Genes 
        plt.figure(figsize=(8,8))
        plt.scatter(exp_mean_genes_arr[pass_cutoff,0], genes_dispersion_scaled[pass_cutoff,0], alpha = 0.5)
        plt.xlabel('Average Expression')
        plt.ylabel('Log Dispersion')
        plt.title('Highly Variable Genes')
        plt.savefig(filename)

        filename = "Figures/scatter_hvg_log_mean_" + str(mean_low_cutoff) + "_disp_" + str(disp_low_cutoff) + "_in.png"

        # Plot Highly variable Genes 
        plt.figure(figsize=(8,8))
        plt.scatter(mean_genes_arr[pass_cutoff,0], genes_dispersion_scaled[pass_cutoff,0], alpha = 0.5)
        plt.xlabel('Log Average Expression')
        plt.ylabel('Log Dispersion')
        plt.title('Highly Variable Genes')
        plt.savefig(filename)

    if (zero_type == 2):
        filename = "Figures/scatter_hvg_mean_" + str(mean_low_cutoff) + "_disp_" + str(disp_low_cutoff) + "_out.png"

        # Plot Highly variable Genes 
        plt.figure(figsize=(8,8))
        plt.scatter(exp_mean_genes_arr[pass_cutoff,0], genes_dispersion_scaled[pass_cutoff,0], alpha = 0.5)
        plt.xlabel('Average Expression')
        plt.ylabel('Log Dispersion')
        plt.title('Highly Variable Genes')
        plt.savefig(filename)

        filename = "Figures/scatter_hvg_log_mean_" + str(mean_low_cutoff) + "_disp_" + str(disp_low_cutoff) + "_out.png"

        # Plot Highly variable Genes 
        plt.figure(figsize=(8,8))
        plt.scatter(mean_genes_arr[pass_cutoff,0], genes_dispersion_scaled[pass_cutoff,0], alpha = 0.5)
        plt.xlabel('Log Average Expression')
        plt.ylabel('Log Dispersion')
        plt.title('Highly Variable Genes')
        plt.savefig(filename)

    list_output = dict()
    list_output["ind_genes"] = pass_cutoff
    list_output["genes_dispersion_scaled"] = genes_dispersion_scaled

    return(list_output)

def subsample(exp_mat):
    m = exp_mat.shape[1]
    num_train = floor(0.95*m)
    num_dev   = floor(0.025*m)
    num_test  = num_train + num_dev
    ind_train = np.random.permutation(m)
    x_train   = exp_mat[:, ind_train[1:int(num_train)]]
    x_dev     = exp_mat[:, ind_train[int(num_train + 1):int(num_test)]]
    x_test    = exp_mat[:, ind_train[int(num_test + 1):m]]
    return x_train, x_dev, x_test

