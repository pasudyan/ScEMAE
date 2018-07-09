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

def array_normalize(x_mat_orig, scale):
    x_mat = x_mat_orig.copy().astype('float64')
    sum_cells = np.sum(x_mat, axis = 0).astype('float64')
    for i in range(x_mat.shape[1]):
        x_mat[:, i] = (x_mat[:, i]/sum_cells[i])*scale
    return(x_mat)

def compute_mean_zero_out(exp_mat):
    dm = exp_mat.shape[0]
    mean_out = np.zeros((dm,))
    for i in range(dm):
        tmp = exp_mat[i,:]
        ind = len(np.where(tmp != 0)[0])
        mean_out[i] = np.sum(tmp)/ind
    return(mean_out)

def compute_sd_zero_out(exp_mat):
    dm = exp_mat.shape[0]
    sd = np.zeros((dm,))
    for d in range(dm):
        row = exp_mat[d,:].astype('float64')
        ind = np.where(row != 0)[0]
        row_sub = row[ind]
        sd[d] = np.std(row_sub)
    return(sd)

def IndxVariableGenes(mean_genes, disp_genes, numgenes, mean_low_cutoff, mean_high_cutoff, disp_low_cutoff, disp_high_cutoff, zero_type):
    # Cutoffs in Log scale
    numbins = 20
    freq, bins = np.histogram(mean_genes, bins = numbins, range = None, 
        normed = False, weights = None)
    pos = np.digitize(mean_genes, bins)
    max_pos = np.where(pos == numbins + 1) 
    pos[max_pos] = numbins
    
    mean_y = np.zeros((numbins,))
    sd_y   = np.zeros((numbins,))
    genes_dispersion_scaled = np.zeros((numgenes,))

    for j in range(numbins):
        y = np.where(pos == (j+1))[0]
        print(len(y))
        mean_y[j] = np.mean(disp_genes[y])
        sd_y[j] = np.std(disp_genes[y])

    # Fix NaN's 
    mean_y[np.isnan(mean_y)] = 0
    sd_y[np.isnan(sd_y)] = 0

    for j in range(numbins+1):
        y = np.where(pos== (j+1))[0]
        if (len(y) != 0):
            genes_dispersion_scaled[y] = (disp_genes[y] - mean_y[j])/sd_y[j]

    genes_dispersion_scaled[np.isnan(genes_dispersion_scaled) == True] = 0

    exp_mean_genes = np.exp(mean_genes)

    if (zero_type == 1):
        # Plot Highly variable Genes 
        plt.figure(figsize=(8,8))
        plt.scatter(exp_mean_genes, genes_dispersion_scaled, alpha = 0.5)
        plt.xlabel('Average Expression')
        plt.ylabel('Log Dispersion')
        plt.title('Highly Variable Genes')
        plt.savefig("Figures/scatter_mean_disp_in.png")

        # Plot Highly variable Genes 
        plt.figure(figsize=(8,8))
        plt.scatter(mean_genes, genes_dispersion_scaled, alpha = 0.5)
        plt.xlabel('Log Average Expression')
        plt.ylabel('Log Dispersion')
        plt.title('Highly Variable Genes')
        plt.savefig("Figures/scatter_log_mean_disp_in.png")

    if (zero_type == 2):
        # Plot Highly variable Genes 
        plt.figure(figsize=(8,8))
        plt.scatter(exp_mean_genes, genes_dispersion_scaled, alpha = 0.5)
        plt.xlabel('Average Expression')
        plt.ylabel('Log Dispersion')
        plt.title('Highly Variable Genes')
        plt.savefig("Figures/scatter_mean_disp_out.png")

        # Plot Highly variable Genes 
        plt.figure(figsize=(8,8))
        plt.scatter(mean_genes, genes_dispersion_scaled, alpha = 0.5)
        plt.xlabel('Log Average Expression')
        plt.ylabel('Log Dispersion')
        plt.title('Highly Variable Genes')
        plt.savefig("Figures/scatter_log_mean_disp_out.png")

    # Genes that passes cutoffs
    pass_cutoff = np.where((mean_genes < np.log(mean_high_cutoff)) & (mean_genes > np.log(mean_low_cutoff)) & 
        (genes_dispersion_scaled < np.log(disp_high_cutoff)) & (genes_dispersion_scaled > np.log(disp_low_cutoff)))[0]

    if (zero_type == 1):
        filename = "Figures/mean_" + str(mean_low_cutoff) + "_disp_" + str(disp_low_cutoff) + "_scatter_hvg_in.png"

        # Plot Highly variable Genes 
        plt.figure(figsize=(8,8))
        plt.scatter(exp_mean_genes[pass_cutoff], genes_dispersion_scaled[pass_cutoff], alpha = 0.5)
        plt.xlabel('Average Expression')
        plt.ylabel('Log Dispersion')
        plt.title('Highly Variable Genes')
        plt.savefig(filename)

        filename = "Figures/log_mean_" + str(mean_low_cutoff) + "_disp_" + str(disp_low_cutoff) + "_scatter_hvg_in.png"

        # Plot Highly variable Genes 
        plt.figure(figsize=(8,8))
        plt.scatter(mean_genes[pass_cutoff], genes_dispersion_scaled[pass_cutoff], alpha = 0.5)
        plt.xlabel('Log Average Expression')
        plt.ylabel('Log Dispersion')
        plt.title('Highly Variable Genes')
        plt.savefig(filename)

    if (zero_type == 2):
        filename = "Figures/mean_" + str(mean_low_cutoff) + "_disp_" + str(disp_low_cutoff) + "_scatter_hvg_out.png"

        # Plot Highly variable Genes 
        plt.figure(figsize=(8,8))
        plt.scatter(exp_mean_genes[pass_cutoff], genes_dispersion_scaled[pass_cutoff], alpha = 0.5)
        plt.xlabel('Average Expression')
        plt.ylabel('Log Dispersion')
        plt.title('Highly Variable Genes')
        plt.savefig(filename)

        filename = "Figures/log_mean_" + str(mean_low_cutoff) + "_disp_" + str(disp_low_cutoff) + "_scatter_hvg_out.png"

        # Plot Highly variable Genes 
        plt.figure(figsize=(8,8))
        plt.scatter(mean_genes[pass_cutoff], genes_dispersion_scaled[pass_cutoff], alpha = 0.5)
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

