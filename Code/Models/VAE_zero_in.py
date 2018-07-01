import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import csv
import collections
import scipy.sparse as sp_sparse
import tables
import h5py

from scipy import stats
from scipy.io import mmread
from random import sample
from math import floor

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt


# ================= Variational Auto-encoder ================== #

load_data = np.load('Preproc_data/millcell_train_test_set_norm.npz')

x_train = load_data['arr_0']
x_test  = load_data['arr_1']

m = x_train.shape[0]
batch_size = 1024
original_dim = m
latent_dim = 2
intermediate_dim = 5
epochs = 5000
epsilon_std = 1.0

x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='relu')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# instantiate VAE model
vae = Model(x, x_decoded_mean)

# def binary_crossentropy(y_true, y_pred):
#     return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

# def mean_squared_error(y_true, y_pred):
#     return K.mean(K.square(y_pred - y_true), axis=-1)

def mean_squared_error_mask(y_true, y_pred, mask_ind):
    dm = y_true.shape[1]
    n  = batch_size
    sq_veq = 0
    for i in range(n):
        s_ind   = np.where(mask_ind[0] == i)[0]
        col_ind = mask_ind[1][s_ind]
        b_ind   = np.ones(dm, dtype = bool)
        b_ind[col_ind] = False
        sq_veq  = sq_veq + K.sum(K.square(y_pred[i, b_ind] - y_true[i, b_ind]))
    return sq_veq/n

# def build_masked_loss(loss_function, mask_value = [-1]):
#     """Builds a loss function that masks based on targets
#     Args:
#         loss_function: The loss function to mask
#         mask_value: The value to mask in the targets
#     Returns:
#         function: a loss function that acts like loss_function with masked inputs
#     """
#     def masked_loss_function(y_true, y_pred):
#         mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
#         return loss_function(y_true * mask, y_pred * mask)
#     return masked_loss_function


# xent_loss = mean_squared_error_mask(x, x_decoded_mean, mask_ind)
# xent_loss = build_masked_loss(metrics.mean_squared_error, mask_value = [-1])(x, x_decoded_mean)

# Compute VAE loss
mask_value = [-1.]
mask       = K.cast(K.not_equal(x, mask_value), K.floatx())
xent_loss  = original_dim * metrics.mean_squared_error(x*mask, x_decoded_mean*mask)
kl_loss    = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss   = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(loss = None, optimizer='adam')
vae.summary()

x_train_subset = x_train.transpose()
x_test_subset  = x_test.transpose()

# =========== Masking drop outs to be -1 ========== #
def mask_dropouts(mat):
    ix  = np.where(mat == 0)
    mat_log = np.log(mat)
    mat_log[ix[0], ix[1]] = -1
    return mat_log, ix

x_train_mask, ix_train = mask_dropouts(x_train_subset)
x_test_mask, ix_test   = mask_dropouts(x_test_subset)

mask_ind = ix_train

# Checkpoint
filepath = "./Models/weights_best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose = 1, save_best_only= True, mode = 'max')
callbacks_list = [checkpoint]

# Fit Model
vae.fit(x_train_mask,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        validation_data=(x_test_mask, None))

encoder = Model(x, z_mean)
x_test_encoded = encoder.predict(x_test_mask, batch_size=batch_size)

plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1])
# plt.colorbar()
plt.savefig("vae_mill_cell.png")

# def plot_model_history(model_history):
#     fig, axs = plt.subplots(1,2,figsize=(15,5))
#     # summarize history for accuracy
#     axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
#     axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
#     axs[0].set_title('Model Accuracy')
#     axs[0].set_ylabel('Accuracy')
#     axs[0].set_xlabel('Epoch')
#     axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
#     axs[0].legend(['train', 'val'], loc='best')
#     # summarize history for loss
#     axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
#     axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
#     axs[1].set_title('Model Loss')
#     axs[1].set_ylabel('Loss')
#     axs[1].set_xlabel('Epoch')
#     axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
#     axs[1].legend(['train', 'val'], loc='best')
#     plt.show()













