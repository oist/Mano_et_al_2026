#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 10:34:19 2025

@author: sam
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, wilcoxon
from sklearn.decomposition import PCA

# Custom one-sided Wilcoxon test that handles all-identical data
def wilcoxon_one_sided_greater(x):
    """
    Returns a one-sided p-value for the hypothesis: median(x) > 0.
    Fallback for all-identical data:
      - all positive => p = (0.5)^n
      - all zero or negative => p = 1.0
    Otherwise, uses standard Wilcoxon and halves the two-sided p if mean>0.
    """
    x = np.asarray(x)
    n = len(x)
    
    if np.allclose(x, x[0]):  # all values identical
        if x[0] > 0:
            return (0.5 ** n)  # sign test
        else:
            return 1.0
    else:
        stat, p_two_sided = wilcoxon(x, zero_method='wilcox')
        mean_x = np.mean(x)
        if mean_x > 0:
            return p_two_sided / 2.0  # one-sided if distribution is mostly > 0
        else:
            return 1.0

 



pop_data=np.load("pop_data_pol.npy")
bin_midpoints=np.load("bins_pol.npy")
an_array=np.load('an_array_pol.npy')
bin_midpoints=(bin_midpoints-0.5)*2
exp_type = 'POL'


# pop_data=np.load("pop_data.npy")
# bin_midpoints=np.load("bins_int.npy")
# an_array=np.load('an_array.npy')
# exp_type = 'GRAY'


null_trial=0.5
                 
# ------------------------------------------------------------------
# 1) USER PARAMETERS
# ------------------------------------------------------------------
time_window    = (0, 200)
apply_pca      = True
n_components   = .7
alpha          = 0.05

pop_data=pop_data[:,:,time_window[0]:time_window[1],:]

# 2) Basic shapes
n_folds, n_bins, n_time, n_neurons = pop_data.shape
null_bin    = int(n_bins * null_trial) 

# 3) Cross-validated SNR (fold by fold)
snr_per_fold = np.zeros((n_folds, n_bins))

for test_fold in range(n_folds):
    # Identify which folds are used for training
    train_folds = [f for f in range(n_folds) if f != test_fold]
    
    # Extract training data for normalization
    train_data_temp = pop_data[train_folds]  # shape: (n_train_folds, n_bins, n_time, n_neurons)
    
    # Compute mean and std across training folds, bins, and time (per neuron)
    neuron_mean = np.mean(train_data_temp, axis=(0, 1, 3), keepdims=True)  # shape: (1,1,1,n_neurons)
    neuron_std  = np.std(train_data_temp, axis=(0, 1, 3), keepdims=True)   # shape: (1,1,1,n_neurons)
    
    # Avoid division by zero
    eps = 1e-8
    neuron_std += eps

    # Normalize training data
    norm_train_data = (train_data_temp - neuron_mean) / neuron_std
    # shape: (n_train_folds, n_bins, n_time, n_neurons)

    # Normalize test data
    norm_test_data  = (pop_data[test_fold] - neuron_mean) / neuron_std
    # shape: (n_bins, n_time, n_neurons)

    # Flatten train/test
    # We combine all training folds, then flatten time & neurons
    train_data = norm_train_data.reshape(-1, n_time * n_neurons)   # shape: (n_train_folds*n_bins, n_time*n_neurons)
    test_data  = norm_test_data.reshape(n_bins, n_time * n_neurons)  # shape: (n_bins, n_time*n_neurons)

    # -----------------------------------------------------------
    # PCA on training folds (optional)
    # -----------------------------------------------------------
    if apply_pca:
        pca = PCA(n_components=n_components)
        pca.fit(train_data)
        
        # Transform the test data
        test_vectors = pca.transform(test_data)  # shape: (n_bins, n_components)
        
        # Build the null vectors from the *normalized* training data, 
        # specifically for the null_bin in each training fold.
        null_vectors = []
        for i in range(len(train_folds)):
            # norm_train_data[i] has shape (n_bins, n_time, n_neurons)
            null_data = norm_train_data[i, null_bin]  # shape: (n_time, n_neurons)
            null_data_flat = null_data.reshape(1, -1) # shape: (1, n_time*n_neurons)
            null_vec = pca.transform(null_data_flat)[0]  # shape: (n_components,)
            null_vectors.append(null_vec)
        null_vectors = np.array(null_vectors)  # shape: (n_train_folds, n_components)

    else:
        # No PCA: just keep the flattened data
        test_vectors = test_data  # shape: (n_bins, n_time*n_neurons)

        # Build null vectors from the normalized training data
        null_vectors = []
        for i in range(len(train_folds)):
            null_data = norm_train_data[i, null_bin]  # (n_time, n_neurons)
            null_data_flat = null_data.reshape(-1)    # (n_time*n_neurons,)
            null_vectors.append(null_data_flat)
        null_vectors = np.array(null_vectors)  # (n_train_folds, n_time*n_neurons)

    # -----------------------------------------------------------
    # Compute distance of each test bin from the null vectors
    # -----------------------------------------------------------
    fold_distances = np.zeros(n_bins)
    for b in range(n_bins):
        # Distance from test_vectors[b] to each vector in null_vectors
        # norm(...) is the Euclidean distance
        dists_b = [np.linalg.norm(test_vectors[b] - nv) for nv in null_vectors]
        fold_distances[b] = np.mean(dists_b)

    # -----------------------------------------------------------
    # Compute SNR
    # -----------------------------------------------------------
    # "null_mean" is the distance of the null bin from itself
    null_mean = fold_distances[null_bin]
    
    # "null_std" is the variation in distance from the test_fold's null bin 
    # to each training null_vector
    null_std = np.std([
        np.linalg.norm(test_vectors[null_bin] - nv)
        for nv in null_vectors
    ], ddof=1)

    # Avoid zero-std or extremely small std
    null_std = null_std if null_std > 1e-12 else 1e-12

    # SNR for this fold: how many std devs away from the null distance is each bin?
    snr_fold = (fold_distances - null_mean) / null_std
    snr_per_fold[test_fold] = snr_fold

# 4) Aggregate SNR across folds
snr_all  = np.mean(snr_per_fold, axis=0)  # shape: (n_bins,)
snr_sems = sem(snr_per_fold, axis=0)      # shape: (n_bins,)

# ------------------------------------------------------------------
# 5) SEQUENTIAL TEST in TWO DIRECTIONS from null_bin
# ------------------------------------------------------------------

negative_indices = range(null_bin - 1, -1, -1)    # e.g. 24, 23, 22... down to 0
positive_indices = range(null_bin + 1, n_bins, 1) # e.g. 26, 27, 28... up to n_bins-1

threshold_bin_neg = None
threshold_contrast_neg = None

for b in negative_indices:
    sr_values = snr_per_fold[:, b]  # SNR across folds at bin b
    pval = wilcoxon_one_sided_greater(sr_values)
    if pval < alpha and np.mean(sr_values) > 0:
        threshold_bin_neg = b
        threshold_contrast_neg = bin_midpoints[b]
        print(f"Negative threshold found at bin {threshold_bin_neg}, "
              f"contrast={threshold_contrast_neg:.3f}, "
              f"mean SNR={snr_all[b]:.3f}, p={pval:.3g}")
        break

if threshold_bin_neg is None:
    print("No negative threshold found.")

threshold_bin_pos = None
threshold_contrast_pos = None

for b in positive_indices:
    sr_values = snr_per_fold[:, b]
    pval = wilcoxon_one_sided_greater(sr_values)
    if pval < alpha and np.mean(sr_values) > 0:
        threshold_bin_pos = b
        threshold_contrast_pos = bin_midpoints[b]
        print(f"Positive threshold found at bin {threshold_bin_pos}, "
              f"contrast={threshold_contrast_pos:.3f}, "
              f"mean SNR={snr_all[b]:.3f}, p={pval:.3g}")
        break

if threshold_bin_pos is None:
    print("No positive threshold found.")

neg_thresh=threshold_contrast_neg
pos_thresh=threshold_contrast_pos
# ------------------------------------------------------------------
# 6) Plot & final report (consistent units)
# ------------------------------------------------------------------

# ------------------------------------------------------------------
plt.figure(figsize=(3,3))  # Use 6x3 for horizontal layout



plt.plot(bin_midpoints, snr_all, label='Mean SNR')
plt.fill_between(
    bin_midpoints, snr_all - snr_sems, snr_all + snr_sems,
    alpha=0.3, label='± SEM'
)
plt.axhline(0, ls='--', color='gray', zorder=-1)

if threshold_bin_neg is not None:
    plt.axvline(neg_thresh, color='k', ls='--', label='Negative threshold')
if threshold_bin_pos is not None:
    plt.axvline(pos_thresh, color='k', ls='--', label='Positive threshold')

plt.xlabel("Contrast (same units as x-axis)")
plt.ylabel("SNR (vs. null bin)")
plt.title("Sequential threshold test")
plt.xlim(-1,1)
# plt.legend()  # Uncomment if legend is desired

plt.tight_layout()
plt.show()


# ----------- single, authoritative report -----------------
if exp_type == 'POL':
    print(f"Negative DoLP threshold  = {neg_thresh:.3f}")
    print(f"Positive DoLP threshold = {pos_thresh:.3f}")
    plt.savefig('pol_thresh.pdf', dpi=300)
else:
    print(f"Negative contrast threshold  = {neg_thresh:.3f}")
    print(f"Positive contrast threshold = {pos_thresh:.3f}")
    plt.savefig('int_thresh.pdf', dpi=300)
 
 
