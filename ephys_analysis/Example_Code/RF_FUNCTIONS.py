#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import os, random, string
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import label, regionprops
from scipy.signal import medfilt
from scipy.signal import medfilt2d
from scipy.ndimage import zoom
from skimage.measure import label
from skimage.measure import regionprops
from scipy.ndimage import gaussian_filter
from tqdm import trange


def compute_on_off_responses(
    dataset_ids, ALL_STIM_WINDOWS, ALL_ids, response_dict,
    ALL_spikes, ALL_stim_FRs, ALL_stim_matrices, COND_VECTORS, MIX_BINAR
):
    ALL_ON_RESPONSES = []
    ALL_OFF_RESPONSES = []
    ALL_ON_RESPONSES_NULL = []
    ALL_OFF_RESPONSES_NULL = []

    for data_index in trange(len(dataset_ids)):
        if dataset_ids[data_index] in response_dict:
            stim_windows = ALL_STIM_WINDOWS[data_index]
            g_ids = ALL_ids[data_index]
            spike_t_matrix = ALL_spikes[data_index]
            response_units = response_dict[dataset_ids[data_index]]
            stim_unit_FRs = ALL_stim_FRs[data_index]

            if MIX_BINAR[data_index] == False:
                sparse_noise_matrix = ALL_stim_matrices[data_index][:len(stim_windows)]
                sparse_noise_matrix_null = np.copy(sparse_noise_matrix)
                np.random.shuffle(sparse_noise_matrix_null)
            else:
                SN_LUM = ALL_stim_matrices[data_index]
                COND_VECTOR = COND_VECTORS[data_index]
                x = np.where(COND_VECTOR == 0)[0]
                x = x[x < len(stim_windows)]
                sparse_noise_matrix = SN_LUM[x[:-2]]
                sparse_noise_matrix_null = np.copy(sparse_noise_matrix)
                np.random.shuffle(sparse_noise_matrix_null)
                stim_unit_FRs = [i[x[:-2]] for i in stim_unit_FRs]
                stim_windows = stim_windows[x[:-2]]


            positive_sum_matrix = np.sum(np.clip(sparse_noise_matrix, 0, None), axis=0)
            negative_sum_matrix = -np.sum(np.clip(sparse_noise_matrix, None, 0), axis=0)

            UNIT_ON_RESPONSES = np.zeros((len(g_ids),) + sparse_noise_matrix.shape[1:])
            UNIT_OFF_RESPONSES = np.zeros_like(UNIT_ON_RESPONSES)
            UNIT_ON_RESPONSES_NULL = np.zeros_like(UNIT_ON_RESPONSES)
            UNIT_OFF_RESPONSES_NULL = np.zeros_like(UNIT_ON_RESPONSES)

            for unit in range(len(g_ids)):
                if g_ids[unit] in response_units:
                    for stim in range(len(stim_windows)):
                        fr = stim_unit_FRs[unit][stim]
                        temp_on = sparse_noise_matrix[stim] > 0
                        temp_off = sparse_noise_matrix[stim] < 0
                        temp_on_null = sparse_noise_matrix_null[stim] > 0
                        temp_off_null = sparse_noise_matrix_null[stim] < 0

                        UNIT_ON_RESPONSES[unit] += temp_on * fr
                        UNIT_OFF_RESPONSES[unit] += temp_off * fr
                        UNIT_ON_RESPONSES_NULL[unit] += temp_on_null * fr
                        UNIT_OFF_RESPONSES_NULL[unit] += temp_off_null * fr

                    UNIT_ON_RESPONSES[unit] /= positive_sum_matrix
                    UNIT_OFF_RESPONSES[unit] /= negative_sum_matrix
                    UNIT_ON_RESPONSES_NULL[unit] /= positive_sum_matrix
                    UNIT_OFF_RESPONSES_NULL[unit] /= negative_sum_matrix

            ALL_ON_RESPONSES.append(UNIT_ON_RESPONSES)
            ALL_OFF_RESPONSES.append(UNIT_OFF_RESPONSES)
            ALL_ON_RESPONSES_NULL.append(UNIT_ON_RESPONSES_NULL)
            ALL_OFF_RESPONSES_NULL.append(UNIT_OFF_RESPONSES_NULL)
        else:
            ALL_ON_RESPONSES.append(np.nan)
            ALL_OFF_RESPONSES.append(np.nan)
            ALL_ON_RESPONSES_NULL.append(np.nan)
            ALL_OFF_RESPONSES_NULL.append(np.nan)

    return ALL_ON_RESPONSES, ALL_OFF_RESPONSES, ALL_ON_RESPONSES_NULL, ALL_OFF_RESPONSES_NULL

def compute_on_none_responses_and_centers(
    dataset_ids, response_dict, ALL_STIM_WINDOWS, ALL_ids,
    ALL_spikes, ALL_stim_FRs, ALL_stim_matrices, COND_VECTORS, MIX_BINAR,
    ALL_ON_RESPONSES, ALL_OFF_RESPONSES, find_zero_neighborhood_indexes_with_previous
):
    ALL_ONs = []
    ALL_NONEs = []
    ALL_CENTREs = []

    for data_index in trange(len(dataset_ids)):
        ONs = []
        NONEs = []
        CENTREs = []

        if dataset_ids[data_index] in response_dict:
            stim_windows = ALL_STIM_WINDOWS[data_index]
            g_ids = ALL_ids[data_index]
            response_units = response_dict[dataset_ids[data_index]]
            spike_t_matrix = ALL_spikes[data_index]
            stim_unit_FRs = ALL_stim_FRs[data_index]

            if MIX_BINAR[data_index]:
                SN_LUM = ALL_stim_matrices[data_index]
                COND_VECTOR = COND_VECTORS[data_index]
                x = np.where(COND_VECTOR == 0)[0]
                x = x[x < len(stim_windows)]
                sparse_noise_matrix = SN_LUM[x[:-2]]
                dn_matrix = sparse_noise_matrix
                stim_windows = stim_windows[x[:-2]]
                stim_unit_FRs = [u[x[:-2]] for u in stim_unit_FRs]
            else:
                dn_matrix = ALL_stim_matrices[data_index][:len(stim_windows)]

            UNIT_ON_RESPONSES = ALL_ON_RESPONSES[data_index]
            UNIT_OFF_RESPONSES = ALL_OFF_RESPONSES[data_index]
            stim_length = np.mean(np.diff(stim_windows, axis=1))

            for unit in range(len(g_ids)):
                if g_ids[unit] in response_units:
                    ON_max = np.max(UNIT_ON_RESPONSES[unit])
                    OFF_max = np.max(UNIT_OFF_RESPONSES[unit])
                    if ON_max > OFF_max:
                        rf_centre_index = np.where(UNIT_ON_RESPONSES[unit] == ON_max)
                        is_on = True
                    else:
                        rf_centre_index = np.where(UNIT_OFF_RESPONSES[unit] == OFF_max)
                        is_on = False

                    if len(rf_centre_index[0]) > 1:
                        rf_centre_index = np.array([rf_centre_index[0][0], rf_centre_index[1][0]])
                    else:
                        rf_centre_index = np.array([i[0] for i in rf_centre_index])

                    CENTREs.append(rf_centre_index)

                    if is_on:
                        pos_peak_indexes = np.where(dn_matrix[:, rf_centre_index[0], rf_centre_index[1]] == 1)[0]
                    else:
                        pos_peak_indexes = np.where(dn_matrix[:, rf_centre_index[0], rf_centre_index[1]] == -1)[0]

                    zero_indexes = find_zero_neighborhood_indexes_with_previous(
                        dn_matrix, rf_centre_index, neighborhood_size=4
                    )
                    pos_windows = stim_windows[pos_peak_indexes]
                    zero_windows = stim_windows[zero_indexes]

                    unit_stim_on = stim_unit_FRs[unit][pos_peak_indexes]
                    unit_stim_none = stim_unit_FRs[unit][zero_indexes]

                    stim_on = np.nansum(unit_stim_on) / len(unit_stim_on) * (1 / stim_length)
                    stim_none = np.nansum(unit_stim_none) / len(unit_stim_none) * (1 / stim_length)
                    stim_none = max(stim_none, 0.001)

                    ONs.append(stim_on)
                    NONEs.append(stim_none)
                else:
                    ONs.append(np.nan)
                    NONEs.append(np.nan)
                    CENTREs.append(np.nan)

        else:
            ONs = [np.nan]
            NONEs = [np.nan]
            CENTREs = [np.nan]

        ALL_ONs.append(np.asarray(ONs))
        ALL_NONEs.append(np.asarray(NONEs))
        ALL_CENTREs.append(np.asarray(CENTREs, dtype='object'))

    return ALL_ONs, ALL_NONEs, ALL_CENTREs

def find_zero_neighborhood_indexes_with_previous(sparse_noise_matrix, rf_centre_index, neighborhood_size=3):
    """
    Finds the indexes of frames in the sparse_noise_matrix where the neighborhood
    around the rf_centre_index is entirely zero for both the current and previous frames.

    Parameters:
    - sparse_noise_matrix (ndarray): Array of 2D matrices (frames).
    - rf_centre_index (tuple or array): Tuple or array of (row, column) specifying the center index.
    - neighborhood_size (int): Size of the neighborhood (default is 3x3).

    Returns:
    - List[int]: List of frame indexes where both current and previous frames
                 have a neighborhood entirely zero.
    """
    half_size = neighborhood_size // 2
    zero_neighborhood_indexes = []

    # Ensure rf_centre_index is a tuple of integers
    center_x, center_y = int(rf_centre_index[0]), int(rf_centre_index[1])

    # Iterate over the frames of the sparse_noise_matrix, starting from the second frame
    for frame_index in range(1, sparse_noise_matrix.shape[0]):
        # Extract the 2D matrices for the current and previous frames
        current_frame = sparse_noise_matrix[frame_index]
        previous_frame = sparse_noise_matrix[frame_index - 1]

        # Define the neighborhood boundaries for the current frame
        x_min = max(0, center_x - half_size)
        x_max = min(current_frame.shape[0], center_x + half_size + 1)
        y_min = max(0, center_y - half_size)
        y_max = min(current_frame.shape[1], center_y + half_size + 1)

        # Extract the neighborhoods
        current_neighborhood = current_frame[x_min:x_max, y_min:y_max]
        previous_neighborhood = previous_frame[x_min:x_max, y_min:y_max]

        # Check if both neighborhoods are entirely zero
        if np.all(current_neighborhood == 0) and np.all(previous_neighborhood == 0):
            zero_neighborhood_indexes.append(frame_index)

    return zero_neighborhood_indexes

def get_rf_PTSH(sn_matrix, resp_on_mat, resp_off_mat, spikes, stim_windows, stim_length, extension, bin_size=0.005):
    resp_on_smoothed = gaussian_filter(resp_on_mat, sigma=0.5)
    resp_off_smoothed = gaussian_filter(resp_off_mat, sigma=0.5)

    on_max = np.max(resp_on_smoothed)
    off_max = np.max(resp_off_smoothed)

    if on_max > off_max:
        rf_center_index = np.where(resp_on_smoothed == on_max)
        on_off = True
    else:
        rf_center_index = np.where(resp_off_smoothed == off_max)
        on_off = False

    bins = np.arange(0, stim_length + extension * 2 + bin_size, bin_size)

    pos_t_indexes = np.where(sn_matrix[:len(stim_windows)][:, rf_center_index[0], rf_center_index[1]] == 1)[0]
    neg_t_indexes = np.where(sn_matrix[:len(stim_windows)][:, rf_center_index[0], rf_center_index[1]] == -1)[0]

    pos_windows = stim_windows[pos_t_indexes]
    neg_windows = stim_windows[neg_t_indexes]

    for pw in range(len(pos_windows)):
        pos_windows[pw][0] -= extension
        pos_windows[pw][1] += extension
    for nw in range(len(neg_windows)):
        neg_windows[nw][0] -= extension
        neg_windows[nw][1] += extension

    pw_times = [spikes[np.logical_and(spikes >= w[0], spikes < w[1])] for w in pos_windows]
    nw_times = [spikes[np.logical_and(spikes >= w[0], spikes < w[1])] for w in neg_windows]

    pos_points = [times - win[0] for times, win in zip(pw_times, pos_windows)]
    neg_points = [times - win[0] for times, win in zip(nw_times, neg_windows)]

    all_pos_points = np.concatenate(pos_points)
    all_neg_points = np.concatenate(neg_points)

    pos_hist, _ = np.histogram(all_pos_points, bins=bins)
    pos_hist = pos_hist / len(pw_times)
    neg_hist, _ = np.histogram(all_neg_points, bins=bins)
    neg_hist = neg_hist / len(nw_times)

    return pos_hist, neg_hist, bins, rf_center_index, on_off

def calc_rf(eta, base_eta, peak_thresh=4, blob_thresh=3, gaus_f=1, med_f=1, upsample=1, force_polarity=None):
    eta = zoom(eta, zoom=upsample, order=1)
    base_eta = zoom(base_eta, zoom=upsample, order=1)
    if med_f>0:
        filt_eta=medfilt2d(eta, kernel_size=med_f)
        filt_eta=gaussian_filter(filt_eta, gaus_f)
        filt_base=medfilt2d(base_eta, kernel_size=med_f)
        filt_base=gaussian_filter(filt_base, gaus_f)
    else:
        filt_eta=gaussian_filter(eta, gaus_f)
        filt_base=gaussian_filter(base_eta, gaus_f)
    mean_eta = np.mean(filt_base)
    etd_eta = np.std(filt_base)
    filt_eta=filt_eta-mean_eta
    max_deviation = np.max(filt_eta)
    min_deviation = np.min(filt_eta)
    if force_polarity == 'positive':
        peak_coords = np.unravel_index(np.argmax(filt_eta), filt_eta.shape)
        binary_rf = filt_eta > (blob_thresh * std_eta)
        cleared_peak_thresh = filt_eta[peak_coords] > (peak_thresh * etd_eta)
        pos = 1
    elif force_polarity == 'negative':
        peak_coords = np.unravel_index(np.argmin(filt_eta), filt_eta.shape)
        binary_rf = filt_eta < (-1 * blob_thresh * std_eta)
        cleared_peak_thresh = filt_eta[peak_coords] < (-1 * peak_thresh * std_eta)
        pos = -1
    else:
        if np.abs(max_deviation) > np.abs(min_deviation):
            peak_coords = np.unravel_index(np.argmax(filt_eta), filt_eta.shape)
            binary_rf = filt_eta > (blob_thresh * std_eta)
            cleared_peak_thresh = filt_eta[peak_coords] > (peak_thresh * std_eta)
            pos = 1
        else:
            peak_coords = np.unravel_index(np.argmin(filt_eta), filt_eta.shape)
            binary_rf = filt_eta < (-1 * blob_thresh * std_eta)
            cleared_peak_thresh = filt_eta[peak_coords] < (-1 * peak_thresh * std_eta)
            pos = -1
    if cleared_peak_thresh:
        labeled_rf, num_features = label(binary_rf)
        regions = regionprops(labeled_rf)
        # Assuming the largest labeled area is the receptive field
        largest_area = max(regions, key=lambda x: x.area) if regions else None
        rf_blob = labeled_rf == largest_area.label if largest_area else np.zeros_like(binary_rf)
        rf_size = largest_area.area if largest_area else 0
        rf_loc = largest_area.centroid if largest_area else (0, 0)
        rf_diameter = largest_area.major_axis_length if largest_area else 0
        major_axis = largest_area.major_axis_length if largest_area else 0
        try:
            minor_axis = largest_area.minor_axis_length if largest_area else 0
        except:
            minor_axis=0
        rf_ellipse = np.pi * (major_axis / 2) * (minor_axis / 2)
        #this is peak deviation, with background subtracted
        rf_peakrate=eta[int(rf_loc[0]),int(rf_loc[1])]
        rf_minrate=np.min(filt_eta)
    else:
        rf_blob=np.zeros_like(filt_eta)
        rf_size=0
        rf_loc=(0,0)
        rf_diameter = 0
        rf_ellipse = 0
        rf_peakrate=mean_eta
        rf_minrate=np.min(filt_eta)
        pos=0
    return rf_blob, rf_size, rf_loc, filt_eta,std_eta, rf_diameter, rf_peakrate,rf_minrate, pos







