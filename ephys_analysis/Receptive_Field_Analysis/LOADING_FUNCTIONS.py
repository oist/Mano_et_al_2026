#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
import numpy as np
from tqdm import trange


def find_first_match(path, keyword):
    files = os.listdir(path)
    return [os.path.join(path, f) for f in files if keyword in f][0]

def find_all_matches(path, keyword):
    files = os.listdir(path)
    return [os.path.join(path, f) for f in files if keyword in f]

def load_stim_and_conditions(rec_path, mix_binar):
    stim_folder = os.path.join(rec_path, 'visual_stim')
    if not mix_binar:
        stim_path = find_first_match(stim_folder, 'npy')
        stim_name = os.path.basename(stim_path)
        cond_vector = np.nan
    else:
        stim_path = find_first_match(stim_folder, 'LUM')
        stim_name = os.path.basename(stim_path)
        cond_path = find_first_match(stim_folder, 'COND')
        cond_vector = np.load(cond_path)
    return stim_path, stim_name, cond_vector

def extract_recording_paths(exp_names, base_path):
    rec_paths = []
    for exp_name in exp_names:
        rec_list = os.listdir(os.path.join(base_path, exp_name))
        found = False
        for i in rec_list:
            if 'SN_LUM' in i:
                rec_paths.append(os.path.join(base_path, exp_name, i))
                found = True
        if not found:
            print(f'No matching recording in {exp_name}!!')
    return rec_paths

def compute_stim_frs(dataset_ids, ALL_STIM_WINDOWS, ALL_ids, ALL_spikes, response_dict):
    ALL_stim_FRs = []
    for data_index in range(len(dataset_ids)):
        if dataset_ids[data_index] in response_dict:
            stim_windows = ALL_STIM_WINDOWS[data_index]
            g_ids = ALL_ids[data_index]
            spike_t_matrix = ALL_spikes[data_index]
            stim_unit_FRs = np.zeros((len(g_ids), len(stim_windows)))
            for unit in trange(len(g_ids)):
                for spike_t in spike_t_matrix[unit]:
                    if stim_windows[0][0] < spike_t < stim_windows[-1][1] + 0.03:
                        window_index = np.searchsorted(stim_windows[:, 0], spike_t) - 1
                        if 0 <= window_index < len(stim_windows) and spike_t < stim_windows[window_index][1] + 0.03:
                            stim_unit_FRs[unit][window_index] += 1
            ALL_stim_FRs.append(stim_unit_FRs)
        else:
            ALL_stim_FRs.append(np.nan)
    return ALL_stim_FRs


# In[ ]:




