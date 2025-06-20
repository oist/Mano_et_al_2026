#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:09:06 2023

@author: sam
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ephys_functions import *
import sys
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.stats import mannwhitneyu
atlas_10 = None
dt1 = None
sys.path.append('./ephys')

on_deigo = False

from matplotlib import font_manager
import matplotlib as mpl
font_manager.fontManager.addfont('/home/sam/bucket/common/fonts/arial.ttf')
mpl.rcParams.update({
    'font.size': 7,
    'axes.titlesize': 7,
    'legend.fontsize': 7,
    'font.family': 'sans-serif',
    'font.sans-serif': "Arial",
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'xtick.major.size': 2,
    'xtick.minor.size': 2,
    'xtick.minor.width': 0.5,
    'ytick.major.width': 0.5,
    'ytick.minor.width': 0.5,
    'ytick.major.size': 2,
    'ytick.minor.size': 2,
})


us=5 #upsample to 1 visual degree
pt=4 #peak threshold
bt=3 #background threshold
gf=1 #gaussian filter
mf=0 #median filter



#%% Good mix datasets

curr_ephys_paths = ['/home/sam/bucket/electrophysiology/squidOL/2024_02_02/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_02_05/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_02_13/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_02_15/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_02_26/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_05_28/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_06_24/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_06_27/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_08_19/', 
                    '/home/sam/bucket/electrophysiology/squidOL/2024_08_22/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_09_10/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_10_22/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_10_25/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_11_01/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_11_07/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_11_12/',
                    '/home/sam/bucket/electrophysiology/squidOL/2025_01_07/']


#3/7 unit 53 is a good one

#%% SN RFs
plot=0
curr_ephys_paths=['/home/sam/bucket/electrophysiology/squidOL/2025_01_17/',
                  '/home/sam/bucket/electrophysiology/squidOL/2025_01_29/',
                  '/home/sam/bucket/electrophysiology/squidOL/2025_01_30/',
                  '/home/sam/bucket/electrophysiology/squidOL/2025_02_04/',
                  '/home/sam/bucket/electrophysiology/squidOL/2025_02_12/',
                  '/home/sam/bucket/electrophysiology/squidOL/2025_02_13/',
                  '/home/sam/bucket/electrophysiology/squidOL/2025_03_07/',
                  '/home/sam/bucket/electrophysiology/squidOL/2025_03_11/']

for curr_ephys_path in curr_ephys_paths:
    
    sn_exp = load_ephys_results(sorted(glob.glob(curr_ephys_path + '/*_SN_POL_VER*'))[0],0,False)
    
    dt = int(np.mean(np.diff(sn_exp['stim_times'][:, 0]))*1000)/1000
      
    first_st=sn_exp['stim_times'][0, 0]
    stimulus_times=sn_exp['stim_times'][:, 0]-first_st
    stim_mat=sn_exp['sn_mat'][0:len(stimulus_times)]
    
    for probe in range(len(sn_exp['st_mat'])):    
        num_units = len(sn_exp['st_mat'][probe])
      
        all_eta0=[]
        all_eta1=[]
        all_nulleta0=[]
        all_nulleta1=[]
        all_psth0=[]
        
        for unit in range(num_units):
            rf_blob=np.zeros(1)
            neuron_rf=[]
       
            # Process spike times
            curr_times = sn_exp['st_mat'][probe][unit] - first_st

            if plot==1:
                plt.close()
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.reshape(2, 2)
            
            X = stim_mat
            bpp, bin_edges = np.histogram(curr_times, bins=stimulus_times)
            bpp = np.concatenate([bpp, [0]]) / dt
            bpp_expanded = bpp[:, np.newaxis, np.newaxis]
            
            eta0, eta1, null_eta0, null_eta1 = compute_real_and_null_etas(X, bpp_expanded)
            
            peak_ind = np.unravel_index(np.argmax(eta0), eta0.shape)
            has_1 = np.where(stim_mat[:,int(peak_ind[0]),int(peak_ind[1])] == 1)[0]
            psth0, bin_edges, aligned_spikes=calc_psth(stimulus_times[has_1], curr_times, [0, .2], bin_width=0.001, trial_samp=1000, plot=0)
            
            all_eta0.append(eta0)
            all_eta1.append(eta1)
            all_nulleta0.append(null_eta0)
            all_nulleta1.append(null_eta1)
            all_psth0.append(psth0)
            
            if plot==1:
                
                # Calculate receptive fields
                rf_blob0, rf_size0, rf_loc, filt_eta0,_,_,_,_ = calc_rf(
                    eta0, null_eta0, peak_thresh=pt, blob_thresh=bt, gaus_f=gf, med_f=mf, upsample=us)
                
                rf_blob1, rf_size1, rf_loc, filt_eta1, _,_,_,_ = calc_rf(
                    eta1, null_eta1, peak_thresh=pt, blob_thresh=bt, gaus_f=gf, med_f=mf, upsample=us)
            
                # Plot positive RF and filtered ETA for the current condition (column)
                axes[0, 0].imshow(filt_eta0, cmap=plt.cm.bwr)
                axes[0, 0].set_title(f'Cond  Filtered ETA0')
                axes[0, 0].axis('off')
        
                axes[1, 0].imshow(rf_blob0)
                axes[1, 0].set_title(str(np.max(eta0)))
                axes[1, 0].axis('off')
        
                # Plot negative RF and filtered ETA for the current condition (column)
                axes[0, 1].imshow(filt_eta1, cmap=plt.cm.bwr)
                axes[0, 1].set_title(str(np.min(eta1)))
                axes[0, 1].axis('off')
        
                axes[1, 1].imshow(rf_blob1)
                axes[1, 1].set_title(f'RF Blob1')
                axes[1, 1].axis('off')
          
                # Adjust the layout and save the figure
                plt.tight_layout()
                plt.savefig(sn_exp['analysis_path'] + f'/probe_{probe}_unit_{unit}_grid.png')
                plt.close()
        
        all_eta0=np.array(all_eta0)
        np.save(sn_exp['analysis_path'] + f'/probe_{probe}_etas0.npy',all_eta0)
        all_eta1=np.array(all_eta1)
        np.save(sn_exp['analysis_path'] + f'/probe_{probe}_etas1.npy',all_eta1)
        all_nulleta0=np.array(all_nulleta0)
        np.save(sn_exp['analysis_path'] + f'/probe_{probe}_nulletas0.npy',all_nulleta0)
        all_nulleta1=np.array(all_nulleta1)
        np.save(sn_exp['analysis_path'] + f'/probe_{probe}_nulletas1.npy',all_nulleta1)
        all_psth0=np.array(all_psth0)
        np.save(sn_exp['analysis_path'] + f'/probe_{probe}_psth0.npy',all_psth0)


#%% SN lum pol mix


curr_ephys_paths=['/home/sam/bucket/electrophysiology/squidOL/2025_01_17/',
                  '/home/sam/bucket/electrophysiology/squidOL/2025_01_29/',
                  '/home/sam/bucket/electrophysiology/squidOL/2025_01_30/',
                  '/home/sam/bucket/electrophysiology/squidOL/2025_02_04/',
                  '/home/sam/bucket/electrophysiology/squidOL/2025_02_12/',
                  '/home/sam/bucket/electrophysiology/squidOL/2025_02_13/',
                  '/home/sam/bucket/electrophysiology/squidOL/2025_03_07/',
                  '/home/sam/bucket/electrophysiology/squidOL/2025_03_11/']


import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, gaussian_filter


plot = 0  # 1 = generate PSTH plots, 0 = skip PSTH plots

for curr_ephys_path in curr_ephys_paths:
    # Sample calls/definitions
    sn_exp = load_ephys_results(glob.glob(curr_ephys_path + '/*SN_LUM_POL_MIX')[0], 9, False)
    dt = int(np.mean(np.diff(sn_exp['stim_times'][:, 0])) * 1000) / 1000
    first_st = sn_exp['stim_times'][0, 0]
    stimulus_times = sn_exp['stim_times'][:, 0] -first_st 
    stimulus_times_pol = np.load(sn_exp['curr_experiment_path'] + '/STIM_WINDOWS.npy', allow_pickle=True)[:, 0] -first_st  + 0.025 #offset because of refresh, doesn't matter so much for ETA.
    stimulus_times_pol = stimulus_times_pol[0:len(stimulus_times)]
    
    int_mat = sn_exp['sn_mat_lum'][0:len(stimulus_times)]
    pol_mat = sn_exp['sn_mat_pol'][0:len(stimulus_times)]
    condition = sn_exp['cond'][0:len(stimulus_times)]
    unique_conds = np.unique(condition)
    unique_conds=range(0,3)
    for probe in range(len(sn_exp['st_mat'])):
        num_units = len(sn_exp['st_mat'][probe])
    
        all_eta0=np.zeros((num_units,len(unique_conds), int_mat.shape[1],int_mat.shape[2]))
        all_eta1=np.zeros((num_units,len(unique_conds), int_mat.shape[1],int_mat.shape[2]))
        all_nulleta0=np.zeros((num_units,len(unique_conds), int_mat.shape[1],int_mat.shape[2]))
        all_nulleta1=np.zeros((num_units,len(unique_conds), int_mat.shape[1],int_mat.shape[2]))
        all_psth0=[]
        
        for unit in range(num_units):
            # Initialize any needed variables
            rf_blob = np.zeros(1)
            neuron_rf=[]
            curr_times = sn_exp['st_mat'][probe][unit] -first_st 
    
            if plot==1:
                fig1, axes = plt.subplots(4, 3, figsize=(15, 10))  
                axes = axes.reshape(4, 3)
    
            all_psth = []
            for cond_idx, cond in enumerate(unique_conds):
    
                if cond == 0:
                    X = int_mat[condition == 0]
                    bpp, bin_edges = np.histogram(curr_times, bins=stimulus_times)
                    bpp = np.concatenate([bpp, [0]])/dt  # pad if needed
    
                elif cond == 1:
                    X = pol_mat[condition == 1]
                    bpp, bin_edges = np.histogram(curr_times, bins=stimulus_times_pol)
                    bpp = np.concatenate([bpp, [0]])/dt

                else:  # cond == 2
                    X = int_mat[condition == 2]
                    bpp, bin_edges = np.histogram(curr_times, bins=stimulus_times)
                    bpp = np.concatenate([bpp, [0]])/dt
    
                # Expand bpp for broadcasting
                bpp_expanded = bpp[:, np.newaxis, np.newaxis]
                curr_bpp = bpp_expanded[condition == cond]
    
                eta0, eta1, null_eta0, null_eta1 = compute_real_and_null_etas(X, curr_bpp)
    
                if cond ==1:
                    curr_stimtimes=stimulus_times_pol[condition ==1]
                    
                    peak_ind = np.unravel_index(np.argmax(eta0), eta0.shape)
                    has_neg1 = np.where(X[:,int(peak_ind[0]),int(peak_ind[1])] > 0 )[0] #
                    psth0, bin_edges, aligned_spikes=calc_psth(curr_stimtimes[has_neg1], curr_times, [0, .2], bin_width=0.001, trial_samp=1000, plot=0)
                    all_psth0.append(psth0)
    
                all_eta0[unit,cond_idx]=eta0
                all_eta1[unit,cond_idx]=eta1
                all_nulleta0[unit,cond_idx]=null_eta0
                all_nulleta1[unit,cond_idx]=null_eta1
                
                if plot == 1:
                    # Calculate receptive fields
                    rf_blob0, rf_size0, _, filt_eta0, _, _, _,_ = calc_rf(
                        eta0, null_eta0, peak_thresh=pt, blob_thresh=bt, gaus_f=gf, med_f=mf, upsample=us)
                        
                    rf_blob1, rf_size1, _, filt_eta1, _, _, _ ,_ = calc_rf(
                        eta1, null_eta1, peak_thresh=pt, blob_thresh=bt, gaus_f=gf, med_f=mf, upsample=us)
        
                    # -- Plot in figure(1) --
                    # Positive RF
                    axes[0, cond_idx].imshow(filt_eta0, cmap=plt.cm.bwr)
                    axes[0, cond_idx].set_title(str(np.max(eta0)))
                    axes[0, cond_idx].axis('off')
        
                    axes[1, cond_idx].imshow(rf_blob0)
                    axes[1, cond_idx].set_title(f'Cond {cond} RF Blob1')
                    axes[1, cond_idx].axis('off')
        
                    # Negative RF
                    axes[2, cond_idx].imshow(filt_eta1, cmap=plt.cm.bwr)
                    axes[2, cond_idx].set_title(str(np.min(eta1)))
                    axes[2, cond_idx].axis('off')
        
                    axes[3, cond_idx].imshow(rf_blob1)
                    axes[3, cond_idx].axis('off')
            
            if plot ==1:
                # -- Save Figure(1): the 4x3 RF grid --
                fig1.tight_layout()
                fig1.savefig(sn_exp['analysis_path'] + f'/probe_{probe}_unit_{unit}_grid.png')
                plt.close(fig1)


        np.save(sn_exp['analysis_path'] + f'/probe_{probe}_etas0.npy',all_eta0)
        np.save(sn_exp['analysis_path'] + f'/probe_{probe}_etas1.npy',all_eta1)
        np.save(sn_exp['analysis_path'] + f'/probe_{probe}_nulletas0.npy',all_nulleta0)
        np.save(sn_exp['analysis_path'] + f'/probe_{probe}_nulletas1.npy',all_nulleta1)
        all_psth0=np.array(all_psth0)
        np.save(sn_exp['analysis_path'] + f'/probe_{probe}_psth0.npy',all_psth0)
    


#%% Example pol receptive field, RF size and pos correlations. Presence of RF for pol drop vs increase
#try using the DG for on-off pol and for horiz/vert responses

#dim0=int
#dim1=horiz_pol
#dim2=vert_pol

use_anatamy=0 #I'll save sizes and depths to combine with the SN5 datasets


curr_anatamy_paths=['/home/sam/bucket/DBS/data/20250127_121_squid_NP_recording_0117_18mm_TOPRO_3',
'/home/sam/bucket/DBS/data/20250213_130_squid_NP_recording_20250129_15mm_TOPRO_3',
'/home/sam/bucket/DBS/data/20250214_131_squid_NP_recording_20250130_17mm_TOPRO_3',
'/home/sam/bucket/DBS/data/20250310_133_squid_NP_recording_20250204_16mm_TO_PRO_3',
'/home/sam/bucket/DBS/data/20250311_135_squid_NP_recording_20250212_16mm_TO_PRO_3',
'/home/sam/bucket/DBS/data/20250313_137_squid_NP_recording_20250213_18mm_TO_PRO_3',
'/home/sam/bucket/DBS/data/20250318_F138_squid_NP_recording_20250307_21mm_TO_PRO_3',
'/home/sam/bucket/DBS/data/20250403_F142_squid_NP_recording_20250311_18mm_TO_PRO_3'

]

curr_ephys_paths=['/home/sam/bucket/electrophysiology/squidOL/2025_01_17/',
                  '/home/sam/bucket/electrophysiology/squidOL/2025_01_29/',
                  '/home/sam/bucket/electrophysiology/squidOL/2025_01_30/',
                  '/home/sam/bucket/electrophysiology/squidOL/2025_02_04/',
                  '/home/sam/bucket/electrophysiology/squidOL/2025_02_12/',
                  '/home/sam/bucket/electrophysiology/squidOL/2025_02_13/',
                  '/home/sam/bucket/electrophysiology/squidOL/2025_03_07/',
                  '/home/sam/bucket/electrophysiology/squidOL/2025_03_11/']


#first get eta and null eta array for all experiments across vert and mix, light on and off
all_eta=[]
all_nulleta=[]
path_list=[]
unit_list=[]
all_depths=[]
all_br=[]
all_psths=[]
all_psthsv=[]
for index in range(len(curr_ephys_paths)):

    curr_ephys_path=curr_ephys_paths[index]
    if use_anatamy:
        curr_anatamy_path=curr_anatamy_paths[index]
        exp = load_ephys_results(glob.glob(curr_ephys_path + '/*SN_LUM_POL_MIX')[0], 9, False,curr_anatamy_path)
    else:
        exp = load_ephys_results(glob.glob(curr_ephys_path + '/*SN_LUM_POL_MIX')[0], 9, False)
        
    vert_exp = load_ephys_results(glob.glob(curr_ephys_path + '/*SN_POL*')[0], 0, False)
    stimulus_times=exp['stim_times'][:, 0]
    stim_mat=exp['sn_mat_pol'][0:len(stimulus_times)]
    condition = exp['cond'][0:len(stimulus_times)]
    unique_conds = np.unique(condition)
    
    for probe in range(len(exp['st_mat'])):
        
        if os.path.exists(exp['analysis_path'] + '/probe_' + str(probe) + '_goodU.npy'):
            
            manual_check = np.load(exp['analysis_path'] + '/probe_' + str(probe) + '_goodU.npy')
            good_units=np.array(exp['unit_quality']['probe' + str(probe)]['good_units'])
           # good_probe = np.intersect1d(manual_check, good_units) 
            good_probe=manual_check
            
            eta0 = np.load(exp['analysis_path'] + '/probe_' + str(probe) +'_etas0.npy')[good_probe]
            eta1 = np.load(exp['analysis_path'] + '/probe_' + str(probe) +'_etas1.npy')[good_probe]
            eta0v = np.load(vert_exp['analysis_path'] + '/probe_' + str(probe) +'_etas0.npy')[good_probe]
            eta1v = np.load(vert_exp['analysis_path'] + '/probe_' + str(probe) +'_etas1.npy')[good_probe]
            psths=np.load(exp['analysis_path'] + f'/probe_{probe}_psth0.npy')[good_probe]
            psthsv=np.load(vert_exp['analysis_path'] + f'/probe_{probe}_psth0.npy')[good_probe]
            
            eta0[:,2,:,:]=eta0v #replace the high pol background with the vertical
            eta1[:,2,:,:]=eta1v #replace the high pol background with the vertical
            
            nulleta0 = np.load(exp['analysis_path'] + '/probe_' + str(probe) +'_nulletas0.npy')[good_probe]
            nulleta1 = np.load(exp['analysis_path'] + '/probe_' + str(probe) +'_nulletas1.npy')[good_probe]
            nulleta0v = np.load(vert_exp['analysis_path'] + '/probe_' + str(probe) +'_nulletas0.npy')[good_probe]
            nulleta1v = np.load(vert_exp['analysis_path'] + '/probe_' + str(probe) +'_nulletas1.npy')[good_probe]
            
            nulleta0[:,2,:,:]=nulleta0v #replace the high pol background with the vertical
            nulleta1[:,2,:,:]=nulleta1v #replace the high pol background with the vertical
            
            eta = np.stack([eta0, eta1], axis=4)
            nulleta = np.stack([nulleta0, nulleta1], axis=4)
            
            all_eta.append(eta)
            all_nulleta.append(nulleta)
            unit_list.extend(good_probe)
            all_psths.append(psths)
            all_psthsv.append(psthsv)
            
            if use_anatamy:
                curr_units = exp['unit_anatomy'][probe]
                curr_depths=curr_units['dist'][good_probe].to_numpy()
                all_br.extend(curr_units['brain_region'][good_probe].to_numpy())
                all_depths.append(curr_depths)
        
            
eta=np.concatenate(all_eta,axis=0)  #neurons x condition x X x Y x light on/off
nulleta=np.concatenate(all_nulleta,axis=0)


neurons, dims, xs, ys, light_cond = eta.shape

rf_sizes=np.zeros((neurons, dims, light_cond))
rf_peak=np.zeros((neurons, dims, light_cond))
rf_loc=np.zeros((neurons, dims, light_cond,2))
rf_mean=np.zeros((neurons, dims, light_cond))
rf_pol=np.zeros((neurons, dims, light_cond))

for i in range(neurons):
    for j in range(dims):
        for k in range(light_cond):
            
            rf, rf_sizes[i,j,k], rf_loc[i,j,k], filt_rf, _, rf_peak[i,j,k], _ ,rf_pol[i,j,k] = calc_rf(
                eta[i, j, :,:,k], nulleta[i, j, :,:,k],
                peak_thresh=pt, blob_thresh=bt,
                gaus_f=gf, med_f=mf, upsample=us)
            rf_peak[i,j,k]=np.max(eta[i, j, :,:,k])
            rf_mean[i,j,k]=np.mean(nulleta[i, j, :,:,k])


valid_neurons = np.all(rf_peak > 0, axis=(1, 2))   # must be >0 everywhere
removed_n = np.sum(~valid_neurons)
print(f"Removed {removed_n} neurons that had a zero RF peak in at least one condition.")

# Keep only the good ones
rf_sizes = rf_sizes[valid_neurons]
rf_peak  = rf_peak [valid_neurons]
rf_loc   = rf_loc  [valid_neurons]
rf_mean  = rf_mean [valid_neurons]
rf_pol   = rf_pol  [valid_neurons]
psth=np.concatenate(all_psths)[valid_neurons]
psth_v=np.concatenate(all_psthsv)[valid_neurons]

# Update neuron count for downstream analyses
neurons = rf_sizes.shape[0]
print(f"{neurons} neurons remain for analysis.\n")

if use_anatamy:
    all_depths=np.concatenate(all_depths)
    all_depths = all_depths[valid_neurons]
    all_sizes=rf_sizes[:,0,0]
    np.save('/home/sam/OL_toplogy/ephys/mix_sizes',all_sizes)
    np.save('/home/sam/OL_toplogy/ephys/mix_depths',all_depths)
    np.save('/home/sam/OL_toplogy/ephys/mix_br',all_br)
    
has_rf=rf_sizes[:,0,0]>0
has_int_rf   = has_rf
# ──────────────────────────────────────────────────────────────────
# 1.  Basic polarity statistics
# ------------------------------------------------------------------
n_total = np.sum(has_rf)

pos_pol_H = np.sum(rf_sizes[has_rf, 1, 0] > 0) / n_total
pos_pol_V = np.sum(rf_sizes[has_rf, 2, 0] > 0) / n_total
neg_pol_H = np.sum(rf_sizes[has_rf, 1, 1] > 0) / n_total
neg_pol_V = np.sum(rf_sizes[has_rf, 2, 1] > 0) / n_total

print("─ Polarity fractions (relative to all neurons with an RF) ─")
print(f"  Positive  H-pol : {pos_pol_H:6.3f}")
print(f"  Positive  V-pol : {pos_pol_V:6.3f}")
print(f"  Negative  H-pol : {neg_pol_H:6.3f}")
print(f"  Negative  V-pol : {neg_pol_V:6.3f}\n")

# ──────────────────────────────────────────────────────────────────
# 2.  Intensity-RF ↔ Polarity-RF overlap & size correlation
# ------------------------------------------------------------------
has_pol_rf   = np.logical_or(rf_sizes[:, 1, 0] > 0, rf_sizes[:, 2, 0] > 0)
has_both_rf  = np.logical_and(rf_sizes[:, 0, 0] > 0, rf_sizes[:, 1, 0] > 0)

frac_int_with_pol = np.sum(rf_sizes[:, 1, 0] > 0) / np.sum(has_int_rf)
print("─ RF overlap ─")
print(f"  Fraction of intensity-RF neurons that also have a polarity RF: {frac_int_with_pol:.3f}\n")

# Size correlation (only cells that have both RF types)
r_size, p_size = pearsonr(rf_sizes[has_both_rf, 0, 0],
                          rf_sizes[has_both_rf, 1, 0])
print("─ RF size correlation (intensity width vs. polarity width) ─")
print(f"  r = {r_size:.3f}   p = {p_size:.3e}\n")

has_horiz_rf = rf_sizes[:, 1, 0] > 0
np.sum(rf_pol[has_horiz_rf, 1,0]>0) / np.sum(has_horiz_rf)



# ──────────────────────────────────────────────────────────────────
# 3.  RF centre correlation (X & Y) between intensity & polarity maps
# ------------------------------------------------------------------
r_x, p_x = pearsonr(rf_loc[has_both_rf, 0, 0, 0],  # intensity-X
                    rf_loc[has_both_rf, 1, 0, 0])  # polarity-X
r_y, p_y = pearsonr(rf_loc[has_both_rf, 0, 0, 1],  # intensity-Y
                    rf_loc[has_both_rf, 1, 0, 1])  # polarity-Y
print("─ RF centre correlation ─")
print(f"  X-coords : r = {r_x:.3f}   p = {p_x:.3e}")
print(f"  Y-coords : r = {r_y:.3f}   p = {p_y:.3e}\n")


curr_psth=psth[has_pol_rf]
curr_psthV=psth_v[has_pol_rf]

import numpy as np
from scipy.stats import pearsonr

f=5
correlations = np.zeros(curr_psth.shape[0])
for i in range(curr_psth.shape[0]):
    correlations[i], _ = pearsonr(gaussian_filter(curr_psth[i,:],f), gaussian_filter(curr_psthV[i,:],f))


ogl_data=np.array([-4.79463882e-01,  3.86508177e-01,  3.22952351e-01, -8.94110848e-01,
       -8.69076307e-01, -8.35422015e-01,  5.20058237e-02, -5.37442341e-01,
        8.41104621e-02, -6.80704483e-01, -3.73879598e-01,  1.25757791e-01,
        6.75771746e-01,  8.05441991e-01,  2.29686904e-02,  1.20304250e-01,
       -5.00645550e-01,  7.53753462e-01,  7.15718899e-01,  5.60528619e-01,
       -1.12700500e-01,  3.80793628e-01, -9.37927882e-02, -6.63357560e-01,
       -6.27146254e-01, -1.35679673e-01,  6.99938567e-01,  5.04955647e-02,
        5.19004727e-01,  2.75316916e-01, -3.94948267e-01,  1.53629536e-01,
       -2.22863074e-01, -1.71195635e-01, -6.08739140e-01,  3.27423013e-01,
       -5.03851209e-01, -1.24829147e-01, -4.84198929e-01,  3.49098667e-01,
       -6.90599213e-01, -5.65046733e-01,  1.25737778e-01, -2.54302510e-01,
       -4.46044292e-01, -4.50717948e-01, -2.70542756e-01, -6.30304236e-01,
       -1.08659427e-01,  3.95542633e-01, -7.36720493e-01, -2.50128521e-01,
       -3.94222244e-01, -6.96872487e-01, -4.42371631e-01, -5.33535220e-01,
       -9.80367423e-02,  9.52260131e-02, -6.39914662e-01, -5.31063328e-01,
        8.94512941e-02, -6.93585654e-01, -9.45416747e-01, -3.30536495e-02,
        2.59129705e-01, -7.69328611e-01, -6.78426992e-01,  5.74192409e-01,
        3.30733325e-02, -4.64680704e-01, -3.00299874e-01, -2.56610905e-01,
       -4.68181433e-01, -8.24399641e-02, -6.81254609e-01, -6.10420063e-01,
       -1.92926958e-01, -3.02041506e-01, -2.40055699e-01,  1.79385799e-01,
       -7.60029511e-02, -3.34552779e-01,  5.40074530e-01,  7.55718239e-01,
       -3.64049477e-01, -6.72646689e-01,  1.22868069e-02, -4.68087947e-01,
       -3.75478261e-01, -7.52608467e-01, -4.46601129e-01, -7.62412767e-01,
       -7.87978416e-01, -1.51541395e-01,  3.10363595e-02, -2.13067216e-01,
       -4.24015349e-01, -3.17919745e-01, -6.43903814e-01, -4.85796200e-01,
       -3.23959668e-01, -4.45575489e-01, -7.15200097e-01, -8.60340802e-01,
       -3.72853380e-01, -7.37051912e-01, -8.72936409e-01, -5.17036823e-01,
       -4.31494982e-01, -8.63994271e-01, -5.51750218e-01, -8.14022867e-01,
       -5.81092706e-01, -1.84029915e-01, -8.86116612e-01, -3.38653794e-01,
       -1.15120712e-01, -2.93900954e-01, -7.29692396e-01, -8.47854785e-01,
       -3.21989605e-01, -4.63703192e-02, -5.01035664e-01, -3.39255203e-01,
       -7.00233055e-01, -5.40829600e-01, -2.58053290e-01,  3.90053129e-01,
       -4.61843586e-01, -3.71895804e-01, -7.76747461e-01, -6.59931546e-01,
        3.01013661e-01, -7.14646120e-01,  1.59460188e-01, -2.65809609e-01,
        7.63470876e-01,  1.34099774e-01, -7.85679477e-01, -4.18579887e-01,
       -1.48171803e-01, -4.61382552e-01, -4.98440410e-01, -1.29949207e-01,
       -7.24169676e-01,  6.26874771e-01, -3.27957159e-01, -4.81480045e-01,
       -7.34681068e-01, -5.62897955e-01, -2.15320782e-01, -5.32186922e-01,
        2.38227623e-02, -4.94730593e-01, -1.55212918e-01, -3.13403825e-01,
       -3.20437983e-01, -7.52570546e-01, -6.28311185e-01, -7.39070872e-01,
       -6.15564016e-01, -7.85083628e-01, -3.39156464e-01, -7.08323629e-01,
        4.52424902e-01, -4.27660945e-01, -7.80810812e-01, -5.39263263e-01,
        5.66302395e-04, -8.81918757e-01, -2.91722898e-01, -6.28319274e-01,
       -5.66466184e-01, -6.70158491e-01, -2.29103472e-01,  2.98624215e-01,
       -5.50347283e-01, -6.06913801e-01,  1.03284780e-02, -6.81140817e-01,
       -4.69086802e-01, -8.08255866e-01, -5.95756510e-01,  6.21965047e-01,
       -7.60978706e-01, -4.85117138e-01,  3.79740894e-01, -6.97398303e-01,
        2.02626246e-01,  2.53532536e-01, -2.98938187e-01, -2.05249459e-01,
        5.22819283e-01, -6.52735890e-01, -3.47127321e-01, -7.73701452e-01,
       -4.49472366e-01, -7.32795639e-01, -5.79347452e-01,  1.68818481e-01,
       -5.21526803e-01, -5.66403011e-01, -5.60849853e-01, -5.15208983e-01,
       -2.59797749e-01, -5.00884916e-01,  2.36721885e-01, -8.04910310e-01,
        9.07619039e-02,  8.80192700e-02, -6.34523613e-01, -5.99038168e-01,
       -6.70804536e-01, -7.58718765e-01])

igl_data=np.array([ 0.41329346,  0.15805228,  0.29012335,  0.67007501, -0.31926045,
        0.50448136, -0.08502833, -0.49564086,  0.25967239, -0.34336603,
        0.3519872 , -0.41893392,  0.5903655 , -0.03939804,  0.50666555,
        0.08043769, -0.20687483, -0.10026138,  0.10598926,  0.54885139,
        0.19993036,  0.07312818,  0.15289678,  0.3991133 , -0.18472735,
       -0.83924636, -0.860218  , -0.78954433, -0.88693271,  0.59839424,
       -0.91241337, -0.9532127 ,  0.02640715,  0.75399291, -0.58752889,
        0.71909581, -0.96529727,  0.70246704,  0.67431362, -0.51386836,
       -0.23358657, -0.71861904, -0.8056891 , -0.78284717, -0.42679273,
        0.25020257,  0.01291117,  0.28602346, -0.95516583, -0.54780273,
       -0.95501015, -0.16068611, -0.94601185, -0.89116778, -0.78756318,
       -0.70436564, -0.83533233, -0.5185852 , -0.94438043, -0.74800168,
        0.39734796, -0.88908583,  0.7367639 , -0.96344848, -0.62379266,
       -0.08011653, -0.95993361, -0.93945412,  0.7111298 , -0.57923154,
       -0.33332217, -0.97917144, -0.93383321, -0.69692639, -0.24696181,
        0.56117121, -0.92977765, -0.65241905,  0.48713126, -0.85011965,
       -0.91924841, -0.91096174,  0.73416329, -0.37514509,  0.52371903,
       -0.9735581 ,  0.56680507, -0.58810425, -0.30751168, -0.90569184,
        0.53669886, -0.9441441 ,  0.24740231, -0.4787107 , -0.52633188,
       -0.77529822, -0.98145658,  0.3547479 ,  0.42163708, -0.69192997,
        0.00954455, -0.57715544,  0.40078228, -0.30815022, -0.38434212,
       -0.80149756, -0.9621475 , -0.36151199, -0.38166943, -0.77145791,
       -0.97243596, -0.43476885, -0.14400762,  0.43826013,  0.03725592,
        0.76131605, -0.73851614, -0.49506157,  0.43097022, -0.59363265,
       -0.38733565,  0.0701185 ,  0.55947405, -0.76069141,  0.87040696,
       -0.82380781, -0.83346159, -0.07543269,  0.82740854, -0.75919956,
       -0.5170218 ,  0.62031633, -0.87952826, -0.86623912, -0.8629589 ,
       -0.54477243, -0.4610283 , -0.4870659 , -0.70510716,  0.76572818,
       -0.80310766,  0.72874514,  0.1427854 , -0.90508868, -0.94368473,
       -0.72748717,  0.02197738, -0.81779846, -0.92838886,  0.33259094,
       -0.62678468,  0.06374826, -0.15767053,  0.05763641, -0.0362997 ,
       -0.74390538, -0.00511512,  0.10219777, -0.13792469,  0.03135711,
       -0.96970779, -0.49880606, -0.17722528, -0.4124028 , -0.87652647,
       -0.10074439, -0.64286713, -0.94361478, -0.94767765, -0.2699315 ,
       -0.89383017, -0.87338641, -0.62785496,  0.27737597, -0.52184869,
        0.00474889,  0.03612798, -0.75146076, -0.20371206,  0.08399973,
       -0.22149375, -0.97576613,  0.44532385, -0.86035112, -0.12631738,
        0.31596518, -0.3580431 , -0.2938339 ,  0.64425549,  0.17872594,
       -0.49959055, -0.61270252, -0.93649451, -0.67608572,  0.44197414,
       -0.19946239, -0.73937531,  0.3864203 , -0.55946094, -0.39115464,
        0.12726708, -0.52222431,  0.16093365, -0.68071784, -0.09176614,
        0.09893213, -0.90088109,  0.131656  ,  0.27726918, -0.69952688,
       -0.72535986,  0.50548482, -0.83334653, -0.63045633, -0.42248571,
       -0.00999702, -0.0616059 , -0.1229713 , -0.71380492, -0.65005232,
       -0.40581474, -0.74314341, -0.80459252, -0.49002565, -0.36028621,
       -0.74929054, -0.626489  , -0.09504125, -0.86974207, -0.21391723,
       -0.63455617, -0.89102271, -0.29834531,  0.00128697, -0.7432691 ,
       -0.20872349, -0.84723128, -0.97881059,  0.36810839, -0.13590059,
       -0.12966161,  0.55208317, -0.92978164,  0.04529008,  0.1522203 ,
       -0.07083694, -0.97464119, -0.33966052,  0.44162356, -0.74184346,
       -0.82378336, -0.04232905, -0.67082164,  0.03296344,  0.71006253,
       -0.16736318, -0.46820147, -0.14127637,  0.14960673, -0.18629746,
        0.00697517, -0.7319949 ,  0.20301622,  0.7574837 , -0.00647137,
       -0.36501487,  0.04228213,  0.47554374, -0.15066922, -0.20223636,
        0.18111204, -0.28824186, -0.95525362, -0.18875035, -0.90059651,
        0.10042609, -0.09936876, -0.97754326,  0.39030802,  0.09400765,
       -0.96808143, -0.96901148, -0.69424247, -0.9218505 ,  0.58049099,
        0.56496021, -0.94351466, -0.93731932,  0.13221428, -0.78164186,
       -0.17131169,  0.71314759,  0.3021104 ,  0.67225629, -0.09656363,
        0.86766181, -0.84306154])

import numpy as np
import matplotlib.pyplot as plt

bins = np.linspace(-1, 1, 15)

plt.figure(figsize=(3, 3))

# Filled histogram for correlations
plt.hist(correlations, bins=bins, density=True, color='k', alpha=0.6, label='correlations')

# Pool ogl and igl
pooled = np.concatenate([ogl_data, igl_data])
pooled_counts, _ = np.histogram(pooled, bins=bins, density=True)

# Repeat edges and counts to span full bin width
step_edges = np.repeat(bins, 2)[1:-1]  # duplicate each edge, but exclude first and last
step_heights = np.repeat(pooled_counts, 2)  # duplicate each count

plt.plot(step_edges, step_heights, color='purple', linewidth=4, label='pooled ogl + igl')

plt.xlabel('degree_ind')
plt.legend()
plt.tight_layout()
plt.show()


plt.savefig('degree_ind.pdf', dpi=600)



#%% RF example
#kostas' screen is 125x70

a=np.load('/home/sam/bucket/electrophysiology/squidOL/2025_03_07/2025_03_07_rec02_SN_POL_VER_SCREEN/analysis_folder/probe_0_etas0.npy')
c=np.load('/home/sam/bucket/electrophysiology/squidOL/2025_03_07/2025_03_07_rec03_SN_LUM_POL_MIX/analysis_folder/probe_0_etas0.npy')
b=np.load('/home/sam/bucket/electrophysiology/squidOL/2025_03_07/2025_03_07_rec02_SN_POL_VER_SCREEN/analysis_folder/probe_0_etas1.npy')
d=np.load('/home/sam/bucket/electrophysiology/squidOL/2025_03_07/2025_03_07_rec03_SN_LUM_POL_MIX/analysis_folder/probe_0_etas1.npy')

fig, axes = plt.subplots(2, 3, figsize=(6,3))
curr_n = 53 #its a good unit, I checked
smoothness = 1  # Set the smoothing parameter
upsample_factor = 5  # Set the upsampling factor
crop_size= (60,80) #degree visual angle

# Define function to process data (Gaussian smoothing + upsampling + vmin/vmax calculation)
import numpy as np
from scipy.ndimage import gaussian_filter, zoom

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

def process_data(data, smoothness, upsample_factor, crop_size, peak):
    """
    Process data with Gaussian smoothing and upsampling, then crop a fixed-size region centered at `peak`,
    padding with zeros if needed.

    Parameters:
        data : 2D numpy array
        smoothness : float
            Gaussian sigma
        upsample_factor : float
            Zoom factor
        crop_size : (int, int)
            Output crop size (height, width)
        peak : (int, int)
            (y, x) coordinates for crop center

    Returns:
        cropped_img : 2D array of shape `crop_size`
        vmin, vmax : display range (1st and 99th percentile)
        peak : location used
    """
    # Smooth and upsample
    smoothed = gaussian_filter(data, sigma=smoothness)
    upsampled = zoom(smoothed, upsample_factor, order=1)

    # Compute display range
    vmin, vmax = np.percentile(upsampled, [1, 99])

    # Ensure peak is int and in (y, x) order
    cy, cx = map(int, peak)
    ch, cw = crop_size

    # Desired crop bounds
    y_start = cy - ch // 2
    y_end   = cy + ch // 2
    x_start = cx - cw // 2
    x_end   = cx + cw // 2

    # Initialize zero-padded image
    cropped_img = np.zeros((ch, cw), dtype=upsampled.dtype)

    # Compute source and destination slice ranges
    src_y_start = max(y_start, 0)
    src_y_end   = min(y_end, upsampled.shape[0])
    src_x_start = max(x_start, 0)
    src_x_end   = min(x_end, upsampled.shape[1])

    dst_y_start = src_y_start - y_start
    dst_y_end   = dst_y_start + (src_y_end - src_y_start)
    dst_x_start = src_x_start - x_start
    dst_x_end   = dst_x_start + (src_x_end - src_x_start)

    # Copy valid region from upsampled to cropped_img
    cropped_img[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        upsampled[src_y_start:src_y_end, src_x_start:src_x_end]

    return cropped_img, vmin, vmax, peak



peak=(33,49)
peak0=(35,70)
peak1= (38,77)


img_a, vmin_a, vmax_a, peak = process_data(a[curr_n], smoothness, upsample_factor, crop_size, peak)
axes[0, 1].imshow(img_a, vmin=vmin_a, vmax=vmax_a, cmap='bone_r')
axes[0, 1].set_title(f"1%: {vmin_a:.2f}, 99%: {vmax_a:.2f}")

img_c0, vmin_c0, vmax_c0, peak0 = process_data(c[curr_n, 0], smoothness, upsample_factor, crop_size, peak0)
axes[0, 0].imshow(img_c0, vmin=vmin_c0, vmax=vmax_c0, cmap='bone_r')
axes[0, 0].set_title(f"1%: {vmin_c0:.2f}, 99%: {vmax_c0:.2f}")

img_c1, vmin_c1, vmax_c1, peak1 = process_data(c[curr_n, 1], smoothness, upsample_factor, crop_size, peak1)
axes[0, 2].imshow(img_c1, vmin=vmin_c1, vmax=vmax_c1, cmap='bone_r')
axes[0, 2].set_title(f"1%: {vmin_c1:.2f}, 99%: {vmax_c1:.2f}")

img_b, vmin_b, vmax_b, _ = process_data(b[curr_n], smoothness, upsample_factor, crop_size, peak)
axes[1, 1].imshow(img_b, vmin=vmin_b, vmax=vmax_b, cmap='bone_r')
axes[1, 1].set_title(f"1%: {vmin_b:.2f}, 99%: {vmax_b:.2f}")

img_d0, vmin_d0, vmax_d0, _ = process_data(d[curr_n, 0], smoothness, upsample_factor, crop_size, peak0)
axes[1, 0].imshow(img_d0, vmin=vmin_d0, vmax=vmax_d0, cmap='bone_r')
axes[1, 0].set_title(f"1%: {vmin_d0:.2f}, 99%: {vmax_d0:.2f}")

img_d1, vmin_d1, vmax_d1, _ = process_data(d[curr_n, 1], smoothness, upsample_factor, crop_size, peak1)
axes[1, 2].imshow(img_d1, vmin=vmin_d1, vmax=vmax_d1, cmap='bone_r')
axes[1, 2].set_title(f"1%: {vmin_d1:.2f}, 99%: {vmax_d1:.2f}")

# Remove axes ticks for a cleaner visualization
for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

plt.savefig('rf_mix.pdf',dpi=600)


#%% DG good experiments

#good but need to recalibrate pol screen!

#Gray
gray_paths=['/home/sam/bucket/electrophysiology/squidOL/2025_01_08/',
            '/home/sam/bucket/electrophysiology/squidOL/2025_01_10/',
            '/home/sam/bucket/electrophysiology/squidOL/2025_01_29/',
            '/home/sam/bucket/electrophysiology/squidOL/2025_01_30/',
            '/home/sam/bucket/electrophysiology/squidOL/2025_02_04/',
            '/home/sam/bucket/electrophysiology/squidOL/2025_03_07/']

#Pol
pol_paths=['/home/sam/bucket/electrophysiology/squidOL/2025_01_29/',
    '/home/sam/bucket/electrophysiology/squidOL/2025_01_30/',
    '/home/sam/bucket/electrophysiology/squidOL/2024_11_12/',
    '/home/sam/bucket/electrophysiology/squidOL/2025_03_25_DONE/',
    '/home/sam/bucket/electrophysiology/squidOL/2025_03_25_H/',
    '/home/sam/bucket/electrophysiology/squidOL/2025_03_25_H_DONE/',
    '/home/sam/bucket/electrophysiology/squidOL/2025_04_15/',
    '/home/sam/bucket/electrophysiology/squidOL/2025_04_28/',
    '/home/sam/bucket/electrophysiology/squidOL/2025_04_28_H/']


#Black
black_paths=['/home/sam/bucket/electrophysiology/squidOL/2025_01_10/',
             '/home/sam/bucket/electrophysiology/squidOL/2025_01_17/',
             '/home/sam/bucket/electrophysiology/squidOL/2025_01_08/',
             '/home/sam/bucket/electrophysiology/squidOL/2025_02_04/',
             '/home/sam/bucket/electrophysiology/squidOL/2025_02_12/',
             '/home/sam/bucket/electrophysiology/squidOL/2025_02_13/',
             '/home/sam/bucket/electrophysiology/squidOL/2025_03_07/',
             '/home/sam/bucket/electrophysiology/squidOL/2025_03_10/',
             '/home/sam/bucket/electrophysiology/squidOL/2025_03_11/',   
             '/home/sam/bucket/electrophysiology/squidOL/2025_03_13/']


#%%  DG analysis

exp_type='BLACK' 
plot=0
num_bins = 18#75
n_folds  = 5                  # CV folds per bin
rng      = np.random.default_rng(seed=0)   # reproducible shuffles

# containers for downstream analyses (unchanged)
all_neurons      = []
all_time_courses = []
all_dsets        = []

# container for population data
pop_data_list = []            # each entry: (folds, bins, time)



if exp_type=='POL': #pol
    curr_ephys_paths=pol_paths
elif exp_type=='GRAY':
    curr_ephys_paths=gray_paths
elif exp_type=='BLACK':
    curr_ephys_paths=black_paths
  
    
dolp_map=np.load('/home/sam/OL_toplogy/ephys/dolp_map_tomo.npy')

for dset_ind, curr_ephys_path in enumerate(curr_ephys_paths):
    
    if exp_type=='POL': #pol
        experiments = sorted(glob.glob(curr_ephys_path + '*DG_POL_HOR_SCREEN*'))

    elif exp_type=='GRAY':
        experiments = glob.glob(curr_ephys_path + '*DG_LUM_GRAY*')
        null_trial=0.5
    
    elif exp_type=='BLACK':
        experiments=glob.glob(curr_ephys_path + '*DG_LUM_BLACK*')
        null_trial=0
    
        
    
    # ── MAIN LOOP ──────────────────────────────────────────────────────
    for exp_name in experiments:
        print(exp_name)
        exp = load_ephys_results(exp_name, 1, False)
    
        # ── stimulus intensities (or DoLP) ────────────────────────────
        intensity = exp['dg_params']['BLUE_int'].to_numpy().astype(float)
        stim_bins = np.linspace(-1, 1, num_bins)          # default (INT) bins
        keep      = ~np.isnan(intensity)                  # remove NaNs
        intensity = intensity[keep]
        stim_times = exp['stim_times'][:, 0][keep]
    
        if exp_type == 'POL':                              # DoLP experiments
            stim_bins  = np.linspace(0, 1, num_bins)
            sub        = intensity < 0.65                 # drop high-DoLP trials
            intensity  = intensity[sub]
            stim_times = stim_times[sub]
            u8         = np.round((intensity + 1) / 2 * 255).astype(np.uint8)
            intensity  = dolp_map[u8]                     # map to DoLP values
            
        if exp_type == 'BLACK':
            stim_bins = np.linspace(-1, 0, num_bins) 
            
        bin_midpoints= (stim_bins[:-1] + stim_bins[1:]) / 2
        
        # basic timing
        stim_dur     = float(np.mean(np.diff(stim_times)))     # seconds
        trial_len    = 1000#int(round(stim_dur * 1000))             # ms at 1 kHz
    
        # ── iterate over probes / units ───────────────────────────────
        for probe in range(len(exp['st_mat'])):
            for unit, spikes in enumerate(exp['st_mat'][probe]):
    
                # psth not needed here except for aligned_spikes
                _, _, aligned = calc_psth(
                    stim_times, spikes, [0, stim_dur],
                    trial_samp=10_000, bin_width=0.001, plot=0
                )
    
                # per-neuron container: (folds, bins, time)
                fold_resp = np.zeros((n_folds, num_bins-1, trial_len), dtype=np.float32)
    
                # which trials land in which stimulus bin
                bin_idx = np.digitize(intensity, stim_bins)    # 1…num_bins
    
                # ── build fold-wise responses ────────────────────────
                for b in range(1, num_bins):                   # skip bin 0 (underflow)
                    trials = np.where(bin_idx == b)[0]
                    if trials.size == 0:
                        continue
    
                    # ensure ≥ n_folds samples by repeating if needed
                    reps   = int(np.ceil(n_folds / trials.size))
                    trials = np.tile(trials, reps)[:n_folds * ((trials.size+reps-1)//reps)]
                    rng.shuffle(trials)                        # in-place shuffle
    
                    # assign every kth trial to fold k
                    for f in range(n_folds):
                        sel = trials[f::n_folds]
                        if sel.size == 0:
                            continue
                        bpp = np.zeros(trial_len, dtype=np.float32)
                        for tr in sel:
                            ms = (np.asarray(aligned[tr]) * 1000).astype(int)
                            ms = ms[(0 <= ms) & (ms < trial_len)]
                            bpp[ms] += 1000.0 / sel.size       # Hz
                        fold_resp[f, b-1] = gaussian_filter(bpp, 5)
    
                # ── apply original quality check & book-keeping ─────
                avg_rate = fold_resp.mean(0)[:, -800:].mean(1)
                sum_dev  = np.abs(fold_resp.mean(0).T - avg_rate).T
                
                if exp_type=='POL':
                    resp = gaussian_filter(sum_dev[1:-1, :200].mean(1), 1)
       
                else:
                    resp = gaussian_filter(sum_dev[:, :200].mean(1), 1)
                    
                if np.max(resp)-np.min(resp)>2:
                    resp = (resp - resp.min()) / resp.ptp()
                    all_neurons.append(resp)
                    all_time_courses.append(sum_dev.copy())
                    all_dsets.append(dset_ind)
    
                    # save fold-wise data for this neuron
                    pop_data_list.append(fold_resp)
    
# ── assemble population matrix & save ─────────────────────────────
pop_data = np.stack(pop_data_list, axis=-1)  # (folds, bins, time, neurons)
print("pop_data shape:", pop_data.shape)      # sanity check

an_array=np.array(all_neurons)

if exp_type == 'GRAY':
    np.save("pop_data.npy", pop_data)
    np.save("bins_int.npy", bin_midpoints)
    np.save("an_array.npy",an_array)
else:
    np.save("pop_data_pol.npy", pop_data[:,1:-1,:,:])
    np.save("bins_pol.npy", bin_midpoints[1:-1])
    np.save("an_array_pol.npy",an_array[:,1:-1])
    
                 
mean_response=np.mean(np.array(all_neurons), axis=0)
std_response = np.std(np.array(all_neurons), axis=0)
sem_response = std_response / np.sqrt(np.array(all_neurons).shape[0])  # SEM formula

if exp_type=='POL':
    bin_midpoints=bin_midpoints[1:-1]

    
# Plot the average response
plt.figure(figsize=(10, 6))
for neuron in all_neurons:
    filt_neuron=gaussian_filter(neuron,0)
    plt.plot(bin_midpoints, filt_neuron,'k',alpha=0.1)
plt.plot(bin_midpoints, mean_response, label='Average Response', color='blue')

# Add shaded region for standard deviation
plt.fill_between(bin_midpoints, 
                 mean_response - sem_response, 
                 mean_response + sem_response, 
                 color='blue', alpha=0.3, label='±1 Standard Error')
sum_drop=np.sum(an_array,axis=1)
int_sort=np.argsort(sum_drop)

plt.figure()
plt.imshow(an_array[int_sort,:],aspect='auto',interpolation='nearest')


# Customize plot
plt.xlabel('contrast')
plt.ylabel('Normalized Response')
plt.title('Average Response')
plt.legend()
plt.grid(True)
plt.show()

if exp_type=='POL':
    plt.savefig('contrast_res_pol.pdf',dpi=600)
else:
    plt.savefig('contrast_res.pdf',dpi=600)

#%% example psths

# exp_name='/home/sam/bucket/electrophysiology/squidOL/2024_11_12/2024_11_12_rec03_DG_POL_HOR_SCREEN/'
# probe=0
# unit=16
# exp_type='POL'
# v_max=20

exp_name='/home/sam/bucket/electrophysiology/squidOL/2025_01_10/2025_01_10_rec05_DG_LUM_GRAY_BG/'
probe=1
unit=1
exp_type='GRAY'
v_max=30



exp = load_ephys_results(exp_name, 1, False)

# ── stimulus intensities (or DoLP) ────────────────────────────
intensity = exp['dg_params']['BLUE_int'].to_numpy().astype(float)
stim_bins = np.linspace(-1, 1, num_bins)          # default (INT) bins
keep      = ~np.isnan(intensity)                  # remove NaNs
intensity = intensity[keep]
stim_times = exp['stim_times'][:, 0][keep]

if exp_type == 'POL':                              # DoLP experiments
    stim_bins  = np.linspace(0, 1, num_bins)
    
    sub        = intensity < 0.65                 # drop high-DoLP trials
    intensity  = intensity[sub]
    stim_times = stim_times[sub]
    u8         = np.round((intensity + 1) / 2 * 255).astype(np.uint8)
    intensity  = dolp_map[u8]                     # map to DoLP values
bin_midpoints= (stim_bins[:-1] + stim_bins[1:]) / 2

ordered_intensity=np.argsort(intensity)

# basic timing
stim_dur     = float(np.mean(np.diff(stim_times)))     # seconds
trial_len    = 1200#int(round(stim_dur * 1000))             # ms at 1 kHz


spikes=exp['st_mat'][probe][unit]
        

_, _, aligned = calc_psth(
    stim_times-.200, spikes, [0, stim_dur+0.3],
    trial_samp=10_000, bin_width=0.001, plot=0)

# per-neuron container: (folds, bins, time)
fold_resp = np.zeros((n_folds, num_bins-1, trial_len), dtype=np.float32)

# which trials land in which stimulus bin
bin_idx = np.digitize(intensity, stim_bins)    # 1…num_bins

# ── build fold-wise responses ────────────────────────
for b in range(1, num_bins):                   # skip bin 0 (underflow)
    trials = np.where(bin_idx == b)[0]
    if trials.size == 0:
        continue

    # ensure ≥ n_folds samples by repeating if needed
    reps   = int(np.ceil(n_folds / trials.size))
    trials = np.tile(trials, reps)[:n_folds * ((trials.size+reps-1)//reps)]
    rng.shuffle(trials)                        # in-place shuffle

    # assign every kth trial to fold k
    for f in range(n_folds):
        sel = trials[f::n_folds]
        if sel.size == 0:
            continue
        bpp = np.zeros(trial_len, dtype=np.float32)
        for tr in sel:
            ms = (np.asarray(aligned[tr]) * 1000).astype(int)
            ms = ms[(0 <= ms) & (ms < trial_len)]
            bpp[ms] += 1000.0 / sel.size       # Hz
        fold_resp[f, b-1] = gaussian_filter(bpp, 5)

# ── apply original quality check & book-keeping ─────
avg_rate = fold_resp.mean(0)[:, -500:].mean(1)
sum_dev  = np.abs(fold_resp.mean(0).T - avg_rate).T

plt.figure(figsize=(5,3))


# Set y-axis ticks to correspond to bin midpoints
if exp_type== 'GRAY':
    plt.imshow(np.mean(fold_resp, axis=0), aspect='auto', interpolation='nearest',cmap='bone_r',vmax=v_max)
    plt.yticks(ticks=np.arange(len(bin_midpoints)), labels=(np.round(bin_midpoints, 2)))

else:
    plt.imshow(np.mean(fold_resp[:, 1:-1], axis=0), aspect='auto', interpolation='nearest',cmap='bone_r',vmax=v_max)
    plt.yticks(ticks=np.arange(len(bin_midpoints[1:-1])), labels=2*(np.round(bin_midpoints[1:-1]-0.5, 2)))

plt.xlabel("Time bin")
plt.ylabel("Contrast bin midpoint")
plt.title("Mean response (fold average)")
plt.colorbar(label="Response amplitude")
plt.tight_layout()
plt.show()

if exp_type == 'GRAY':
    plt.savefig('gray_ex.pdf',dpi=600)
else:
    plt.savefig('pol_ex.pdf',dpi=600)
#%% Contrast sensitivity to black. As above, but pooling over different datasets

#-2025_02_04 ND1 filter (90%)
#-2025_02_06 used only red lights. Experiment didn't work well, don't use.
#-2025_02_12 ND2 filter (99%)
#-2025_02_13_ND2 filter
#-2025_03_07 ND3 filter (99.9%)
#-2025_03_10 ND3
#-2025_03_11 ND3
#-2025_03_13 ND3+ND2 (99.999%) (rec08).

plt.figure(figsize=(5, 3))

all_dsets = np.array(all_dsets)
all_neurons = np.array(all_neurons)
filt_type = np.array([0, 0, 0, 1, 2, 2, 3, 3, 3, 4])
filt_names = ['full', '0.1', '0.01', '0.001', '0.000001']

num_filters = np.max(filt_type) + 1  # +1 because filter types go from 0 to 4
colors = plt.cm.plasma(np.linspace(0, 1, num_filters))  # Or any colormap you like

for curr_type in range(num_filters):
    # Get dataset IDs that belong to this filter type
    target_dsets = np.where(filt_type == curr_type)[0]
    # Get indices in all_dsets that match
    matching_indices = np.where(np.isin(all_dsets, target_dsets))[0]
    
    # Extract the neurons for these indices
    curr_neurons = all_neurons[matching_indices]
    print(len(curr_neurons))
    if len(curr_neurons) == 0:
        continue  # Skip if there are no neurons for this filter
    
    mean_response = np.mean(curr_neurons, axis=0)
    std_response = np.std(curr_neurons, axis=0)
    sem_response = std_response / np.sqrt(curr_neurons.shape[0])

    # Plot mean response
    plt.plot(bin_midpoints, mean_response, label=filt_names[curr_type], color=colors[curr_type])
    
    # Plot shaded SEM region
    plt.fill_between(bin_midpoints, 
                     mean_response - sem_response, 
                     mean_response + sem_response, 
                     color=colors[curr_type], alpha=0.3)

# Final plot styling
plt.xlabel('Contrast')
plt.ylabel('Normalized Response')
plt.title('Average Response per Filter Type')
plt.legend(title='Filter')
#plt.grid(True)
plt.tight_layout()
plt.show()

plt.savefig('black_contrast_thresh.pdf', bbox_inches='tight', transparent=True)


#%% POL/Int bar


import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm          # <-- NEW (needed for ScalarMappable)
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------
curr_ephys_paths = [
    '/home/sam/bucket/electrophysiology/squidOL/2025_03_07/',
    '/home/sam/bucket/electrophysiology/squidOL/2025_01_30/',
    '/home/sam/bucket/electrophysiology/squidOL/2025_01_29/',
    '/home/sam/bucket/electrophysiology/squidOL/2025_02_04/',
    '/home/sam/bucket/electrophysiology/squidOL/2025_02_12/',
    '/home/sam/bucket/electrophysiology/squidOL/2025_02_13/',
    '/home/sam/bucket/electrophysiology/squidOL/2025_03_10/',
    '/home/sam/bucket/electrophysiology/squidOL/2025_03_11/',
    '/home/sam/bucket/electrophysiology/squidOL/2025_03_13/'
]

dolp_bg        = 0.9509421092629422     # DoLP in background condition
dolp_map       = np.load('/home/sam/OL_toplogy/ephys/dolp_map_tomo.npy')
plot           = 0                      # 1 = diagnostic single-cell plots
resp_thresh    = 15                      # Hz
num_bins       = 10                     # intensity × DoLP matrix size
num_bins_pol=10

all_neurons    = []
all_pol_vals   = []
all_lum_vals   = []

# ---------------------------------------------------------------------
# Helper : safe imshow with fixed axis meaning
# ---------------------------------------------------------------------
def imshow_int_pol(mat, inten_bins, pol_bins, **kw):
    """
    Consistent imshow : matrix[row=intensity, col=polarisation].
    X-axis = ΔDoLP,   Y-axis = Michelson Contrast.
    """
    return plt.imshow(
        mat,
        extent=[pol_bins[0],  pol_bins[-1],
                inten_bins[0], inten_bins[-1]],
        origin='lower',
        aspect='auto',
        **kw
    )

# ---------------------------------------------------------------------
# Main loop over experiments
# ---------------------------------------------------------------------
for curr_ephys_path in curr_ephys_paths:

    exp              = load_ephys_results(glob.glob(curr_ephys_path + '*ALIGNED_BARS_DLP_BG_1')[0],
                                          11, False)
    stimulus_times_dg = exp['stim_times'][:, 0]

    # Stimulus properties ------------------------------------------------
    lum_vals = (exp['lum_vals'][:len(stimulus_times_dg)] + 1) / 2                # Michelson
    pol_vals = exp['pol_vals'][:len(stimulus_times_dg)]
    pol_power      = np.round(((pol_vals + 1) / 2) * 255).astype(np.uint8)
    pol_vals_cal   = dolp_bg - dolp_map[pol_power]                               # ΔDoLP (drop)

    # Accumulate for later coverage plot
    all_pol_vals.extend(pol_vals_cal)
    all_lum_vals.extend(lum_vals)

    # Bin edges / centres (kept constant within the dataset loop)
    intensity_bins          = np.linspace(lum_vals.min(),  lum_vals.max(),  num_bins + 1)
    polarization_bins       = np.linspace(pol_vals_cal.min(), pol_vals_cal.max(), num_bins_pol + 1)
    intensity_bin_centers   = 0.5 * (intensity_bins[:-1]   + intensity_bins[1:])
    polarization_bin_centers = 0.5 * (polarization_bins[:-1] + polarization_bins[1:])

    # -----------------------------------------------------------------
    # Iterate over probes / good units
    # -----------------------------------------------------------------
    for probe in range(len(exp['st_mat'])):
        num_units = len(exp['st_mat'][probe])
        #good_units = np.array(exp['unit_quality'][f'probe{probe}']['good_units'])

        for unit in range(num_units): #np.range(exp[good_units:

            curr_times = exp['st_mat'][probe][unit]
            psth, _, aligned_spikes = calc_psth(
                stimulus_times_dg, curr_times, [0, 1],
                trial_samp=10000, bin_width=0.001, plot=0
            )

            # Assign each trial to its 2-D bin
            intensity_idx     = np.digitize(lum_vals, intensity_bins)     - 1
            polarization_idx  = np.digitize(pol_vals_cal, polarization_bins) - 1

            trial_length = 1000 + 1                       # ms
            mean_resp    = np.zeros((num_bins, num_bins_pol, trial_length))

            # ---------------------------------------------------------
            # Build intensity × polarisation matrix of PSTHs
            # ---------------------------------------------------------
            for i in range(num_bins):         # intensity rows
                for j in range(num_bins_pol):     # pol columns
                    trials_ij = np.where((intensity_idx == i) &
                                         (polarization_idx == j))[0]
                    bpp = np.zeros(trial_length, dtype=np.float32)

                    if len(trials_ij):
                        # concatenate spike times across those trials
                        for tr in trials_ij:
                            bpp[(np.array(aligned_spikes[tr]) * 1000).astype(int)] += \
                                1000 / len(trials_ij)    # spikes/s

                    mean_resp[i, j] = gaussian_filter(bpp, 5)

            # Baseline subtraction / peak response matrix
            baseline   = mean_resp[:, :, -100:].mean()
            dev        = np.abs(mean_resp - baseline)
            total_resp = gaussian_filter(dev.max(axis=2),1)

            if total_resp.max() > resp_thresh:
                all_neurons.append(total_resp)

                # -------------- per-cell diagnostic heat-map ----------
                if plot:
                    plt.figure()
                    imshow_int_pol(total_resp, intensity_bin_centers,
                                   polarization_bin_centers, cmap='plasma')
                    plt.xlabel('ΔDoLP (drop)')
                    plt.ylabel('Michelson Contrast')
                    plt.colorbar(label='Spike Rate (Hz)')
                    plt.title(f'Probe {probe}  Unit {unit}')
                    plt.savefig(exp['analysis_path'] + f'/probe_{probe}_unit_{unit}_sum.pdf',
                                dpi=300, bbox_inches='tight')
                    plt.close()

# ---------------------------------------------------------------------
# Population average
# ---------------------------------------------------------------------
all_neurons      = np.array(all_neurons)
all_neurons_norm = (all_neurons - all_neurons.min((1, 2), keepdims=True)) / \
                   (np.maximum(all_neurons.max((1, 2), keepdims=True) -
                               all_neurons.min((1, 2), keepdims=True), 1))

avg_response_norm = all_neurons_norm.mean(axis=0)
sem_response_norm = all_neurons_norm.std(axis=0) / np.sqrt(all_neurons_norm.shape[0])

# ----- Heat-map of population mean -----------------------------------
plt.figure(figsize=(8, 6))
imshow_int_pol(avg_response_norm, intensity_bin_centers, polarization_bin_centers,
               cmap='jet')
plt.xlabel('ΔDoLP (drop)')
plt.ylabel('Michelson Contrast')
plt.title('Average Normalized Response Across Neurons')
cbar = plt.colorbar()
cbar.set_label('Normalized Response (0–1)')
plt.tight_layout()
plt.savefig('average_response_heatmap.pdf', bbox_inches='tight')
plt.show()

# ----- Intensity tuning at each DoLP level ---------------------------
fig, ax = plt.subplots(figsize=(4,3))

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import bone_r

# 1. Define discrete colormap and boundaries
n_bins = len(polarization_bin_centers)
cmap = bone_r
colors = [cmap(i / (n_bins - 1)) for i in range(n_bins)]
discrete_cmap = ListedColormap(colors)

# Compute boundaries halfway between bin centers for sharp transitions
boundaries = np.concatenate([
    [polarization_bin_centers[0] - (polarization_bin_centers[1] - polarization_bin_centers[0]) / 2],
    (polarization_bin_centers[:-1] + polarization_bin_centers[1:]) / 2,
    [polarization_bin_centers[-1] + (polarization_bin_centers[-1] - polarization_bin_centers[-2]) / 2]
])

norm = BoundaryNorm(boundaries, n_bins)

# 2. Plot using this discrete colormap
for idx, pol_val in enumerate(polarization_bin_centers):
    color      = discrete_cmap(idx)
    mean_vals  = avg_response_norm[:, idx]
    sem_vals   = sem_response_norm[:, idx]

    ax.plot(intensity_bin_centers, mean_vals, marker='o', color=color)
    ax.fill_between(intensity_bin_centers,
                    mean_vals - sem_vals, mean_vals + sem_vals,
                    color=color, alpha=0.25)

# 3. Discrete colorbar
sm = cm.ScalarMappable(norm=norm, cmap=discrete_cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, ticks=polarization_bin_centers)
cbar.set_label('ΔDoLP (drop)')
cbar.ax.set_yticklabels([f"{val:.2f}" for val in polarization_bin_centers])


ax.set_xlabel('Michelson Contrast')
ax.set_ylabel('Normalized Spike Rate (0–1)')
ax.set_title('Intensity Tuning Curves at Different Polarisation Levels')
#ax.set_xticks(intensity_bin_centers) 
fig.tight_layout()
fig.savefig('average_response_lineplot_colormap_sem.pdf',
            dpi=300, bbox_inches='tight')
plt.show()
plt.savefig('total_lineplot.pdf', dpi=300)

# ----- Stimulus coverage scatter -------------------------------------
plt.figure()
plt.scatter(all_lum_vals, all_pol_vals, s=2)
plt.xlabel('Stimulus Intensity (Michelson Contrast)')
plt.ylabel('Stimulus Polarization (ΔDoLP)')
plt.title('Stimulus Coverage')
plt.tight_layout()
plt.savefig('stimulus_coverage.pdf', dpi=300)
plt.show()


fig, ax = plt.subplots(figsize=(3,3))

cmap = plt.cm.plasma
norm = mcolors.Normalize(vmin=intensity_bin_centers.min(),
                         vmax=intensity_bin_centers.max())

for idx, int_val in enumerate(intensity_bin_centers):
    color      = cmap(norm(int_val))
    mean_vals  = avg_response_norm[idx, :]      # varying DoLP at fixed intensity
    sem_vals   = sem_response_norm[idx, :]

    ax.plot(polarization_bin_centers, mean_vals, marker='o', color=color)
    ax.fill_between(polarization_bin_centers,
                    mean_vals - sem_vals, mean_vals + sem_vals,
                    color=color, alpha=0.25)

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Michelson Contrast (intensity)')

ax.set_xlabel('ΔDoLP (drop)')
ax.set_ylabel('Normalized Spike Rate (0–1)')
ax.set_title('Polarisation Tuning Curves at Different Intensity Levels')
fig.tight_layout()
fig.savefig('polarisation_tuning_by_intensity_sem.pdf',
            dpi=300, bbox_inches='tight')
plt.show()



#%%
"""
Consensus-Leiden clustering of neuronal response matrices
---------------------------------------------------------
• Builds a k-NN graph in high-dimensional z-scored space
• Runs Leiden N times with different random seeds
• Converts the N partitions into a consensus (co-association) matrix
• Runs Leiden once more on that consensus graph  →  final labels
• Produces:
    1. UMAP scatter (coloured by consensus clusters + cluster IDs)
    2. Mean ΔDoLP × Contrast map for every cluster
"""

# ──────────────────────────  LIBRARIES  ─────────────────────────────
import numpy as np, random
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
import igraph                 # python-igraph
import leidenalg              # pip install leidenalg
import umap                   # visualisation only

# ───────────────────────────  SETTINGS  ─────────────────────────────
RNG_SEED        = 1        # reproducibility of the *whole* pipeline
N_CONS_RUNS     = 1       # how many Leiden runs for consensus
HD_K_NEIGHBOURS = 25       # neighbours for high-D k-NN graph
CONS_THRESHOLD  = 0.5      # keep edges seen in ≥ 50 % of runs
UMAP_NEIGHBORS = 25
# ───────────────────────────  DATA IN  ──────────────────────────────
# ---- You already have these objects in memory ---------------------
# all_neurons            : (n_neurons, R1, R2)  raw (or norm’d) responses
# all_neurons_norm       : same shape, 0–1 normalised
# intensity_bin_centers  : 1-D (R1,)   Michelson contrast centres
# polarization_bin_centers : 1-D (R2,) ΔDoLP centres
# imshow_int_pol(mat, inten_bins, pol_bins, **kw) -> helper to plot
# -------------------------------------------------------------------

# ───────────────────  1. PRE-PROCESS & GLOBAL SEEDS  ────────────────
np.random.seed(RNG_SEED)
random.seed(RNG_SEED)


n_neurons, R1, R2 = all_neurons.shape
flat   = all_neurons_norm.reshape(n_neurons, R1 * R2)
flat_z = (flat - flat.mean(0)) / flat.std(0)

knn_pc = kneighbors_graph(
    flat_z,
    n_neighbors=HD_K_NEIGHBOURS,
    mode='connectivity',
    include_self=False
)

src, trg = knn_pc.nonzero()
g_pc = igraph.Graph(
    n=n_neurons,
    edges=list(zip(src, trg)),
    directed=False
)
g_pc.es['weight'] = [1.0] * g_pc.ecount()


# ───────────────────  3. MULTIPLE LEIDEN RUNS  ──────────────────────
all_labels = np.empty((N_CONS_RUNS, n_neurons), dtype=int)

for run in range(N_CONS_RUNS):
    part = leidenalg.find_partition(
        g_pc,
        leidenalg.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=1.0,
        seed=RNG_SEED                       # different seed each run
    )
    all_labels[run] = part.membership
print(f"Finished {N_CONS_RUNS} Leiden runs")

# ───────────────────  4. CONSENSUS MATRIX  ─────────────────────────
# co-association frequency matrix  (symmetric, float in [0,1])
cons = np.zeros((n_neurons, n_neurons), dtype=np.float32)
for lab in all_labels:
    same = (lab[:, None] == lab[None, :])     # boolean matrix
    cons += same
cons /= N_CONS_RUNS

# Threshold weak links to keep graph sparse / faster
cons[cons < CONS_THRESHOLD] = 0.0
np.fill_diagonal(cons, 0)                     # no self-loops

g_cons = igraph.Graph.Weighted_Adjacency(
    cons.tolist(), mode='UNDIRECTED', loops=False
)

# ───────────────────  5. FINAL CONSENSUS LEIDEN  ───────────────────
cons_part = leidenalg.find_partition(
    g_cons,
    leidenalg.RBConfigurationVertexPartition,
    weights='weight',
    resolution_parameter=1,
    seed=RNG_SEED
)
labels  = np.array(cons_part.membership)
n_clust = labels.max() + 1
print(f"Consensus Leiden produced {n_clust} clusters")

# ───────────────────  6. UMAP (visualisation only)  ─────────────────
reducer = umap.UMAP(
    n_neighbors=UMAP_NEIGHBORS, min_dist=0, metric='euclidean',
    n_components=2, random_state=RNG_SEED
)
embedding = reducer.fit_transform(flat_z)

# ─────────────  FIGURE 1: UMAP + consensus cluster IDs  ────────────
plt.figure(figsize=(5, 5))
plt.scatter(embedding[:, 0], embedding[:, 1],
            c=labels, cmap='tab20', s=50, alpha=0.8, edgecolor='none')

for cid in range(n_clust):
    idx      = labels == cid
    x_c, y_c = embedding[idx, 0].mean(), embedding[idx, 1].mean()
    plt.text(x_c, y_c, str(cid), ha='center', va='center',
             fontsize=8, fontweight='bold',
             color='black',
             bbox=dict(boxstyle='round,pad=0.2',
                       fc='white', alpha=0.7, lw=0))

plt.xlabel('UMAP-1');  plt.ylabel('UMAP-2')
plt.title('UMAP (consensus Leiden clusters)')
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()
plt.savefig('umap.pdf', dpi=300)
plt.show()

# ─── FIGURE 2: MEAN ΔDoLP × CONTRAST MAP FOR EACH CONSENSUS CLUSTER ─
import numpy as np
import matplotlib.pyplot as plt


fig_rows = int(np.ceil(n_clust / 4))
fig_cols = min(n_clust, 4)

plt.figure(figsize=(3 * fig_cols, 3 * fig_rows))
for cid in range(n_clust):
    idx   = labels == cid
    mean_ = all_neurons_norm[idx].mean(axis=0)          # (R1, R2)

    ax = plt.subplot(fig_rows, fig_cols, cid + 1)
    im = imshow_int_pol(mean_.T, polarization_bin_centers,
                        intensity_bin_centers,
                         cmap='bone_r',vmin=0.1, vmax=0.8)

    ax.set_yticks(polarization_bin_centers)
    ax.set_yticklabels(np.round(polarization_bin_centers, 2),
                       rotation=45, ha='right')
    ax.set_xticks(intensity_bin_centers)
    ax.set_xticklabels(np.round(intensity_bin_centers, 2))

    ax.set_xlabel('Michelson Contrast')
    if cid % fig_cols == 0:
        ax.set_ylabel('ΔDoLP (drop)')

    ax.set_title(f'Cluster {cid} (n={idx.sum()})', fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle('Mean response per consensus Leiden cluster', y=1.02)
plt.tight_layout()
plt.savefig('cluster_means.pdf', dpi=300)
plt.show()



# ---------------------------------------------------------------------
# 
# ---------------------------------------------------------------------
n_cols   = 6          # how many neurons to show *per cluster* (i.e. # columns)
rng_seed = 0          # set to None for a new random draw each run
# ---------------------------------------------------------------------

rng       = np.random.default_rng(rng_seed)
fig_rows  = n_clust            # one row per cluster
fig_cols  = n_cols             # fixed-width “cluster × neuron” matrix

plt.figure(figsize=(3 * fig_cols, 3 * fig_rows))

for cid in range(n_clust):
    # -----------------------------------------------------------------
    # pick up to n_cols random neurons from this cluster
    # -----------------------------------------------------------------
    cl_idx   = np.where(labels == cid)[0]
    n_neu    = cl_idx.size
    sel_num  = min(n_neu, n_cols)
    picked   = rng.choice(cl_idx, size=sel_num, replace=False) if sel_num else []

    for j in range(n_cols):
        ax = plt.subplot(fig_rows, fig_cols, cid * fig_cols + j + 1)

        if j < sel_num:
            nid   = picked[j]
            resp  = all_neurons_norm[nid]            # shape: (R1, R2)

            im = imshow_int_pol(resp.T,
                                polarization_bin_centers,
                                intensity_bin_centers,
                                cmap='bone_r',
                                vmin=0, vmax=1)

            # tick labels
            ax.set_yticks([])
            ax.set_xticks([])

        else:
            # if the cluster contains fewer than n_cols neurons, leave blank
            ax.axis('off')

plt.suptitle('Random neurons per consensus Leiden cluster', y=1.02)
plt.tight_layout()
plt.savefig('cluster_random_neurons.pdf', dpi=300)
plt.show()


#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leak-free additive linear model for 10 × 10 intensity × polarisation maps
=======================================================================

*  The 10 × 10 matrices were **Gaussian-blurred (σ = 1 bin)**, so adjacent bins
   are not independent.  Ordinary K-fold CV therefore “leaks”.
*  We **fit once on the whole map** for the point estimates γ, α, β.
*  Uncertainty is quantified with a **block bootstrap that resamples complete
   rows and columns** (with replacement).  Because every resample keeps the
   row/column structure intact, dependencies created by the blur are preserved.
*  95 % confidence intervals and two-sided bootstrap *p*-values are reported,
   followed by FDR correction.

"""

# ---------------------------------------------------------------------
#  SETTINGS
# ---------------------------------------------------------------------
halo        = 1                          # one-bin halo ⇒ 3×3 blocks
block_size  = 2*halo + 1                 # here: 3
B           = 1000                       # bootstrap replicates
rng         = np.random.default_rng(0)

# ---------------------------------------------------------------------
#  1. build list of non-overlapping 3×3 blocks (tiles)
# ---------------------------------------------------------------------
tiles = []                               # list of index arrays
for r0 in range(0, R1, block_size):      # row start of block
    for c0 in range(0, R2, block_size):  # col start of block
        rows = np.arange(r0, min(r0 + block_size, R1))
        cols = np.arange(c0, min(c0 + block_size, R2))
        # linear indices of that tile in the flattened 10×10 map
        idx  = np.array([r*R2 + c for r in rows for c in cols], dtype=int)
        tiles.append(idx)

n_tiles = len(tiles)                     # here: ceil(10/3)² = 16

# ---------------------------------------------------------------------
#  2. point estimates on the full map  (unchanged)
# ---------------------------------------------------------------------
coef_hat      = np.linalg.lstsq(X_full, Y.T, rcond=None)[0].T
gamma, alpha, beta = coef_hat.T

# coefficient of determination for each neuron (fit vs map)
y_hat   = X_full @ coef_hat.T           # (100, neur)  -> transpose later
ss_tot  = ((Y - Y.mean(axis=1, keepdims=True))**2).sum(axis=1)
ss_res  = ((Y - y_hat.T)**2).sum(axis=1)
r2_neur = 1.0 - ss_res / ss_tot         # shape (n_neurons,)
r2_mean = r2_neur.mean()
r2_sem  = r2_neur.std(ddof=1) / np.sqrt(n_neurons)   # ← NEW

# ---------------------------------------------------------------------
#  3. tile bootstrap  (resample tiles with replacement)
# ---------------------------------------------------------------------
coef_bs = np.empty((B, n_neurons, 3), float)

for b in range(B):
    # pick 16 tiles with replacement
    chosen = rng.integers(0, n_tiles, size=n_tiles)
    rows   = np.unique(np.hstack([tiles[t] for t in chosen]))
    X_b    = X_full[rows]
    Y_b    = Y[:, rows]
    coef_bs[b] = np.linalg.lstsq(X_b, Y_b.T, rcond=None)[0].T

# ---------------------------------------------------------------------
#  4. 95 % CIs and FDR-corrected significance
# ---------------------------------------------------------------------
ci_lo  = np.percentile(coef_bs,  2.5, axis=0)
ci_hi  = np.percentile(coef_bs, 97.5, axis=0)

def two_sided_p(point, samples):
    flips = (np.sign(samples) != np.sign(point[:, None])).sum(axis=1)
    flips += (point == 0)                    # handle exact zeros
    return 2 * np.minimum(flips, B - flips) / B

p_alpha = two_sided_p(alpha, coef_bs[..., 1].T)
p_beta  = two_sided_p(beta,  coef_bs[..., 2].T)

rejected, _        = fdrcorrection(np.hstack([p_alpha, p_beta]), alpha=0.05)
sig_alpha          = rejected[:n_neurons]
sig_beta           = rejected[n_neurons:]

# convenience masks
sig_both   =  sig_alpha &  sig_beta
sig_alpha_only =  sig_alpha & ~sig_beta
sig_beta_only  =  sig_beta  & ~sig_alpha
sig_any    =  sig_alpha |  sig_beta

# ---------------------------------------------------------------------
#  5. console summary
# ---------------------------------------------------------------------
print('Additive model  (σ = 1-bin Gaussian blur, tile-bootstrap CI)')
print('─────────────────────────────────────────────────────────────')
print(f'Neurons analysed                : {n_neurons}')
print(f'Median α  (Hz/contrast)         : {np.median(alpha):6.2f}  '
      f'CI [{np.median(ci_lo[:,1]):6.2f} … {np.median(ci_hi[:,1]):6.2f}]')
print(f'Median β  (Hz/ΔDoLP)            : {np.median(beta):6.2f}  '
      f'CI [{np.median(ci_lo[:,2]):6.2f} … {np.median(ci_hi[:,2]):6.2f}]')
print('─────────────────────────────────────────────────────────────')
print(f'Significant α only  : {sig_alpha_only.sum():3d}  '
      f'({sig_alpha_only.mean()*100:5.1f} %)')
print(f'Significant β only  : {sig_beta_only.sum():3d}  '
      f'({sig_beta_only.mean()*100:5.1f} %)')
print(f'Significant both    : {sig_both.sum():3d}  '
      f'({sig_both.mean()*100:5.1f} %)')
print(f'Significant α or β  : {sig_any.sum():3d}  '
      f'({sig_any.mean()*100:5.1f} %)')
print('─────────────────────────────────────────────────────────────')
print(f'Mean R² ± SEM                : {r2_mean:6.3f} ± {r2_sem:6.3f}')
# ---------------- 5. slope–slope plot --------------------------------
if 'labels' in globals() and len(labels) == n_neurons:
    colours = labels
    cmap    = 'tab20'
else:
    colours = np.where(sig_alpha | sig_beta, 1, 0)
    cmap    = 'coolwarm'

plt.figure(figsize=(5, 5))
plt.scatter(alpha, beta, c=colours, cmap=cmap, s=50, alpha=0.85,
            edgecolors='none')
plt.errorbar(alpha, beta,
             xerr=[alpha - ci_lo[:,1], ci_hi[:,1] - alpha],
             yerr=[beta  - ci_lo[:,2], ci_hi[:,2] - beta],
             fmt='none', ecolor='k', elinewidth=.5, alpha=.3)
plt.axhline(0, ls='--', c='k', lw=.7)
plt.axvline(0, ls='--', c='k', lw=.7)
plt.xlabel('α  (intensity slope, Hz · contrast⁻¹)')
plt.ylabel('β  (polarisation slope, Hz · ΔDoLP⁻¹)')
plt.title('Tuning slopes per neuron\n(error bars = 95 % block-bootstrap CI)')
plt.tight_layout()
plt.savefig('alphabeta_bootstrap.pdf', dpi=600)
plt.show()




#%% Generate RFs for LFP from sparse noise experiments


data_folder='/home/sam/Pictures/vision_figs/'
curr_ephys_path='/home/sam/bucket/electrophysiology/squidOL/2024_02_05/'
curr_anatomy_path='/home/sam/bucket/DBS/data/20240221_F66_squid_Neuropixels_record0205'


curr_ephys_paths = ['/home/sam/bucket/electrophysiology/squidOL/2024_02_05/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_02_13/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_05_28/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_06_24/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_06_27/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_09_10/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_10_22/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_11_01/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_11_12/']

for curr_ephys_path in curr_ephys_paths:
    exp = load_ephys_results(sorted(glob.glob(curr_ephys_path + '*SN_LUM_size_5*'))[-1],0, False )
  
    for probe in range(len(exp['st_mat'])):    

        #stimulus stuff
        first_st = exp['stim_times'][0, 0]
        stimulus_times = exp['stim_times'][:, 0]
        stimulus_times_rel = exp['stim_times'][:, 0]-first_st
        X=exp['sn_mat']
        X=X[:len(stimulus_times)]
        
        #load lfp
        tStart = stimulus_times[0] # in seconds
        tEnd = stimulus_times[-1]+1  # in seconds
        lfp_spatial_ds = 1  # take every N channels of lfp
        chanList = np.array(range(0, 384))# list of channels to extract, by index in saved file
        
        convData, d_times, sRate, meta = load_data(exp,probe, 'lfp', chanList, tStart, tEnd)
        lfp_ds, ds_times=lfp_preprocessing(convData,d_times,sRate,359,360,lfp_spatial_ds,1,resamp=False) # out of brain from anatomy, but not used
        lfp_ms1, lfp_times = resample_mat(lfp_ds, ds_times*1000, 1)
        filtered_lfp = butter_bandpass(lfp_ms1, 1, 100, 1000)
        
        chan_diff=np.concatenate(([0],np.mean(np.abs(np.diff(filtered_lfp[:,0:60000],axis=0)),axis=1)))

        xy_list=[]
        # Loop over all possible (X, Y) positions
        for x in range(X.shape[1]):  # Loop over the 14 X positions
            for y in range(X.shape[2]):  # Loop over the 25 Y positions
                trials = np.where((X[:, x, y] == 1) & (np.roll(X[:, x, y], 1) != 1))[0]
                # Exclude the first index if it wraps around due to np.roll
                if trials[0] == 0 and X[0, x, y] == 1:
                    trials = trials[1:]
                xy_list.append(trials)
        
        all_rfs=[]
        for chan in tqdm(range(filtered_lfp.shape[0])):
            all_avg=[]
            for pos in xy_list:
                ev_avg,ev_mat=get_event_trig_avg(filtered_lfp[chan], np.uint32(stimulus_times_rel[pos]*1000),0, 125)
                all_avg.append(ev_avg)
            all_avg=np.array(all_avg)
            #remove nonspecific signal, probably an artifact from out of brain I haven't removed?
            sig_mean=np.mean(all_avg,axis=0)
            filt_avg=all_avg-sig_mean
            all_avg[all_avg>0]=0
            deflection=np.sum(all_avg,axis=1)
            lfp_eta=np.reshape(deflection,(X.shape[1],X.shape[2]))
            all_rfs.append(lfp_eta)
            
            
        #do shuffle for null ETA
        X_shuff = np.random.permutation(X)
        xy_listnull=[]
        # Loop over all possible (X, Y) positions
        for x in range(X_shuff.shape[1]):  # Loop over the 14 X positions
            for y in range(X_shuff.shape[2]):  # Loop over the 25 Y positions
                trials = np.where((X_shuff[:, x, y] == 1) & (np.roll(X_shuff[:, x, y], 1) != 1))[0]
                # Exclude the first index if it wraps around due to np.roll
                if trials[0] == 0 and X_shuff[0, x, y] == 1:
                    trials = trials[1:]
                xy_listnull.append(trials)
        
        all_rfsnull=[]
        for chan in tqdm(range(filtered_lfp.shape[0])):
            all_avgnull=[]
            for pos in xy_listnull:
                ev_avgnull,ev_mat=get_event_trig_avg(filtered_lfp[chan], np.uint32(stimulus_times_rel[pos]*1000),0, 125)
                all_avgnull.append(ev_avgnull)
            all_avgnull=np.array(all_avgnull)
            #remove nonspecific signal, probably an artifact from out of brain I haven't removed?
            sig_mean=np.mean(all_avgnull,axis=0)
            filt_avg=all_avgnull-sig_mean
            all_avgnull[all_avgnull>0]=0
            deflection=np.sum(all_avgnull,axis=1)
            lfp_eta=np.reshape(deflection,(X_shuff.shape[1],X_shuff.shape[2]))
            all_rfsnull.append(lfp_eta)
            
        np.save(exp['analysis_path'] + '/probe_' + str(probe) + 'lfp_rf.npy',np.array(all_rfs))
        np.save(exp['analysis_path'] + '/probe_' + str(probe) + 'lfp_rfnull.npy',np.array(all_rfsnull))
        np.save(exp['analysis_path'] + '/probe_' + str(probe) + 'chan_diff.npy',np.array(chan_diff))




#%% calculate receptive fields for LFP

curr_ephys_paths = ['/home/sam/bucket/electrophysiology/squidOL/2024_02_05/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_02_13/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_05_28/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_06_24/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_06_27/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_09_10/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_10_22/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_11_01/',
                    '/home/sam/bucket/electrophysiology/squidOL/2024_11_12/']


curr_anatamy_paths = ['/home/sam/bucket/DBS/data/20240221_F66_squid_Neuropixels_record0205',
                      '/home/sam/bucket/DBS/data/20240223_F68_squid_Neuropixels_record0213',
                      '/home/sam/bucket/DBS/data/20240609_F89_NP_recording_0528',
                      '/home/sam/bucket/DBS/data/20240712_F99_squid_NP_recording_0624_TOPRO_3',
                      '/home/sam/bucket/DBS/data/20240716_F100_squid_NP_recording_0627_TOPRO_3',
                      '/home/sam/bucket/DBS/data/20240923_F105_squid_NP_recording_0910_TOPRO_3',
                      '/home/sam/bucket/DBS/data/20241101_F110_squid_kuwa_NP_recording_1022_TOPRO_3',
                      '/home/sam/bucket/DBS/data/20241118_F113_squid_kuwa_NP_recording_1101_TOPRO_3',
                      '/home/sam/bucket/DBS/data/20241125_F115_squid_kuwa_NP_recording_1112_TOPRO_3']

all_depths=[]
all_sizes=[]
all_br=[]
all_spD=[]
all_chanD=[]
all_inds=[]

chan_diff_thresh=3

for index in range(len(curr_ephys_paths)):

    curr_ephys_path=curr_ephys_paths[index]
    curr_anatamy_path=curr_anatamy_paths[index]
    curr_experiment_path = glob.glob(curr_ephys_path + '/' + '*' + 'SN_LUM_size_5*')[0]

    exp = load_ephys_results(curr_experiment_path,
                             0, False, curr_anatamy_path, atlas_10, dt1)

    for probe in range(len(exp['st_mat'])):
        
        if os.path.exists(exp['analysis_path'] + '/probe_' + str(probe) + 'good_lfp_chans.npy'):
            all_rfs=np.load(exp['analysis_path'] + '/probe_' + str(probe) + 'lfp_rf.npy')
            all_rfsnull=np.load(exp['analysis_path'] + '/probe_' + str(probe) + 'lfp_rfnull.npy')
            good_mask=np.load(exp['analysis_path'] + '/probe_' + str(probe) + 'good_lfp_chans.npy')
            chan_diff=np.load(exp['analysis_path'] + '/probe_' + str(probe) + 'chan_diff.npy')[0:len(good_mask)]
            
            standout_chans=np.concatenate(([0],np.diff(chan_diff)))
            good_mask[standout_chans>chan_diff_thresh]=False
            
            if np.sum(good_mask)>0: #at least one good channel
                print('good: ' + curr_ephys_path + '_' + str(probe))
                all_centers=[]
                all_chans=[]
                brs=[]
                sizes=[]
                locs=[]
                for chan in range(len(all_rfs)):
                    rf_blob1, rf_size, rf_loc, filt_eta1, rf_diameter,rf_peakrate,rf_minrate, pos  =    calc_rf(all_rfs[chan],all_rfsnull[chan],peak_thresh=pt,blob_thresh=4,gaus_f=0,med_f=mf,upsample=5,force_polarity='negative')
            
                    all_chans.append(chan)
                    if rf_size>0:
    
                        sizes.append(rf_size)
                        locs.append(rf_loc)
                    else:
                        
                        sizes.append(np.nan)
                        locs.append([np.nan,np.nan])

                curr_depths=exp['elec_depths'][probe][0:len(all_rfs)]
                curr_br=exp['elec_brs'][probe][0:len(all_rfs)]
                 
                good_mask[curr_br!=1]=False
                curr_chans=np.where(good_mask)[0]

                # Filter depths and sizes using the mask
                filtered_depths = np.array(curr_depths)[good_mask]
                filtered_sizes = np.array(sizes)[0:len(curr_depths)][good_mask]
                filtered_locs = np.array(locs)[0:len(curr_depths)][good_mask]
                
               
                all_inds.append(index*np.ones_like(filtered_depths))
                all_depths.append(filtered_depths)
                all_sizes.append(filtered_sizes)
                all_br.append(exp['elec_brs'][probe][0:len(curr_depths)][good_mask])
                
                spatial_dists = pdist(filtered_locs, metric='euclidean')
                spatial_dists_probe = pdist(np10_site_positions()[curr_chans], metric='euclidean')  # 1-D vector

                all_spD.append(spatial_dists)
                all_chanD.append(spatial_dists_probe)

            else:
                print(exp['analysis_path'] + '/probe_' + str(probe))
        
all_depths=np.concatenate(all_depths)+0.0001 #don't want divide by zero
all_sizes=np.concatenate(all_sizes)
all_br=np.concatenate(all_br)
all_spD=np.concatenate(all_spD)
all_chanD=np.concatenate(all_chanD)
all_inds=np.concatenate(all_inds)

   
good_neurons = (all_br < 3) & (~np.isnan(all_sizes))
good_br=all_br[good_neurons]
good_depths=all_depths[good_neurons]
good_sizes=all_sizes[good_neurons]
good_inds=all_inds[good_neurons]

good_sizes_r=2*np.sqrt(good_sizes/np.pi)

plot_smoothed_rf_size_vs_depth(good_depths*10, good_sizes,good_inds, title="Receptive Field Size vs Depth",
                                   bandwidth=50.0, n_eval=100, save_path='lfp_rf_size.pdf')


group_low  = good_sizes[good_depths < 50]
group_high = good_sizes[good_depths > 50]

stat, pval = mannwhitneyu(group_low, group_high,alternative='two-sided')

print(f"Mann–Whitney U test statistic: {stat:.3f}")
print(f"Two-sided p-value: {pval:.4e}")

good_comparisons=(~np.isnan(all_spD))
good_spD=all_spD[good_comparisons]
good_chanD=all_chanD[good_comparisons]

r, p = pearsonr(good_spD, good_chanD)

# Plot
plt.figure(figsize=(3,3))
plt.scatter(good_chanD,good_spD, 5, 'k',alpha=0.7)
plt.xlabel('distance in brain (um)')
plt.ylabel('distance on screen (visual degree)')
plt.title(f'Correlation: r = {r:.2f}, p = {p:.3f}')
plt.tight_layout()
plt.show()
plt.savefig('retinotopy_lfp.pdf',dpi=600)

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1)  BIN-BY-DISTANCE IN BRAIN -------------------------------------
# ------------------------------------------------------------------
NUM_BINS = 20                         # change if needed
bins        = np.linspace(good_chanD.min(),
                          good_chanD.max(),
                          NUM_BINS + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_idx     = np.digitize(good_chanD, bins) - 1          # 0-based indices

mean_sp = np.full(NUM_BINS, np.nan)
std_sp  = np.full(NUM_BINS, np.nan)

for i in range(NUM_BINS):
    vals = good_spD[bin_idx == i]
    if vals.size:                     # skip empty bins
        mean_sp[i] = vals.mean()
        std_sp[i]  = np.std(vals, ddof=1)                # sample SD

# ------------------------------------------------------------------
# 2)  PLOT  — histogram with shaded error bars
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4, 3))

# bars ── very light fill, no edge so whiskers stand out
bar_w = bins[1] - bins[0]
ax.bar(bin_centers, mean_sp,
       width=bar_w, color='skyblue', alpha=0.6, edgecolor='none',
       label='Binned mean')

# slim vertical whiskers + small caps
mask = ~np.isnan(mean_sp)           # skip empty bins
ax.errorbar(bin_centers[mask], mean_sp[mask], yerr=std_sp[mask],
            fmt='none',                     # no marker
            ecolor='midnightblue',
            elinewidth=1.2,
            capsize=3, capthick=1.2,
            zorder=3)

ax.set_xlabel('distance in brain (µm)')
ax.set_ylabel('distance on screen (visual °)')
ax.set_title(f'Correlation: r = {r:.2f}, p = {p:.3f}')
ax.legend(frameon=False)
plt.tight_layout()

# save and show
plt.savefig('retinotopy_lfp.pdf', dpi=600)
plt.show()


#%% #Single dataset example figs from 2_13. I load things up as above and just plot intermediate steps

curr_ephys_paths = ['/home/sam/bucket/electrophysiology/squidOL/2024_02_13/']
curr_anatamy_paths = ['/home/sam/bucket/DBS/data/20240223_F68_squid_Neuropixels_record0213']

all_depths=[]
all_sizes=[]
all_br=[]

chan_diff_thresh=3
probe=1

for index in range(len(curr_ephys_paths)):

    curr_ephys_path=curr_ephys_paths[index]
    curr_anatamy_path=curr_anatamy_paths[index]
    curr_experiment_path = glob.glob(curr_ephys_path + '/' + '*' + 'SN_LUM_size_5*')[0]

    exp = load_ephys_results(curr_experiment_path,
                             0, False, curr_anatamy_path, atlas_10, dt1)

    if os.path.exists(exp['analysis_path'] + '/probe_' + str(probe) + 'good_lfp_chans.npy'):
        all_rfs=np.load(exp['analysis_path'] + '/probe_' + str(probe) + 'lfp_rf.npy')
        all_rfsnull=np.load(exp['analysis_path'] + '/probe_' + str(probe) + 'lfp_rfnull.npy')
        good_mask=np.load(exp['analysis_path'] + '/probe_' + str(probe) + 'good_lfp_chans.npy')
        chan_diff=np.load(exp['analysis_path'] + '/probe_' + str(probe) + 'chan_diff.npy')[0:len(good_mask)]
        
        standout_chans=np.concatenate(([0],np.diff(chan_diff)))
        good_mask[standout_chans>chan_diff_thresh]=False

      
        
        if np.sum(good_mask)>0: #at least one good channel
            print('good: ' + curr_ephys_path + '_' + str(probe))
            all_centers=[]
            all_chans=[]
            brs=[]
            sizes=[]
            for chan in range(len(all_rfs)):
                rf_blob1, rf_size, rf_loc, filt_eta1, rf_diameter,rf_peakrate,rf_minrate, pos  =    calc_rf(np.fliplr(all_rfs[chan]),all_rfsnull[chan],peak_thresh=pt,blob_thresh=4,gaus_f=0,med_f=mf,upsample=5,force_polarity='negative')
        
                all_chans.append(chan)
                if rf_size>0:
                    all_centers.append(rf_loc)
                    sizes.append(rf_size)
                else:
                    all_centers.append((np.nan,np.nan))
                    sizes.append(np.nan)

            curr_depths=exp['elec_depths'][probe][0:len(all_rfs)]
            curr_pos=exp['elec_positions'][probe][0:len(all_rfs)]
            curr_br=exp['elec_brs'][probe][0:len(all_rfs)]
            good_mask[curr_br!=1]=False

            # Filter depths and sizes using the mask
            filtered_depths = np.array(curr_depths)[good_mask]
            filtered_pos = np.array(curr_pos)[good_mask]
            filtered_sizes = np.array(sizes)[0:len(curr_depths)][good_mask]
            filtered_chans =np.array(all_chans)[0:len(curr_depths)][good_mask]
            filtered_centers = np.array(all_centers)[0:len(curr_depths)][good_mask]
            filtered_d=2*np.sqrt(filtered_sizes/np.pi)
            
            plot_smoothed_rf_size_vs_depth(filtered_depths*10, filtered_sizes,exp['elec_brs'][probe][0:len(curr_depths)][good_mask], title="Receptive Field Size vs Depth 2_13",
                                               bandwidth=50.0, n_eval=100, save_path='2_13_lfp_rf_size.pdf')


            plot_smoothed_rf_size_vs_depth(filtered_chans, filtered_sizes,exp['elec_brs'][probe][0:len(curr_depths)][good_mask], title="Receptive Field Size vs channels 2_13",                                                                  bandwidth=50.0, n_eval=100, save_path='2_13_lfp_rf_size_chan.pdf')

            plt.figure()
            plt.scatter(filtered_chans,filtered_centers[:,1]-50)
            ax.set_title('ev_triggered_avg')
            plt.show()
            plt.savefig('lfp_retinotopy_azimuth.pdf',dpi=600)

            plt.figure()
            plt.scatter(filtered_chans,30-filtered_centers[:,0])
            ax.set_title('ev_triggered_avg')
            plt.show()
            plt.savefig('lfp_retinotopy_elevation.pdf',dpi=600)
                              
            #stimulus stuff
            first_st = exp['stim_times'][0, 0]
            stimulus_times = exp['stim_times'][:, 0]
            stimulus_times_rel = exp['stim_times'][:, 0]-first_st
            X=exp['sn_mat']
            X=X[:len(stimulus_times)]
            
            #load lfp
            tStart = stimulus_times[0] # in seconds
            tEnd = stimulus_times[-1]+1  # in seconds
            lfp_spatial_ds = 1  # take every N channels of lfp
            chanList = np.array(range(0, 384))# list of channels to extract, by index in saved file
            
            convData, d_times, sRate, meta = load_data(exp,probe, 'lfp', chanList, tStart, tEnd)
            lfp_ds, ds_times=lfp_preprocessing(convData,d_times,sRate,359,360,lfp_spatial_ds,1,resamp=False) # out of brain from anatomy, but not used
            lfp_ms1, lfp_times = resample_mat(lfp_ds, ds_times*1000, 1)
            filtered_lfp = butter_bandpass(lfp_ms1, 1, 100, 1000)
            
                        
            xy_list=[]
            # Loop over all possible (X, Y) positions
            for x in range(X.shape[1]):  # Loop over the 14 X positions
                for y in range(X.shape[2]):  # Loop over the 25 Y positions
                    trials = np.where((X[:, x, y] == 1) & (np.roll(X[:, x, y], 1) != 1))[0]
                    # Exclude the first index if it wraps around due to np.roll
                    if trials[0] == 0 and X[0, x, y] == 1:
                        trials = trials[1:]
                    xy_list.append(trials)
            
            all_rfs=[]
            for chan in tqdm(range(filtered_lfp.shape[0])):
                all_avg=[]
                for pos in xy_list:
                    ev_avg,ev_mat=get_event_trig_avg(filtered_lfp[chan], np.uint32(stimulus_times_rel[pos]*1000),0, 125)
                    all_avg.append(ev_avg)
                all_avg=np.array(all_avg)
                #remove nonspecific signal, probably an artifact from out of brain I haven't removed?
                sig_mean=np.mean(all_avg,axis=0)
                filt_avg=all_avg-sig_mean
                all_avg[all_avg>0]=0
                deflection=np.sum(all_avg,axis=1)
                lfp_eta=np.reshape(deflection,(X.shape[1],X.shape[2]))
                all_rfs.append(lfp_eta)
                
                
            #do shuffle for null ETA
            X_shuff = np.random.permutation(X)
            xy_listnull=[]
            # Loop over all possible (X, Y) positions
            for x in range(X_shuff.shape[1]):  # Loop over the 14 X positions
                for y in range(X_shuff.shape[2]):  # Loop over the 25 Y positions
                    trials = np.where((X_shuff[:, x, y] == 1) & (np.roll(X_shuff[:, x, y], 1) != 1))[0]
                    # Exclude the first index if it wraps around due to np.roll
                    if trials[0] == 0 and X_shuff[0, x, y] == 1:
                        trials = trials[1:]
                    xy_listnull.append(trials)
            
            all_rfsnull=[]
            for chan in tqdm(range(filtered_lfp.shape[0])):
                all_avgnull=[]
                for pos in xy_listnull:
                    ev_avgnull,ev_mat=get_event_trig_avg(filtered_lfp[chan], np.uint32(stimulus_times_rel[pos]*1000),0, 125)
                    all_avgnull.append(ev_avgnull)
                all_avgnull=np.array(all_avgnull)
                #remove nonspecific signal, probably an artifact from out of brain I haven't removed?
                sig_mean=np.mean(all_avgnull,axis=0)
                filt_avg=all_avgnull-sig_mean
                all_avgnull[all_avgnull>0]=0
                deflection=np.sum(all_avgnull,axis=1)
                lfp_eta=np.reshape(deflection,(X_shuff.shape[1],X_shuff.shape[2]))
                all_rfsnull.append(lfp_eta)
                


data_folder='/home/sam/OL_toplogy/ephys'
#figure for example LFP
chan=190
all_avg=[]
for pos in xy_list:
    ev_avg,ev_mat=get_event_trig_avg(filtered_lfp[chan], np.uint32(stimulus_times_rel[pos]*1000),0, 300)
    all_avg.append(ev_avg)
all_avg=np.array(all_avg)
sig_mean=np.mean(all_avg,axis=0)
filt_avg=all_avg-sig_mean
all_avg[all_avg>0]=0
deflection=np.sum(all_avg,axis=1)
lfp_eta=np.reshape(deflection,(X.shape[1],X.shape[2]))


fig, ax = plt.subplots(figsize=(3,3))
cax = ax.imshow(filt_avg, aspect='auto', interpolation='nearest', cmap='bone_r', vmin=-50, vmax=50)
ax.set_title('ev_triggered_avg')
fig.colorbar(cax, ax=ax, label='Event Value')
plt.show()
plt.savefig(data_folder +'/lfp_eta.pdf',dpi=600)

fig, ax = plt.subplots(figsize=(3,3))
cax = ax.plot(filt_avg[119])
ax.set_title('ev_triggered_avg')
plt.show()
plt.savefig(data_folder +'/lfp_eta_single.pdf',dpi=600)


rf_blob1, rf_size, rf_loc, filt_eta1, rf_diameter,rf_peakrate,rf_minrate, pos  =    calc_rf(all_rfs[chan],all_rfsnull[chan],peak_thresh=pt,blob_thresh=4,gaus_f=0,med_f=mf,upsample=5,force_polarity='negative')
 

vmin, vmax = np.percentile(-filt_eta1, [1, 99])

fig, ax = plt.subplots(figsize=(3, 3))
cax = ax.imshow(
    -filt_eta1,
    aspect="equal",
    interpolation="nearest",
    vmin=vmin,
    vmax=vmax,
)
ax.set_title("ev_triggered_avg")
fig.colorbar(cax, ax=ax, label="Event Value")

plt.savefig(data_folder + "/lfp_eta_screen.pdf", dpi=600)
plt.show()

fig, ax = plt.subplots(figsize=(3,3))
cax = ax.imshow(rf_blob1, aspect='equal', interpolation='nearest',vmin=0)
ax.set_title('ev_triggered_avg')
fig.colorbar(cax, ax=ax, label='Event Value')
plt.show()
plt.savefig(data_folder +'/lfp_rf.pdf',dpi=600)

#figure for example 2
chan=270
all_avg=[]
for pos in xy_list:
    ev_avg,ev_mat=get_event_trig_avg(filtered_lfp[chan], np.uint32(stimulus_times_rel[pos]*1000),0, 300)
    all_avg.append(ev_avg)
all_avg=np.array(all_avg)
sig_mean=np.mean(all_avg,axis=0)
filt_avg=all_avg-sig_mean
all_avg[all_avg>0]=0
deflection=np.sum(all_avg,axis=1)
lfp_eta=np.reshape(deflection,(X.shape[1],X.shape[2]))


fig, ax = plt.subplots(figsize=(3,3))
cax = ax.imshow(filt_avg, aspect='auto', interpolation='nearest', cmap='bone_r', vmin=-50, vmax=50)
ax.set_title('ev_triggered_avg')
fig.colorbar(cax, ax=ax, label='Event Value')
plt.show()
plt.savefig(data_folder +'/lfp_eta2.pdf',dpi=600)

rf_blob1, rf_size, rf_loc, filt_eta1, rf_diameter,rf_peakrate,rf_minrate, pos  =    calc_rf(all_rfs[chan],all_rfsnull[chan],peak_thresh=pt,blob_thresh=4,gaus_f=0,med_f=mf,upsample=5,force_polarity='negative')
 

vmin, vmax = np.percentile(-filt_eta1, [1, 99])

fig, ax = plt.subplots(figsize=(3, 3))
cax = ax.imshow(
    -filt_eta1,
    aspect="equal",
    interpolation="nearest",
    vmin=vmin,
    vmax=vmax,
)
ax.set_title("ev_triggered_avg")
fig.colorbar(cax, ax=ax, label="Event Value")

plt.savefig(data_folder + "/lfp_eta_screen2.pdf", dpi=600)
plt.show()


fig, ax = plt.subplots(figsize=(3,3))
cax = ax.imshow(rf_blob1, aspect='equal', interpolation='nearest',vmin=0)
ax.set_title('ev_triggered_avg')
fig.colorbar(cax, ax=ax, label='Event Value')
plt.show()
plt.savefig(data_folder +'/lfp_rf2.pdf',dpi=600)


