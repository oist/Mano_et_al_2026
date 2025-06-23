#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:33:20 2024

@author: sam
"""
import numpy as np
import rawpy
import imageio
import matplotlib.pyplot as plt
import glob
import pandas as pd
import cv2
import os
import re
import h5py
from scipy.ndimage import label, center_of_mass


# ---------------------------------------------------------------------
# Utility: circular mean of AoP (0–180°) over a mask
# ---------------------------------------------------------------------
def circular_mean_aop_deg(aop_img_deg, mask):
    """
    Return mean AoP in degrees (0–180) using the doubled-angle method.
    """
    rad = np.deg2rad(aop_img_deg[mask] * 2.0)           # 0–360° → 0–2π
    sin_m, cos_m = np.sin(rad).mean(), np.cos(rad).mean()
    mean_rad = 0.5 * np.arctan2(sin_m, cos_m)           # back to 0–π
    mean_deg = np.mod(np.rad2deg(mean_rad), 180.0)
    return mean_deg

    
def extract_green(raw_image):
    raw     = raw_image.raw_image_visible.astype(np.uint16)
    pattern = raw_image.raw_pattern              # 2×2 indices
    cdesc   = raw_image.color_desc.decode()      # e.g. 'RGBG'

    # Indices in the pattern whose colour is Green
    g_indices = [i for i, c in enumerate(cdesc) if c == 'G']
    if len(g_indices) != 2:
        raise RuntimeError("Expected exactly two green indices, got {}".format(g_indices))

    # Collect the two half‑resolution green planes
    g_planes = []
    for idx in g_indices:
        y_off, x_off = np.argwhere(pattern == idx)[0]   # the offset within 2×2 tile
        g_planes.append(raw[y_off::2, x_off::2].astype(np.float32))

    # Average
    green_avg = ((g_planes[0] + g_planes[1]) * 0.5).astype(np.uint16)  # H//2 × W//2

    
    return green_avg.astype(np.uint16)



def preprocess_green_for_dolp(raw, flat_field=None):
    """
    Extract and preprocess the green channel for DoLP analysis.

    Steps:
    1. Extract average green from Bayer pattern (H/2 × W/2).
    2. Subtract black level.
    3. Normalize by exposure time and ISO.
    4. (Optional) Flat-field correction.

    Parameters
    ----------
    raw : rawpy.RawPy
        Opened raw image.
    flat_field : np.ndarray, optional
        Normalized flat-field map (same shape as output green image).
        If provided, the image will be divided by this.

    Returns
    -------
    green_processed : np.ndarray
        Preprocessed green image as float32, shape (H//2, W//2).
    """

    # Step 1: Extract average green channel
    raw_data = raw.raw_image_visible.astype(np.uint16)
    pattern = raw.raw_pattern
    cdesc = raw.color_desc.decode()

    # Get green pixel indices
    g_indices = [i for i, c in enumerate(cdesc) if c == 'G']
    if len(g_indices) != 2:
        raise RuntimeError("Expected exactly two green indices, got {}".format(g_indices))

    # Extract green planes
    g_planes = []
    for idx in g_indices:
        y_off, x_off = np.argwhere(pattern == idx)[0]
        g_planes.append(raw_data[y_off::2, x_off::2].astype(np.float32))

    green = (g_planes[0] + g_planes[1]) * 0.5

    # Step 2: Subtract black level
    black_level = raw.black_level_per_channel[g_indices[0]]
    green -= black_level
    green[green < 0] = 0

    # Step 3: Normalize by exposure time and ISO gain
    # exposure = raw.metadata.exposure              # in seconds
    # iso = raw.metadata.iso_speed     
    # gain = iso / 100.0
    # green /= (exposure * gain)

    # Step 4: Flat-field correction (optional)
    if flat_field is not None:
        if flat_field.shape != green.shape:
            raise ValueError("Flat-field map shape mismatch")
        green /= flat_field

    return green.astype(np.float32)


def load_orf_extract_green(filepath):
    """
    Load an Olympus .ORF file and extract the green channel image.

    Parameters:
    -----------
    filepath : str
        Full path to the .ORF file.

    Returns:
    --------
    green_image : np.ndarray
        2D image containing green channel pixel values.
    """
    # with rawpy.imread("flat_frame.ORF") as raw_flat:
    #     flat_raw = preprocess_green_for_dolp(raw_flat)
    #     flat_map = cv2.GaussianBlur(flat_raw, (0, 0), 51)
    #     flat_map /= np.mean(flat_map)
    
    with rawpy.imread(filepath) as raw:
        green_image = preprocess_green_for_dolp(raw, flat_field=None)
    return green_image

def save_green_as_png(data_path, hf):
    # Load and extract green image
    green_float32 = load_orf_extract_green(data_path + hf + '.ORF').astype('float32')

    # Normalize to 0–255 and convert to uint8
    green_norm = cv2.normalize(green_float32, None, 0, 255, cv2.NORM_MINMAX)
    green_uint8 = green_norm.astype('uint8')

    # Construct output filename
    out_path = os.path.join(data_path, hf + "_green.png")

    # Save image
    cv2.imwrite(out_path, green_uint8)

    return out_path, green_uint8


def save_image_with_scaling(image, filename, scale_to_uint8=True):
    if scale_to_uint8:
        # Scale the image to the 0-255 range
        image[image>0.5]=0.5
        scaled_img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        # Convert to uint8
        scaled_img = scaled_img.astype(np.uint8)
    else:
        scaled_img = image

    # Save the image as PNG using cv2
    cv2.imwrite(filename, scaled_img)

    
def modify_filename(filename):
    # Pattern to match the last underscore followed by digits
    pattern = r'_*(\d+)$'
    return re.sub(pattern, lambda m: f"{m.group(1)[-2:]}", filename)


def pad_single_digits(name):
    # Ensure z has two digits
    name = re.sub(r'(\D)(\d)(z)', r'\g<1>0\2\3', name)  # Add leading zero if 1-digit z
    # Ensure m has two digits
    name = re.sub(r'(\D)(\d)(m)', r'\g<1>0\2\3', name)  # Add leading zero if 1-digit m
    return name


def align_images_sift(curr_h, curr_v, num_matches_to_draw=50, visualize=True):
    """
    Align `curr_v` image to `curr_h` image using SIFT keypoints and homography.

    Args:
        curr_h (np.ndarray): The reference (horizontal) image.
        curr_v (np.ndarray): The image to align (vertical).
        num_matches_to_draw (int): Number of matches to draw in visualization.
        visualize (bool): Whether to show the matched keypoints.

    Returns:
        aligned_v (np.ndarray): The aligned version of curr_v.
        M (np.ndarray): The estimated homography matrix.
        mask (np.ndarray): The inlier mask returned by RANSAC.
    """
    # Create SIFT detector and compute descriptors
    sift = cv2.SIFT_create()
    kp_h, desc_h = sift.detectAndCompute(curr_h, None)
    kp_v, desc_v = sift.detectAndCompute(curr_v, None)

    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc_h, desc_v)
    matches = sorted(matches, key=lambda x: x.distance)

    # Select top matches
    best_matches = matches[:num_matches_to_draw]

    # Visualization
    if visualize:
        matched_vis = cv2.drawMatches(
            curr_h, kp_h,
            curr_v, kp_v,
            best_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        plt.figure()
        plt.title(f'Top {num_matches_to_draw} SIFT Matches')
        plt.imshow(matched_vis, cmap='gray')
        plt.axis('off')
        plt.show()

    # Compute homography with RANSAC
    src_pts = np.float32([kp_h[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_v[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # Warp image
    h, w = curr_h.shape[:2]
    aligned_v = cv2.warpPerspective(curr_v, M, (w, h))

    return aligned_v, M, mask

#%% Get DoLP and intensity data from raw images

#need to shift the images, click on center of the square in all images, align vert/horiz to that

dataset_paths=['/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250328-back-forth/',
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250328-back-forth/Reverse/',
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250404_temp/',
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250404_temp/Reverse/',
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250408/',
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250417/',
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250430/dive1/series1/',
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250430/dive1/series1/reverse/',
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250501-depth-cloudy/series1/',
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250501-depth-cloudy/series1/reverse/',
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250502-depth-sunny/',
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250503-shimp-on-transect/',
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250503-shimp-on-transect/reverse/' ]


dataf=pd.read_csv(glob.glob(data_path + '*data.csv')[0])

dataf[['date', 'z', 'm']] = dataf['name_tag'].str.extract(r'(\d{8})-(\d+)z(\d+)m')

# Group by 'date' and 'zm' to operate over X values for matching z and m
grouped_files = dataf.groupby(['date', 'z', 'm']).agg({
    'name_tag': list,
    'pfilter_position': list
}).reset_index()

all_dists=[]
all_int=[]
    
for files, pols in zip(grouped_files['name_tag'],
                       grouped_files['pfilter_position']):

    # Convert to NumPy for easy indexing
    files = np.asarray(files)
    pols  = np.asarray(pols)


 # locate filenames by polarizer position
    horiz_idx = np.where(pols == "horizontal")[0]
    vert_idx  = np.where(pols == "vertical")[0]
    diag_idx  = np.where(pols == "45-deg")[0]       # may be empty

    # ensure at least H & V are present
    if len(horiz_idx) != len(vert_idx):
        print(f"❌ skipping group (H {len(horiz_idx)} ≠ V {len(vert_idx)})")
        continue

    # iterate over matched indices
    for i in range(len(horiz_idx)):
        hf = modify_filename(files[horiz_idx[i]])
        vf = modify_filename(files[vert_idx[i]])
        df = (
            modify_filename(files[diag_idx[i]])
            if i < len(diag_idx)
            else None
        )

        # depth & distance tags
        m = re.search(r"(\d{2})z(\d{2})m", hf)
        curr_depth, curr_dist = map(int, m.groups())

        # ── load H & V
        curr_h = load_orf_extract_green(f"{data_path}{hf}.ORF").astype("float32")
        curr_v = load_orf_extract_green(f"{data_path}{vf}.ORF").astype("float32")

        # ── optional 45-deg image
        has_diag = df is not None
        if has_diag:
            curr_d = load_orf_extract_green(f"{data_path}{df}.ORF").astype("float32")
        else:
            curr_d = np.zeros_like(curr_h)          # placeholder for stacking

        # ── align V (always) and D (only if it exists) to H
        # load center-of-mass points
        curr_pt1 = np.load(f"{data_path}{hf}_green_point.npy")
        curr_pt2 = np.load(f"{data_path}{vf}_green_point.npy")

        y0, x0 = curr_pt1
        y1, x1 = curr_pt2
        dx, dy = int(y0 - y1), int(x0 - x1)
        rows, cols = curr_v.shape
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned_v = cv2.warpAffine(curr_v, M, (cols, rows))

        if has_diag:
            curr_pt3 = np.load(f"{data_path}{df}_green_point.npy")
            y2, x2 = curr_pt3
            dx, dy = int(y0 - y2), int(x0 - x2)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            aligned_d = cv2.warpAffine(curr_d, M, (cols, rows))
        else:
            aligned_d = curr_d                       # still zeros

        # ── Stokes parameters
        epsilon = 1e-8
        S0 = curr_h + aligned_v
        S1 = curr_h - aligned_v
        if has_diag:
            S2 = 2 * aligned_d - S0
        else:
            S2 = np.zeros_like(S0)

        # DoLP & AoP
        dolp_img = np.sqrt(S1**2 + S2**2) / (S0 + epsilon)

        if has_diag:
            aop_img = 0.5 * np.degrees(np.arctan2(S2, S1)) % 180
        else:
            # two-point → AoP can only be 0 or 90 °
            aop_img = np.where(S1 >= 0, 0, 90).astype("float32")

        int_img = S0 / 2      # mean intensity

        # ── save multi-layer raw stack
        all_imgs = np.stack(
            (dolp_img, int_img, aop_img, curr_h, aligned_v, aligned_d)
        )
        np.save(f"{data_path}{hf}_raw.npy", all_imgs)

        # ── save DoLP PNG for QA
        save_image_with_scaling(
            dolp_img,
            f"{data_path}{hf}_dolp.png",
            scale_to_uint8=True,
        )

        # ── accumulate summary metrics
        if has_diag:
            all_int.extend([curr_h.mean(), aligned_v.mean(), aligned_d.mean()])
            all_dists.extend([curr_dist] * 3)
        else:
            all_int.extend([curr_h.mean(), aligned_v.mean()])
            all_dists.extend([curr_dist] * 2)

# convert summaries to NumPy for later analysis
all_int  = np.asarray(all_int)
all_dists = np.asarray(all_dists)
print("✅ processing finished →", len(all_int), "intensity points")


    
#%% Calculate information of interest from field data
    
dataset_paths=['/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250501-depth-cloudy/series1/',
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250501-depth-cloudy/series1/reverse/',
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250502-depth-sunny/',
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250328-back-forth/',
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250328-back-forth/Reverse/',
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250404_temp/',
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250404_temp/Reverse/',
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250408/',
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250417/',
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250430/dive1/series1/',
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250430/dive1/series1/reverse/',
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250501-depth-cloudy/series1/',
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250501-depth-cloudy/series1/reverse/',
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250502-depth-sunny/',
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250503-shimp-on-transect/',
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250503-shimp-on-transect/reverse/' ,
               '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250325_temp_folder/']

    
dataset_paths= ['/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250502-depth-sunny/']



for data_path in dataset_paths:
        
    dataf=pd.read_csv(glob.glob(data_path + '*data.csv')[0])
    dataf[['date', 'z', 'm']] = dataf['name_tag'].str.extract(r'(\d{8})-(\d+)z(\d+)m')
    # Group by 'date' and 'zm' to operate over X values for matching z and m
    grouped_files = dataf.groupby(['date', 'z', 'm']).agg({
        'name_tag': list,
        'pfilter_position': list
    }).reset_index()
    # after extracting forground and background roi
    
   # ---------------------------------------------------------------------
    # NPY files -----------------------------------------------------------
    # ---------------------------------------------------------------------
    all_npy_files = glob.glob(os.path.join(data_path, '*raw.npy'))
    npy_files = [f for f in all_npy_files if 'point' not in os.path.basename(f)]
    
    # ---------------------------------------------------------------------
    # Containers
    # ---------------------------------------------------------------------
    depth, dist              = [], []
    contrast, int_contrast   = [], []
    dolp_fg, dolp_bg         = [], []
    aop_fg_deg, aop_bg_deg   = [], []
    int_fg, int_bg           = [], []
    h_fg, h_bg               = [], []
    v_fg, v_bg               = [], []
    
    PD_list                  = []          # <<< NEW
    
    # ---------------------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------------------
    
    for file in npy_files:
    
        # -- meta data -----------------------------------------------------
        z_digits, m_digits = re.search(r'(\d{2})z(\d{2})m', file).groups()
        curr_depth, curr_dist = int(z_digits), int(m_digits)
    
        all_imgs  = np.load(file)
        dolp_img  = all_imgs[0]       # DoLP
        int_img   = all_imgs[1]       # Intensity
        aop_img   = all_imgs[2]       # AoP (deg 0–180)
        horiz_img = all_imgs[3]       # H
        vert_img  = all_imgs[4]       # V
        diag_img  = all_imgs[5]
        # -- filename base -------------------------------------------------
        base = os.path.splitext(os.path.basename(file))[0].rsplit('_', 1)[0]
        base = base.split('_')[0]
    
        # -- masks ---------------------------------------------------------
        fg = cv2.imread(sorted(glob.glob(data_path + base[:-2] + '*fg_image_raw.png'))[0],
                        0) > 0
        
        if len(sorted(glob.glob(data_path + base[:-2] + '*bg_image_raw.png')))>0:
            bg = cv2.imread(sorted(glob.glob(data_path + base[:-2] + '*bg_image_raw.png'))[0],
                       0) > 0
        else:
            bg = cv2.imread(sorted(glob.glob(data_path + '*bg_image_raw.png'))[0],0) > 0
    
        # -- contrasts -----------------------------------------------------
        contrast.append((dolp_img[fg].mean() - dolp_img[bg].mean()) /
                        dolp_img[bg].mean())
        int_contrast.append((int_img[fg].mean() - int_img[bg].mean()) /
                            int_img[bg].mean())
    
        # -- DoLP & Int means ---------------------------------------------
        dolp_fg.append(dolp_img[fg].mean())
        dolp_bg.append(dolp_img[bg].mean())
        int_fg.append(int_img[fg].mean())
        int_bg.append(int_img[bg].mean())
    
        # -- H & V means ---------------------------------------------------
        hf = horiz_img[fg].mean()
        vf = vert_img[fg].mean()
        hb = horiz_img[bg].mean()
        vb = vert_img[bg].mean()
    
        h_fg.append(hf); v_fg.append(vf)
        h_bg.append(hb); v_bg.append(vb)
    
        # -- AoP circular means -------------------------------------------
        aop_fg_deg.append(circular_mean_aop_deg(aop_img, fg))
        aop_bg_deg.append(circular_mean_aop_deg(aop_img, bg))
    
        # -- Polarisation distance (Sp form) ------------------------------  <<< NEW
        PD_val, theta_opt = max_pd_phys(horiz_img, diag_img, vert_img,
                                        fg, bg,
                                        Sp=10.0, theta_step_deg=1)
        PD_list.append(float(PD_val))   # convert from 0-D ndarray to scalar
    
        # -- bookkeeping ---------------------------------------------------
        depth.append(curr_depth)
        dist.append(curr_dist)
    
    # ---------------------------------------------------------------------
    # Convert to NumPy arrays
    # ---------------------------------------------------------------------
    depth        = np.asarray(depth)
    dist         = np.asarray(dist)
    contrast     = np.asarray(contrast)
    int_contrast = np.asarray(int_contrast)
    dolp_fg      = np.asarray(dolp_fg)
    dolp_bg      = np.asarray(dolp_bg)
    aop_fg_deg   = np.asarray(aop_fg_deg)
    aop_bg_deg   = np.asarray(aop_bg_deg)
    int_fg       = np.asarray(int_fg)
    int_bg       = np.asarray(int_bg)
    PD_data      = np.asarray(PD_list)            # <<< NEW
    
    # ---------------------------------------------------------------------
    # Save to HDF5
    # ---------------------------------------------------------------------
    summary_path = os.path.join(data_path,
                                f"{dataf['name_tag'][0]}_data_summary.h5")
    
    with h5py.File(summary_path, "w") as h5f:
        h5f.create_dataset("depth",          data=depth)
        h5f.create_dataset("dist",           data=dist)
        h5f.create_dataset("pol_contrast",   data=contrast)
        h5f.create_dataset("int_contrast",   data=int_contrast)
        h5f.create_dataset("dolp_fg",        data=dolp_fg)
        h5f.create_dataset("dolp_bg",        data=dolp_bg)
        h5f.create_dataset("aop_fg_deg",     data=aop_fg_deg)
        h5f.create_dataset("aop_bg_deg",     data=aop_bg_deg)
        h5f.create_dataset("h_fg",           data=np.asarray(h_fg))
        h5f.create_dataset("v_fg",           data=np.asarray(v_fg))
        h5f.create_dataset("h_bg",           data=np.asarray(h_bg))
        

#%% DoLP background (EDF 13a)

import os, glob, h5py
import numpy as np
import matplotlib.pyplot as plt

# ── Dataset folders ───────────────────────────────────────────────
dataset_paths = [
    '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250501-depth-cloudy/series1/',
    '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250501-depth-cloudy/series1/reverse/',
    '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250502-depth-sunny/',
    '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250328-back-forth/',
    '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250328-back-forth/Reverse/',
    '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250404_temp/',
    '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250404_temp/Reverse/',
    '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250408/',
    '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250417/',
    '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250430/dive1/series1/',
    '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250430/dive1/series1/reverse/',
    '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250503-shimp-on-transect/',
    '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250325_temp_folder/'
]



# ── Gather DoLP_BG plus labels ───────────────────────────────────
means, sds, labels = [], [], []

for path in dataset_paths:
    try:
        h5_file = glob.glob(os.path.join(path, '*_data_summary.h5'))[0]
    except IndexError:
        print(f"⚠️  Skipped (no summary file): {path}")
        continue

    with h5py.File(h5_file, 'r') as h5f:
        dolp_bg = np.asarray(h5f['dolp_bg'])

    means.append(dolp_bg.mean())
    sds.append(dolp_bg.std(ddof=1))          # sample SD

    base   = os.path.basename(os.path.normpath(path))
    parent = os.path.basename(os.path.dirname(os.path.normpath(path)))
    label  = f"{parent}/{base}" if base.lower() in {'series1', 'reverse'} else base
    labels.append(label)

# ── Plot mean ± SD ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(8, len(labels)*0.8), 5))

xpos = np.arange(1, len(means)+1)

# vertical SD bars
for x, m, sd in zip(xpos, means, sds):
    ax.plot([x, x], [m - sd, m + sd], color='black', linewidth=1, zorder=1)

# mean points (larger dot, in front)
ax.scatter(xpos, means, s=60, color='black', zorder=2)

ax.set_xticks(xpos)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel('DoLP_BG')
ax.set_title('Mean ± SD of background DoLP per dataset')
ax.grid(axis='y', linestyle=':', alpha=0.4)

plt.tight_layout()
plt.savefig('dolp_bg.pdf', dpi=600)
plt.show()

#%% dolp image of white standard (EDF 13b)



file='/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250503-shimp-on-transect/20250503-06z03m02_raw.npy'
all_imgs = np.load(file)
dolp_img=all_imgs[0]
int_img=all_imgs[1]


dolp_crop = dolp_img[400:900,500:1500]
int_crop  = int_img[400:900,500:1500]

#dolp_crop = dolp_img[530:800,750:1250]
#int_crop  = int_img[530:800,750:1250]

# Suppose you have a known background region or background value
background_value_dolp = np.mean(dolp_img[500:700:,500:700])  # or a region index
background_value_int  = np.mean(int_img[500:700:,500:700])   # likewise

# Compute Weber contrast: (I - I_background) / I_background
delta_DoLP = (dolp_crop - background_value_dolp) 
weber_int  = (int_crop  - background_value_int)  / background_value_int
weber_int[np.abs(weber_int)<0.118]=0

mask = (delta_DoLP > -0.059) & (delta_DoLP < 0.177)
delta_DoLP[mask] = 0


fig, ax = plt.subplots(3, 1, figsize=(9, 4))

# ── 1) intensity ────────────────────────────────────────────────────
im0 = ax[0].imshow(int_crop , cmap='gray', vmin=0, vmax=600)
ax[0].set_title('intensity')
fig.colorbar(im0, ax=ax[0], orientation='horizontal',
             shrink=0.8, pad=0.08)

# ── 2) Weber contrast (auto-scaled) ─────────────────────────────────
wc_vmax = np.percentile(np.abs(weber_int), 99)

im1 = ax[1].imshow(np.abs(weber_int), cmap='plasma',
                   vmin=0, vmax=wc_vmax)
ax[1].set_title('weber contrast')
fig.colorbar(im1, ax=ax[1], orientation='horizontal',
             shrink=0.8, pad=0.08)

# ── 3) ΔDoLP (auto-scaled) ─────────────────────────────────────────
dd_vmax = np.percentile(np.abs(delta_DoLP), 99)

im2 = ax[2].imshow(np.abs(delta_DoLP), cmap='plasma',
                   vmin=0, vmax=dd_vmax)
ax[2].set_title('delta dolp')
fig.colorbar(im2, ax=ax[2], orientation='horizontal',
             shrink=0.8, pad=0.08)


for a in ax:
    a.set_xticks([]); a.set_yticks([])

plt.tight_layout()
plt.savefig('dolp_int_card.pdf',dpi=600)
plt.show()






#%% Extended Data block of field data (EDF 13c-l)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined contrast-decay plotting plus sighting-distance summary.

Adds:
• Saves decay rates k_I (Weber) and k_D (ΔDoLP) alongside sighting distances
• Caps sighting distances at 25 m, converts NaNs → 0
• Computes per-row mean of the two sighting distances
• Writes table to ‘sighting_distance_summary.csv’
"""

import math, glob, os, h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  scipy.optimize import curve_fit

# ───────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────

def fit_exp(x, y, p0=(0.3, 0.1)):
    popt, _ = curve_fit(lambda d, A, k: A * np.exp(-k * d),
                        x, y, p0=p0, maxfev=10000)
    return popt  # A, k

def mean_by_dist(d, v):
    u = np.unique(d)
    return u, np.array([v[d == ui].mean() for ui in u])

mm2in = lambda mm: mm / 25.4

# ───────────────────────────────────────────────────────────────
# Parameters and paths
# ───────────────────────────────────────────────────────────────
dataset_paths = [
    '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250325_temp_folder/',
    '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250328-back-forth/',
    '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250328-back-forth/Reverse/',
    '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250404_temp/',
    '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250404_temp/Reverse/',
    '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250408/',
    '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250417/',
    '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250430/dive1/series1/',
    '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250430/dive1/series1/reverse/',
    '/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250503-shimp-on-transect/'
]
T_I, T_D   = 0.118, 0.059
CAP_MAX    = 25.0
OUT_CSV    = 'sighting_distance_summary.csv'

# ───────────────────────────────────────────────────────────────
# Figure & grid
# ───────────────────────────────────────────────────────────────
n_cols, n_rows = 3, 4
fig = plt.figure(figsize=(mm2in(180), mm2in(100)))
gs  = fig.add_gridspec(n_rows, n_cols, wspace=0.40, hspace=0.38)

rows_out = []

# ───────────────────────────────────────────────────────────────
# Plot each dataset
# ───────────────────────────────────────────────────────────────
all_bg=[]
for idx, path in enumerate(dataset_paths):
    r, c = divmod(idx, n_cols)
    ax   = fig.add_subplot(gs[r, c])

    # Use folder name (or its parent) as label
    label = (os.path.basename(os.path.normpath(path)) or
             os.path.basename(os.path.dirname(os.path.normpath(path))))

    h5f = glob.glob(os.path.join(path, '*_data_summary.h5'))[0]
    with h5py.File(h5f) as h5:
        dist       = np.asarray(h5['dist'])
        int_contr  = np.asarray(h5['int_contrast'])
        d_fg, d_bg = np.asarray(h5['dolp_fg']), np.asarray(h5['dolp_bg'])
        a_fg, a_bg = np.asarray(h5['aop_fg_deg']), np.asarray(h5['aop_bg_deg'])

    ΔDoLP = np.abs(d_fg - d_bg)
    all_bg.append(d_bg)
    # Fits
    A_I, k_I = fit_exp(dist, np.abs(int_contr))
    A_D, k_D = fit_exp(dist, ΔDoLP)

    # Plot fitted curves
    d_axis = np.linspace(0, 25, 501)
    ax.plot(d_axis, (A_I * np.exp(-k_I * d_axis)) / T_I,
            lw=1, color='k', label='intensity')
    ax.plot(d_axis, (A_D * np.exp(-k_D * d_axis)) / T_D,
            lw=1, color='tab:blue', label='|ΔDoLP|')

    # Plot means
    u_d, m_I = mean_by_dist(dist, np.abs(int_contr))
    _,  m_D  = mean_by_dist(dist, ΔDoLP)
    ax.scatter(u_d, m_I / T_I, s=28, color='k',        edgecolor='w', zorder=3)
    ax.scatter(u_d, m_D / T_D, s=28, color='tab:blue', edgecolor='w', zorder=3)

    # Formatting
    ax.axhline(1, ls='--', lw=1, color='grey', alpha=0.6)
    if r == n_rows - 1:
        ax.set_xlabel('distance (m)')
    ax.legend(frameon=False, fontsize=6, loc='upper right')

    # Sighting-distance calculation
    d_I = math.log(A_I / T_I) / k_I if A_I > T_I and k_I > 0 else np.nan
    d_D = math.log(A_D / T_D) / k_D if A_D > T_D and k_D > 0 else np.nan

    # Store results
    rows_out.append({
        'Dataset'                        : label,
        'Sighting_Distance_Weber(m)'     : d_I,
        'Sighting_Distance_ΔDoLP(m)'     : d_D,
        'k_Weber(1/m)'                   : k_I,
        'k_ΔDoLP(1/m)'                   : k_D
    })

# ───────────────────────────────────────────────────────────────
# Assemble summary table
# ───────────────────────────────────────────────────────────────
sighting_df = pd.DataFrame(rows_out)

# Cap sighting distances and handle NaNs
numeric_dist_cols = ['Sighting_Distance_Weber(m)', 'Sighting_Distance_ΔDoLP(m)']
sighting_df[numeric_dist_cols] = sighting_df[numeric_dist_cols].clip(upper=CAP_MAX)
sighting_df.fillna(0, inplace=True)

# Row-wise mean of the (capped) sighting distances
numeric_dist_cols = ['k_Weber(1/m)', 'k_ΔDoLP(1/m)']
sighting_df['Mean_Sighting_Distance(m)'] = sighting_df[numeric_dist_cols].mean(axis=0)

# ───────────────────────────────────────────────────────────────
# Output
# ───────────────────────────────────────────────────────────────
print("\nSighting-distance summary (NaNs→0, values>25→25)\n" + "-"*55)
print(sighting_df.to_string(index=False))

sighting_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved ➜ {os.path.abspath(OUT_CSV)}")

# ───────────────────────────────────────────────────────────────
# Save figure
# ───────────────────────────────────────────────────────────────
fig.tight_layout()
fig.savefig('gio_combined_2x5.pdf', dpi=600, bbox_inches='tight')
plt.show()

# ───────────────────────────────────────────────────────────────
# CSV summary table
# ───────────────────────────────────────────────────────────────
df = pd.DataFrame(rows_out)
df.to_csv('sighting_distances.csv', index=False)
print('\nSaved → gio_combined_2x5.pdf  and  sighting_distances.csv')



#%% DoLP shrimp image (Fig. 5j)

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from pathlib import Path


tau    = 0.059                       # DoLP JND (fraction, 0.05 = 5 %)
omega  = np.deg2rad(5.0)            # AoP JND (radians, 2° default)
crop   = (slice(560, 710), slice(760, 1190))   # crop for display maps
bkg_sl = (slice(300, 600), slice(700, 1200))   # background sampling


data_path='/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250503-shimp-on-transect/raws/shrimp/extras/'


curr_h = load_orf_extract_green(data_path + 'P5033002.ORF').astype('float32')
curr_v = load_orf_extract_green(data_path + 'P5033003.ORF').astype('float32')
curr_d = load_orf_extract_green(data_path + 'P5033004.ORF').astype('float32')

_,scaled_h=save_green_as_png(data_path, 'P5033002')
_,scaled_v=save_green_as_png(data_path, 'P5033003')
_,scaled_d=save_green_as_png(data_path, 'P5033004')


aligned_v_tmp, Mv, mask = align_images_sift(scaled_h, scaled_v, num_matches_to_draw=50, visualize=False)
aligned_v = cv2.warpPerspective(curr_v, Mv, (curr_h.shape[1], curr_h.shape[0]))

aligned_d_tmp, Md, mask = align_images_sift(scaled_h, scaled_d, num_matches_to_draw=50, visualize=False)
aligned_d = cv2.warpPerspective(curr_d, Md, (curr_h.shape[1], curr_h.shape[0]))


epsilon = 1e-8
S0 = curr_h + aligned_v
S1 = curr_h - aligned_v
S2 = 2 * aligned_d - curr_h - aligned_v

DoLP = np.sqrt(S1**2 + S2**2) / (S0 + epsilon)
AoP  = 0.5 * np.arctan2(S2, S1)             # radians, −π/2 … π/2

# Intensity image (mean of the two analyser channels)
int_img = S0 / 2


bkg_int  = np.mean(int_img[bkg_sl])
bkg_dolp = np.mean(DoLP[bkg_sl])
bkg_aop  = np.mean(AoP[bkg_sl])


delta_DoLP = DoLP - bkg_dolp

mask = (delta_DoLP > -0.059) & (delta_DoLP < 0.177)
delta_DoLP[mask] = 0

delta_AoP  = np.abs(AoP - bkg_aop)
delta_AoP  = np.where(delta_AoP > np.pi/2, np.pi - delta_AoP, delta_AoP)  # wrap 0–90°

Dbar = 0.5 * (DoLP + bkg_dolp)           # mean DoLP term
PD_sq = (delta_DoLP / tau) ** 2 + (2 * Dbar * np.sin(delta_AoP) / omega) ** 2
PD    = np.sqrt(PD_sq)

weber_int = (int_img  - bkg_int)  / bkg_int
weber_int[np.abs(weber_int)<0.118]=0
shrimp_mask=cv2.imread('/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250503-shimp-on-transect/raws/shrimp/extras/mask_image.jpg',0)
shrimp_mask=shrimp_mask>0

ddolp_vals     = delta_DoLP[shrimp_mask]

print(f"Mean polarisation distance (shrimp vs bkg): delta_dolp = {ddolp_vals.mean():.2f}")

# ---------------------------------------------------------------------
fig, ax = plt.subplots(3, 1, figsize=(9, 4))

# ── 1) intensity ────────────────────────────────────────────────────
im0 = ax[0].imshow(int_img[crop], cmap='gray', vmin=200, vmax=1200)
ax[0].set_title('intensity')
fig.colorbar(im0, ax=ax[0], orientation='horizontal',
             shrink=0.8, pad=0.08)

# ── 2) Weber contrast (auto-scaled) ─────────────────────────────────
wc_vmin = np.percentile(np.abs(weber_int[crop]), 1)
wc_vmax = np.percentile(np.abs(weber_int[crop]), 99)

im1 = ax[1].imshow(np.abs(weber_int[crop]), cmap='plasma',
                   vmin=0, vmax=wc_vmax)
ax[1].set_title('weber contrast')
fig.colorbar(im1, ax=ax[1], orientation='horizontal',
             shrink=0.8, pad=0.08)

# ── 3) ΔDoLP (auto-scaled) ─────────────────────────────────────────
dd_vmin = np.percentile(np.abs(delta_DoLP[crop]), 1)
dd_vmax = np.percentile(np.abs(delta_DoLP[crop]), 99)

im2 = ax[2].imshow(np.abs(delta_DoLP[crop]), cmap='plasma',
                   vmin=0, vmax=dd_vmax)
ax[2].set_title('delta dolp')
fig.colorbar(im2, ax=ax[2], orientation='horizontal',
             shrink=0.8, pad=0.08)


for a in ax:
    a.set_xticks([]); a.set_yticks([])

plt.tight_layout()
plt.savefig('shrimp_dolp_map.pdf', dpi=600)
plt.show()


#%% Fig 5i

import math
from pathlib import Path
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from  scipy.optimize import curve_fit

# ───────────────────────────────────────────────────────────────
# 1)  Load CSV  (edit path if needed)
csv_path = Path("/home/sam/OL_toplogy/field/2025.06.16_transect_06_OIST.csv")
df = pd.read_csv(csv_path)

# ───────────────────────────────────────────────────────────────
# 2)  Exponential model  a·exp(–b·d) + c
def exp_model(d, a, b, c):
    return a * np.exp(-b * d) + c

# ───────────────────────────────────────────────────────────────
# 3)  Style dict for G1 / G3 / G6 markers & lines
style = {
    "G1": dict(markerfacecolor="white", markeredgecolor="black",
               markeredgewidth=1.2,   markersize=6,
               linecolor="black",     linestyle="--"),
    "G3": dict(markerfacecolor="0.5",  markeredgecolor="0.5",
               markeredgewidth=1.2,   markersize=6,
               linecolor="0.5",       linestyle="-"),
    "G6": dict(markerfacecolor="black", markeredgecolor="black",
               markeredgewidth=1.2,   markersize=6,
               linecolor="black",     linestyle="-."),
}

# ───────────────────────────────────────────────────────────────
# 4)  Helper: plot one subset and collect sighting distances
def plot_group(group_df, *, title, thr_mag):
    """
    group_df : DataFrame to plot
    title    : figure title
    thr_mag  : detection-threshold magnitude (positive value)
    Returns  : list of dict rows  {metric, sighting_distance}
    """
    plt.figure(figsize=(4, 3))

    plt.axhline(thr_mag, color="red", linestyle=":", linewidth=1)

    rows = []   # accumulate table data

    for metric, sub in group_df.groupby("metric"):
        suffix = metric.split("_")[-1]
        st     = style.get(suffix, style["G3"])   # default grey

        x = sub["distance"].to_numpy()
        y = np.abs(sub["value"].to_numpy())        # ← absolute value

        # ── scatter points ─────────────────────────────────────
        plt.scatter(
            x, y,
            facecolors=st["markerfacecolor"],
            edgecolors=st["markeredgecolor"],
            linewidths=st["markeredgewidth"],
            s=st["markersize"]**2,
            label=metric
        )

        # ── exponential fit ───────────────────────────────────
        x_sorted = np.argsort(x)
        x_fitdata, y_fitdata = x[x_sorted], y[x_sorted]

        a0, b0, c0 = y_fitdata[0] - y_fitdata[-1], 0.1, y_fitdata[-1]
        sighting = math.nan      # default if fit fails

        try:
            popt, _ = curve_fit(exp_model, x_fitdata, y_fitdata,
                                p0=(a0, b0, c0), maxfev=10000)
            a, b, c = popt

            # ── EXTRAPOLATE (include d = 0) ──────────────────
            x_line = np.linspace(0.0, x.max(), 400)
            plt.plot(
                x_line,
                exp_model(x_line, *popt),
                color=st["linecolor"],
                linestyle=st["linestyle"],
                linewidth=1.5
            )

            # ── Sighting distance (fit = thr_mag) ───────────
            if a > 0 and b > 0 and (a + c) > thr_mag >= c:
                sighting = -math.log((thr_mag - c) / a) / b

        except RuntimeError:
            print(f"[warn] fit failed for {metric}; plotted points only.")

        rows.append(dict(metric=metric,
                         sighting_distance=round(sighting, 3)))

    plt.xlabel("Distance")
    plt.ylabel("|Value|")
    plt.title(title)
    plt.legend(fontsize="x-small", ncol=1)
    plt.tight_layout()

    return rows

# ───────────────────────────────────────────────────────────────
# 5)  Plot Weber and non-Weber groups, gather table rows
is_weber = df["metric"].str.startswith("weber")

table_rows  = plot_group(
    df[is_weber],
    title="Weber metrics (|value|)",
    thr_mag=0.118
)

plt.show()
plt.savefig("weber.pdf", dpi=600)   # active figure saved (last one)

table_rows += plot_group(
    df[~is_weber],
    title="Other metrics (|value|)",
    thr_mag=0.059
)

plt.show()
plt.savefig("dolp.pdf", dpi=600)   # active figure saved (last one)

# ───────────────────────────────────────────────────────────────
# 6)  Assemble, print, save table
table_df = (pd.DataFrame(table_rows)
            .sort_values("metric")
            .reset_index(drop=True))

print("\n──────── Sighting distance (threshold crossing) ────────")
print(table_df.to_string(index=False))
print("─────────────────────────────────────────────────────────")

out_csv = csv_path.with_name("sighting_distances_per_metric.csv")
table_df.to_csv(out_csv, index=False)
print(f"Table saved → {out_csv}")





