#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 08:30:39 2024

@author: sam
"""
from matplotlib.colors import ListedColormap
from numpy.polynomial.polynomial import Polynomial
import sys
sys.path.append('./utils')
sys.path.append('./ephys')
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb
from skimage.filters import threshold_local
import sys
import matplotlib.colors as mcolors
from skimage.morphology import skeletonize_3d
import nrrd
from skimage import measure, morphology
import cv2
from skimage.feature import peak_local_max
import os
from skimage import io
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.patches import Rectangle


def load_nifti(src: str):
    """
    Load nifti image. Array is rearranged so that it aligns with TIFF or HDF5 format.
    Parameters
    ----------
    src : str
    """
    stack = nb.load(src).get_fdata()
    # the below line is needed to make it consistent with TIFF array order
    stack = np.swapaxes(stack, 0, 2)
    return stack


def save_nifti(stack, niftiname, spx, spy, spz):
    """
    Write array as NIfTI format.
    Note: array layout is reordered so that tiff and nifti can be treated equally;
    In tiff, z is 0th index, i.e. to make xy slice,
    slice = stack[ i, :, : ]
    On the other hand, in nifti, z is the 2nd index, i.e. to make xy slice,
    slice = stack[ :, :, i ]
    Parameters
    ----------
    stack : np.ndarray
    niftiname : str
    spx : float
    spy : float
    spz : float
    Returns
    -------
    None
    Examples
    --------
    vol = save_nifti(stack, 'image.nii.gz', 3, 3, 3)
    """
    # swap axis!
    stack = np.swapaxes(stack,0,2)
    nim = nb.Nifti1Image(stack, affine=None)
    # define voxel spacing and orientation
    # ANTS uses qform when reading NIFTI-1 images
    aff = np.diag([-spx,-spy,spz,1]) # sign is due to the inconsistency between nibabel and ANTS (or ITK)
    nim.header.set_qform(aff, code=2)
    # write!
    nim.to_filename(niftiname)
    
    
    
def plot_gradient(grad_x,grad_y,z_val):
     # Compute gradient magnitude and angle
    magnitude = np.sqrt(grad_x[z_val]**2 + grad_y[z_val]**2)
    angle = np.arctan2(grad_y[z_val], grad_x[z_val])

    # Normalize magnitude to be between 0 and 1 for brightness
    normalized_magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())

    # Convert angle to hue (0 to 1 for full color wheel)
    hue = (angle + np.pi) / (2 * np.pi)

    # Create HSV image
    hsv_image = np.zeros(grad_x[z_val].shape + (3,))
    hsv_image[..., 0] = hue  # Hue
    hsv_image[..., 1] = 1  # Full Saturation
    hsv_image[..., 2] = normalized_magnitude  # Value as normalized magnitude

    # Convert HSV to RGB for display
    rgb_image = mcolors.hsv_to_rgb(hsv_image)
    
    return rgb_image

    
def plot_polar_gradient_hsv():
    # Number of points for demonstration
    num_angles = 36
    num_magnitudes = 10

    # Create angle and magnitude arrays
    angles = np.linspace(0, 2 * np.pi, num_angles)
    magnitudes = np.linspace(0, 1, num_magnitudes)

    # Create figure and polar subplot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Loop through angles and magnitudes
    for angle in angles:
        for magnitude in magnitudes:
            # Convert angle to hue (0 to 1 for the full color wheel)
            hue = angle / (2 * np.pi)

            # Use magnitude for brightness (value in HSV)
            color = mcolors.hsv_to_rgb([hue, 1, magnitude])

            # Plot the point
            ax.scatter(angle, magnitude, color=color, s=600*magnitude)

    # Set plot properties
    ax.set_title('Polar Plot: Angle as Hue, Magnitude as Brightness')
    ax.set_rticks([0.5, 1])  # Less radial ticks
    ax.grid(True)

    # Show the plot
    plt.show()



def plot_polar_angles_hue(polar_angles):
    """
    Plot an array of polar angles, mapping each angle to a hue in the HSV color space.

    Parameters:
    - polar_angles: A 2D array of polar angles in degrees, ranging from 0 to 180.
    """
    # Normalize the polar angles to [0, 1] for mapping to hue
    normalized_angles = polar_angles / 180.0

    # Create an HSV image: Hue from normalized polar angles, full saturation, and full value
    hsv_image = np.zeros((*polar_angles.shape, 3))  # Shape is 2D array shape + 3 for HSV channels
    hsv_image[..., 0] = normalized_angles  # Hue based on polar angle
    hsv_image[..., 1] = 1  # Saturation
    hsv_image[..., 2] = 1  # Value/Brightness

    # Convert the HSV image to RGB for plotting
    rgb_image = mcolors.hsv_to_rgb(hsv_image)

    # Plotting the RGB image
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_image)
    plt.title("Polar Angles Mapped to Hue")
    plt.axis('off')  # Hide axes for better visualization
    plt.show()

def logistic_growth(t, K, r):
    P0 = t[0]  # Assuming the first y-value is P0
    return K / (1 + ((K - P0) / P0) * np.exp(-r * t))

def binarize_vol(vol, mask, filt1=21,filt2=61):
    
    thresh=threshold_local(vol,filt1)
    thresh_img1=(vol>thresh)
    thresh=threshold_local(vol,filt2)
    thresh_img2=(vol>thresh)
    
    binary_volume= (np.logical_or(thresh_img1,thresh_img2))
    binary_volume=binary_volume*mask
    
    return binary_volume


def compute_iou_within_mask(volume1, volume2, mask):
    # Identify the indices within the mask
    mask_indices = mask == 1

    # Extract the relevant regions from both volumes
    masked_volume1 = volume1[mask_indices]
    masked_volume2 = volume2[mask_indices]

    # Calculate intersection and union
    intersection = np.logical_and(masked_volume1, masked_volume2).sum()
    union = np.logical_or(masked_volume1, masked_volume2).sum()

    # Compute IoU
    iou = intersection / union if union != 0 else 0  # Avoid division by zero
    return iou


def calculate_nonzero_means(volume, cube_size):
    # Calculate trimmed dimensions to be divisible by cube_size
    trimmed_size = (
        (volume.shape[0] // cube_size[0]) * cube_size[0],
        (volume.shape[1] // cube_size[1]) * cube_size[1],
        (volume.shape[2] // cube_size[2]) * cube_size[2]
    )
    # Trim the volume
    volume = volume[:trimmed_size[0], :trimmed_size[1], :trimmed_size[2]]
    
    # Calculate number of cubes along each dimension
    num_cubes_x = trimmed_size[0] // cube_size[0]
    num_cubes_y = trimmed_size[1] // cube_size[1]
    num_cubes_z = trimmed_size[2] // cube_size[2]
    
    # Array to hold the means of non-zero pixels in each cube
    means = np.zeros((num_cubes_x, num_cubes_y, num_cubes_z))
    
    # Iterate through each cube
    for i in range(num_cubes_x):
        for j in range(num_cubes_y):
            for k in range(num_cubes_z):
                # Extract the cube
                x_start = i * cube_size[0]
                x_end = x_start + cube_size[0]
                y_start = j * cube_size[1]
                y_end = y_start + cube_size[1]
                z_start = k * cube_size[2]
                z_end = z_start + cube_size[2]
                cube = volume[x_start:x_end, y_start:y_end, z_start:z_end]
                
                # Calculate the mean of non-zero elements
                non_zero_elements = cube[cube != 0]
                if non_zero_elements.size > 0:
                    means[i, j, k] = np.mean(non_zero_elements)
                else:
                    means[i, j, k] = 0  # If no non-zero elements, set mean to zero

    return means

def distance_transform_watershed(image,thresh,k=5):
    
    binary_image=np.uint8(image>thresh)
    image_3ch = cv2.cvtColor(binary_image*255, cv2.COLOR_GRAY2BGR)
  
        # noise removal
    kernel = np.ones((k,k),np.uint8)
    opening = cv2.morphologyEx(binary_image,cv2.MORPH_OPEN,kernel, iterations = 2)
     
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
     
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    sure_fg = find_local_maxima(dist_transform, min_distance=20, threshold_abs=0.1, labels=binary_image)
     
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
     
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
     
    # Now, mark the region of unknown with zero
    markers[unknown==1] = 0
    
    markers = cv2.watershed(image_3ch,markers)
    image_3ch[markers == -1] = [255,0,0]

    return image_3ch, markers


def find_local_maxima(image, min_distance=10, threshold_abs=0.1, labels=None):
    coordinates = peak_local_max(image, min_distance=min_distance, threshold_abs=threshold_abs, labels=labels)
    mask = np.zeros_like(image, dtype=bool)
    mask[tuple(coordinates.T)] = True
    return mask

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

alphabet_colors = [
    "#F0A0FF", "#0075DC", "#993F00", "#4C005C", "#191919", "#005C31",
    "#2BCE48", "#FFCC99", "#808080", "#94FFB5", "#8F7C00", "#9DCC00",
    "#C20088", "#003380", "#FFA405", "#FFA8BB", "#426600", "#FF0010",
    "#5EF1F2", "#00998F", "#E0FF66", "#740AFF", "#990000", "#FFFF80",
    "#FFE100", "#FF5005"
]

#%% Good coronal section

sl = 119820

tmp = []
for i in range(4):
    tmp.append(io.imread(f"/home/sam/bucket/DBS/data/20231018_F24_squid_33dph_TO_PRO_3_alpha_tubulin/stitched_692BP/RES(10609x9316x1336)/624550_266910_{sl+25*i}.tif"))
img1 = np.array(tmp).max(axis=0)

tmp = []
for i in range(4):
    tmp.append(io.imread(f"/home/sam/bucket/DBS/data/20231018_F24_squid_33dph_TO_PRO_3_alpha_tubulin/stitched_624BP/RES(10609x9316x1336)/624550_266910_{sl+25*i}.tif"))
img2 = np.array(tmp).max(axis=0)


raw = np.stack((img1, img2))
print(raw.shape)

cropped = raw[:, 800:9000, 2250:8800]
raw2 = []
for i in range(2):
    raw2.append(transform.rotate(cropped[i], -105, resize=True, preserve_range=True))
raw2 = np.array(raw2)
raw2 = raw2[:, 1300:7200, 200:-200]

gamma = 1.5
vmax1, vmin1 = 4000, 500
vmax2, vmin2 = 25000, 500
img = np.power(raw2[:, ::1, ::1], 1/gamma)
nuclear_norm = np.clip((img[0] - np.power(vmin1, 1/gamma)) / (np.power(vmax1, 1/gamma) - np.power(vmin1, 1/gamma)), 0, 1)
tubulin_norm = np.clip((img[1] - np.power(vmin2, 1/gamma)) / (np.power(vmax2, 1/gamma) - np.power(vmin2, 1/gamma)), 0, 1)

cm1 = plt.get_cmap('Blues')
colors = cm1(np.linspace(0, 0.7, 256))
colors[0] = (1,1,1,1)
interted_cm1 = ListedColormap(1 - colors[:, :3])

cm2 = plt.get_cmap('Reds')
colors = cm2(np.linspace(0.0, 0.8, 256))
colors[0] = (1,1,1,1)
inverted_cm2 = ListedColormap(1 - colors[:, :3])

nuclear_rgb = interted_cm1(nuclear_norm)[..., :3]
tubulin_rgb = inverted_cm2(tubulin_norm)[..., :3]

overlay = nuclear_rgb + tubulin_rgb
overlay = np.clip(overlay, 0, 1)
inverted = 1 - overlay

plt.imshow(inverted[500:3000,3000:3500,:])

# Show
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.font_manager as fm
import numpy as np

x, y, w, h = 2370, 250, 800, 2900   # adjust as needed

fig, (ax_main, ax_zoom) = plt.subplots(
    1, 2,
    dpi=500,
    figsize=(4, 2),
    gridspec_kw={'width_ratios': [1, 0.45]}
)
fig.subplots_adjust(wspace=0.02)   # gap between panels

# ------------------------------------------------------------------
# MAIN IMAGE  ───────────────────────────────────────────────────────
# ------------------------------------------------------------------
ax_main.imshow(inverted, cmap='gray')
ax_main.axis('off')

# Rectangle marking zoom region
rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='red', facecolor='none')
ax_main.add_patch(rect)

# Scale‑bar for main image (500 µm)
real_length_um = 500      # printed length
pixel_size_um  = 0.65     # µm per pixel
fontprops      = fm.FontProperties(size=6)

scalebar_main = AnchoredSizeBar(ax_main.transData,
                                real_length_um / pixel_size_um,
                                f'{real_length_um} µm',
                                loc='lower left',
                                pad=0.2,
                                color='black',
                                frameon=False,
                                size_vertical=10,
                                fontproperties=fontprops)
ax_main.add_artist(scalebar_main)

# ------------------------------------------------------------------
# ZOOMED‑IN VIEW  ───────────────────────────────────────────────────
# ------------------------------------------------------------------
zoomed_img = inverted[y:y + h, x:x + w]
ax_zoom.imshow(zoomed_img, cmap='gray')
ax_zoom.axis('off')
ax_zoom.set_aspect('equal')   # fill the narrower panel

# Scale‑bar for zoomed image (100 µm)
real_length_zoom_um = 100
scalebar_zoom = AnchoredSizeBar(ax_zoom.transData,
                                real_length_zoom_um / pixel_size_um,
                                f'{real_length_zoom_um} µm',
                                loc='lower left',
                                pad=0.2,
                                color='black',
                                frameon=False,
                                size_vertical=10,
                                fontproperties=fontprops)
ax_zoom.add_artist(scalebar_zoom)

# ------------------------------------------------------------------
# SAVE & SHOW  ──────────────────────────────────────────────────────
# ------------------------------------------------------------------
plt.savefig("OL_medulla_anatamy.pdf", bbox_inches='tight', transparent=True)
plt.show()




#%% tree branch volume with depth

# turn to a loop over datasets. This calculates island volume as a function of distanct from the iGL, its repeatable


datasets=['/home/sam/bucket/DBS/data/20231208_F47_squid_NPrec_1130/',
          '/home/sam/bucket/DBS/data/20231214_F48_squid_Neuropixels_record_1205_TO_PRO_3/',
          '/home/sam/bucket/DBS/data/20231218_F49_squid_Neuropixels_record_1207_TO_PRO_3/',
          '/home/sam/bucket/DBS/data/20240112_F55_squid_Neuropixels_record_1211_TO_PRO_3/',
          '/home/sam/bucket/DBS/data/20240119_F56_squid_Neuropixels_record_1214_TO_PRO_3/'  #check mismatch between atlas and brain
          ]


all_v=[]
all_d=[]
for dataset in datasets:
    vol=load_nifti(dataset + 'analysis/raw/692BP_10um.nii')
    atlas_10=load_nifti(dataset + 'analysis/transformed_inv/atlas_trans_seg10.nii.gz') 
    
    medulla_mask=atlas_10==1
   # medulla_mask = binary_erosion(medulla_mask, ball(5)) 

    dt1 = distance_transform_edt(atlas_10 != 5.) 
    grad_z, grad_y, grad_x = np.gradient(dt1)*(atlas_10==1) 
    dt1=dt1*(medulla_mask) # outer granual layer
    binary_volume=binarize_vol(vol, medulla_mask)
  
    # Label all connected components in the volume
    labels = measure.label(binary_volume)
    
    # Remove small objects; 'min_size' is the minimum size threshold
    min_size = 10000  # You can adjust this value based on your needs
    cleaned_volume = morphology.remove_small_objects(labels, min_size=min_size)
    
    # Convert labels back to binary format (True for object, False for background)
    cleaned_binary_volume = cleaned_volume > 0


    dist_trans=distance_transform_edt(cleaned_binary_volume)
    skeleton=skeletonize_3d(binary_volume)
    
  
    for_dist = dist_trans * (skeleton>0)  
    for_dist=4/3*np.pi*np.power(for_dist,3)*10 #volume in um
    
    mask = (for_dist> 0) & (dt1 > 0)
    filtered_dist_trans = for_dist*mask

    cube_size = (50,50,50)  # Define the cube size
    means = calculate_nonzero_means(filtered_dist_trans, cube_size)
    means_d=calculate_nonzero_means(dt1, cube_size)
    
    distances=means_d.ravel()
    volumes=means.ravel()
    good_d=np.logical_and(distances>0,volumes>0) 
    distances=distances[good_d]
    volumes=volumes[good_d]
  
    all_v.append(volumes)
    all_d.append(distances)
    
volumes=np.concatenate(all_v)
distances=np.concatenate(all_d)*10

sorted_indices = np.argsort(distances)
x_sorted = distances[sorted_indices]
y_sorted = volumes[sorted_indices]


alphabet_colors = [
    "#F0A0FF", "#0075DC", "#993F00", "#4C005C", "#191919", "#005C31",
    "#2BCE48", "#FFCC99", "#808080", "#94FFB5", "#8F7C00", "#9DCC00",
    "#C20088", "#003380", "#FFA405", "#FFA8BB", "#426600", "#FF0010",
    "#5EF1F2", "#00998F", "#E0FF66", "#740AFF", "#990000", "#FFFF80",
    "#FFE100", "#FF5005"
]

alphabet_colors = [alphabet_colors[i] for i in [1, 6, 0, 12, 14]]

custom_cmap = ListedColormap(alphabet_colors)

plt.figure(figsize=(4,3))
for ind in range(0,len(all_v)):
    
    curr_d=all_d[ind]*10
    curr_v=all_v[ind]
    colour = custom_cmap(ind / len(all_v))
    plt.scatter(curr_d,curr_v, 10,color=colour,alpha=0.5)
    

p = Polynomial.fit(x_sorted,y_sorted, deg=5)  # Adjust degree as necessary
x_line = np.linspace(min(x_sorted),950, 300)
y_line = p(x_line)

plt.plot(x_line, y_line, 'k',linewidth=1.5)

plt.xlabel('distance from the iGL (μm)')
plt.ylabel('mean island volume (μm^3)')

plt.ylim([0 ,1950])
plt.tight_layout()
plt.show()   
          
    
plt.savefig('treeGrowth.pdf', bbox_inches='tight', transparent=True)



#%% comparing islands after warping

atlas_10, _ = nrrd.read('/home/sam/bucket/DBS/atlas/S_lessoniana_OL_v0.3/segmentation/Segmentation.seg.nrrd')
atlas_10 = np.swapaxes(atlas_10, 0, 2)

datasets=['/home/sam/bucket/DBS/data/20231208_F47_squid_NPrec_1130/',
          '/home/sam/bucket/DBS/data/20231214_F48_squid_Neuropixels_record_1205_TO_PRO_3/',
          '/home/sam/bucket/DBS/data/20231218_F49_squid_Neuropixels_record_1207_TO_PRO_3/',
          '/home/sam/bucket/DBS/data/20240112_F55_squid_Neuropixels_record_1211_TO_PRO_3/']#,
         # '/home/sam/bucket/DBS/data/20240119_F56_squid_Neuropixels_record_1214_TO_PRO_3/'  #check mismatch between atlas and brain
         # ]


binary_volumes=[]
shapes=[]
for dataset in datasets:

    vol=load_nifti(dataset + '/analysis/transformed/transformed_692BP_10um.nii')
    atlas_tmp=atlas_10[0:vol.shape[0],0:vol.shape[1],0:vol.shape[2]] #
    medulla_mask=atlas_tmp==1
    bv=binarize_vol(vol, medulla_mask)
    binary_volumes.append(bv)
    shapes.append(bv.shape)
    
iou=np.zeros((len(binary_volumes),len(binary_volumes)))
for i,ds1 in enumerate(binary_volumes):
    for j,ds2 in enumerate(binary_volumes):
        iou[i,j] = compute_iou_within_mask(ds1,ds2,medulla_mask)
        
# Flatten the correlation matrix, excluding the diagonal and duplicate values
upper_tri = np.triu(iou,k=1).ravel()
upper_tri =upper_tri [upper_tri >0.]

mean_val = np.mean(upper_tri)
std_val = np.std(upper_tri)

print(f"Mean: {mean_val:.6f}")
print(f"Standard Deviation: {std_val:.6f}")

#Mean: 0.300680
#Standard Deviation: 0.005383


# Plotting the histogram of correlation values
bins = np.linspace(0, 1, num=51)
plt.figure()
plt.hist(upper_tri, bins=bins, edgecolor='k')
plt.title('Histogram of IOU Values')
plt.xlabel('Jaccard index')
plt.ylabel('Frequency')
plt.show()

    
#%% 
from skimage.transform import resize
from skimage.morphology import disk

datasets=['/home/sam/bucket/DBS/data/20231208_F47_squid_NPrec_1130/',
          '/home/sam/bucket/DBS/data/20231214_F48_squid_Neuropixels_record_1205_TO_PRO_3/']


binary_volumes=[]
shapes=[]
for dataset in datasets:

    img=np.load(dataset + 'img_650.npy')
    medulla_mask=np.ones_like(img)#atlas_tmp[130]>0
    bv=binarize_vol(img, medulla_mask,filt1=21*5,filt2=61*5)
    binary_volumes.append(bv)
    shapes.append(bv.shape)
    
    
Y,X=binary_volumes[0].shape
composite_image = np.zeros((Y,X,3),dtype=np.uint8)

# Set the red and blue channels for magenta
zplane=130


composite_image[:, :, 0] = binary_volumes[1] * 255  # Red
composite_image[:, :, 2] = binary_volumes[1]* 255  # Blue
composite_image[:, :, 1] = binary_volumes[0] * 255  # Green


# Slice and rotate the image 90° counterclockwise
img = composite_image


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm



pixperum = 2.5                       # 2.5 pixels / µm  (⇢ 0.4 µm / pixel)
crop      = img[700:1000, 700:-90]       # grab ROI
rot_crop  = np.rot90(crop)           # 90 ° CCW


fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
ax.imshow(rot_crop, cmap='gray')
ax.axis('off')                       # remove ticks & frame

# ── scale‑bar ──────────────────────────────────────────────────────────
bar_um = 200                         # visible bar length in µm
bar_px = bar_um / pixperum
font   = fm.FontProperties(size=6)

scalebar = AnchoredSizeBar(
    ax.transData, bar_px, f'{bar_um} µm',
    loc='lower right', pad=0.2, color='w',
    frameon=False, size_vertical=bar_px * 0.05,
    fontproperties=font
)
ax.add_artist(scalebar)
# ───────────────────────────────────────────────────────────────────────

plt.tight_layout()
plt.show()

plt.savefig('non_repeating_islands.pdf', bbox_inches='tight', transparent=True)


#%% New diI injection analysis
import napari
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from skimage.transform import resize


dataset='/home/sam/bucket/DBS/data/20250314_T01_squid_DiI_2_1200um_20250226-0304_TO_PRO_3/analysis2/'
#dataset='/home/sam/bucket/DBS/data//20231114_F38_squid_52dph_DiI_inject_inOL_1101_TO_PRO_3/analysis2/'
#dataset='/home/sam/bucket/DBS/data/20240530_F84_squid_DiI_injected0513_1000um_2/analysis2/'

tracer=load_nifti(dataset + 'raw/593BP_10um.nii')
atlas=load_nifti(dataset + '/transformed_inv/transformed_Segmentation.seg.nii.gz').astype('uint8')
mask=load_nifti(dataset + '/needle_mask/Segmentation.nii.gz').astype('uint8')


ds_size=[int(tracer.shape[0]),int(tracer.shape[1]),int(tracer.shape[2])]


# tracer_downscaled = resize(
#     tracer,
#     output_shape=ds_size,
#     order=1,            # linear interpolation; use 0 for nearest-neighbor
#     mode='edge',        # or 'constant', 'reflect' depending on needs
#     preserve_range=True # keeps original value range (no normalization)
# )

mask_downscaled = resize(
    mask,
    output_shape=ds_size,
    order=0,            # linear interpolation; use 0 for nearest-neighbor
    mode='edge',        # or 'constant', 'reflect' depending on needs
    preserve_range=True # keeps original value range (no normalization)
)

atlas_downscaled = resize(
    atlas,
    output_shape=ds_size,
    order=0,            # linear interpolation; use 0 for nearest-neighbor
    mode='edge',        # or 'constant', 'reflect' depending on needs
    preserve_range=True # keeps original value range (no normalization)
)

save_nifti(tracer, dataset + 'sam/tracer_ds.nii.gz', 10,10,10)
save_nifti(mask_downscaled, dataset + 'sam/mask_ds.nii.gz', 10,10,10)
save_nifti(atlas_downscaled, dataset + 'sam/atlas_ds.nii.gz', 10,10,10)


#%% injection following gradient and spreading non isotropically
from skimage import feature
import joblib
import scipy.ndimage
import cv2
import numpy as np
import scipy.ndimage
from skimage import measure
from scipy.ndimage import gaussian_filter
from scipy.ndimage import distance_transform_edt, center_of_mass, binary_dilation
import scipy.ndimage as ndi



def rotate_image_around_point(image, point1, point2):

    # Calculate the rotation angle needed
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)   # Subtract 90 to align vertically

    # Rotate the image around point1
    # Calculate the center relative to point1
    height, width = image.shape[:2]

    # Calculate new center after rotation to keep the image center
    rot_center = (width // 2, height // 2)

    # Translate the image so point1 is at the center
    shift_x = rot_center[0] - point1[0]
    shift_y = rot_center[1] - point1[1]
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    translated_image = cv2.warpAffine(image, translation_matrix, (width, height))

    # Rotate the image
    rotated_image = scipy.ndimage.rotate(translated_image, -angle_deg, reshape=True)

    return rotated_image

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional

# ────────────────────────────────────────────────────────────────────
# 1)  Build and (optionally) smooth a 3‑D polar histogram
# ────────────────────────────────────────────────────────────────────
def create_polar_histogram_3d(
    binary_volume: np.ndarray,
    center: Optional[Tuple[int, int, int]] = None,
    *,
    num_radius_bins: int = 20,
    num_theta_bins: int = 36,
    max_distance: Optional[float] = 100,
    smooth_sigma: Optional[float] = None
):
    """
    Convert a 3‑D binary volume into a polar (θ, r) histogram.

    Returns
    -------
    norm_hist : (N_theta, N_r) ndarray
        Smoothed (if requested) and column‑normalised histogram.
    r_bins    : (N_r + 1,) ndarray
    theta_bins: (N_theta + 1,) ndarray
    """
    # ----- gather coordinates of “1” voxels -------------------------
    depth, height, width = binary_volume.shape
    if center is None:
        center = (depth // 2, height // 2, width // 2)

    z_idx, y_idx, x_idx = np.where(binary_volume == 1)
    x = x_idx - center[2]
    y = y_idx - center[1]
    z = z_idx - center[0]

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)

    # ----- optional radial cutoff -----------------------------------
    if max_distance is not None:
        keep = r <= max_distance
        r, theta = r[keep], theta[keep]
    else:
        max_distance = r.max()

    # ----- binning ---------------------------------------------------
    r_bins = np.linspace(0, max_distance, num_radius_bins + 1)
    theta_bins = np.linspace(-np.pi, np.pi, num_theta_bins + 1)

    hist, _, _ = np.histogram2d(theta, r, bins=[theta_bins, r_bins])

    # ----- smooth RAW counts (if requested) -------------------------
    if smooth_sigma is not None and smooth_sigma > 0:
        hist = gaussian_filter(hist, sigma=smooth_sigma, mode="wrap")

    # ----- column‑wise normalisation (PDF per radius bin) -----------
    col_sums = hist.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_hist = np.where(col_sums > 0, hist / col_sums, 0)

    return norm_hist, r_bins, theta_bins



import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
# no inset_axes needed any more

def plot_polar_histogram(
    histogram: np.ndarray,
    r_bins_px: np.ndarray,        # radial bin edges in *pixels*
    theta_bins: np.ndarray,       # angular bin edges in radians
    *,
    px_per_um: float = 5.0,       # pixels per µm
    vmax: Optional[float] = None
):
    """
    Display a column-normalised polar histogram as a polar heat-map
    with a semi-transparent bull’s-eye grid.

    Parameters
    ----------
    histogram : (N_r, N_theta) array
        Column-normalised counts.
    r_bins_px : (N_r+1,) array
        Radial bin edges in pixels.
    theta_bins : (N_theta+1,) array
        Angular bin edges in radians.
    px_per_um : float, optional
        Pixels per micrometre (default 5).
    vmax : float, optional
        Upper colour-scale limit.  Defaults to histogram.max().
    """
    # ── Convert radial edges to µm ──────────────────────────────────
    r_bins_um = r_bins_px / px_per_um
    max_r     = r_bins_um[-1]

    # ── Figure 35 mm × 35 mm ───────────────────────────────────────
    mm = 1 / 25.4                                   # inches per mm
    fig, ax = plt.subplots(
        subplot_kw={'projection': 'polar'},
        figsize=(35 * mm, 35 * mm),
        dpi=300
    )

    # ── θ-wrap for seamless first/last column ──────────────────────
    hist_ext  = np.vstack([histogram, histogram[:1, :]])           # wrap
    theta_ext = np.append(theta_bins,
                          theta_bins[-1] + (theta_bins[1] - theta_bins[0]))

    Θ, R = np.meshgrid(theta_ext, r_bins_um)

    # Colour limits
    vmin = hist_ext.min()
    vmax_use = vmax if vmax is not None else hist_ext.max()

    pc = ax.pcolormesh(
        Θ, R, hist_ext.T,
        cmap='jet',
        shading='auto',
        vmin=vmin,
        vmax=vmax_use
    )

    # ── Axis appearance ────────────────────────────────────────────
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_xticks([])                              # remove angle ticks

    ax.set_rticks([max_r])                         # one radial tick
    ax.set_yticklabels([f"{max_r:.0f} µm"])
    ax.set_rlim(0, max_r)
    ax.grid(False)                                 # suppress default grid

    # ── Custom bull’s-eye grid (α = 0.5) ───────────────────────────
    spokes = np.linspace(0, 2*np.pi, 12, endpoint=False)   # every 30°
    for θ in spokes:
        ax.plot([θ, θ], [0, max_r],
                color='black', lw=0.4, alpha=0.5, zorder=3)

    circles = np.linspace(max_r/4, max_r, 4)               # ¼,½,¾,1×R
    θ_dense = np.linspace(0, 2*np.pi, 720)
    for r in circles:
        ax.plot(θ_dense, np.full_like(θ_dense, r),
                color='black', lw=0.4, alpha=0.5, zorder=3)

    ax.set_title('Polar histogram (column normalised)', pad=12)

    # ── vmin / vmax annotation bottom-left ─────────────────────────
    ax.text(
        0.02, 0.02,
        f"vmin = {vmin:.2g}\nvmax = {vmax_use:.2g}",
        transform=ax.transAxes,
        fontsize=6,
        va='bottom', ha='left',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7)
    )

    plt.tight_layout()
    plt.show()



def rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the minimal‑angle 3×3 rotation matrix that brings vector *b* onto vector *a*."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(b, a)
    c = np.dot(b, a)
    s = np.linalg.norm(v)
    if s == 0:          # already parallel
        return np.eye(3)
    vx = np.array([[    0, -v[2],  v[1]],
                   [ v[2],     0, -v[0]],
                   [-v[1],  v[0],     0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)



def align_medulla(volume, p1, p2,
                  medulla_mask=None,
                  buffer=120,
                  upsample=2,          # 2× in each axis  (nearest-neighbour)
                  clip_dist=100,       # keep only a 200×200×200 cube (before rotation)
                  order=0):
    """
    Align `p1→p2` with +Y, preserving tiny binary structures.

    Parameters
    ----------
    volume : ndarray  (bool / uint8)   original stack
    p1, p2 : array-like (3,)           points in *original* voxel coords
    medulla_mask : ndarray or None     apply mask before processing
    buffer : int                       symmetric padding before any shift
    upsample : int                     integer zoom factor (>1 = safer)
    clip_dist : int                    keep voxels with |x|,|y|,|z| ≤ clip_dist
    order : int                        scipy interpolation order (keep 0!)

    Returns
    -------
    vol_rot : ndarray (same dtype as `volume`)
              padded, upsampled, clipped, rotated, *binary* stack
    """
    # ---------------------------------------------------------------
    # 0 — binary start
    # ---------------------------------------------------------------
    vol = volume.astype(bool).copy()
    if medulla_mask is not None:
        vol &= medulla_mask

    # ---------------------------------------------------------------
    # 1 — pad so nothing clips during the initial shift
    # ---------------------------------------------------------------
    vol = np.pad(vol, [(buffer, buffer)] * 3, mode='constant')
    p1_pad = np.asarray(p1) + buffer
    p2_pad = np.asarray(p2) + buffer

    # ---------------------------------------------------------------
    # 2 — translate so p1 is at the exact centre
    # ---------------------------------------------------------------
    centre = np.array(vol.shape) // 2
    shift  = centre - p1_pad
    vol    = ndi.shift(vol, shift=shift, order=0, prefilter=False)
    # (p2 position after shift—not needed later, but kept for clarity)
    # p2_shift = p2_pad + shift

    # ---------------------------------------------------------------
    # 3 — upsample (nearest-neighbour keeps mask crisp)
    # ---------------------------------------------------------------
    if upsample > 1:
        vol = ndi.zoom(
            vol, zoom=upsample,
            order=0,
            mode="grid-constant",  # <<< suppresses the warning
            grid_mode=True
        )

    # ---------------------------------------------------------------
    # 4 — clip to a cube ±clip_dist voxels (in the *upsampled* grid)
    # ---------------------------------------------------------------
    if clip_dist is not None:
        clip = int(clip_dist * upsample)
        c    = np.array(vol.shape) // 2       # integer centre
        xs0  = int(max(c[0] - clip, 0))
        xs1  = int(min(c[0] + clip + 1, vol.shape[0]))
        ys0  = int(max(c[1] - clip, 0))
        ys1  = int(min(c[1] + clip + 1, vol.shape[1]))
        zs0  = int(max(c[2] - clip, 0))
        zs1  = int(min(c[2] + clip + 1, vol.shape[2]))
        vol  = vol[xs0:xs1, ys0:ys1, zs0:zs1]

    # ---------------------------------------------------------------
    # 5 — build rotation that sends  p1→p2  onto +Y
    #       (source, destination  **in that order**)
    # ---------------------------------------------------------------
    R = rotation_matrix_from_vectors(np.array([0, 1, 0]),p2 - p1)
    M = R.T                               # inverse for affine_transform

    # ---------------------------------------------------------------
    # 6 — rotate about the *current* centre
    # ---------------------------------------------------------------
    centre = np.array(vol.shape, dtype=float) / 2
    offset = centre - M @ centre
    vol_rot = ndi.affine_transform(vol.astype(float), M,
                                   offset=offset,
                                   order=order,
                                   mode='constant', cval=0)

    # ---------------------------------------------------------------
    # 7 — re-binarise and heal 1-voxel breaches
    # ---------------------------------------------------------------
    vol_rot = (vol_rot > 0.5)
    #vol_rot = ndi.binary_closing(vol_rot, iterations=1)

    return vol_rot.astype(volume.dtype)



# dataset='/home/sam/bucket/DBS/data/20250314_T01_squid_DiI_2_1200um_20250226-0304_TO_PRO_3/analysis2/sam/'
# closest_pt=np.array([202,52,104])
# injection_site=np.array([179,137,135])
# call_sign='T01'


dataset='/home/sam/bucket/DBS/data/20231114_F38_squid_52dph_DiI_inject_inOL_1101_TO_PRO_3/analysis2/sam/'
closest_pt=np.array([254,52,139])
injection_site=np.array([180,133,162])
call_sign='F38'

# dataset='/home/sam/bucket/DBS/data/20240530_F84_squid_DiI_injected0513_1000um_2/analysis2/sam/'
# closest_pt=np.array([320,59,130])
# injection_site=np.array([309,160,212])
# call_sign='F84'

tracer=load_nifti(dataset + 'tracer_ds.nii.gz')
atlas_10=load_nifti(dataset + 'atlas_ds.nii.gz').astype('uint8')
mask=load_nifti(dataset + 'mask_ds.nii.gz').astype('uint8')


medulla_mask=atlas_10==1
igl=atlas_10==5

sigma1 = .1
sigma2 = 5


blob_img=np.zeros_like(tracer).astype('bool')
for i,slice in tqdm(enumerate(tracer)):
    # Apply Gaussian filters
    slice[slice>1000]=1000
    blurred1 = gaussian_filter(slice, sigma=sigma1)
    blurred2 = gaussian_filter(slice, sigma=sigma2)
    
    dog = blurred1 - blurred2
    dog[mask[i]==1]=0
    dog[medulla_mask[i]!=1]=0
    blob_img[i]=dog>100


rv = align_medulla(blob_img, injection_site, closest_pt,  medulla_mask)

hist, r_edges, th_edges = create_polar_histogram_3d(
    rv>0,
    num_radius_bins=36, 
    num_theta_bins=180,
    smooth_sigma=2,   # or None / 0 to disable smoothing
    max_distance=200
)

plot_polar_histogram(hist, r_edges, th_edges, px_per_um=0.2)

plt.savefig('injection_spread_' + call_sign + '.pdf', bbox_inches='tight', transparent=True)

# viewer = napari.Viewer()
# viewer.add_image(blob_img, name='Mask 3D') 
# viewer.add_image(tracer, name='Mask 3D') 
# napari.run()

# viewer = napari.Viewer()
# viewer.add_image(rv, name='Mask 3D')  # or viewer.add_image if it's not binary
# napari.run()


#%% Rudi single neurons

"""
Polar scatter of cell processes.
• θ  = XY direction of each 3‑D point
• r  = XY distance from the origin
• colour = smoothed point density in (θ, r) space
"""

import numpy as np
import matplotlib.pyplot as plt
import glob, os
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

PX2UM = 1

# ---------------------------------------------------------------------
# 1)  Parent folder containing one sub‑folder per cell  ───────────────
# ---------------------------------------------------------------------
directory = '/home/sam/bucket/sam/OL/rudi_neurons2/'

# ---------------------------------------------------------------------
# 2)  Load coordinates, convert to polar  ─────────────────────────────
# ---------------------------------------------------------------------
all_angles = []
all_radii  = []

for  cell_file in glob.glob(os.path.join(directory, "*.npy")):

    cell = np.load(cell_file)         # shape (N, 3)
    xy = cell[:, :2]                      # (x, y)
    r     = np.linalg.norm(xy, axis=1)    # radius
    theta = np.arctan2(xy[:, 1], xy[:, 0])  # angle in radians

    all_angles.append(theta)
    all_radii.append(r)

if not all_angles:
    raise RuntimeError(f"No .npy files found under {directory}")

angles = np.concatenate(all_angles)
radii  = np.concatenate(all_radii)




import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar      # pip install matplotlib-scalebar
import numpy as np
from scipy.ndimage import gaussian_filter

# ------------------------------------------------------------------
# 1)  HISTOGRAM PARAMETERS  ----------------------------------------
# ------------------------------------------------------------------
N_ANGLE_BINS  = 180          # 2° per bin   → 180 bins cover (–π, π]
N_RADIUS_BINS = 36           # linear radius bins
GAUSS_SIGMA   = 2            # Gaussian blur (in bins)

UM2PX = 1               # µm per pixel ( ⇒ 1 px = 0.0791 µm )

# Bin edges
angle_edges  = np.linspace(-np.pi, np.pi,  N_ANGLE_BINS + 1)
radius_edges = np.linspace(0, radii.max(), N_RADIUS_BINS + 1)

# ------------------------------------------------------------------
# 2)  BUILD, SMOOTH & NORMALISE 2-D HISTOGRAM  ---------------------
# ------------------------------------------------------------------
hist, _, _    = np.histogram2d(angles, radii, bins=[angle_edges, radius_edges])
hist_smoothed = gaussian_filter(hist, sigma=GAUSS_SIGMA, mode=("wrap", "nearest"))
hist_density  = hist_smoothed / hist_smoothed.sum()
hist_plot     = hist_density.T        # pcolormesh expects (r, θ)

# ------------------------------------------------------------------
# 3)  POLAR PLOT  ---------------------------------------------------
# ------------------------------------------------------------------
fig = plt.figure(figsize=(3, 3), dpi=150)
ax  = fig.add_subplot(111, projection="polar")

pcm = ax.pcolormesh(
    angle_edges,
    radius_edges,
    hist_plot,
    cmap="jet",
    shading="auto"
)
ax.grid(False)

# ---- Aesthetics ---------------------------------------------------
ax.set_theta_zero_location("E")          # 0° points right
ax.set_theta_direction(1)                # clockwise positive
ax.set_rlim(0, radius_edges[-1])         # radial limits

# Only outer radial tick
ax.set_rticks([radius_edges[-1]])
ax.set_yticklabels([f"{radius_edges[-1] /UM2PX:.0f} µm"])   # outer radius in µm


# ── Custom bull’s-eye grid (α = 0.5) ───────────────────────────
spokes = np.linspace(0, 2*np.pi, 12, endpoint=False)   # every 30°
for θ in spokes:
    ax.plot([θ, θ], [0, radius_edges[-1]],
            color='white', lw=0.4, alpha=0.5, zorder=3)

circles = np.linspace(50, 250, 5)               # ¼,½,¾,1×R
θ_dense = np.linspace(0, 2*np.pi, 720)
for r in circles:
    ax.plot(θ_dense, np.full_like(θ_dense, r),
            color='white', lw=0.4, alpha=0.5, zorder=3)


# ---- µm scale-bar -------------------------------------------------
# Choose a convenient bar length (e.g. 10 µm)
bar_len_um  = 50                         # µm
bar_len_px  = bar_len_um / PX2UM


# ---- vmin / vmax annotation --------------------------------------
vmin, vmax = pcm.get_clim()
fig.text(0.02, 0.02, f"vmin = {vmin:.3e}\nvmax = {vmax:.3e}",
         ha="left", va="bottom", fontsize=6)

ax.set_title("Polar density histogram")
plt.tight_layout()
plt.show()



plt.savefig('cellHist.pdf', bbox_inches='tight', transparent=True)





