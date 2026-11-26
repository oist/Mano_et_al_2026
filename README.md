# Mano_et_al_2026

Analysis code and figure-generation notebooks for Mano et al. (2026), a study of squid vision. The repository covers calcium imaging and electrophysiology of the squid visual system, optic-lobe anatomy, and measurements of light intensity and polarization in the field.

## Start here with the toy data and analysis

The toy dataset and analysis are in [`ephys_analysis/Example_Code/`](ephys_analysis/Example_Code/). Open [`EXAMPLE_CODE_RF_ANALYSIS.ipynb`](ephys_analysis/Example_Code/EXAMPLE_CODE_RF_ANALYSIS.ipynb) to see how spike times from a single probe are aligned to sparse-noise stimulus frames. The notebook builds ON and OFF receptive-field maps, plots peristimulus time histograms, and estimates receptive-field centers and sizes. The analysis functions are in [`RF_FUNCTIONS.py`](ephys_analysis/Example_Code/RF_FUNCTIONS.py).

| Bundled data file | Contents |
| --- | --- |
| [`spike_t_matrix.npy`](ephys_analysis/Example_Code/spike_t_matrix.npy) | Spike times in seconds for 24 units |
| [`g_ids.npy`](ephys_analysis/Example_Code/g_ids.npy) | Unit IDs in the same order as the spike-time array |
| [`g_channels.npy`](ephys_analysis/Example_Code/g_channels.npy) | Peak probe channel for each unit |
| [`stim_windows.npy`](ephys_analysis/Example_Code/stim_windows.npy) | Start and end times in seconds for the stimulus frames |

**Required to rerun the example.** The notebook also reads `sparse_noise_matrix.npy`, which contains the sparse-noise frames corresponding to `stim_windows.npy`. The stimulus matrix is missing from this repository. With the matching file in `ephys_analysis/Example_Code/`, open the notebook in Jupyter with that directory as the working directory. The Python dependencies are NumPy, SciPy, pandas, Matplotlib, scikit-image, and tqdm.

## Paper analyses and supporting code

| Location | Contents |
| --- | --- |
| [`ephys_analysis/`](ephys_analysis/) | Electrophysiology analyses and plots, including [receptive-field mapping and property extraction](ephys_analysis/Receptive_Field_Analysis/), [direction selectivity](ephys_analysis/Direction_Selectivity_Analysis/), and [unit-quality plots and SpikeGLX file readers](ephys_analysis/Ephys_Processing/). The [figure script](ephys_analysis/ephys_analysis.py) also covers contrast sensitivity, polarization responses, and local field potentials. The [atlas notebook](ephys_analysis/ATLAS_SUPP_PLOTS.ipynb) generates supplementary atlas plots. |
| [`calcium_analysis/`](calcium_analysis/) | Recording-level and summary notebooks for the optic lobe’s outer and inner granular layers (OGL and IGL). These cover receptive fields and retinotopy ([OGL](calcium_analysis/OGL_receptive_field/), [IGL](calcium_analysis/IGL_receptive_field/)), direction and orientation selectivity ([OGL](calcium_analysis/DS_OS_OGL/), [IGL](calcium_analysis/DS_OS_IGL/)), [response clustering](calcium_analysis/OGL_IGL_clustering/), [spontaneous activity](calcium_analysis/OGL_IGL_spontaneous_activity/), and [IGL activity waves](calcium_analysis/IGL_wave/). This directory also includes [supplementary movie notebooks](calcium_analysis/sup_movies/). |
| [`calcium_scripts/`](calcium_scripts/) | Python and Slurm jobs for [motion correction and DeepInterpolation denoising](calcium_scripts/preprocessing/README.md), [cell detection and ΔF/F extraction](calcium_scripts/ROI_detection/README.md), and [ImageJ ROI and file-conversion utilities](calcium_scripts/utils/). |
| [`anatomical_analysis/`](anatomical_analysis/) | Analysis of optic-lobe image volumes, including segmentation, layer organization, dye injections, and single-neuron reconstructions. |
| [`fieldwork_analysis/`](fieldwork_analysis/) | Image processing and analysis for field measurements of intensity and polarization, with tools for drawing contours and marking points. |
| [`monitor_cal.ipynb`](monitor_cal.ipynb) | Display calibration and plotting. Additional [LCD calibration and calcium-response notebooks](calcium_analysis/sup_calcium_LCD/) are under `calcium_analysis/`. |
| [`singularity/`](singularity/) | Container definitions and build instructions for [calcium imaging analysis](singularity/ophys_v3/README.md) and [DeepInterpolation](singularity/deepinterpolation/README.md). |

Many paper analyses reference recordings and intermediate files on OIST lab storage. Running those workflows requires access to the corresponding inputs and updating the data and cluster paths for your environment.

## Figure formatting

Use widths of 184 mm for two-column figures and 92 mm for single-column figures, with 7-point Arial. The Matplotlib settings below embed TrueType fonts for editing PDFs in Adobe Illustrator. Replace the font path with the location of Arial on your system.

```python
from matplotlib import font_manager
import matplotlib as mpl

font_manager.fontManager.addfont('/bucket/ReiterU/common/fonts/arial.ttf')
mpl.rcParams.update({
    'font.size': 7,
    'axes.titlesize': 7,
    'legend.fontsize': 7,
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'xtick.major.size': 2,
    'xtick.minor.size': 2,
    'xtick.minor.width': 0.5,
    'ytick.major.width': 0.5,
    'ytick.major.size': 2,
    'ytick.minor.size': 2,
    'ytick.minor.width': 0.5,
})
```
