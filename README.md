# Mano_et_al_2026

Analysis code and figure-generation notebooks for Mano et al. (2026), a study of how the optic lobe of the bigfin reef squid (*Sepioteuthis lessoniana*) processes light intensity and polarization. The code covers calcium imaging, electrophysiology, optic-lobe anatomy, and underwater measurements of light intensity and polarization. [Read the bioRxiv preprint](https://doi.org/10.64898/2026.01.08.698312).

## Toy example for electrophysiology receptive fields

Open [`EXAMPLE_CODE_RF_ANALYSIS.ipynb`](ephys_analysis/Example_Code/EXAMPLE_CODE_RF_ANALYSIS.ipynb) in Jupyter to view the code and saved figures. The notebook aligns spikes from a single probe to sparse-noise stimulus frames, maps responses to bright (ON) and dark (OFF) stimuli, plots peristimulus time histograms, and estimates receptive-field centers and sizes. [Analysis functions](ephys_analysis/Example_Code/RF_FUNCTIONS.py).

### Bundled data

All five NumPy inputs are in [`ephys_analysis/Example_Code/`](ephys_analysis/Example_Code/) alongside the notebook. To download them individually on GitHub, open a file below and select **Download raw file**.

| Bundled data file | Contents |
| --- | --- |
| [`spike_t_matrix.npy`](ephys_analysis/Example_Code/spike_t_matrix.npy) | Spike times in seconds for 24 units, stored as a NumPy object array with one entry per unit |
| [`g_ids.npy`](ephys_analysis/Example_Code/g_ids.npy) | Unit IDs in the same order as the spike-time array |
| [`g_channels.npy`](ephys_analysis/Example_Code/g_channels.npy) | Peak probe channel for each unit |
| [`stim_windows.npy`](ephys_analysis/Example_Code/stim_windows.npy) | Start and end times in seconds for 16,800 stimulus frames, in presentation order |
| [`sparse_noise_matrix.npy`](ephys_analysis/Example_Code/sparse_noise_matrix.npy) | The 16,800 sparse-noise frames in the same order as `stim_windows.npy`, on a 12 × 21 grid (32 MiB) |

### Rerun the notebook

From the repository root, install the Python dependencies and launch JupyterLab from the example folder so the notebook can load all five inputs by filename.

```bash
python -m pip install jupyterlab numpy scipy pandas matplotlib scikit-image tqdm
cd ephys_analysis/Example_Code
jupyter lab EXAMPLE_CODE_RF_ANALYSIS.ipynb
```

Select **Run → Run All Cells**.

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

### Run a paper analysis

- **Electrophysiology receptive fields.** Start with [`EXTRACT_RF_PROPERTIES.ipynb`](ephys_analysis/Receptive_Field_Analysis/EXTRACT_RF_PROPERTIES.ipynb) for estimates from recordings or with [`RF_ANALYSIS.ipynb`](ephys_analysis/Receptive_Field_Analysis/RF_ANALYSIS.ipynb) for population plots using processed receptive-field measurements.
- **Calcium imaging.** Start with motion correction and denoising, followed by cell detection and fluorescence trace extraction for cell-based analyses. The recording-level and summary notebooks live in the analysis folder for each experiment. [Preprocessing](calcium_scripts/preprocessing/README.md) · [Cell detection and trace extraction](calcium_scripts/ROI_detection/README.md).

Many paper workflows read recordings, stimulus files, atlases, or intermediate results from OIST lab storage. Obtain the inputs for the workflow you want to run, then update its data and output paths for your environment. The calcium imaging batch jobs also use Slurm and the Singularity environments linked above.
