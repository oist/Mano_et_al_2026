#!/bin/bash
#SBATCH -p compute
#SBATCH -t 0-4
#SBATCH -c 24
#SBATCH --mem=128G
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=tomoyuki.mano1@oist.jp

#modules
module load singularity

#### Parameters ####
rootDir=/bucket/ReiterU/OLdev/experiments/20241022_squid_cal520
recName=rec8
rawImg=${rootDir}/2P/${recName}/mcorr_nonrigid/corrected.hdf5
roi=${rootDir}/2P/${recName}/ROI_tomo_renamed.zip
tdmsPath=${rootDir}/sync/${recName}.tdms
badFrames=${rootDir}/2P/${recName}/mcorr_nonrigid/QC/bad_frames.npy

saveDir=/bucket/ReiterU/OLdev/analysis/20250327_OGL_receptive_field/20241022_rec8
####################

tempDir=$(mktemp -d /flash/ReiterU/tmp/tomo.XXXXXX)

# main
singularity exec --bind `pwd`:/workdir /bucket/ReiterU/singularity/caiman_v2.sif \
python /home/t/tomoyuki-mano1/projects/OL_calcium/scripts/manual_calcium/compute_dff_v7.py \
--img_path ${rawImg} \
--roi_path ${roi} \
--tdms_path ${tdmsPath} \
--save_dir ${tempDir} \
--dataset_name "mov" \
--max_workers 24 \
--pad_before 30 \
--pad_after 30 \
--input_sampling_freq 5000 \
--output_sampling_freq 1000 \
--baseline_window 200 \
--baseline_percentile 0.2 \
--bad_frames ${badFrames} \
--neuropil_radius 30 \
--neuropil_r 0.5

echo "Copying files to /bucket..."
if [[ ! -e ${saveDir} ]]; then
    ssh deigo mkdir -p ${saveDir}
fi
scp -r ${tempDir}/* deigo:${saveDir}
echo "Done."

echo "Deleting temporary directory..."
rm -r ${tempDir}
echo "Done."
