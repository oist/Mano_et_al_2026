#!/bin/bash
#SBATCH -p compute
#SBATCH -t 0-6
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=tomoyuki.mano1@oist.jp

#modules
module load singularity

#### Parameters ####
rootDir=/bucket/ReiterU/OLdev/analysis/20250623_calcium_test_data/20250317_rec6
rawImg=${rootDir}/mcorr_nonrigid/corrected.hdf5
roi=${rootDir}/ROI_renamed.zip
tdmsPath=${rootDir}/sync.tdms
# badFrames=${rootDir}/mcorr_nonrigid/QC/bad_frames.npy
saveDir=${rootDir}/dff

singularityImage=/bucket/ReiterU/singularity/ophys_v3.sif
scriptDir=/home/t/tomoyuki-mano1/projects/squid_vision_paper_draft/calcium_scripts/ROI_detection
####################

tempDir=$(mktemp -d /flash/ReiterU/tmp/tomo.XXXXXX)

# main
singularity exec --bind `pwd`:/workdir ${singularityImage} \
python ${scriptDir}/compute_dff_v7.py \
--img_path ${rawImg} \
--roi_path ${roi} \
--tdms_path ${tdmsPath} \
--save_dir ${tempDir} \
--dataset_name "mov" \
--max_workers 32 \
--pad_before 30 \
--pad_after 30 \
--input_sampling_freq 5000 \
--output_sampling_freq 1000 \
--baseline_window 200 \
--baseline_percentile 0.2 \
--neuropil_radius 30 \
--neuropil_r 0.5 \
--no-crop_last_two_frames

echo "Copying files to /bucket..."
if [[ ! -e ${saveDir} ]]; then
    ssh deigo mkdir -p ${saveDir}
fi
scp -r ${tempDir}/* deigo:${saveDir}
echo "Done."

echo "Deleting temporary directory..."
rm -r ${tempDir}
echo "Done."
